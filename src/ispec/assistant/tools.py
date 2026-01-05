from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ispec.assistant.context import person_summary, project_summary
from ispec.db.models import AuthUser, Experiment, ExperimentRun, Person, Project, UserRole
from ispec.schedule.models import ScheduleRequest, ScheduleSlot


TOOL_CALL_PREFIX = "TOOL_CALL"
TOOL_RESULT_PREFIX = "TOOL_RESULT"

CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

SCHEDULE_SLOT_STATUSES = {"available", "booked", "closed"}
SCHEDULE_REQUEST_STATUSES = {"requested", "confirmed", "declined", "cancelled"}


def tool_prompt() -> str:
    """Short tool list for the system prompt."""

    return (
        "Available tools (read-only):\n"
        "- search_projects(query: str, limit: int = 5)\n"
        "- get_project(id: int)\n"
        "- latest_experiments(limit: int = 5)\n"
        "- get_experiment(id: int)\n"
        "- latest_experiment_runs(limit: int = 5)\n"
        "- get_experiment_run(id: int)\n"
        "- search_people(query: str, limit: int = 5)\n"
        "- get_person(id: int)\n"
        "- list_schedule_slots(start: YYYY-MM-DD, end: YYYY-MM-DD, status: str | None = None, limit: int = 50)\n"
        "- list_schedule_requests(limit: int = 20, status: str | None = None)  # admin-only\n"
        "- get_schedule_request(id: int)  # admin-only\n"
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed >= 0 else None
    return None


def _safe_str(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _clamp_int(value: int | None, *, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    return max(min_value, min(max_value, value))


def _safe_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC_TZ)
    return value.astimezone(UTC_TZ)


def _range_bounds_local(start: date, end: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(start, time.min, tzinfo=CENTRAL_TZ)
    end_local = datetime.combine(end, time.max, tzinfo=CENTRAL_TZ)
    return start_local.astimezone(UTC_TZ).replace(tzinfo=None), end_local.astimezone(UTC_TZ).replace(
        tzinfo=None
    )


def _require_admin(user: AuthUser | None) -> str | None:
    if user is None:
        return "Not authenticated."
    if user.role != UserRole.admin:
        return "Admin access required."
    return None


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a TOOL_CALL request from assistant output.

    Expected format (single line):
      TOOL_CALL {"name": "...", "arguments": {...}}
    """

    stripped = (text or "").strip()
    if not stripped.startswith(TOOL_CALL_PREFIX):
        return None

    raw = stripped[len(TOOL_CALL_PREFIX) :].strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments") or payload.get("args") or {}
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(args, dict):
        return None
    return name.strip(), args


def format_tool_result_message(tool_name: str, payload: dict[str, Any]) -> str:
    return f"{TOOL_RESULT_PREFIX} {tool_name} (JSON):\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def run_tool(
    *,
    name: str,
    args: dict[str, Any],
    core_db: Session,
    schedule_db: Session | None = None,
    user: AuthUser | None = None,
) -> dict[str, Any]:
    try:
        if name == "latest_experiments":
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=50)
            rows = core_db.query(Experiment).order_by(Experiment.id.desc()).limit(limit).all()

            experiments: list[dict[str, Any]] = []
            for experiment in rows:
                created = getattr(experiment, "Experiment_CreationTS", None)
                experiments.append(
                    {
                        "id": int(experiment.id),
                        "record_no": experiment.record_no,
                        "name": experiment.exp_Name,
                        "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                        "date": experiment.exp_Date.isoformat() if experiment.exp_Date else None,
                        "project_id": int(experiment.project_id)
                        if experiment.project_id is not None
                        else None,
                        "created": created.isoformat() if created else None,
                        "links": {
                            "ui": f"/experiment/{experiment.id}",
                            "api": f"/api/experiments/{experiment.id}",
                        },
                    }
                )
            return {"ok": True, "tool": name, "result": {"experiments": experiments}}

        if name == "get_experiment":
            experiment_id = _safe_int(args.get("id"))
            if experiment_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            experiment = core_db.get(Experiment, experiment_id)
            if experiment is None:
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"Experiment {experiment_id} not found.",
                }
            created = getattr(experiment, "Experiment_CreationTS", None)
            modified = getattr(experiment, "Experiment_ModificationTS", None)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(experiment.id),
                    "record_no": experiment.record_no,
                    "name": experiment.exp_Name,
                    "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                    "date": experiment.exp_Date.isoformat() if experiment.exp_Date else None,
                    "project_id": int(experiment.project_id)
                    if experiment.project_id is not None
                    else None,
                    "created": created.isoformat() if created else None,
                    "modified": modified.isoformat() if modified else None,
                    "links": {
                        "ui": f"/experiment/{experiment.id}",
                        "api": f"/api/experiments/{experiment.id}",
                    },
                },
            }

        if name == "latest_experiment_runs":
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=50)
            rows = core_db.query(ExperimentRun).order_by(ExperimentRun.id.desc()).limit(limit).all()

            runs: list[dict[str, Any]] = []
            for run in rows:
                created = getattr(run, "ExperimentRun_CreationTS", None)
                runs.append(
                    {
                        "id": int(run.id),
                        "experiment_id": int(run.experiment_id),
                        "run_no": int(run.run_no),
                        "search_no": int(run.search_no),
                        "search_engine": run.search_engine,
                        "search_state": run.search_state,
                        "created": created.isoformat() if created else None,
                        "links": {
                            "ui": f"/experiment-run/{run.id}",
                            "api": f"/api/experiment-runs/{run.id}",
                        },
                    }
                )
            return {"ok": True, "tool": name, "result": {"runs": runs}}

        if name == "get_experiment_run":
            run_id = _safe_int(args.get("id"))
            if run_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            run = core_db.get(ExperimentRun, run_id)
            if run is None:
                return {"ok": False, "tool": name, "error": f"ExperimentRun {run_id} not found."}
            created = getattr(run, "ExperimentRun_CreationTS", None)
            modified = getattr(run, "ExperimentRun_ModificationTS", None)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(run.id),
                    "experiment_id": int(run.experiment_id),
                    "run_no": int(run.run_no),
                    "search_no": int(run.search_no),
                    "search_engine": run.search_engine,
                    "search_state": run.search_state,
                    "created": created.isoformat() if created else None,
                    "modified": modified.isoformat() if modified else None,
                    "links": {
                        "ui": f"/experiment-run/{run.id}",
                        "api": f"/api/experiment-runs/{run.id}",
                    },
                },
            }

        if name == "get_project":
            project_id = _safe_int(args.get("id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            project = core_db.get(Project, project_id)
            if project is None:
                return {"ok": False, "tool": name, "error": f"Project {project_id} not found."}
            return {"ok": True, "tool": name, "result": project_summary(core_db, project)}

        if name == "search_projects":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=20)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}

            matches: list[dict[str, Any]] = []
            project_id = _safe_int(query)
            if project_id is not None:
                project = core_db.get(Project, project_id)
                if project is not None:
                    matches.append(
                        {
                            "id": int(project.id),
                            "title": project.prj_ProjectTitle,
                            "status": project.prj_Status,
                            "current": bool(getattr(project, "prj_Current_FLAG", False)),
                        }
                    )
                return {"ok": True, "tool": name, "result": {"matches": matches}}

            pattern = f"%{query}%"
            rows = (
                core_db.query(Project)
                .filter(
                    or_(
                        Project.prj_ProjectTitle.ilike(pattern),
                        Project.prj_PI.ilike(pattern),
                        Project.prj_Project_LabContact.ilike(pattern),
                    )
                )
                .order_by(Project.id.asc())
                .limit(limit)
                .all()
            )
            for project in rows:
                matches.append(
                    {
                        "id": int(project.id),
                        "title": project.prj_ProjectTitle,
                        "status": project.prj_Status,
                        "current": bool(getattr(project, "prj_Current_FLAG", False)),
                    }
                )
            return {"ok": True, "tool": name, "result": {"matches": matches}}

        if name == "get_person":
            person_id = _safe_int(args.get("id"))
            if person_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            person = core_db.get(Person, person_id)
            if person is None:
                return {"ok": False, "tool": name, "error": f"Person {person_id} not found."}
            return {"ok": True, "tool": name, "result": person_summary(person)}

        if name == "search_people":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=5, min_value=1, max_value=20)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}

            matches: list[dict[str, Any]] = []
            person_id = _safe_int(query)
            if person_id is not None:
                person = core_db.get(Person, person_id)
                if person is not None:
                    matches.append(person_summary(person))
                return {"ok": True, "tool": name, "result": {"matches": matches}}

            pattern = f"%{query}%"
            rows = (
                core_db.query(Person)
                .filter(
                    or_(
                        Person.ppl_Name_First.ilike(pattern),
                        Person.ppl_Name_Last.ilike(pattern),
                        Person.ppl_Email.ilike(pattern),
                    )
                )
                .order_by(Person.id.asc())
                .limit(limit)
                .all()
            )
            matches.extend(person_summary(person) for person in rows)
            return {"ok": True, "tool": name, "result": {"matches": matches}}

        if name == "list_schedule_slots":
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            start = _safe_date(args.get("start"))
            end = _safe_date(args.get("end"))
            if start is None or end is None:
                today = datetime.now(CENTRAL_TZ).date()
                start = today
                end = today + timedelta(days=28)
            if end < start:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "end must be on or after start (YYYY-MM-DD).",
                }

            raw_status = _safe_str(args.get("status"), max_len=32)
            status: str | None = None
            if raw_status:
                normalized = raw_status.strip().lower()
                if normalized not in SCHEDULE_SLOT_STATUSES:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"status must be one of {sorted(SCHEDULE_SLOT_STATUSES)}",
                    }
                status = normalized

            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=500)

            start_utc, end_utc = _range_bounds_local(start, end)
            query = (
                schedule_db.query(ScheduleSlot)
                .filter(ScheduleSlot.start_at >= start_utc)
                .filter(ScheduleSlot.start_at <= end_utc)
            )
            if status is not None:
                query = query.filter(ScheduleSlot.status == status)
            rows = query.order_by(ScheduleSlot.start_at.asc()).limit(limit).all()

            items: list[dict[str, Any]] = []
            for row in rows:
                start_utc_aware = _as_utc_aware(row.start_at)
                end_utc_aware = _as_utc_aware(row.end_at)
                items.append(
                    {
                        "id": int(row.id),
                        "status": row.status,
                        "start_at": start_utc_aware.isoformat(),
                        "end_at": end_utc_aware.isoformat(),
                        "start_at_central": start_utc_aware.astimezone(CENTRAL_TZ).isoformat(),
                        "end_at_central": end_utc_aware.astimezone(CENTRAL_TZ).isoformat(),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "timezone": str(CENTRAL_TZ),
                    "count": len(items),
                    "items": items,
                },
            }

        if name == "list_schedule_requests":
            auth_error = _require_admin(user)
            if auth_error:
                return {"ok": False, "tool": name, "error": auth_error}
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            raw_status = _safe_str(args.get("status"), max_len=32)
            status: str | None = None
            if raw_status:
                normalized = raw_status.strip().lower()
                if normalized not in SCHEDULE_REQUEST_STATUSES:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"status must be one of {sorted(SCHEDULE_REQUEST_STATUSES)}",
                    }
                status = normalized

            query = schedule_db.query(ScheduleRequest)
            if status is not None:
                query = query.filter(ScheduleRequest.status == status)
            rows = query.order_by(ScheduleRequest.created_at.desc()).limit(limit).all()

            items: list[dict[str, Any]] = []
            for row in rows:
                slot_ids = [link.slot_id for link in sorted(row.slots, key=lambda link: link.rank)]
                items.append(
                    {
                        "id": int(row.id),
                        "status": row.status,
                        "created_at": _as_utc_aware(row.created_at).isoformat(),
                        "requester_name": row.requester_name,
                        "requester_email": row.requester_email,
                        "requester_org": row.requester_org,
                        "project_title": row.project_title,
                        "cancer_related": bool(row.cancer_related),
                        "slot_ids": slot_ids,
                    }
                )

            return {"ok": True, "tool": name, "result": {"count": len(items), "items": items}}

        if name == "get_schedule_request":
            auth_error = _require_admin(user)
            if auth_error:
                return {"ok": False, "tool": name, "error": auth_error}
            if schedule_db is None:
                return {"ok": False, "tool": name, "error": "Schedule DB is not available."}

            request_id = _safe_int(args.get("id"))
            if request_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: id"}
            row = schedule_db.get(ScheduleRequest, request_id)
            if row is None:
                return {"ok": False, "tool": name, "error": f"Request {request_id} not found."}

            slot_ids = [link.slot_id for link in sorted(row.slots, key=lambda link: link.rank)]
            description = row.project_description or ""
            if len(description) > 2000:
                description = description[:1999] + "…"

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "id": int(row.id),
                    "status": row.status,
                    "created_at": _as_utc_aware(row.created_at).isoformat(),
                    "updated_at": _as_utc_aware(row.updated_at).isoformat(),
                    "requester_name": row.requester_name,
                    "requester_email": row.requester_email,
                    "requester_org": row.requester_org,
                    "requester_phone": row.requester_phone,
                    "project_title": row.project_title,
                    "project_description": description,
                    "cancer_related": bool(row.cancer_related),
                    "slot_ids": slot_ids,
                },
            }

        return {
            "ok": False,
            "tool": name,
            "error": f"Unknown tool '{name}'.",
            "available": [
                "search_projects",
                "get_project",
                "latest_experiments",
                "get_experiment",
                "latest_experiment_runs",
                "get_experiment_run",
                "search_people",
                "get_person",
                "list_schedule_slots",
                "list_schedule_requests",
                "get_schedule_request",
            ],
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": name,
            "error": f"{type(exc).__name__}: {exc}",
        }

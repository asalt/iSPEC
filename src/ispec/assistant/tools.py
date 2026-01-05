from __future__ import annotations

from datetime import date, datetime, time, timedelta
import enum
import json
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ispec.assistant.context import person_summary, project_summary
from ispec.db.models import (
    AuthUser,
    E2G,
    Experiment,
    ExperimentRun,
    Person,
    Project,
    ProjectComment,
    UserRole,
)
from ispec.schedule.models import ScheduleRequest, ScheduleSlot


TOOL_CALL_PREFIX = "TOOL_CALL"
TOOL_RESULT_PREFIX = "TOOL_RESULT"

CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

SCHEDULE_SLOT_STATUSES = {"available", "booked", "closed"}
SCHEDULE_REQUEST_STATUSES = {"requested", "confirmed", "declined", "cancelled"}


class ToolScope(str, enum.Enum):
    public = "public"
    user = "user"
    staff = "staff"
    admin = "admin"


_TOOL_SCOPES: dict[str, ToolScope] = {
    "project_counts_snapshot": ToolScope.staff,
    "latest_activity": ToolScope.staff,
    "billing_category_counts": ToolScope.staff,
    "db_file_stats": ToolScope.staff,
    "count_projects": ToolScope.staff,
    "project_status_counts": ToolScope.staff,
    "latest_projects": ToolScope.staff,
    "latest_project_comments": ToolScope.staff,
    "search_projects": ToolScope.staff,
    "get_project": ToolScope.staff,
    "search_api": ToolScope.user,
    "experiments_for_project": ToolScope.staff,
    "latest_experiments": ToolScope.staff,
    "get_experiment": ToolScope.staff,
    "latest_experiment_runs": ToolScope.staff,
    "get_experiment_run": ToolScope.staff,
    "e2g_search_genes_in_project": ToolScope.staff,
    "e2g_gene_in_project": ToolScope.staff,
    "search_people": ToolScope.staff,
    "get_person": ToolScope.staff,
    "list_schedule_slots": ToolScope.user,
    "list_schedule_requests": ToolScope.admin,
    "get_schedule_request": ToolScope.admin,
}


def _scope_error(scope: ToolScope, user: AuthUser | None) -> str | None:
    if user is None:
        return None
    if scope == ToolScope.public:
        return None
    if scope == ToolScope.user:
        return None
    if scope == ToolScope.staff:
        if user.role in {UserRole.viewer, UserRole.editor, UserRole.admin}:
            return None
        return "Staff access required."
    if scope == ToolScope.admin:
        if user.role == UserRole.admin:
            return None
        return "Admin access required."
    return "Access denied."


def tool_prompt() -> str:
    """Short tool list for the system prompt."""

    return (
        "Available tools (read-only):\n"
        "- project_counts_snapshot(max_categories: int = 20)\n"
        "- latest_activity(limit: int = 20, kinds: list[str] | None = None, current_only: bool = false)\n"
        "- billing_category_counts(current_only: bool = false, limit: int = 20)\n"
        "- db_file_stats()  # show sqlite DB file sizes\n"
        "- count_projects(current_only: bool = false, status: str | None = None)  # counts projects\n"
        "- project_status_counts(current_only: bool = false)\n"
        "- latest_projects(sort: str = 'modified', limit: int = 10, current_only: bool = false)\n"
        "- latest_project_comments(limit: int = 10, project_id: int | None = None)\n"
        "- search_projects(query: str, limit: int = 5)\n"
        "- get_project(id: int)\n"
        "- search_api(query: str, limit: int = 10)  # search FastAPI/OpenAPI endpoints\n"
        "- experiments_for_project(project_id: int, limit: int = 20)\n"
        "- latest_experiments(limit: int = 5)\n"
        "- get_experiment(id: int)\n"
        "- latest_experiment_runs(limit: int = 5)\n"
        "- get_experiment_run(id: int)\n"
        "- e2g_search_genes_in_project(project_id: int, query: str, limit: int = 10)\n"
        "- e2g_gene_in_project(project_id: int, gene_id: int, limit: int = 50)\n"
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


def _human_bytes(size: int | None) -> str | None:
    if size is None:
        return None
    if size < 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit_index = 0
    while value >= 1024 and unit_index < (len(units) - 1):
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _sqlite_path_from_session(db: Session) -> Path | None:
    try:
        bind = db.get_bind()
        url = getattr(bind, "url", None)
        if url is None:
            return None
        if getattr(url, "get_backend_name", lambda: None)() != "sqlite":
            return None
        database = getattr(url, "database", None)
        if not database or database == ":memory:":
            return None
        return Path(str(database))
    except Exception:
        return None


def _stat_path(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "size_bytes": None, "size_human": None}
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False, "size_bytes": None, "size_human": None}
    except Exception as exc:
        return {"exists": False, "size_bytes": None, "size_human": None, "error": f"{type(exc).__name__}: {exc}"}
    size = int(stat.st_size)
    return {"exists": True, "size_bytes": size, "size_human": _human_bytes(size)}


def _coerce_str_list(value: Any, *, max_items: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [text[:256]]
    if isinstance(value, list):
        items: list[str] = []
        for item in value[: max(0, max_items)]:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            items.append(text[:256])
        return items or None
    return None


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a TOOL_CALL request from assistant output.

    Expected format (single line, may appear anywhere in the response):
      TOOL_CALL {"name":"...","arguments":{...}}
    """

    line = extract_tool_call_line(text)
    if not line:
        return None

    raw = line[len(TOOL_CALL_PREFIX) :].strip()
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


def extract_tool_call_line(text: str) -> str | None:
    """Return the first TOOL_CALL line, if any.

    We intentionally support tool calls that appear alongside extra text
    because some models will not reliably output "tool call only" despite
    being instructed to do so.
    """

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(TOOL_CALL_PREFIX):
            return line
    return None


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
    api_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        scope = _TOOL_SCOPES.get(name, ToolScope.staff)
        scope_error = _scope_error(scope, user)
        if scope_error:
            return {"ok": False, "tool": name, "error": scope_error}

        if name == "project_counts_snapshot":
            max_categories = _clamp_int(
                _safe_int(args.get("max_categories")),
                default=20,
                min_value=1,
                max_value=100,
            )
            total_count = int(core_db.query(func.count(Project.id)).scalar() or 0)
            current_count = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .scalar()
                or 0
            )

            status_rows = (
                core_db.query(Project.prj_Status, func.count(Project.id))
                .group_by(Project.prj_Status)
                .all()
            )
            status_total = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in status_rows
            ]
            status_total.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )

            status_rows_current = (
                core_db.query(Project.prj_Status, func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .group_by(Project.prj_Status)
                .all()
            )
            status_current = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in status_rows_current
            ]
            status_current.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )

            price_level_rows = (
                core_db.query(Project.prj_ProjectPriceLevel, func.count(Project.id))
                .group_by(Project.prj_ProjectPriceLevel)
                .all()
            )
            price_levels = [
                {
                    "category": (category.strip() if isinstance(category, str) and category.strip() else None),
                    "count": int(count or 0),
                }
                for category, count in price_level_rows
            ]
            price_levels.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["category"] is None else str(item["category"]),
                )
            )
            if len(price_levels) > max_categories:
                price_levels = price_levels[:max_categories]

            ready_to_bill_total = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Billing_ReadyToBill.is_(True))
                .scalar()
                or 0
            )
            ready_to_bill_current = int(
                core_db.query(func.count(Project.id))
                .filter(Project.prj_Current_FLAG.is_(True))
                .filter(Project.prj_Billing_ReadyToBill.is_(True))
                .scalar()
                or 0
            )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "generated_at": datetime.now(tz=UTC_TZ).isoformat(),
                    "projects": {
                        "total": total_count,
                        "current": current_count,
                        "status_counts_total": status_total,
                        "status_counts_current": status_current,
                        "billing_ready_to_bill": {
                            "total": ready_to_bill_total,
                            "current": ready_to_bill_current,
                        },
                        "billing_categories": {
                            "field": "prj_ProjectPriceLevel",
                            "max_categories": max_categories,
                            "items": price_levels,
                        },
                    },
                },
            }

        if name == "latest_activity":
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=200)
            current_only = bool(args.get("current_only"))
            kinds = _coerce_str_list(args.get("kinds"), max_items=8)
            allowed_kinds = {
                "project",
                "project_comment",
                "experiment",
                "experiment_run",
            }
            if kinds is None:
                selected_kinds = allowed_kinds
            else:
                selected_kinds = {kind.strip().lower() for kind in kinds if kind.strip()}
                unknown = sorted(selected_kinds - allowed_kinds)
                if unknown:
                    return {
                        "ok": False,
                        "tool": name,
                        "error": f"Unknown kinds: {unknown}. Allowed: {sorted(allowed_kinds)}",
                    }

            events: list[dict[str, Any]] = []

            if "project" in selected_kinds:
                query = core_db.query(Project)
                if current_only:
                    query = query.filter(Project.prj_Current_FLAG.is_(True))
                rows = (
                    query.order_by(Project.prj_ModificationTS.desc(), Project.id.desc())
                    .limit(limit)
                    .all()
                )
                for project in rows:
                    modified = getattr(project, "prj_ModificationTS", None)
                    created = getattr(project, "prj_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "project",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(project.id),
                            "title": project.prj_ProjectTitle,
                            "status": project.prj_Status,
                            "current": bool(project.prj_Current_FLAG),
                            "links": {"ui": f"/project/{project.id}", "api": f"/api/projects/{project.id}"},
                        }
                    )

            if "project_comment" in selected_kinds:
                query = (
                    core_db.query(ProjectComment, Project)
                    .join(Project, ProjectComment.project_id == Project.id)
                    .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
                )
                if current_only:
                    query = query.filter(Project.prj_Current_FLAG.is_(True))
                rows = query.limit(limit).all()
                for comment, project in rows:
                    created = getattr(comment, "com_CreationTS", None)
                    text = (comment.com_Comment or "").strip()
                    if len(text) > 240:
                        text = text[:239] + "…"
                    events.append(
                        {
                            "kind": "project_comment",
                            "timestamp": created.isoformat() if created else None,
                            "id": int(comment.id),
                            "project_id": int(project.id),
                            "project_title": project.prj_ProjectTitle,
                            "type": comment.com_CommentType,
                            "added_by": comment.com_AddedBy,
                            "comment": text or None,
                            "links": {"ui": f"/project/{project.id}", "api": f"/api/projects/{project.id}"},
                        }
                    )

            if "experiment" in selected_kinds:
                query = core_db.query(Experiment)
                if current_only:
                    query = query.join(Project, Experiment.project_id == Project.id).filter(
                        Project.prj_Current_FLAG.is_(True)
                    )
                rows = (
                    query.order_by(Experiment.Experiment_ModificationTS.desc(), Experiment.id.desc())
                    .limit(limit)
                    .all()
                )
                for experiment in rows:
                    modified = getattr(experiment, "Experiment_ModificationTS", None)
                    created = getattr(experiment, "Experiment_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "experiment",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(experiment.id),
                            "record_no": experiment.record_no,
                            "name": experiment.exp_Name,
                            "type": str(experiment.exp_Type) if experiment.exp_Type is not None else None,
                            "project_id": int(experiment.project_id)
                            if experiment.project_id is not None
                            else None,
                            "links": {"ui": f"/experiment/{experiment.id}", "api": f"/api/experiments/{experiment.id}"},
                        }
                    )

            if "experiment_run" in selected_kinds:
                query = core_db.query(ExperimentRun)
                if current_only:
                    query = (
                        query.join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                        .join(Project, Experiment.project_id == Project.id)
                        .filter(Project.prj_Current_FLAG.is_(True))
                    )
                rows = (
                    query.order_by(
                        ExperimentRun.ExperimentRun_ModificationTS.desc(),
                        ExperimentRun.id.desc(),
                    )
                    .limit(limit)
                    .all()
                )
                for run in rows:
                    modified = getattr(run, "ExperimentRun_ModificationTS", None)
                    created = getattr(run, "ExperimentRun_CreationTS", None)
                    ts = modified or created
                    events.append(
                        {
                            "kind": "experiment_run",
                            "timestamp": ts.isoformat() if ts else None,
                            "id": int(run.id),
                            "experiment_id": int(run.experiment_id),
                            "run_no": int(run.run_no),
                            "search_no": int(run.search_no),
                            "search_engine": getattr(run, "search_engine", None),
                            "search_state": getattr(run, "search_state", None),
                            "links": {
                                "ui": f"/experiment-run/{run.id}",
                                "api": f"/api/experiment-runs/{run.id}",
                            },
                        }
                    )

            def sort_key(item: dict[str, Any]) -> tuple[int, str, int]:
                ts = item.get("timestamp")
                if isinstance(ts, str) and ts:
                    return (1, ts, int(item.get("id") or 0))
                return (0, "", int(item.get("id") or 0))

            events.sort(key=sort_key, reverse=True)
            events = events[:limit]

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "limit": limit,
                    "current_only": current_only,
                    "kinds": sorted(selected_kinds),
                    "count": len(events),
                    "events": events,
                },
            }

        if name == "billing_category_counts":
            current_only = bool(args.get("current_only"))
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)

            query = core_db.query(Project.prj_ProjectPriceLevel, func.count(Project.id))
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))
            rows = query.group_by(Project.prj_ProjectPriceLevel).all()
            items = [
                {
                    "category": (category.strip() if isinstance(category, str) and category.strip() else None),
                    "count": int(count or 0),
                }
                for category, count in rows
            ]
            items.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["category"] is None else str(item["category"]),
                )
            )
            if len(items) > limit:
                items = items[:limit]
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "current_only": current_only,
                    "field": "prj_ProjectPriceLevel",
                    "count": len(items),
                    "items": items,
                },
            }

        if name == "db_file_stats":
            core_path = _sqlite_path_from_session(core_db)
            assistant_path = (core_path.parent / "ispec-assistant.db") if core_path is not None else None
            schedule_path = _sqlite_path_from_session(schedule_db) if schedule_db is not None else None
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "core_db": _stat_path(core_path),
                    "assistant_db": _stat_path(assistant_path),
                    "schedule_db": _stat_path(schedule_path) if schedule_db is not None else None,
                },
            }

        if name == "count_projects":
            current_only = bool(args.get("current_only"))
            status = _safe_str(args.get("status"), max_len=64)

            query = core_db.query(func.count(Project.id))
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))
            if status:
                query = query.filter(Project.prj_Status == status)
            count = int(query.scalar() or 0)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "count": count,
                    "current_only": current_only,
                    "status": status,
                },
            }

        if name == "project_status_counts":
            current_only = bool(args.get("current_only"))
            query = core_db.query(Project.prj_Status, func.count(Project.id))
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))
            rows = query.group_by(Project.prj_Status).all()
            items = [
                {"status": (status if status is not None else None), "count": int(count or 0)}
                for status, count in rows
            ]
            items.sort(
                key=lambda item: (
                    -int(item["count"]),
                    "" if item["status"] is None else str(item["status"]),
                )
            )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "current_only": current_only,
                    "total": int(sum(item["count"] for item in items)),
                    "items": items,
                },
            }

        if name == "latest_projects":
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            current_only = bool(args.get("current_only"))
            sort_raw = (_safe_str(args.get("sort"), max_len=32) or "modified").strip().lower()

            query = core_db.query(Project)
            if current_only:
                query = query.filter(Project.prj_Current_FLAG.is_(True))

            if sort_raw in {"created", "creation"}:
                query = query.order_by(Project.prj_CreationTS.desc(), Project.id.desc())
                sort = "created"
            elif sort_raw in {"modified", "modification", "updated", "update"}:
                query = query.order_by(Project.prj_ModificationTS.desc(), Project.id.desc())
                sort = "modified"
            elif sort_raw in {"id"}:
                query = query.order_by(Project.id.desc())
                sort = "id"
            else:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "sort must be one of: created, modified, id",
                }

            rows = query.limit(limit).all()
            projects = []
            for project in rows:
                projects.append(
                    {
                        "id": int(project.id),
                        "title": project.prj_ProjectTitle,
                        "status": project.prj_Status,
                        "current": bool(project.prj_Current_FLAG),
                        "created": project.prj_CreationTS.isoformat()
                        if getattr(project, "prj_CreationTS", None)
                        else None,
                        "modified": project.prj_ModificationTS.isoformat()
                        if getattr(project, "prj_ModificationTS", None)
                        else None,
                        "links": {
                            "ui": f"/project/{project.id}",
                            "api": f"/api/projects/{project.id}",
                        },
                    }
                )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "sort": sort,
                    "current_only": current_only,
                    "count": len(projects),
                    "projects": projects,
                },
            }

        if name == "latest_project_comments":
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            project_id = _safe_int(args.get("project_id"))
            query = (
                core_db.query(ProjectComment, Project)
                .join(Project, ProjectComment.project_id == Project.id)
                .order_by(ProjectComment.com_CreationTS.desc(), ProjectComment.id.desc())
            )
            if project_id is not None:
                query = query.filter(ProjectComment.project_id == project_id)
            rows = query.limit(limit).all()

            comments: list[dict[str, Any]] = []
            for comment, project in rows:
                created = getattr(comment, "com_CreationTS", None)
                text = (comment.com_Comment or "").strip()
                if len(text) > 240:
                    text = text[:239] + "…"
                comments.append(
                    {
                        "id": int(comment.id),
                        "project_id": int(project.id),
                        "project_title": project.prj_ProjectTitle,
                        "type": comment.com_CommentType,
                        "added_by": comment.com_AddedBy,
                        "created": created.isoformat() if created else None,
                        "comment": text or None,
                        "links": {
                            "ui": f"/project/{project.id}",
                            "api": f"/api/projects/{project.id}",
                        },
                    }
                )
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "count": len(comments),
                    "comments": comments,
                },
            }

        if name == "search_api":
            query = _safe_str(args.get("query"), max_len=200)
            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            if not query:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}
            if not isinstance(api_schema, dict):
                return {"ok": False, "tool": name, "error": "API schema is not available."}

            matches = _search_openapi_schema(api_schema, query=query, limit=limit)
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "query": query,
                    "count": len(matches),
                    "matches": matches,
                    "docs_hint": "/docs (FastAPI Swagger UI), /openapi.json",
                },
            }

        if name == "experiments_for_project":
            project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}
            limit = _clamp_int(_safe_int(args.get("limit")), default=20, min_value=1, max_value=100)

            rows = (
                core_db.query(Experiment)
                .filter(Experiment.project_id == project_id)
                .order_by(Experiment.id.desc())
                .limit(limit)
                .all()
            )

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
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "count": len(experiments),
                    "experiments": experiments,
                },
            }

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
                        "search_engine": getattr(run, "search_engine", None),
                        "search_state": getattr(run, "search_state", None),
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
                    "search_engine": getattr(run, "search_engine", None),
                    "search_state": getattr(run, "search_state", None),
                    "created": created.isoformat() if created else None,
                    "modified": modified.isoformat() if modified else None,
                    "links": {
                        "ui": f"/experiment-run/{run.id}",
                        "api": f"/api/experiment-runs/{run.id}",
                    },
                },
            }

        if name == "e2g_search_genes_in_project":
            project_id = _safe_int(args.get("project_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}

            query_text = _safe_str(args.get("query"), max_len=200)
            if not query_text:
                return {"ok": False, "tool": name, "error": "Missing string argument: query"}
            if query_text.strip().lower() in {"*", "all"}:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "Use e2g_gene_in_project for a specific GeneID, or provide a keyword (GeneSymbol/description).",
                }

            limit = _clamp_int(_safe_int(args.get("limit")), default=10, min_value=1, max_value=50)
            pattern = f"%{query_text}%"

            rows = (
                core_db.query(
                    E2G.gene,
                    E2G.gene_symbol,
                    E2G.description,
                    func.count(E2G.id).label("hits"),
                )
                .join(ExperimentRun, E2G.experiment_run_id == ExperimentRun.id)
                .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                .filter(Experiment.project_id == project_id)
                .filter(
                    or_(
                        E2G.gene.ilike(pattern),
                        E2G.gene_symbol.ilike(pattern),
                        E2G.description.ilike(pattern),
                    )
                )
                .group_by(E2G.gene, E2G.gene_symbol, E2G.description)
                .order_by(func.count(E2G.id).desc(), E2G.gene.asc())
                .limit(limit)
                .all()
            )

            matches: list[dict[str, Any]] = []
            for gene, symbol, description, hits in rows:
                desc_text = (description or "").strip()
                if len(desc_text) > 240:
                    desc_text = desc_text[:239] + "…"
                gene_id = _safe_int(gene)
                matches.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": (symbol.strip() if isinstance(symbol, str) and symbol.strip() else None),
                        "description": desc_text or None,
                        "hits": int(hits or 0),
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "query": query_text,
                    "count": len(matches),
                    "matches": matches,
                },
            }

        if name == "e2g_gene_in_project":
            project_id = _safe_int(args.get("project_id"))
            gene_id = _safe_int(args.get("gene_id"))
            if project_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: project_id"}
            if gene_id is None:
                return {"ok": False, "tool": name, "error": "Missing integer argument: gene_id"}
            limit = _clamp_int(_safe_int(args.get("limit")), default=50, min_value=1, max_value=200)

            rows = (
                core_db.query(E2G, ExperimentRun, Experiment)
                .join(ExperimentRun, E2G.experiment_run_id == ExperimentRun.id)
                .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
                .filter(Experiment.project_id == project_id)
                .filter(E2G.geneidtype == "GeneID")
                .filter(E2G.gene == str(gene_id))
                .order_by(
                    func.coalesce(E2G.psms_u2g, E2G.psms, 0).desc(),
                    func.coalesce(E2G.iBAQ_dstrAdj, 0.0).desc(),
                    Experiment.id.desc(),
                    ExperimentRun.id.desc(),
                    E2G.id.asc(),
                )
                .limit(limit)
                .all()
            )

            hits: list[dict[str, Any]] = []
            for e2g, run, experiment in rows:
                peptideprint = getattr(e2g, "peptideprint", None)
                peptideprint_len = len(peptideprint) if isinstance(peptideprint, str) else None
                peptideprint_preview = None
                if isinstance(peptideprint, str) and peptideprint:
                    peptideprint_preview = peptideprint[:800]
                    if len(peptideprint) > 800:
                        peptideprint_preview = peptideprint_preview[:799] + "…"

                hits.append(
                    {
                        "experiment_id": int(experiment.id),
                        "experiment_record_no": experiment.record_no,
                        "experiment_name": experiment.exp_Name,
                        "experiment_run_id": int(run.id),
                        "run_no": int(run.run_no),
                        "search_no": int(run.search_no),
                        "label": run.label,
                        "gene_id": gene_id,
                        "gene_symbol": getattr(e2g, "gene_symbol", None),
                        "taxon_id": getattr(e2g, "taxon_id", None),
                        "sra": getattr(e2g, "sra", None),
                        "psms": getattr(e2g, "psms", None),
                        "psms_u2g": getattr(e2g, "psms_u2g", None),
                        "peptide_count": getattr(e2g, "peptide_count", None),
                        "peptide_count_u2g": getattr(e2g, "peptide_count_u2g", None),
                        "coverage": getattr(e2g, "coverage", None),
                        "coverage_u2g": getattr(e2g, "coverage_u2g", None),
                        "area_sum_u2g_all": getattr(e2g, "area_sum_u2g_all", None),
                        "iBAQ_dstrAdj": getattr(e2g, "iBAQ_dstrAdj", None),
                        "peptideprint_len": peptideprint_len,
                        "peptideprint_preview": peptideprint_preview,
                        "links": {
                            "project_ui": f"/project/{project_id}",
                            "experiment_ui": f"/experiment/{experiment.id}",
                            "experiment_run_ui": f"/experiment-run/{run.id}",
                        },
                    }
                )

            return {
                "ok": True,
                "tool": name,
                "result": {
                    "project_id": project_id,
                    "gene_id": gene_id,
                    "count": len(hits),
                    "hits": hits,
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
            if query.strip().lower() in {"*", "all"}:
                return {
                    "ok": False,
                    "tool": name,
                    "error": "Use count_projects to answer 'how many projects'. search_projects expects a keyword (title/PI/contact) or an id.",
                }

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
                "project_counts_snapshot",
                "latest_activity",
                "billing_category_counts",
                "db_file_stats",
                "count_projects",
                "project_status_counts",
                "latest_projects",
                "latest_project_comments",
                "search_projects",
                "get_project",
                "search_api",
                "experiments_for_project",
                "latest_experiments",
                "get_experiment",
                "latest_experiment_runs",
                "get_experiment_run",
                "e2g_search_genes_in_project",
                "e2g_gene_in_project",
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


_OPENAI_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "project_counts_snapshot": {
        "type": "function",
        "function": {
            "name": "project_counts_snapshot",
            "description": "Return a snapshot of project counts (total/current/status/billing categories).",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_categories": {
                        "type": "integer",
                        "description": "Max billing categories to include.",
                    }
                },
            },
        },
    },
    "latest_activity": {
        "type": "function",
        "function": {
            "name": "latest_activity",
            "description": "Return recent iSPEC activity across projects, comments, experiments, and runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max events to return."},
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of kinds: project, project_comment, experiment, experiment_run.",
                    },
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, restrict activity to current projects.",
                    },
                },
            },
        },
    },
    "billing_category_counts": {
        "type": "function",
        "function": {
            "name": "billing_category_counts",
            "description": "Count projects grouped by billing category (ProjectPriceLevel).",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, only current projects.",
                    },
                    "limit": {"type": "integer", "description": "Max categories to return."},
                },
            },
        },
    },
    "db_file_stats": {
        "type": "function",
        "function": {
            "name": "db_file_stats",
            "description": "Return sqlite DB file sizes (core, assistant, and schedule DB).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "count_projects": {
        "type": "function",
        "function": {
            "name": "count_projects",
            "description": "Count projects in the iSPEC database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, count only current projects.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional status filter (e.g. inquiry, closed).",
                    },
                },
            },
        },
    },
    "project_status_counts": {
        "type": "function",
        "function": {
            "name": "project_status_counts",
            "description": "Count projects by status (optionally restricted to current projects).",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, include only current projects.",
                    }
                },
            },
        },
    },
    "latest_projects": {
        "type": "function",
        "function": {
            "name": "latest_projects",
            "description": "Fetch the latest projects by modified/created timestamp (or id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort": {"type": "string", "description": "created, modified, or id."},
                    "limit": {"type": "integer", "description": "Max projects to return."},
                    "current_only": {
                        "type": "boolean",
                        "description": "If true, only current projects.",
                    },
                },
            },
        },
    },
    "latest_project_comments": {
        "type": "function",
        "function": {
            "name": "latest_project_comments",
            "description": "Fetch latest project comments, optionally for a single project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max comments to return."},
                    "project_id": {"type": "integer", "description": "Optional project id filter."},
                },
            },
        },
    },
    "search_projects": {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": "Search projects by keyword (title/PI/contact) or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or id."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "get_project": {
        "type": "function",
        "function": {
            "name": "get_project",
            "description": "Fetch a project by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Project id."}},
                "required": ["id"],
            },
        },
    },
    "search_api": {
        "type": "function",
        "function": {
            "name": "search_api",
            "description": "Search FastAPI/OpenAPI endpoints by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword query."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "experiments_for_project": {
        "type": "function",
        "function": {
            "name": "experiments_for_project",
            "description": "List experiments for a given project id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "limit": {"type": "integer", "description": "Max experiments to return."},
                },
                "required": ["project_id"],
            },
        },
    },
    "latest_experiments": {
        "type": "function",
        "function": {
            "name": "latest_experiments",
            "description": "Fetch latest experiments.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max experiments to return."}},
            },
        },
    },
    "get_experiment": {
        "type": "function",
        "function": {
            "name": "get_experiment",
            "description": "Fetch an experiment by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Experiment id."}},
                "required": ["id"],
            },
        },
    },
    "latest_experiment_runs": {
        "type": "function",
        "function": {
            "name": "latest_experiment_runs",
            "description": "Fetch latest experiment runs.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max runs to return."}},
            },
        },
    },
    "get_experiment_run": {
        "type": "function",
        "function": {
            "name": "get_experiment_run",
            "description": "Fetch an experiment run by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "ExperimentRun id."}},
                "required": ["id"],
            },
        },
    },
    "e2g_search_genes_in_project": {
        "type": "function",
        "function": {
            "name": "e2g_search_genes_in_project",
            "description": "Search E2G (experiment_to_gene) rows within a project by GeneID, GeneSymbol, or description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "query": {"type": "string", "description": "GeneID, gene symbol, or keyword."},
                    "limit": {"type": "integer", "description": "Max genes to return."},
                },
                "required": ["project_id", "query"],
            },
        },
    },
    "e2g_gene_in_project": {
        "type": "function",
        "function": {
            "name": "e2g_gene_in_project",
            "description": "Return E2G hits (per experiment run) for a GeneID within a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer", "description": "Project id."},
                    "gene_id": {"type": "integer", "description": "GeneID (canonical identifier)."},
                    "limit": {"type": "integer", "description": "Max hits to return."},
                },
                "required": ["project_id", "gene_id"],
            },
        },
    },
    "search_people": {
        "type": "function",
        "function": {
            "name": "search_people",
            "description": "Search people by keyword (name/email) or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or id."},
                    "limit": {"type": "integer", "description": "Max matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    "get_person": {
        "type": "function",
        "function": {
            "name": "get_person",
            "description": "Fetch a person by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Person id."}},
                "required": ["id"],
            },
        },
    },
    "list_schedule_slots": {
        "type": "function",
        "function": {
            "name": "list_schedule_slots",
            "description": "List schedule slots over a date range (America/Chicago).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date YYYY-MM-DD."},
                    "end": {"type": "string", "description": "End date YYYY-MM-DD."},
                    "status": {"type": "string", "description": "available, booked, or closed."},
                    "limit": {"type": "integer", "description": "Max slots to return."},
                },
            },
        },
    },
    "list_schedule_requests": {
        "type": "function",
        "function": {
            "name": "list_schedule_requests",
            "description": "Admin-only: list schedule requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max requests to return."},
                    "status": {
                        "type": "string",
                        "description": "requested, confirmed, declined, cancelled.",
                    },
                },
            },
        },
    },
    "get_schedule_request": {
        "type": "function",
        "function": {
            "name": "get_schedule_request",
            "description": "Admin-only: fetch a schedule request by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "Request id."}},
                "required": ["id"],
            },
        },
    },
}


def openai_tools_for_user(user: AuthUser | None) -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool schemas, filtered by tool scope."""

    tools: list[dict[str, Any]] = []
    for name, spec in _OPENAI_TOOL_SPECS.items():
        scope = _TOOL_SCOPES.get(name, ToolScope.staff)
        if _scope_error(scope, user) is None:
            tools.append(spec)
    return tools


_OPENAPI_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "options",
    "head",
}


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _search_openapi_schema(api_schema: dict[str, Any], *, query: str, limit: int) -> list[dict[str, Any]]:
    """Search an OpenAPI schema for matching endpoints.

    Returns a list of {method, path, summary, tags, operation_id}.
    """

    tokens = _tokenize(query)
    paths = api_schema.get("paths")
    if not isinstance(paths, dict):
        return []

    scored: list[tuple[int, str, str, dict[str, Any]]] = []
    for path, methods in paths.items():
        if not isinstance(path, str) or not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(method, str) or method.lower() not in _OPENAPI_METHODS:
                continue
            if not isinstance(op, dict):
                continue

            summary = (op.get("summary") or "").strip()
            description = (op.get("description") or "").strip()
            operation_id = (op.get("operationId") or "").strip()
            tags = op.get("tags") if isinstance(op.get("tags"), list) else []
            tag_text = " ".join(str(t) for t in tags if t)

            haystack = " ".join(
                part
                for part in [
                    method.lower(),
                    path.lower(),
                    summary.lower(),
                    description.lower(),
                    operation_id.lower(),
                    tag_text.lower(),
                ]
                if part
            )
            score = 0
            for token in tokens:
                if token in haystack:
                    score += 1
                    if token in path.lower():
                        score += 1
                    if summary and token in summary.lower():
                        score += 1
            if score <= 0:
                continue
            scored.append((score, path, method.upper(), op))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    results: list[dict[str, Any]] = []
    for score, path, method, op in scored[: max(0, limit)]:
        summary = (op.get("summary") or "").strip() or None
        operation_id = (op.get("operationId") or "").strip() or None
        tags = op.get("tags") if isinstance(op.get("tags"), list) else []
        tags_clean = [str(tag) for tag in tags if isinstance(tag, str) and tag.strip()]
        results.append(
            {
                "method": method,
                "path": path,
                "summary": summary,
                "operation_id": operation_id,
                "tags": tags_clean,
                "score": score,
            }
        )
    return results

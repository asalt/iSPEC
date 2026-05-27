from __future__ import annotations

import json
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ispec.agent_state.connect import get_agent_state_session_dep
from ispec.agent_state.store import list_heads
from ispec.db.connect import get_session_dep
from ispec.agent.connect import get_agent_session_dep
from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy
from ispec.api.routes.schema import build_form_schema
from ispec.api.security import require_assistant_access
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview
from ispec.db.models import AuthUser, LegacySyncState, UserRole


router = APIRouter(prefix="/ops", tags=["Ops"])


def utcnow() -> datetime:
    return datetime.now(UTC)


def _truncate_text(value: str | None, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _legacy_sync_allowed_usernames() -> set[str]:
    raw = os.getenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS")
    if raw is None or not raw.strip():
        return {"alex"}

    users: set[str] = set()
    for item in raw.replace(",", " ").split():
        username = item.strip().lower()
        if username:
            users.add(username)
    return users or {"alex"}


def _require_legacy_sync_operator(user: AuthUser | None) -> AuthUser:
    if (
        user is None
        or not getattr(user, "is_active", False)
        or user.role not in {UserRole.admin, UserRole.editor}
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff access required.",
        )

    username = str(getattr(user, "username", "") or "").strip().lower()
    if username not in _legacy_sync_allowed_usernames():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Legacy sync manual run is restricted to whitelisted staff.",
        )
    return user


def _require_schema_staff(user: AuthUser | None) -> AuthUser:
    if (
        user is None
        or not getattr(user, "is_active", False)
        or user.role not in {UserRole.admin, UserRole.editor}
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff access required.",
        )
    return user


def _schema_snapshot_dir() -> Path:
    raw = (os.getenv("ISPEC_SCHEMA_SNAPSHOT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.cwd() / "data" / "schema_snapshots"


def _schema_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_resource_specs() -> list[dict[str, Any]]:
    from ispec.db.models import (
        Assay,
        E2G,
        Experiment,
        ExperimentRun,
        Job,
        LetterOfSupport,
        MSRawFile,
        PSM,
        Person,
        Project,
        ProjectComment,
        ProjectPerson,
        Reagent,
    )

    return [
        {"resource": "people", "prefix": "/people", "model": Person, "tag": "Person"},
        {
            "resource": "projects",
            "prefix": "/projects",
            "model": Project,
            "tag": "Project",
        },
        {
            "resource": "reagents",
            "prefix": "/reagents",
            "model": Reagent,
            "tag": "Reagent",
        },
        {"resource": "assays", "prefix": "/assays", "model": Assay, "tag": "Assay"},
        {
            "resource": "experiments",
            "prefix": "/experiments",
            "model": Experiment,
            "tag": "Experiment",
        },
        {
            "resource": "experiment_runs",
            "prefix": "/experiment_runs",
            "model": ExperimentRun,
            "tag": "ExperimentRun",
        },
        {
            "resource": "experiment_to_gene",
            "prefix": "/experiment_to_gene",
            "model": E2G,
            "tag": "E2G",
        },
        {"resource": "jobs", "prefix": "/jobs", "model": Job, "tag": "Job"},
        {"resource": "psms", "prefix": "/psms", "model": PSM, "tag": "PSM"},
        {
            "resource": "msraw_files",
            "prefix": "/msraw_files",
            "model": MSRawFile,
            "tag": "MSRawFile",
        },
        {
            "resource": "project_comment",
            "prefix": "/project_comment",
            "model": ProjectComment,
            "tag": "ProjectComment",
        },
        {
            "resource": "project_person",
            "prefix": "/project_person",
            "model": ProjectPerson,
            "tag": "ProjectPerson",
        },
        {
            "resource": "letter_of_support",
            "prefix": "/letter_of_support",
            "model": LetterOfSupport,
            "tag": "LetterOfSupport",
        },
    ]


def _create_exclude_fields(model) -> set[str]:
    return {
        "id",
        *{
            column.name
            for column in model.__table__.columns
            if column.name.endswith("_CreationTS")
            or column.name.endswith("_ModificationTS")
        },
    }


def _build_schema_catalog() -> dict[str, Any]:
    from ispec.api.routes.routes import get_exposed_resources

    specs = _schema_resource_specs()
    exposed = get_exposed_resources()
    prefix_by_table = {
        spec["model"].__table__.name: spec["prefix"]
        for spec in specs
        if spec["resource"] in exposed
    }

    def route_prefix_for_table(table: str) -> str:
        return prefix_by_table.get(table, f"/{table}")

    resources: list[dict[str, Any]] = []
    for spec in specs:
        if spec["resource"] not in exposed:
            continue
        model = spec["model"]
        create_model = make_pydantic_model_from_sqlalchemy(
            model,
            name_suffix="Create",
            exclude_fields=_create_exclude_fields(model),
        )
        schema = build_form_schema(
            model,
            create_model,
            route_prefix_for_table=route_prefix_for_table,
        )
        resources.append(
            {
                "resource": spec["resource"],
                "prefix": spec["prefix"],
                "table": model.__table__.name,
                "model": model.__name__,
                "tag": spec["tag"],
                "field_count": len(schema.get("properties") or {}),
                "required": list(schema.get("required") or []),
                "schema_hash": _schema_hash(schema),
                "schema": schema,
            }
        )

    generated_at = utcnow()
    return {
        "ok": True,
        "generated_at": generated_at.isoformat(),
        "openapi_url": "/openapi.json",
        "redoc_url": "/redoc",
        "resource_count": len(resources),
        "resources": resources,
    }


class LegacySyncRunRequest(BaseModel):
    dry_run: bool | None = None
    reset_cursor: bool | None = None
    limit: int | None = Field(default=None, ge=1)
    max_pages: int | None = Field(default=None, ge=0)
    max_project_comments: int | None = Field(default=None, ge=0)
    max_experiment_runs: int | None = Field(default=None, ge=0)
    recent_project_comment_days: int | None = Field(default=None, ge=0)
    recent_project_comment_scan_limit: int | None = Field(default=None, ge=0)
    backfill_missing: bool | None = None


class LegacySyncRunResponse(BaseModel):
    ok: bool = True
    queued: bool
    command_id: int
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AssistantSessionItem(BaseModel):
    session_id: str
    user_id: int | None = None
    updated_at: str | None = None
    message_count: int
    last_message_id: int
    last_message_role: str | None = None
    last_assistant_message_id: int
    last_user_message: str | None = None
    reviewed_up_to_id: int
    review_updated_at: str | None = None
    needs_review: bool = False


class AssistantSnapshot(BaseModel):
    sessions_total: int = 0
    messages_total: int = 0
    reviews_total: int = 0
    latest_review_id: int = 0
    latest_review_at: str | None = None
    sessions_needing_review: int = 0
    recent_sessions: list[AssistantSessionItem] = Field(default_factory=list)


class AgentCommandItem(BaseModel):
    id: int
    command_type: str
    status: str
    priority: int = 0
    available_at: str | None = None
    updated_at: str | None = None
    attempts: int = 0
    max_attempts: int = 0
    error: str | None = None


class AgentStepItem(BaseModel):
    id: int
    kind: str
    ok: bool = True
    severity: str | None = None
    duration_ms: int | None = None
    ended_at: str | None = None
    error: str | None = None


class AgentSnapshot(BaseModel):
    commands_queued: int = 0
    commands_running: int = 0
    commands_failed: int = 0
    latest_commands: list[AgentCommandItem] = Field(default_factory=list)
    latest_steps: list[AgentStepItem] = Field(default_factory=list)
    latest_supervisor_run: dict[str, Any] | None = None
    state_heads: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class OpsSnapshotResponse(BaseModel):
    ok: bool = True
    ts: datetime
    assistant: AssistantSnapshot
    agent: AgentSnapshot


@router.post("/legacy-sync/run", response_model=LegacySyncRunResponse)
def run_legacy_sync(
    request: LegacySyncRunRequest | None = None,
    agent_db: Session = Depends(get_agent_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
) -> LegacySyncRunResponse:
    operator = _require_legacy_sync_operator(user)
    existing = (
        agent_db.query(AgentCommand)
        .filter(AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL)
        .filter(AgentCommand.status.in_(["queued", "running"]))
        .order_by(AgentCommand.id.asc())
        .first()
    )
    if existing is not None:
        return LegacySyncRunResponse(
            ok=True,
            queued=False,
            command_id=int(existing.id),
            status="existing",
            message=f"Legacy sync command is already {existing.status}.",
            payload=dict(existing.payload_json or {}),
        )

    now = utcnow()
    request_payload = (request or LegacySyncRunRequest()).model_dump(exclude_none=True)
    payload: dict[str, Any] = {
        **request_payload,
        "meta": {
            "enqueued_by": "api_ops_legacy_sync",
            "username": str(operator.username),
            "user_id": int(operator.id) if operator.id is not None else None,
            "requested_at": now.isoformat(),
        },
    }

    cmd = AgentCommand(
        command_type=COMMAND_LEGACY_SYNC_ALL,
        status="queued",
        priority=0,
        created_at=now,
        updated_at=now,
        available_at=now,
        attempts=0,
        max_attempts=1,
        payload_json=payload,
        result_json={},
    )
    agent_db.add(cmd)
    agent_db.commit()
    agent_db.refresh(cmd)

    return LegacySyncRunResponse(
        ok=True,
        queued=True,
        command_id=int(cmd.id),
        status="queued",
        message="Legacy sync command queued.",
        payload=dict(cmd.payload_json or {}),
    )


@router.get("/schema/catalog")
def schema_catalog(
    user: AuthUser | None = Depends(require_assistant_access),
) -> dict[str, Any]:
    _require_schema_staff(user)
    catalog = _build_schema_catalog()
    catalog["catalog_hash"] = _schema_hash(catalog)
    return catalog


@router.post("/schema/snapshot")
def snapshot_schema_catalog(
    user: AuthUser | None = Depends(require_assistant_access),
) -> dict[str, Any]:
    operator = _require_schema_staff(user)
    catalog = _build_schema_catalog()
    catalog_hash = _schema_hash(catalog)
    catalog["catalog_hash"] = catalog_hash
    catalog["snapshot_by"] = {
        "username": str(operator.username),
        "user_id": int(operator.id) if operator.id is not None else None,
    }

    out_dir = _schema_snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"schema_catalog_{stamp}_{catalog_hash[:8]}.json"
    path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "path": str(path),
        "filename": path.name,
        "catalog_hash": catalog_hash,
        "resource_count": catalog["resource_count"],
        "generated_at": catalog["generated_at"],
    }


@router.get("/snapshot", response_model=OpsSnapshotResponse)
def snapshot(
    assistant_db: Session = Depends(get_assistant_session_dep),
    agent_db: Session = Depends(get_agent_session_dep),
    agent_state_db: Session = Depends(get_agent_state_session_dep),
    core_db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
) -> OpsSnapshotResponse:
    _ = user  # reserved for future access control
    now = utcnow()

    sessions_total = int(
        assistant_db.query(func.count(SupportSession.id)).scalar() or 0
    )
    messages_total = int(
        assistant_db.query(func.count(SupportMessage.id)).scalar() or 0
    )
    reviews_total = int(
        assistant_db.query(func.count(SupportSessionReview.id)).scalar() or 0
    )

    latest_review_id = 0
    latest_review_at: str | None = None
    latest_review = (
        assistant_db.query(SupportSessionReview)
        .order_by(SupportSessionReview.id.desc())
        .first()
    )
    if latest_review is not None:
        latest_review_id = int(latest_review.id or 0)
        latest_review_at = (
            latest_review.updated_at.isoformat() if latest_review.updated_at else None
        )

    sessions = (
        assistant_db.query(SupportSession)
        .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
        .limit(12)
        .all()
    )
    session_pks = [int(s.id) for s in sessions]
    review_by_session_pk: dict[int, dict[str, Any]] = {}
    if session_pks:
        review_rows = (
            assistant_db.query(
                SupportSessionReview.session_pk,
                SupportSessionReview.target_message_id,
                SupportSessionReview.updated_at,
            )
            .filter(SupportSessionReview.session_pk.in_(session_pks))
            .order_by(
                SupportSessionReview.session_pk.asc(),
                SupportSessionReview.target_message_id.desc(),
                SupportSessionReview.updated_at.desc(),
                SupportSessionReview.id.desc(),
            )
            .all()
        )
        for session_pk, target_id, updated_at in review_rows:
            key = int(session_pk)
            if key in review_by_session_pk:
                continue
            review_by_session_pk[key] = {
                "reviewed_up_to_id": int(target_id or 0),
                "review_updated_at": updated_at.isoformat() if updated_at else None,
            }

    recent_sessions: list[AssistantSessionItem] = []
    sessions_needing_review = 0
    for session in sessions:
        msg_count = int(
            assistant_db.query(func.count(SupportMessage.id))
            .filter(SupportMessage.session_pk == session.id)
            .scalar()
            or 0
        )
        last_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .first()
        )
        last_user = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "user")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        last_assistant = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )

        review_info = review_by_session_pk.get(int(session.id))
        reviewed_up_to_id = (
            int(review_info.get("reviewed_up_to_id") or 0)
            if isinstance(review_info, dict)
            else 0
        )
        review_updated_at = (
            review_info.get("review_updated_at")
            if isinstance(review_info, dict)
            else None
        )

        last_id = int(last_message.id) if last_message is not None else 0
        last_role = (
            getattr(last_message, "role", None) if last_message is not None else None
        )
        last_assistant_id = int(last_assistant.id) if last_assistant is not None else 0

        needs_review = bool(
            last_role == "assistant"
            and last_assistant_id > 0
            and last_assistant_id > reviewed_up_to_id
        )
        if needs_review:
            sessions_needing_review += 1

        recent_sessions.append(
            AssistantSessionItem(
                session_id=str(session.session_id),
                user_id=int(session.user_id) if session.user_id is not None else None,
                updated_at=(
                    session.updated_at.isoformat() if session.updated_at else None
                ),
                message_count=msg_count,
                last_message_id=last_id,
                last_message_role=last_role,
                last_assistant_message_id=last_assistant_id,
                last_user_message=_truncate_text(
                    getattr(last_user, "content", None), limit=240
                ),
                reviewed_up_to_id=reviewed_up_to_id,
                review_updated_at=review_updated_at,
                needs_review=needs_review,
            )
        )

    commands_queued = int(
        agent_db.query(func.count(AgentCommand.id))
        .filter(AgentCommand.status == "queued")
        .scalar()
        or 0
    )
    commands_running = int(
        agent_db.query(func.count(AgentCommand.id))
        .filter(AgentCommand.status == "running")
        .scalar()
        or 0
    )
    commands_failed = int(
        agent_db.query(func.count(AgentCommand.id))
        .filter(AgentCommand.status == "failed")
        .scalar()
        or 0
    )

    command_rows = (
        agent_db.query(AgentCommand).order_by(AgentCommand.id.desc()).limit(25).all()
    )
    latest_commands = [
        AgentCommandItem(
            id=int(cmd.id),
            command_type=str(cmd.command_type),
            status=str(cmd.status),
            priority=int(cmd.priority or 0),
            available_at=(
                cmd.available_at.isoformat()
                if getattr(cmd, "available_at", None)
                else None
            ),
            updated_at=(
                cmd.updated_at.isoformat() if getattr(cmd, "updated_at", None) else None
            ),
            attempts=int(cmd.attempts or 0),
            max_attempts=int(cmd.max_attempts or 0),
            error=cmd.error,
        )
        for cmd in command_rows
    ]

    step_rows = agent_db.query(AgentStep).order_by(AgentStep.id.desc()).limit(25).all()
    latest_steps = [
        AgentStepItem(
            id=int(step.id),
            kind=str(step.kind),
            ok=bool(step.ok),
            severity=step.severity,
            duration_ms=int(step.duration_ms) if step.duration_ms is not None else None,
            ended_at=(
                step.ended_at.isoformat() if getattr(step, "ended_at", None) else None
            ),
            error=step.error,
        )
        for step in step_rows
    ]

    run_row = (
        agent_db.query(AgentRun)
        .filter(AgentRun.kind == "supervisor")
        .order_by(AgentRun.id.desc())
        .first()
    )
    latest_supervisor_run = None
    if run_row is not None:
        orchestrator_state = None
        orchestrator_action_summary = None
        orchestrator_thought = None
        if isinstance(run_row.summary_json, dict):
            orchestrator_state = run_row.summary_json.get("orchestrator")
            if isinstance(orchestrator_state, dict):
                action_obj = orchestrator_state.get("last_action")
                if isinstance(action_obj, dict):
                    orchestrator_action_summary = action_obj
                action_text = orchestrator_state.get("last_action_summary")
                if isinstance(action_text, str) and action_text.strip():
                    if not isinstance(orchestrator_action_summary, dict):
                        orchestrator_action_summary = {}
                    orchestrator_action_summary = {
                        **orchestrator_action_summary,
                        "summary": action_text.strip(),
                    }
                thought_text = orchestrator_state.get("last_thought")
                if isinstance(thought_text, str) and thought_text.strip():
                    orchestrator_thought = thought_text.strip()
        latest_supervisor_run = {
            "run_id": getattr(run_row, "run_id", None),
            "agent_id": getattr(run_row, "agent_id", None),
            "status": getattr(run_row, "status", None),
            "updated_at": (
                getattr(run_row, "updated_at", None).isoformat()
                if getattr(run_row, "updated_at", None)
                else None
            ),
            "status_bar": getattr(run_row, "status_bar", None),
            "orchestrator": orchestrator_state,
            "orchestrator_action_summary": orchestrator_action_summary,
            "orchestrator_thought_advisory": orchestrator_thought,
            "checks": (
                run_row.state_json.get("checks")
                if isinstance(run_row.state_json, dict)
                else None
            ),
        }
        try:
            latest_supervisor_run["config_json"] = (
                run_row.config_json if isinstance(run_row.config_json, dict) else None
            )
        except Exception:
            latest_supervisor_run["config_json"] = None

    legacy_cursors: list[dict[str, Any]] = []
    try:
        rows = (
            core_db.query(LegacySyncState)
            .order_by(LegacySyncState.legacy_table.asc())
            .all()
        )
        for row in rows:
            legacy_cursors.append(
                {
                    "legacy_table": row.legacy_table,
                    "since": (
                        row.since.isoformat() if getattr(row, "since", None) else None
                    ),
                    "since_pk": int(row.since_pk) if row.since_pk is not None else None,
                    "updated_at": (
                        row.legsync_ModificationTS.isoformat()
                        if getattr(row, "legsync_ModificationTS", None)
                        else None
                    ),
                }
            )
    except Exception:
        legacy_cursors = []

    latest_legacy_sync: dict[str, Any] | None = None
    try:
        step = (
            agent_db.query(AgentStep)
            .filter(AgentStep.kind == COMMAND_LEGACY_SYNC_ALL)
            .order_by(AgentStep.id.desc())
            .first()
        )
        if step is not None:
            result_payload = None
            if isinstance(step.tool_results_json, list) and step.tool_results_json:
                candidate = step.tool_results_json[-1]
                if isinstance(candidate, dict):
                    result_payload = candidate
            latest_legacy_sync = {
                "id": int(step.id),
                "ok": bool(step.ok),
                "ended_at": (
                    step.ended_at.isoformat()
                    if getattr(step, "ended_at", None)
                    else None
                ),
                "duration_ms": (
                    int(step.duration_ms) if step.duration_ms is not None else None
                ),
                "error": step.error,
                "result": result_payload,
            }
    except Exception:
        latest_legacy_sync = None

    state_heads: list[dict[str, Any]] = []
    try:
        state_heads = list_heads(agent_state_db, limit=12)
    except Exception:
        state_heads = []

    return OpsSnapshotResponse(
        ok=True,
        ts=now,
        assistant=AssistantSnapshot(
            sessions_total=sessions_total,
            messages_total=messages_total,
            reviews_total=reviews_total,
            latest_review_id=latest_review_id,
            latest_review_at=latest_review_at,
            sessions_needing_review=sessions_needing_review,
            recent_sessions=recent_sessions,
        ),
        agent=AgentSnapshot(
            commands_queued=commands_queued,
            commands_running=commands_running,
            commands_failed=commands_failed,
            latest_commands=latest_commands,
            latest_steps=latest_steps,
            latest_supervisor_run=latest_supervisor_run,
            state_heads=state_heads,
            legacy_sync_cursors=legacy_cursors,
            latest_legacy_sync=latest_legacy_sync,
        ),
    )

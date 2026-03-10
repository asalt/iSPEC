from __future__ import annotations

import json
import re
from datetime import timedelta
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ispec.agent_state.connect import get_agent_state_session_dep
from ispec.agent_state.store import append_observation, get_schema, list_heads, register_schema_version
from ispec.agent.commands import COMMAND_ASSESS_TACKLE_RESULTS, COMMAND_RUN_TACKLE_PROMPT
from ispec.agent.connect import get_agent_session_dep
from ispec.agent.models import AgentCommand, AgentEvent
from ispec.db.connect import get_session_dep
from ispec.db.models import Project

router = APIRouter(prefix="/agents", tags=["Agents"])


def utcnow() -> datetime:
    return datetime.now(UTC)


class AgentEventIn(BaseModel):
    type: str = Field(min_length=1, max_length=64)
    agent_id: str = Field(min_length=1, max_length=256)
    ts: datetime | None = None

    name: str | None = Field(default=None, max_length=256)
    severity: str | None = Field(default=None, max_length=32)
    trace_id: str | None = Field(default=None, max_length=128)
    correlation_id: str | None = Field(default=None, max_length=128)

    dimensions: dict[str, Any] = Field(default_factory=dict)
    value: Any | None = None

    model_config = ConfigDict(extra="allow")


class IngestResponse(BaseModel):
    ingested: int


class CommandPollResponse(BaseModel):
    commands: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/events", response_model=IngestResponse)
def ingest_events(
    events: list[AgentEventIn],
    db: Session = Depends(get_agent_session_dep),
) -> IngestResponse:
    now = utcnow()
    rows: list[AgentEvent] = []
    for event in events:
        payload = event.model_dump(mode="json")
        rows.append(
            AgentEvent(
                agent_id=event.agent_id,
                event_type=event.type,
                ts=event.ts or now,
                received_at=now,
                name=event.name,
                severity=event.severity,
                trace_id=event.trace_id,
                correlation_id=event.correlation_id,
                payload_json=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            )
        )

    db.add_all(rows)
    db.commit()
    return IngestResponse(ingested=len(rows))


@router.get("/commands/poll", response_model=CommandPollResponse)
def poll_commands(
    agent_id: str = Query(min_length=1, max_length=256),
) -> CommandPollResponse:
    # v0: plumbing-only stub so agents can safely "pull" commands without
    # requiring inbound ports or a full queue implementation yet.
    return CommandPollResponse(commands=[])


def _clamp_int(value: int, *, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _prune_json_for_storage(
    value: Any,
    *,
    max_depth: int = 6,
    max_list_items: int = 200,
    max_dict_items: int = 300,
    max_str_chars: int = 10_000,
) -> Any:
    if max_depth <= 0:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value[:max_str_chars]
        return "<truncated>"

    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        return value[:max_str_chars]

    if isinstance(value, list):
        items = [
            _prune_json_for_storage(
                item,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_str_chars=max_str_chars,
            )
            for item in value[: max(0, int(max_list_items))]
        ]
        if len(value) > int(max_list_items):
            items.append(f"<truncated {len(value) - int(max_list_items)} more items>")
        return items

    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        keys = [k for k in value.keys() if isinstance(k, str)]
        for key in keys[: max(0, int(max_dict_items))]:
            pruned[key] = _prune_json_for_storage(
                value.get(key),
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_str_chars=max_str_chars,
            )
        if len(keys) > int(max_dict_items):
            pruned["_truncated_keys"] = int(len(keys) - int(max_dict_items))
        return pruned

    return str(value)


class EnqueueCommandRequest(BaseModel):
    command_type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    delay_seconds: int = 0
    max_attempts: int = 3


class EnqueueCommandResponse(BaseModel):
    command_id: int


class AgentCommandOut(BaseModel):
    id: int
    command_type: str
    status: str
    priority: int
    available_at: str | None = None
    updated_at: str | None = None
    claimed_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    attempts: int = 0
    max_attempts: int = 0
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class AgentStateSchemaDimIn(BaseModel):
    dim_index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)


class AgentStateSchemaUpsertRequest(BaseModel):
    schema_id: int = Field(ge=1)
    version: int = Field(ge=1)
    state_scope: str = Field(min_length=1, max_length=64)
    codec: str = Field(default="f32le", min_length=1, max_length=32)
    notes: str | None = Field(default=None, max_length=2000)
    dims: list[AgentStateSchemaDimIn] = Field(min_length=1, max_length=256)


class AgentStateSchemaDimOut(BaseModel):
    dim_index: int
    name: str
    description: str | None = None


class AgentStateSchemaOut(BaseModel):
    schema_id: int
    version: int
    state_scope: str
    dim_count: int
    codec: str
    created_at: str | None = None
    notes: str | None = None
    dims: list[AgentStateSchemaDimOut] = Field(default_factory=list)


class AgentStateObservationRequest(BaseModel):
    schema_id: int = Field(ge=1)
    schema_version: int = Field(ge=1)
    state_scope: str = Field(min_length=1, max_length=64)
    vector: list[float] = Field(min_length=1, max_length=2048)
    ts_ms: int | None = Field(default=None, ge=0)
    agent_id: str | None = Field(default=None, max_length=256)
    job_id: str | None = Field(default=None, max_length=256)
    task_id: str | None = Field(default=None, max_length=256)
    step_index: int | None = Field(default=None, ge=0)
    reward: float | None = None
    source_kind: str | None = Field(default=None, max_length=128)
    source_ref: str | None = Field(default=None, max_length=512)
    update_head: bool = True


class AgentStateHeadOut(BaseModel):
    agent_id: str
    state_scope: str
    schema_id: int
    schema_version: int
    ts_ms: int
    observation_id: int | None = None
    vector: list[float] = Field(default_factory=list)
    dim_names: list[str] = Field(default_factory=list)


class AgentStateObservationResponse(BaseModel):
    observation_id: int
    head_updated: bool
    schema_info: AgentStateSchemaOut
    vector: list[float] = Field(default_factory=list)
    head: AgentStateHeadOut | None = None


class AgentStateHeadsResponse(BaseModel):
    heads: list[AgentStateHeadOut] = Field(default_factory=list)


_REMOTE_ALLOWED_COMMANDS = {
    COMMAND_ASSESS_TACKLE_RESULTS,
    COMMAND_RUN_TACKLE_PROMPT,
}


@router.post("/commands", response_model=EnqueueCommandResponse)
def enqueue_command(
    request: EnqueueCommandRequest,
    db: Session = Depends(get_agent_session_dep),
) -> EnqueueCommandResponse:
    command_type = (request.command_type or "").strip()
    if command_type not in _REMOTE_ALLOWED_COMMANDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported command_type: {command_type}. Allowed: {sorted(_REMOTE_ALLOWED_COMMANDS)}",
        )

    now = utcnow()
    delay_seconds = _clamp_int(int(request.delay_seconds or 0), lo=0, hi=86400)
    priority = _clamp_int(int(request.priority or 0), lo=-50, hi=100)
    max_attempts = _clamp_int(int(request.max_attempts or 0), lo=1, hi=10)

    payload = dict(request.payload or {})
    # Avoid unbounded DB growth when callers send huge structures (e.g. full
    # limma tables). Store a conservative pruned version; callers can retain
    # raw artifacts in their own telemetry store.
    payload = _prune_json_for_storage(payload)

    row = AgentCommand(
        command_type=command_type,
        status="queued",
        priority=int(priority),
        created_at=now,
        updated_at=now,
        available_at=now + timedelta(seconds=int(delay_seconds)),
        attempts=0,
        max_attempts=int(max_attempts),
        payload_json=payload,
        result_json={},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return EnqueueCommandResponse(command_id=int(row.id))


@router.get("/commands/{command_id}", response_model=AgentCommandOut)
def get_command(
    command_id: int,
    db: Session = Depends(get_agent_session_dep),
) -> AgentCommandOut:
    row = db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Command not found.")

    return AgentCommandOut(
        id=int(row.id),
        command_type=str(row.command_type or ""),
        status=str(row.status or ""),
        priority=int(row.priority or 0),
        available_at=row.available_at.isoformat() if getattr(row, "available_at", None) else None,
        updated_at=row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
        claimed_at=row.claimed_at.isoformat() if getattr(row, "claimed_at", None) else None,
        started_at=row.started_at.isoformat() if getattr(row, "started_at", None) else None,
        ended_at=row.ended_at.isoformat() if getattr(row, "ended_at", None) else None,
        attempts=int(row.attempts or 0),
        max_attempts=int(row.max_attempts or 0),
        error=row.error,
        payload=dict(row.payload_json or {}),
        result=dict(row.result_json or {}),
    )


@router.post("/state/schema", response_model=AgentStateSchemaOut)
def upsert_state_schema(
    request: AgentStateSchemaUpsertRequest,
    db: Session = Depends(get_agent_state_session_dep),
) -> AgentStateSchemaOut:
    try:
        payload = register_schema_version(
            db,
            schema_id=int(request.schema_id),
            version=int(request.version),
            state_scope=str(request.state_scope),
            codec=str(request.codec),
            notes=request.notes,
            dims=[item.model_dump() for item in request.dims],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AgentStateSchemaOut.model_validate(payload)


@router.get("/state/schema/{schema_id}/{version}", response_model=AgentStateSchemaOut)
def get_state_schema(
    schema_id: int,
    version: int,
    db: Session = Depends(get_agent_state_session_dep),
) -> AgentStateSchemaOut:
    payload = get_schema(db, schema_id=int(schema_id), version=int(version))
    if payload is None:
        raise HTTPException(status_code=404, detail="State schema not found.")
    return AgentStateSchemaOut.model_validate(payload)


@router.post("/state/observations", response_model=AgentStateObservationResponse)
def observe_state(
    request: AgentStateObservationRequest,
    db: Session = Depends(get_agent_state_session_dep),
) -> AgentStateObservationResponse:
    try:
        payload = append_observation(
            db,
            schema_id=int(request.schema_id),
            schema_version=int(request.schema_version),
            state_scope=str(request.state_scope),
            vector=list(request.vector),
            ts_ms=request.ts_ms,
            agent_id=request.agent_id,
            job_id=request.job_id,
            task_id=request.task_id,
            step_index=request.step_index,
            reward=request.reward,
            source_kind=request.source_kind,
            source_ref=request.source_ref,
            update_head=bool(request.update_head),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    head_payload = None
    if request.update_head and request.agent_id:
        heads = list_heads(
            db,
            agent_id=str(request.agent_id),
            state_scope=str(request.state_scope),
            limit=1,
        )
        if heads:
            head_payload = AgentStateHeadOut.model_validate(heads[0])

    return AgentStateObservationResponse(
        observation_id=int(payload["observation_id"]),
        head_updated=bool(payload["head_updated"]),
        schema_info=AgentStateSchemaOut.model_validate(payload["schema"]),
        vector=list(payload["vector"]),
        head=head_payload,
    )


@router.get("/state/heads", response_model=AgentStateHeadsResponse)
def get_state_heads(
    agent_id: str | None = Query(default=None, max_length=256),
    state_scope: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_agent_state_session_dep),
) -> AgentStateHeadsResponse:
    payload = list_heads(
        db,
        agent_id=agent_id,
        state_scope=state_scope,
        limit=int(limit),
    )
    return AgentStateHeadsResponse(
        heads=[AgentStateHeadOut.model_validate(item) for item in payload]
    )


_TOKEN_ID_RE = re.compile(r"^(?P<prefix>[a-zA-Z]+)?0*(?P<num>[0-9]{1,10})$")


def _normalize_token(token: str) -> str:
    raw = token.strip()
    if not raw:
        return ""
    raw = raw.replace("\\", "/").rstrip("/")
    return raw.split("/")[-1].strip()


def _token_to_int_id(token: str) -> int | None:
    if not token:
        return None
    if token.isdigit():
        try:
            return int(token)
        except Exception:
            return None
    match = _TOKEN_ID_RE.match(token)
    if not match:
        return None
    try:
        return int(match.group("num"))
    except Exception:
        return None


class ProjectsResolveRequest(BaseModel):
    tokens: list[str] = Field(default_factory=list)


class ResolvedProject(BaseModel):
    token: str
    project_id: int
    display_id: str
    status: str | None = None
    title: str | None = None


class ProjectsResolveResponse(BaseModel):
    projects: list[ResolvedProject] = Field(default_factory=list)
    unknown_tokens: list[str] = Field(default_factory=list)


@router.post("/projects/resolve", response_model=ProjectsResolveResponse)
def resolve_projects(
    request: ProjectsResolveRequest,
    db: Session = Depends(get_session_dep),
) -> ProjectsResolveResponse:
    """Resolve filesystem tokens (e.g. folder names) to iSPEC projects.

    This endpoint exists so remote agents do not need to hardcode parsing rules
    for converting folder names like MSPC001498 into integer project ids.
    """

    normalized: list[str] = []
    for token in request.tokens:
        if not isinstance(token, str):
            continue
        base = _normalize_token(token)
        if base:
            normalized.append(base)

    # Bound request size to keep LAN agents from accidentally sending huge payloads.
    if len(normalized) > 20_000:
        raise HTTPException(status_code=413, detail="Too many tokens.")

    # Build query filters.
    display_ids_upper = {t.upper() for t in normalized if t}
    numeric_ids = {i for t in normalized if (i := _token_to_int_id(t)) is not None}

    if not display_ids_upper and not numeric_ids:
        return ProjectsResolveResponse(projects=[], unknown_tokens=normalized)

    clauses = []
    if numeric_ids:
        clauses.append(Project.id.in_(numeric_ids))
    if display_ids_upper:
        clauses.append(func.upper(Project.prj_PRJ_DisplayID).in_(display_ids_upper))

    query = db.query(Project).filter(or_(*clauses))
    rows = query.all()

    by_id: dict[int, Project] = {int(r.id): r for r in rows}
    by_display_upper: dict[str, Project] = {}
    for row in rows:
        display = getattr(row, "prj_PRJ_DisplayID", None)
        if isinstance(display, str) and display:
            by_display_upper[display.upper()] = row

    projects: list[ResolvedProject] = []
    unknown: list[str] = []
    for token in normalized:
        if not token:
            continue
        proj = by_display_upper.get(token.upper())
        if proj is None:
            token_id = _token_to_int_id(token)
            if token_id is not None:
                proj = by_id.get(int(token_id))

        if proj is None:
            unknown.append(token)
            continue

        display_id = getattr(proj, "prj_PRJ_DisplayID", None)
        if not (isinstance(display_id, str) and display_id.strip()):
            display_id = f"MSPC{int(proj.id):06d}"

        projects.append(
            ResolvedProject(
                token=token,
                project_id=int(proj.id),
                display_id=str(display_id),
                status=getattr(proj, "prj_Status", None),
                title=getattr(proj, "prj_ProjectTitle", None),
            )
        )

    return ProjectsResolveResponse(projects=projects, unknown_tokens=unknown)


class ProjectIndexRow(BaseModel):
    project_id: int
    display_id: str
    status: str | None = None
    title: str | None = None


class ProjectsIndexResponse(BaseModel):
    projects: list[ProjectIndexRow] = Field(default_factory=list)


@router.get("/projects/index", response_model=ProjectsIndexResponse)
def list_projects_index(
    min_id: int | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_session_dep),
) -> ProjectsIndexResponse:
    """Return a lightweight project index for agents (API-key only)."""

    query = db.query(Project)
    if min_id is not None:
        query = query.filter(Project.id >= int(min_id))
    rows = query.order_by(Project.id.asc()).limit(int(limit)).all()

    projects: list[ProjectIndexRow] = []
    for proj in rows:
        display_id = getattr(proj, "prj_PRJ_DisplayID", None)
        if not (isinstance(display_id, str) and display_id.strip()):
            display_id = f"MSPC{int(proj.id):06d}"
        projects.append(
            ProjectIndexRow(
                project_id=int(proj.id),
                display_id=str(display_id),
                status=getattr(proj, "prj_Status", None),
                title=getattr(proj, "prj_ProjectTitle", None),
            )
        )

    return ProjectsIndexResponse(projects=projects)

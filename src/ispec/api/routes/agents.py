from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ispec.agent.connect import get_agent_session_dep
from ispec.agent.models import AgentEvent
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

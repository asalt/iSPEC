from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ispec.agent.connect import get_agent_session_dep
from ispec.agent.models import AgentEvent

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

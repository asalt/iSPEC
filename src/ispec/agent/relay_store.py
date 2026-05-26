from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ispec.agent.commands import COMMAND_LOCAL_RELAY_REQUEST
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent
from ispec.agent.relay_constants import (
    EVENT_RELAY_RECEIPT,
    EVENT_RELAY_REQUEST_ENQUEUED,
    FAILURE_INVALID_REQUEST,
    RELAY_AGENT_ID,
    RELAY_SCHEMA_VERSION,
)
from ispec.agent.relay_normalize import normalize_relay_request
from ispec.agent.relay_utils import now_plus, utcnow
from ispec.assistant.slack_tmux_bridge import stable_json


def enqueue_relay_request(
    db: Session,
    *,
    request: dict[str, Any],
    priority: int = 0,
    delay_seconds: int = 0,
    max_attempts: int = 1,
) -> tuple[AgentCommand, dict[str, Any]]:
    normalized, error = normalize_relay_request(request)
    if normalized is None:
        raise ValueError(error or FAILURE_INVALID_REQUEST)

    now = utcnow()
    row = AgentCommand(
        command_type=COMMAND_LOCAL_RELAY_REQUEST,
        status="queued",
        priority=max(-50, min(1000, int(priority or 0))),
        created_at=now,
        updated_at=now,
        available_at=now_plus(int(delay_seconds or 0)),
        attempts=0,
        max_attempts=max(1, min(10, int(max_attempts or 1))),
        payload_json={"relay_request": normalized},
        result_json={},
    )
    db.add(row)
    db.flush()

    event_payload = {
        "schema_version": RELAY_SCHEMA_VERSION,
        "request": normalized,
        "command_id": int(row.id),
        "enqueued_at": now.isoformat(),
    }
    db.add(
        AgentEvent(
            agent_id=RELAY_AGENT_ID,
            event_type=EVENT_RELAY_REQUEST_ENQUEUED,
            ts=now,
            received_at=now,
            name="relay_request_enqueued",
            severity="info",
            correlation_id=str(normalized.get("request_id") or ""),
            payload_json=stable_json(event_payload),
        )
    )
    db.commit()
    db.refresh(row)
    return row, event_payload


def record_relay_receipt(
    *,
    command_id: int | None,
    request: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    now = utcnow()
    with get_agent_session() as db:
        db.add(
            AgentEvent(
                agent_id=RELAY_AGENT_ID,
                event_type=EVENT_RELAY_RECEIPT,
                ts=now,
                received_at=now,
                name="relay_receipt",
                severity="info" if bool(receipt.get("ok")) else "warning",
                correlation_id=str(request.get("request_id") or ""),
                payload_json=stable_json(
                    {
                        "schema_version": RELAY_SCHEMA_VERSION,
                        "command_id": int(command_id) if command_id is not None else None,
                        "request": request,
                        "receipt": receipt,
                    }
                ),
            )
        )
        db.commit()

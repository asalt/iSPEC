from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import delete, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ispec.agent.commands import COMMAND_LOCAL_RELAY_REQUEST
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent, RelayDedupKey
from ispec.agent.relay_dedup import (
    KNOWN_OUTCOMES,
    expired_keys,
    payload_fingerprint,
    request_key,
    retention_deadline,
)
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

    for attachment in normalized["attachments"]:
        attachment["path"] = str(Path(attachment["path"]).expanduser().absolute())
    fingerprint = payload_fingerprint(normalized)
    now = utcnow()
    identity_key = request_key(normalized)
    payload_key = "payload:" + fingerprint
    try:
        # First statement is a write. The unique key reserves the operation across
        # processes, and the job + both keys + event commit (or roll back) together.
        inserted = db.execute(
            insert(RelayDedupKey)
            .values(key=identity_key, fingerprint=fingerprint, created_at=now)
            .on_conflict_do_nothing(index_elements=["key"])
        ).rowcount
        identity = db.get(RelayDedupKey, identity_key, populate_existing=True)
        if not inserted:
            if identity.fingerprint != fingerprint:
                raise ValueError("relay_request_id_conflict")
            row = db.get(AgentCommand, identity.command_id, populate_existing=True)
            if row is None:
                raise ValueError("relay_dedup_missing_command")
            return _enqueue_result(
                db, row, normalized, deduplicated=True, match="request_id"
            )

        db.execute(
            delete(RelayDedupKey).where(
                RelayDedupKey.key == payload_key,
                RelayDedupKey.key.in_(
                    expired_keys(db, now=now, limit=10_000).where(
                        RelayDedupKey.key == payload_key
                    )
                ),
            ),
            execution_options={"synchronize_session": False},
        )
        claimed = db.execute(
            insert(RelayDedupKey)
            .values(key=payload_key, fingerprint=fingerprint, created_at=now)
            .on_conflict_do_nothing(index_elements=["key"])
        ).rowcount
        payload_claim = db.get(RelayDedupKey, payload_key, populate_existing=True)
        if not claimed:
            row = db.get(AgentCommand, payload_claim.command_id, populate_existing=True)
            if row is None:
                raise ValueError("relay_dedup_missing_command")
            identity.command_id = row.id
            if payload_claim.expires_at is not None:
                identity.expires_at = retention_deadline(identity_key, now=now)
            return _enqueue_result(
                db, row, normalized, deduplicated=True, match="payload"
            )

        row = _new_relay_command(
            normalized,
            priority=priority,
            delay_seconds=delay_seconds,
            max_attempts=max_attempts,
        )
        db.add(row)
        db.flush()
        identity.command_id = payload_claim.command_id = row.id
        event_payload = _enqueue_payload(
            row, normalized, deduplicated=False, match=None
        )
        db.add(
            AgentEvent(
                agent_id=RELAY_AGENT_ID,
                event_type=EVENT_RELAY_REQUEST_ENQUEUED,
                ts=now,
                received_at=now,
                name="relay_request_enqueued",
                severity="info",
                correlation_id=normalized["request_id"],
                payload_json=stable_json(event_payload),
            )
        )
        db.commit()
        db.refresh(row)
        return row, event_payload
    except Exception:
        db.rollback()
        raise


def _new_relay_command(
    normalized: dict[str, Any], *, priority: int, delay_seconds: int, max_attempts: int
) -> AgentCommand:
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
    return row


def _enqueue_payload(
    row: AgentCommand,
    submitted: dict[str, Any],
    *,
    deduplicated: bool,
    match: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": RELAY_SCHEMA_VERSION,
        "request": row.payload_json["relay_request"],
        "submitted_request_id": submitted["request_id"],
        "deduplicated": deduplicated,
        "duplicate_match": match,
        "command_id": int(row.id),
        "enqueued_at": row.created_at.isoformat(),
    }


def _enqueue_result(
    db: Session,
    row: AgentCommand,
    submitted: dict[str, Any],
    *,
    deduplicated: bool,
    match: str | None,
) -> tuple[AgentCommand, dict[str, Any]]:
    payload = _enqueue_payload(row, submitted, deduplicated=deduplicated, match=match)
    db.commit()
    return row, payload


def previous_relay_receipt(command_id: int | None) -> dict[str, Any] | None:
    if command_id is None:
        return None
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        if row is None or row.command_type != COMMAND_LOCAL_RELAY_REQUEST:
            return None
        receipt = dict(row.result_json or {})
        return receipt if receipt.get("delivery_outcome") else None


def begin_relay_delivery(
    command_id: int | None, request: dict[str, Any]
) -> dict[str, Any] | None:
    """Commit a one-shot execution marker before external IO; never expire it."""
    if command_id is None:
        return None  # Direct adapter calls are not the durable queue API.
    fingerprint = payload_fingerprint(request)
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        if row is None:
            # Compatibility for existing direct callers with correlation IDs.
            return None
        if row.command_type != COMMAND_LOCAL_RELAY_REQUEST:
            raise ValueError("relay_command_type_mismatch")
        prior = dict(row.result_json or {})
        if prior.get("delivery_outcome"):
            return prior
        if prior.get("relay_delivery_started"):
            return uncertain_receipt(request, error="relay_delivery_already_started")
        original, _ = normalize_relay_request(row.payload_json)
        if original is None:
            raise ValueError("relay_invalid_stored_request")
        claim = db.get(RelayDedupKey, request_key(original))
        expected = (
            claim.fingerprint if claim is not None else payload_fingerprint(original)
        )
        if expected != fingerprint or (
            claim is not None and claim.command_id != command_id
        ):
            return {
                "ok": False,
                "sent": False,
                "delivery_outcome": "failed",
                "error_type": "relay_payload_changed",
                "request_id": request["request_id"],
            }
        result = db.execute(
            update(AgentCommand)
            .where(
                AgentCommand.id == command_id,
                AgentCommand.status.in_(("queued", "running")),
                AgentCommand.result_json == row.result_json,
            )
            .values(
                result_json={**prior, "relay_delivery_started": utcnow().isoformat()}
            )
        )
        if result.rowcount != 1:
            return uncertain_receipt(request, error="relay_delivery_claim_lost")
    return None


def uncertain_receipt(request: dict[str, Any], *, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "sent": None,
        "delivery_outcome": "unknown",
        "error_type": error,
        "request_id": request.get("request_id"),
        "requires_reconciliation": True,
    }


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
                        "command_id": (
                            int(command_id) if command_id is not None else None
                        ),
                        "request": request,
                        "receipt": receipt,
                    }
                ),
            )
        )
        row = db.get(AgentCommand, command_id) if command_id is not None else None
        if row is not None and row.command_type == COMMAND_LOCAL_RELAY_REQUEST:
            row.result_json = {**dict(row.result_json or {}), **receipt}
            keys = (
                db.query(RelayDedupKey)
                .filter(RelayDedupKey.command_id == command_id)
                .all()
            )
            for key in keys:
                key.expires_at = (
                    retention_deadline(key.key, now=now)
                    if receipt.get("delivery_outcome") in KNOWN_OUTCOMES
                    else None
                )
        db.commit()

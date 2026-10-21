"""Relay identity and bounded retention, without a second coordination service."""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ispec.agent.models import AgentCommand, RelayDedupKey

PAYLOAD_RETENTION = timedelta(minutes=10)
REQUEST_RETENTION = timedelta(days=30)
# Hashing is outside database transactions and bounded independently of upload policy.
MAX_HASH_BYTES = 100 * 1024 * 1024
KNOWN_OUTCOMES = {"sent", "staged", "recorded", "failed"}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def request_key(request: dict[str, Any]) -> str:
    source = request["source"]
    return "request:" + digest([source["kind"], source["id"], request["request_id"]])


def payload_fingerprint(request: dict[str, Any]) -> str:
    payload = {key: value for key, value in request.items() if key != "request_id"}
    # Launching the same source from another directory does not make a new operation.
    payload["source"] = {key: request["source"][key] for key in ("kind", "id")}
    attachments = []
    for item in request["attachments"]:
        path = Path(item["path"]).expanduser().absolute()
        attachment = {**item, "path": str(path)}
        try:
            before = path.stat()
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_HASH_BYTES:
                raise ValueError("relay_attachment_not_hashable")
            with path.open("rb") as handle:
                sha = hashlib.sha256()
                size = 0
                while chunk := handle.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_HASH_BYTES:
                        raise ValueError("relay_attachment_not_hashable")
                    sha.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise ValueError("relay_attachment_changed_while_hashing")
            attachment["sha256"] = sha.hexdigest()
        except OSError:
            # Missing files can still be staged; becoming readable changes the identity.
            attachment["sha256"] = None
        attachments.append(attachment)
    payload["attachments"] = attachments
    return digest(payload)


def retention_deadline(key: str, *, now: datetime) -> datetime:
    return now + (
        REQUEST_RETENTION if key.startswith("request:") else PAYLOAD_RETENTION
    )


def expired_keys(db: Session, *, now: datetime, limit: int = 1000):
    return (
        select(RelayDedupKey.key)
        .join(AgentCommand, AgentCommand.id == RelayDedupKey.command_id)
        .where(
            RelayDedupKey.expires_at <= now,
            AgentCommand.status.in_(("succeeded", "failed")),
            AgentCommand.result_json["delivery_outcome"]
            .as_string()
            .in_(KNOWN_OUTCOMES),
        )
        .order_by(RelayDedupKey.expires_at, RelayDedupKey.key)
        .limit(max(1, min(limit, 10_000)))
    )


def prune_relay_keys(
    db: Session, *, now: datetime, confirm: bool = False, limit: int = 1000
) -> dict[str, Any]:
    eligible = expired_keys(db, now=now, limit=limit)
    if not confirm:
        return {
            "dry_run": True,
            "eligible": len(db.scalars(eligible).all()),
            "deleted": 0,
        }
    # Eligibility and deletion share a SQL statement, including the pending/unknown guard.
    result = db.execute(delete(RelayDedupKey).where(RelayDedupKey.key.in_(eligible)))
    db.commit()
    return {"dry_run": False, "deleted": result.rowcount}

"""Read legacy-sync state from durable command receipts, not a polling tick."""

from datetime import UTC, datetime, timedelta
import os

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.models import AgentCommand
from ispec.db.legacy_sync_observability import sync_log_directory


def sync_enabled() -> bool:
    return (os.getenv("ISPEC_LEGACY_SYNC_ENABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def sync_interval_seconds() -> int:
    try:
        value = int(os.getenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS") or 86400)
    except ValueError:
        value = 86400
    return max(60, min(7 * 86400, value))


def as_utc(value):
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def sync_commands(db):
    return db.query(AgentCommand).filter(
        AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL
    )


def completed_sync_commands(db):
    return sync_commands(db).filter(
        AgentCommand.status.in_(["succeeded", "failed"]),
        AgentCommand.payload_json["dry_run"].as_boolean().is_not(True),
        AgentCommand.result_json["cancelled"].as_boolean().is_not(True),
    )


def last_sync_attempt(db):
    return (
        completed_sync_commands(db)
        .order_by(AgentCommand.ended_at.desc(), AgentCommand.id.desc())
        .first()
    )


def command_summary(row):
    if row is None:
        return None
    started, ended = as_utc(row.started_at), as_utc(row.ended_at)
    return {
        "id": row.id,
        "status": row.status,
        "attempts": row.attempts,
        "max_attempts": row.max_attempts,
        "error": row.error,
        "dry_run": bool((row.payload_json or {}).get("dry_run")),
        "created_at": as_utc(row.created_at).isoformat(),
        "updated_at": as_utc(row.updated_at).isoformat(),
        "started_at": started.isoformat() if started else None,
        "ended_at": ended.isoformat() if ended else None,
        "available_at": (
            as_utc(row.available_at).isoformat() if row.available_at else None
        ),
        "duration_ms": (
            int((ended - started).total_seconds() * 1000) if started and ended else None
        ),
        "result": row.result_json,
    }


def legacy_sync_status(db, *, now=None):
    now = as_utc(now) or datetime.now(UTC)
    active = (
        sync_commands(db)
        .filter(AgentCommand.status.in_(["queued", "running"]))
        .order_by(AgentCommand.id)
        .first()
    )
    last = last_sync_attempt(db)
    success = (
        completed_sync_commands(db)
        .filter(AgentCommand.status == "succeeded")
        .order_by(AgentCommand.ended_at.desc(), AgentCommand.id.desc())
        .first()
    )
    recent = sync_commands(db).order_by(AgentCommand.id.desc()).limit(15).all()
    enabled = sync_enabled()
    interval = sync_interval_seconds()
    next_due = (
        (as_utc(last.ended_at or last.updated_at) + timedelta(seconds=interval))
        if last
        else now
    )
    return {
        "enabled": enabled,
        "interval_seconds": interval,
        "next_due_at": next_due.isoformat() if enabled and not active else None,
        "due": bool(enabled and not active and next_due <= now),
        "active": command_summary(active),
        "last_attempt": command_summary(last),
        "last_success": command_summary(success),
        "recent_runs": [command_summary(row) for row in recent],
        "log_directory": str(sync_log_directory()),
    }

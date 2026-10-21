"""Observed process state, kept separate from historical run records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psutil


def process_is_running(pid: Any, create_time: Any = None) -> bool:
    try:
        process = psutil.Process(int(pid))
        if (
            create_time is not None
            and abs(process.create_time() - float(create_time)) > 0.01
        ):
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (ValueError, TypeError, psutil.NoSuchProcess):
        return False
    except psutil.AccessDenied:
        return True


def observed_state(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    recorded = payload.get("status")
    running = recorded == "running" and process_is_running(
        payload.get("pid"), payload.get("process_create_time")
    )
    result.update(recorded_status=recorded, running=running)
    if recorded == "running" and not running:
        result.update(status="stale", status_reason="recorded_process_not_running")
    return result


def observed_run_status(
    status: str, updated_at: datetime | None, *, max_age_seconds: float = 120
) -> str:
    """Heartbeat-only view also works when the worker is on another host."""
    if status != "running":
        return status
    if updated_at is None:
        return "stale"
    timestamp = (
        updated_at.replace(tzinfo=UTC) if updated_at.tzinfo is None else updated_at
    )
    return (
        "running"
        if (datetime.now(UTC) - timestamp).total_seconds() <= max_age_seconds
        else "stale"
    )

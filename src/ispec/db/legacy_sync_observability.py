"""Compact per-run evidence for legacy sync, independent of the supervisor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from uuid import uuid4

from ispec.config.paths import resolve_log_dir
from ispec.logging import get_logger

logger = get_logger(__name__)


def sync_log_directory() -> Path:
    location = resolve_log_dir()
    return Path(location.path or location.value).expanduser() / "legacy-sync"


def safe_sync_error(exc: Exception) -> str:
    # Request exceptions can contain credential-bearing URLs. Keep diagnostics,
    # but never copy URL credentials or query strings into the operations view.
    message = re.sub(r"https?://[^\s)]+", "<legacy endpoint>", str(exc))
    # urllib3 renders the path/query separately from HTTPConnectionPool's host.
    # Remove that query too, keeping the trailing connection-failure cause.
    message = re.sub(
        r"with url: (?!<legacy endpoint>)[^\s)]+",
        "with url: <legacy endpoint>",
        message,
    )
    return f"{type(exc).__name__}: {message}"[:1000]


def append_sync_event(event: dict[str, Any]) -> None:
    now = datetime.now(UTC)
    directory = sync_log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y-%m-%d}.jsonl"
    encoded = (json.dumps(event, ensure_ascii=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)
    if event["event"] == "started":
        cutoff = now.date() - timedelta(days=30)
        for old in directory.glob("*.jsonl"):
            try:
                day = datetime.strptime(old.stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                old.unlink(missing_ok=True)


class LegacySyncFailure(RuntimeError):
    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        super().__init__(summary["error"])


class SyncReport:
    def __init__(self, *, dry_run: bool, command_id: int | None, progress=None):
        self.progress = progress
        self.started = time.monotonic()
        self.summary: dict[str, Any] = {}
        self.run: dict[str, Any] = {
            "sync_id": uuid4().hex,
            "command_id": command_id,
            "dry_run": dry_run,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "phases": [],
            "log_directory": str(sync_log_directory()),
        }
        self.emit("started")

    def emit(self, event: str) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **self.run}
        try:
            append_sync_event(record)
        except OSError as exc:
            self.run["log_error"] = safe_sync_error(exc)
            logger.warning(
                "Legacy sync event log unavailable: %s", self.run["log_error"]
            )
        if self.progress is not None:
            try:
                self.progress(dict(self.run))
            except Exception as exc:
                self.run["progress_error"] = safe_sync_error(exc)
                logger.warning(
                    "Legacy sync progress update failed: %s", self.run["progress_error"]
                )

    def phase(self, name: str, operation: Callable, **kwargs):
        phase = {"name": name, "status": "running"}
        self.run["phases"].append(phase)
        started = time.monotonic()
        self.emit("phase_started")
        try:
            result = operation(**kwargs)
        except Exception as exc:
            phase.update(
                status="failed", duration_ms=int((time.monotonic() - started) * 1000)
            )
            self.run.update(status="failed", failed_phase=name)
            self.summary.update(ok=False, error=safe_sync_error(exc))
            self.finish()
            raise LegacySyncFailure(self.summary) from exc
        phase.update(
            status="succeeded", duration_ms=int((time.monotonic() - started) * 1000)
        )
        self.summary[name] = result
        # Summaries contain counts/IDs, never source rows or comment bodies.
        phase["counts"] = {
            key: result[key]
            for key in ("items", "inserted", "updated", "conflicted", "backfilled")
            if key in result
        }
        self.emit("phase_finished")
        return result

    def finish(self) -> dict[str, Any]:
        if self.run["status"] == "running":
            self.run["status"] = "succeeded"
            self.summary["ok"] = True
        self.run["ended_at"] = datetime.now(UTC).isoformat()
        self.run["duration_ms"] = int((time.monotonic() - self.started) * 1000)
        self.run["error"] = self.summary.get("error")
        self.summary["sync_run"] = self.run
        self.emit("finished")
        return self.summary

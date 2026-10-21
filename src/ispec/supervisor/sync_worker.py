"""One claimed inbound sync in a CLI child process, not a second scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.config.paths import resolve_log_dir
from ispec.logging import get_logger
from ispec.supervisor.runtime_status import process_is_running

logger = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class SyncWorker:
    process: subprocess.Popen
    command_id: int
    run_id: str
    attempt: int
    log_path: str
    started_monotonic: float


def worker_is_running(result: dict) -> bool:
    worker = result.get("worker") or {}
    return bool(worker.get("process_create_time")) and process_is_running(
        worker.get("pid"), worker.get("process_create_time")
    )


def reap_lost_sync_workers(db, *, agent_id: str) -> None:
    """Reconcile observed dead children during ordinary queue inspection.

    No retry is implied: pages may have committed. Live or just-starting
    workers are left alone, including children inherited across a restart.
    """
    rows = (
        db.query(AgentCommand)
        .filter(
            AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL,
            AgentCommand.status == "running",
            AgentCommand.claimed_by_agent_id == agent_id,
        )
        .all()
    )
    for row in rows:
        previous = dict(row.result_json or {})
        worker = dict(previous.get("worker") or {})
        if not worker.get("process_create_time") or worker_is_running(previous):
            continue
        worker.update(state="lost", observed_dead_at=datetime.now(UTC).isoformat())
        db.query(AgentCommand).filter(
            AgentCommand.id == row.id,
            AgentCommand.status == "running",
            AgentCommand.result_json == previous,
        ).update(
            {
                AgentCommand.status: "failed",
                AgentCommand.error: "sync_worker_lost",
                AgentCommand.ended_at: datetime.now(UTC),
                AgentCommand.updated_at: datetime.now(UTC),
                AgentCommand.result_json: {**previous, "worker": worker},
            },
            synchronize_session=False,
        )


def _owned_command(db, *, command_id: int, run_id: str, attempt: int):
    return db.query(AgentCommand).filter(
        AgentCommand.id == command_id,
        AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL,
        AgentCommand.claimed_by_run_id == run_id,
        AgentCommand.attempts == attempt,
    )


def launch_sync_worker(*, command_id: int, run_id: str, attempt: int) -> SyncWorker:
    location = resolve_log_dir()
    directory = Path(location.path or location.value) / "supervisor-jobs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"sync-{command_id}-attempt-{attempt}.log"
    token = uuid.uuid4().hex
    with get_agent_session() as db:
        query = _owned_command(
            db, command_id=command_id, run_id=run_id, attempt=attempt
        )
        row = query.filter(AgentCommand.status == "running").one()
        result = dict(row.result_json or {})
        if worker_is_running(result):
            raise RuntimeError("sync_worker_already_running")
        result["worker"] = {
            "token": token,
            "log_path": str(log_path),
            "state": "starting",
        }
        row.result_json = result

    # No shell, credentials, or user-supplied SQL in argv. The payload stays in
    # the claimed receipt; the child inherits the parent's resolved environment.
    argv = [
        sys.executable,
        "-m",
        "ispec",
        "supervisor",
        "sync-worker",
        "--command-id",
        str(command_id),
        "--run-id",
        run_id,
        "--attempt",
        str(attempt),
        "--worker-token",
        token,
    ]
    with os.fdopen(
        os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab"
    ) as log:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
    try:
        created = psutil.Process(process.pid).create_time()
    except psutil.Error:
        pass  # Parent polling records an early exit even before Python imports.
    else:
        try:
            with get_agent_session() as db:
                query = _owned_command(
                    db, command_id=command_id, run_id=run_id, attempt=attempt
                )
                # Do not overwrite the child's claim or first phase update.
                identity = {
                    **result["worker"],
                    "pid": process.pid,
                    "process_create_time": created,
                }
                query.filter(
                    AgentCommand.status == "running", AgentCommand.result_json == result
                ).update(
                    {AgentCommand.result_json: {**result, "worker": identity}},
                    synchronize_session=False,
                )
        except Exception:
            # The process exists now: never lose its handle because optional
            # early PID reporting failed. The child also records its identity.
            logger.warning(
                "Early worker identity reporting failed command_id=%s", command_id
            )
    return SyncWorker(
        process=process,
        command_id=command_id,
        run_id=run_id,
        attempt=attempt,
        log_path=str(log_path),
        started_monotonic=time.monotonic(),
    )


def run_sync_worker(
    *, command_id: int, run_id: str, attempt: int, worker_token: str
) -> int:
    from ispec.concurrency.thread_context import set_main_thread
    from ispec.supervisor.loop import _run_legacy_sync_all

    set_main_thread(owner="legacy-sync-worker")
    with get_agent_session() as db:
        query = _owned_command(
            db, command_id=command_id, run_id=run_id, attempt=attempt
        )
        row = query.filter(AgentCommand.status == "running").first()
        if row is None:
            raise RuntimeError("sync_worker_claim_mismatch")
        previous = dict(row.result_json or {})
        worker = dict(previous.get("worker") or {})
        if worker.get("token") != worker_token or worker.get("state") != "starting":
            raise RuntimeError("sync_worker_claim_mismatch")
        worker.update(
            state="running",
            pid=os.getpid(),
            process_create_time=psutil.Process().create_time(),
            started_at=datetime.now(UTC).isoformat(),
        )
        result = {**previous, "worker": worker}
        # A repeated invocation of this internal CLI must not run the same job.
        claimed = query.filter(
            AgentCommand.status == "running",
            AgentCommand.result_json["worker"]["state"].as_string() == "starting",
            AgentCommand.result_json["worker"]["token"].as_string() == worker_token,
        ).update(
            {
                AgentCommand.result_json: result,
                AgentCommand.updated_at: datetime.now(UTC),
            },
            synchronize_session=False,
        )
        if claimed != 1:
            raise RuntimeError("sync_worker_already_claimed")
        payload = dict(row.payload_json or {})

    execution = _run_legacy_sync_all(payload, command_id=command_id)
    code = 0 if execution.ok else 1
    with get_agent_session() as db:
        query = _owned_command(
            db, command_id=command_id, run_id=run_id, attempt=attempt
        )
        row = query.filter(AgentCommand.status == "running").first()
        if (
            row is None
            or (row.result_json or {}).get("worker", {}).get("token") != worker_token
        ):
            return 2
        worker.update(
            state="finished", exit_code=code, ended_at=datetime.now(UTC).isoformat()
        )
        finished = query.filter(
            AgentCommand.status == "running",
            AgentCommand.result_json == row.result_json,
        ).update(
            {
                AgentCommand.result_json: {**execution.result, "worker": worker},
                AgentCommand.status: "succeeded" if execution.ok else "failed",
                AgentCommand.error: execution.error,
                AgentCommand.ended_at: datetime.now(UTC),
                AgentCommand.updated_at: datetime.now(UTC),
            },
            synchronize_session=False,
        )
        if finished != 1:
            return 2
    return code


def stop_worker(worker: SyncWorker) -> None:
    if worker.process.poll() is None:
        worker.process.terminate()
        try:
            worker.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait(timeout=5)


def worker_timeout_seconds() -> float:
    try:
        return max(
            60.0, float(os.getenv("ISPEC_SUPERVISOR_SYNC_TIMEOUT_SECONDS", "21600"))
        )
    except ValueError:
        return 21600.0

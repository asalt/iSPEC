from __future__ import annotations

import subprocess

import pytest

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL, COMMAND_SLACK_POST_MESSAGE
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor import loop, sync_worker


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    set_main_thread(owner="pytest")
    with get_agent_session() as db:
        db.add(
            AgentRun(
                run_id="run-test", agent_id="test", kind="supervisor", status="running"
            )
        )
    return tmp_path


class FakeProcess:
    pid = 12345

    def __init__(self):
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout):
        return self.code


def claimed_sync():
    command_id = loop._enqueue_command(
        command_type=COMMAND_LEGACY_SYNC_ALL, payload={"dry_run": True}
    )
    cmd = loop._claim_next_command(agent_id="test", run_id="run-test")
    assert cmd.id == command_id
    return cmd


def prepared_worker(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(sync_worker.subprocess, "Popen", lambda *a, **k: process)
    cmd = claimed_sync()
    worker = sync_worker.launch_sync_worker(
        command_id=cmd.id, run_id="run-test", attempt=cmd.attempts
    )
    with get_agent_session() as db:
        token = db.get(AgentCommand, cmd.id).result_json["worker"]["token"]
    return cmd, worker, token


def test_slow_sync_does_not_block_slack_or_claim_second_sync(queue, monkeypatch):
    process = FakeProcess()
    launches = []

    def launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return process

    monkeypatch.setattr(sync_worker.subprocess, "Popen", launch)
    monkeypatch.setattr(
        loop,
        "_run_legacy_sync_all",
        lambda *a, **k: pytest.fail("parent must not sync"),
    )
    monkeypatch.setattr(
        loop,
        "_run_slack_post_message",
        lambda _: loop.CommandExecution(ok=True, result={"stub": True}),
    )
    sync_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    second_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    slack_id = loop._enqueue_command(command_type=COMMAND_SLACK_POST_MESSAGE)
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="run-test", broker=None
    )
    assert processor.tick()
    assert processor.has_inflight_work
    assert processor.tick()
    with get_agent_session() as db:
        assert db.get(AgentCommand, sync_id).status == "running"
        assert db.get(AgentCommand, second_id).status == "queued"
        assert db.get(AgentCommand, slack_id).status == "succeeded"
    assert len(launches) == 1
    argv, kwargs = launches[0]
    assert argv[1:5] == ["-m", "ispec", "supervisor", "sync-worker"]
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "shell" not in kwargs
    processor.stop()
    assert process.terminated
    with get_agent_session() as db:
        row = db.get(AgentCommand, sync_id)
        assert row.status == "failed"
        assert row.error == "sync_worker_interrupted"
        assert row.result_json["worker"]["exit_code"] == -15


@pytest.mark.parametrize("ok", [True, False])
def test_worker_finishes_own_receipt_and_rejects_duplicate(queue, monkeypatch, ok):
    cmd, worker, token = prepared_worker(monkeypatch)
    calls = []

    def execute(payload, *, command_id):
        calls.append(command_id)
        # Progress reporting retains worker identity.
        with get_agent_session() as db:
            assert (
                db.get(AgentCommand, command_id).result_json["worker"]["state"]
                == "running"
            )
        return loop.CommandExecution(
            ok=ok,
            result={"sync_run": {"status": "succeeded" if ok else "failed"}},
            error=None if ok else "stub_remote_failure",
        )

    monkeypatch.setattr(loop, "_run_legacy_sync_all", execute)
    assert sync_worker.run_sync_worker(
        command_id=cmd.id, run_id="run-test", attempt=cmd.attempts, worker_token=token
    ) == (0 if ok else 1)
    with get_agent_session() as db:
        row = db.get(AgentCommand, cmd.id)
        assert row.status == ("succeeded" if ok else "failed")
        assert row.result_json["worker"]["log_path"] == worker.log_path
        assert row.ended_at is not None
    with pytest.raises(RuntimeError, match="claim_mismatch"):
        sync_worker.run_sync_worker(
            command_id=cmd.id,
            run_id="run-test",
            attempt=cmd.attempts,
            worker_token=token,
        )
    assert calls == [cmd.id]


@pytest.mark.parametrize(
    "change", [{"run_id": "other"}, {"attempt": 20}, {"worker_token": "wrong"}]
)
def test_worker_rejects_wrong_claim_before_execution(queue, monkeypatch, change):
    cmd, _, token = prepared_worker(monkeypatch)
    monkeypatch.setattr(
        loop, "_run_legacy_sync_all", lambda *a, **k: pytest.fail("must not execute")
    )
    args = dict(
        command_id=cmd.id, run_id="run-test", attempt=cmd.attempts, worker_token=token
    )
    with pytest.raises(RuntimeError, match="claim_mismatch"):
        sync_worker.run_sync_worker(**{**args, **change})


def test_crash_keeps_partial_progress(queue, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(sync_worker.subprocess, "Popen", lambda *a, **k: process)
    command_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="run-test", broker=None
    )
    processor.tick()
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        row.result_json = {
            **row.result_json,
            "sync_run": {"phases": [{"phase": "projects", "status": "succeeded"}]},
        }
    process.code = -11
    assert processor.tick()
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        assert row.status == "failed"
        assert row.error == "sync_worker_exited:-11"
        assert row.result_json["sync_run"]["phases"][0]["status"] == "succeeded"


def test_timeout_stops_child(queue, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(sync_worker.subprocess, "Popen", lambda *a, **k: process)
    command_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="run-test", broker=None
    )
    processor.tick()
    monkeypatch.setattr(loop, "worker_timeout_seconds", lambda: -1)
    assert processor.tick()
    assert process.terminated
    with get_agent_session() as db:
        assert db.get(AgentCommand, command_id).error == "sync_worker_timeout"


def test_launch_failure_has_terminal_receipt(queue, monkeypatch):
    def fail(*a, **k):
        raise OSError("stub launch failure")

    monkeypatch.setattr(sync_worker.subprocess, "Popen", fail)
    command_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="run-test", broker=None
    )
    assert processor.tick()
    assert not processor.has_inflight_work
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        assert row.status == "failed"
        assert row.error == "sync_worker_launch_failed: OSError"
        assert row.result_json["worker"]["log_path"]


def test_recorded_success_survives_later_process_exit_failure(queue, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(sync_worker.subprocess, "Popen", lambda *a, **k: process)
    command_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    processor = loop._SupervisorCommandProcessor(
        agent_id="test", run_id="run-test", broker=None
    )
    processor.tick()
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        row.status = "succeeded"
        row.result_json = {
            **row.result_json,
            "ok": True,
            "sync_run": {"status": "succeeded"},
        }
    process.code = 1
    assert processor.tick()
    with get_agent_session() as db:
        row = db.get(AgentCommand, command_id)
        assert row.status == "succeeded"
        assert row.result_json["worker"]["exit_code"] == 1
        assert row.result_json["sync_run"]["status"] == "succeeded"


def test_orphan_live_child_is_not_recovered_or_duplicated(queue, monkeypatch):
    from datetime import timedelta

    cmd, _, _ = prepared_worker(monkeypatch)
    with get_agent_session() as db:
        row = db.get(AgentCommand, cmd.id)
        row.updated_at = loop.utcnow() - timedelta(hours=2)
        db.query(AgentRun).filter_by(run_id="run-test").one().status = "stopped"
    monkeypatch.setattr(loop, "worker_is_running", lambda _: True)
    result = loop._recover_stale_running_commands(agent_id="test", run_id="replacement")
    assert result["recovered"] == 0
    loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    assert loop._claim_next_command(agent_id="test", run_id="replacement") is None


def test_dead_orphan_is_failed_without_waiting_for_stale_timeout(queue, monkeypatch):
    cmd, _, _ = prepared_worker(monkeypatch)
    with get_agent_session() as db:
        row = db.get(AgentCommand, cmd.id)
        row.result_json = {
            "worker": {"pid": 12345, "process_create_time": 123.0},
            "sync_run": {"status": "running", "completed_phase": "projects"},
        }
    monkeypatch.setattr(sync_worker, "worker_is_running", lambda _: False)
    next_id = loop._enqueue_command(command_type=COMMAND_LEGACY_SYNC_ALL)
    next_cmd = loop._claim_next_command(agent_id="test", run_id="replacement")
    assert next_cmd.id == next_id
    with get_agent_session() as db:
        row = db.get(AgentCommand, cmd.id)
        assert row.status == "failed"
        assert row.error == "sync_worker_lost"
        assert row.result_json["sync_run"]["completed_phase"] == "projects"
        assert row.attempts == 1

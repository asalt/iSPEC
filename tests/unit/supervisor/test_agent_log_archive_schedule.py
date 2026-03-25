from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ispec.agent.commands import COMMAND_ARCHIVE_AGENT_LOGS
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import (
    CommandExecution,
    _ensure_agent_log_archive_scheduled_commands,
    _process_one_command,
)


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def _seed_supervisor_run(agent_db_path, *, now: datetime) -> None:
    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=now,
                updated_at=now,
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        agent_db.commit()


def test_supervisor_seeds_agent_log_archive_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    archive_db_path = tmp_path / "agent-archive.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_AGENT_ARCHIVE_DB_PATH", str(archive_db_path))
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_ENABLED", "1")
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_INTERVAL_SECONDS", "1800")
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_OLDER_THAN_DAYS", "30")
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_BATCH_SIZE", "123")
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_MAX_BATCHES", "4")

    fixed_now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    _seed_supervisor_run(agent_db_path, now=fixed_now)

    seeded = _ensure_agent_log_archive_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["ok"] is True
    assert seeded["scheduled"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_ARCHIVE_AGENT_LOGS)
            .filter(AgentCommand.status == "queued")
            .one()
        )
        assert cmd.payload_json["archive_db_file_path"] == str(archive_db_path)
        assert cmd.payload_json["older_than_days"] == 30
        assert cmd.payload_json["batch_size"] == 123
        assert cmd.payload_json["max_batches"] == 4
        assert cmd.payload_json["prune_live"] is True


def test_supervisor_processes_agent_log_archive_command_and_records_summary(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    archive_db_path = tmp_path / "agent-archive.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_AGENT_ARCHIVE_DB_PATH", str(archive_db_path))
    monkeypatch.setenv("ISPEC_AGENT_LOG_ARCHIVE_ENABLED", "1")

    fixed_now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    _seed_supervisor_run(agent_db_path, now=fixed_now)

    seeded = _ensure_agent_log_archive_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["scheduled"] == 1

    monkeypatch.setattr(
        supervisor_loop,
        "_run_agent_log_archive",
        lambda payload: CommandExecution(
            ok=True,
            result={
                "ok": True,
                "older_than_days": int(payload.get("older_than_days") or 14),
                "dry_run": False,
                "prune_live": True,
                "runs_archived": 2,
                "steps": {"matched": 5, "archived": 5, "pruned": 5},
                "events": {"matched": 3, "archived": 3, "pruned": 3},
                "commands": {"matched": 4, "archived": 4, "pruned": 4},
            },
        ),
    )

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.command_type == COMMAND_ARCHIVE_AGENT_LOGS).one()
        assert cmd.status == "succeeded"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        scheduler = run.summary_json.get("scheduler") if isinstance(run.summary_json, dict) else None
        assert isinstance(scheduler, dict)
        archive_state = scheduler.get("agent_log_archive")
        assert isinstance(archive_state, dict)
        assert archive_state.get("last_attempted_ok") is True
        assert archive_state.get("last_command_id") == int(cmd.id)
        assert archive_state.get("next_command_id") is None
        last_summary = archive_state.get("last_summary")
        assert isinstance(last_summary, dict)
        assert last_summary["steps"]["archived"] == 5
        assert last_summary["events"]["pruned"] == 3
        assert last_summary["commands"]["matched"] == 4

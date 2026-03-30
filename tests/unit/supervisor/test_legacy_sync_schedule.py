from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import _ensure_legacy_sync_scheduled_commands


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


def test_supervisor_seeds_legacy_sync_command_with_recent_comment_window(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS", "1800")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_LIMIT", "222")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_MAX_PROJECT_COMMENTS", "11")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_MAX_EXPERIMENT_RUNS", "9")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_RECENT_PROJECT_COMMENT_DAYS", "45")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_RECENT_PROJECT_COMMENT_SCAN_LIMIT", "3333")

    fixed_now = datetime(2026, 3, 27, 15, 0, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    _seed_supervisor_run(agent_db_path, now=fixed_now)

    seeded = _ensure_legacy_sync_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["ok"] is True
    assert seeded["scheduled"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL)
            .filter(AgentCommand.status == "queued")
            .one()
        )
        assert cmd.payload_json["limit"] == 222
        assert cmd.payload_json["max_project_comments"] == 11
        assert cmd.payload_json["max_experiment_runs"] == 9
        assert cmd.payload_json["recent_project_comment_days"] == 45
        assert cmd.payload_json["recent_project_comment_scan_limit"] == 3333
        assert cmd.payload_json["meta"]["run_id"] == "run-1"

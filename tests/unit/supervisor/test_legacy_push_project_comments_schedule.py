from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ispec.agent.commands import COMMAND_LEGACY_PUSH_PROJECT_COMMENTS
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import (
    CommandExecution,
    _ensure_legacy_push_project_comments_scheduled_commands,
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


def test_supervisor_seeds_legacy_project_comment_writeback_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    core_db_path = tmp_path / "core.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_DB_PATH", str(core_db_path))
    monkeypatch.setenv("ISPEC_LEGACY_API_URL", "http://legacy.example")
    monkeypatch.setenv("ISPEC_LEGACY_SCHEMA_PATH", str(tmp_path / "ispec-legacy-schema.json"))
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_INTERVAL_SECONDS", "900")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_PROJECT_ID", "1351")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_LIMIT", "77")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_DRY_RUN", "1")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_RECENT_DAYS", "14")

    fixed_now = datetime(2026, 3, 23, 15, 0, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    _seed_supervisor_run(agent_db_path, now=fixed_now)

    seeded = _ensure_legacy_push_project_comments_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["ok"] is True
    assert seeded["scheduled"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS)
            .filter(AgentCommand.status == "queued")
            .one()
        )
        assert cmd.payload_json["db_file_path"] == str(core_db_path)
        assert cmd.payload_json["legacy_url"] == "http://legacy.example"
        assert cmd.payload_json["schema_path"] == str(tmp_path / "ispec-legacy-schema.json")
        assert cmd.payload_json["project_id"] == 1351
        assert cmd.payload_json["limit"] == 77
        assert cmd.payload_json["dry_run"] is True
        assert cmd.payload_json["recent_days"] == 14
        assert cmd.payload_json["meta"]["run_id"] == "run-1"


def test_supervisor_processes_legacy_project_comment_writeback_and_records_summary(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "1")

    fixed_now = datetime(2026, 3, 23, 15, 0, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    _seed_supervisor_run(agent_db_path, now=fixed_now)

    seeded = _ensure_legacy_push_project_comments_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["scheduled"] == 1

    monkeypatch.setattr(
        supervisor_loop,
        "_run_legacy_push_project_comments",
        lambda payload: CommandExecution(
            ok=True,
            result={
                "selected": 9,
                "candidate_comments": 5,
                "projects": 2,
                "legacy_table": "iSPEC_ProjectHistory",
                "legacy_existing_items": 12,
                "already_present": 3,
                "would_insert": 2,
                "inserted": 2,
                "skipped_blank": 1,
                "skipped_system": 2,
                "duplicates_skipped": 1,
                "dry_run": bool(payload.get("dry_run")),
            },
        ),
    )

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS)
            .one()
        )
        assert cmd.status == "succeeded"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        scheduler = run.summary_json.get("scheduler") if isinstance(run.summary_json, dict) else None
        assert isinstance(scheduler, dict)
        push_state = scheduler.get("legacy_push_project_comments")
        assert isinstance(push_state, dict)
        assert push_state.get("last_attempted_ok") is True
        assert push_state.get("last_command_id") == int(cmd.id)
        assert push_state.get("next_command_id") is None
        last_summary = push_state.get("last_summary")
        assert isinstance(last_summary, dict)
        assert last_summary["legacy_table"] == "iSPEC_ProjectHistory"
        assert last_summary["candidate_comments"] == 5
        assert last_summary["inserted"] == 2
        assert last_summary["skipped_system"] == 2

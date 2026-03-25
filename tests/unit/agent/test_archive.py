from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ispec.agent.archive import archive_agent_logs
from ispec.agent.connect import _get_engine, get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun, AgentStep


def _utcnow() -> datetime:
    return datetime.now(UTC)


def test_archive_agent_logs_moves_old_terminal_rows_and_keeps_recent_rows(tmp_path):
    live_db = tmp_path / "agent-live.db"
    archive_db = tmp_path / "agent-archive.db"
    now = _utcnow()
    old_ts = now - timedelta(days=45)
    recent_ts = now - timedelta(days=2)

    with get_agent_session(live_db) as db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            created_at=old_ts,
            updated_at=recent_ts,
        )
        db.add(run)
        db.flush()

        db.add_all(
            [
                AgentStep(
                    run_pk=run.id,
                    step_index=0,
                    kind="old_step",
                    started_at=old_ts,
                    ended_at=old_ts,
                    duration_ms=10,
                    ok=False,
                    error="old",
                    chosen_json={"command_id": 1},
                    tool_results_json=[{"ok": False}],
                ),
                AgentStep(
                    run_pk=run.id,
                    step_index=1,
                    kind="recent_step",
                    started_at=recent_ts,
                    ended_at=recent_ts,
                    duration_ms=11,
                    ok=True,
                    chosen_json={"command_id": 2},
                    tool_results_json=[{"ok": True}],
                ),
                AgentEvent(
                    agent_id="agent-1",
                    event_type="metric",
                    ts=old_ts,
                    received_at=old_ts,
                    payload_json='{"old":true}',
                ),
                AgentEvent(
                    agent_id="agent-1",
                    event_type="metric",
                    ts=recent_ts,
                    received_at=recent_ts,
                    payload_json='{"recent":true}',
                ),
                AgentCommand(
                    command_type="old_cmd",
                    status="failed",
                    created_at=old_ts,
                    updated_at=old_ts,
                    started_at=old_ts,
                    ended_at=old_ts,
                    payload_json={"old": True},
                    result_json={"ok": False},
                    error="old",
                ),
                AgentCommand(
                    command_type="recent_cmd",
                    status="running",
                    created_at=recent_ts,
                    updated_at=recent_ts,
                    started_at=recent_ts,
                    payload_json={"recent": True},
                    result_json={},
                ),
            ]
        )

    summary = archive_agent_logs(
        agent_db_file_path=str(live_db),
        archive_db_file_path=str(archive_db),
        older_than_days=14,
        batch_size=2,
        max_batches=5,
        prune_live=True,
    )

    assert summary["steps"]["archived"] == 1
    assert summary["steps"]["pruned"] == 1
    assert summary["events"]["archived"] == 1
    assert summary["events"]["pruned"] == 1
    assert summary["commands"]["archived"] == 1
    assert summary["commands"]["pruned"] == 1
    assert summary["runs_archived"] == 1

    with get_agent_session(live_db) as db:
        assert db.query(AgentStep).count() == 1
        assert db.query(AgentStep).one().kind == "recent_step"
        assert db.query(AgentEvent).count() == 1
        assert db.query(AgentCommand).count() == 1
        assert db.query(AgentCommand).one().status == "running"
        assert db.query(AgentRun).count() == 1

    with get_agent_session(archive_db) as db:
        assert db.query(AgentRun).count() == 1
        assert db.query(AgentStep).count() == 1
        assert db.query(AgentStep).one().kind == "old_step"
        assert db.query(AgentEvent).count() == 1
        assert db.query(AgentCommand).count() == 1
        assert db.query(AgentCommand).one().command_type == "old_cmd"


def test_archive_agent_logs_dry_run_does_not_copy_or_delete(tmp_path):
    live_db = tmp_path / "agent-live.db"
    archive_db = tmp_path / "agent-archive.db"
    old_ts = _utcnow() - timedelta(days=30)

    with get_agent_session(live_db) as db:
        run = AgentRun(run_id="run-1", agent_id="agent-1", created_at=old_ts, updated_at=old_ts)
        db.add(run)
        db.flush()
        db.add(
            AgentStep(
                run_pk=run.id,
                step_index=0,
                kind="old_step",
                started_at=old_ts,
                ended_at=old_ts,
                ok=False,
            )
        )

    summary = archive_agent_logs(
        agent_db_file_path=str(live_db),
        archive_db_file_path=str(archive_db),
        older_than_days=14,
        batch_size=10,
        dry_run=True,
        prune_live=True,
    )

    assert summary["dry_run"] is True
    assert summary["steps"]["matched"] == 1
    assert summary["steps"]["archived"] == 0
    assert summary["steps"]["pruned"] == 0
    assert not archive_db.exists()

    with get_agent_session(live_db) as db:
        assert db.query(AgentStep).count() == 1


def test_archive_agent_logs_requires_distinct_archive_db(tmp_path):
    live_db = tmp_path / "agent-live.db"
    with get_agent_session(live_db):
        pass

    with pytest.raises(ValueError, match="must differ"):
        archive_agent_logs(
            agent_db_file_path=str(live_db),
            archive_db_file_path=str(live_db),
        )


def test_archive_agent_logs_requires_archive_path_when_not_dry_run(tmp_path, monkeypatch):
    live_db = tmp_path / "agent-live.db"
    with get_agent_session(live_db):
        pass
    monkeypatch.delenv("ISPEC_AGENT_ARCHIVE_DB_PATH", raising=False)

    with pytest.raises(ValueError, match="Missing archive database path"):
        archive_agent_logs(agent_db_file_path=str(live_db), archive_db_file_path=None, dry_run=False)


def teardown_function() -> None:
    _get_engine.cache_clear()

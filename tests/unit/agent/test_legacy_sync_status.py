from datetime import UTC, datetime, timedelta

import pytest

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.legacy_sync_status import legacy_sync_status
from ispec.agent.models import AgentCommand, AgentRun
from ispec.api.routes.ops import get_legacy_sync_status
from ispec.db.models import AuthUser, UserRole
from ispec.supervisor.loop import _ensure_legacy_sync_scheduled_commands
from ispec.concurrency.thread_context import set_main_thread
from fastapi import HTTPException


NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def completed(db, *, status, age, dry_run=False):
    end = NOW - timedelta(hours=age)
    row = AgentCommand(
        command_type=COMMAND_LEGACY_SYNC_ALL,
        status=status,
        started_at=end - timedelta(seconds=12),
        ended_at=end,
        updated_at=end,
        payload_json={"dry_run": dry_run},
        result_json={},
    )
    db.add(row)
    db.flush()
    return row


def test_status_separates_success_failure_preview_and_active_commands(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS", "86400")
    with get_agent_session(tmp_path / "agent.db") as db:
        success = completed(db, status="succeeded", age=30)
        failure = completed(db, status="failed", age=2)
        completed(db, status="succeeded", age=1, dry_run=True)
        active = AgentCommand(
            command_type=COMMAND_LEGACY_SYNC_ALL, status="queued", payload_json={}
        )
        db.add(active)
        db.flush()
        db.add_all(
            AgentCommand(command_type="unrelated", status="succeeded")
            for _ in range(40)
        )
        db.flush()
        result = legacy_sync_status(db, now=NOW)
        assert result["last_success"]["id"] == success.id
        assert result["last_attempt"]["id"] == failure.id
        assert result["active"]["id"] == active.id
        assert result["last_attempt"]["duration_ms"] == 12000
        assert result["last_attempt"]["ended_at"].endswith("+00:00")
        assert len(result["recent_runs"]) == 4
        assert result["next_due_at"] is None


@pytest.mark.parametrize("age,scheduled", [(2, 0), (26, 1)])
def test_daily_schedule_uses_durable_receipts_after_restart(
    tmp_path, monkeypatch, age, scheduled
):
    path = tmp_path / "agent.db"
    set_main_thread(owner="pytest")
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(path))
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS", "86400")
    monkeypatch.setattr("ispec.supervisor.loop.utcnow", lambda: NOW)
    with get_agent_session(path) as db:
        completed(db, status="failed", age=age)
        completed(db, status="succeeded", age=0, dry_run=True)
        db.add(
            AgentRun(
                run_id="new-run", agent_id="test", kind="supervisor", summary_json={}
            )
        )
    result = _ensure_legacy_sync_scheduled_commands(agent_id="test", run_id="new-run")
    assert result["scheduled"] == scheduled
    if not scheduled:
        assert result["reason"] == "interval_not_elapsed"


def test_disabled_status_and_auth_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "0")
    monkeypatch.delenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS", raising=False)
    with get_agent_session(tmp_path / "agent.db") as db:
        result = legacy_sync_status(db, now=NOW)
        assert result["next_due_at"] is None
        assert result["due"] is False
        for role in (UserRole.client, UserRole.viewer):
            with pytest.raises(HTTPException) as error:
                get_legacy_sync_status(
                    agent_db=db,
                    user=AuthUser(username="alex", role=role, is_active=True),
                )
            assert error.value.status_code == 403
        allowed = get_legacy_sync_status(
            agent_db=db,
            user=AuthUser(username="alex", role=UserRole.admin, is_active=True),
        )
        assert allowed["enabled"] is False


def test_cancelled_receipt_does_not_postpone_next_import(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS", "86400")
    with get_agent_session(tmp_path / "agent.db") as db:
        imported = completed(db, status="succeeded", age=26)
        cancelled = completed(db, status="failed", age=0)
        cancelled.result_json = {"cancelled": True}
        db.flush()
        result = legacy_sync_status(db, now=NOW)
        assert result["last_attempt"]["id"] == imported.id
        assert result["due"] is True
        assert result["next_due_at"] == (NOW - timedelta(hours=2)).isoformat()
        assert result["recent_runs"][0]["id"] == cancelled.id

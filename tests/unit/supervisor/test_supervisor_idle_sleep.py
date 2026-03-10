from __future__ import annotations

from datetime import timedelta

from ispec.agent.models import AgentCommand, AgentRun
from ispec.supervisor.loop import _supervisor_dynamic_idle_sleep_seconds, utcnow


def test_supervisor_dynamic_sleep_follows_orchestrator_idle_backoff(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SUPERVISOR_IDLE_MAX_SECONDS", "600")

    from ispec.agent.connect import get_agent_session

    now = utcnow()
    with get_agent_session(agent_db_path) as db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            created_at=now,
            updated_at=now,
            config_json={},
            state_json={"checks": {}},
            summary_json={
                "orchestrator": {
                    "schema_version": 1,
                    "next_tick_reason": "idle_backoff",
                    "next_tick_seconds": 480,
                    "idle_streak": 4,
                }
            },
        )
        db.add(run)
        db.flush()

        sleep_seconds = _supervisor_dynamic_idle_sleep_seconds(
            run=run,
            state_after={"checks": {}},
            base_interval_seconds=10,
            now=now,
            db=db,
        )

    assert sleep_seconds == 120


def test_supervisor_dynamic_sleep_caps_to_due_queued_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SUPERVISOR_IDLE_MAX_SECONDS", "600")

    from ispec.agent.connect import get_agent_session

    now = utcnow()
    with get_agent_session(agent_db_path) as db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            created_at=now,
            updated_at=now,
            config_json={},
            state_json={"checks": {}},
            summary_json={
                "orchestrator": {
                    "schema_version": 1,
                    "next_tick_reason": "idle_backoff",
                    "next_tick_seconds": 480,
                    "idle_streak": 4,
                }
            },
        )
        db.add(run)
        db.add(
            AgentCommand(
                command_type="orchestrator_tick_v1",
                status="queued",
                priority=-5,
                available_at=now + timedelta(seconds=15),
                payload_json={"source": "test"},
                result_json={},
            )
        )
        db.flush()

        sleep_seconds = _supervisor_dynamic_idle_sleep_seconds(
            run=run,
            state_after={"checks": {}},
            base_interval_seconds=10,
            now=now,
            db=db,
        )

    assert 1 <= sleep_seconds <= 15


def test_supervisor_dynamic_sleep_uses_base_when_checks_fail(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    from ispec.agent.connect import get_agent_session

    now = utcnow()
    with get_agent_session(agent_db_path) as db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            created_at=now,
            updated_at=now,
            config_json={},
            state_json={"checks": {}},
            summary_json={
                "orchestrator": {
                    "schema_version": 1,
                    "next_tick_reason": "idle_backoff",
                    "next_tick_seconds": 480,
                    "idle_streak": 5,
                }
            },
        )
        db.add(run)
        db.flush()

        sleep_seconds = _supervisor_dynamic_idle_sleep_seconds(
            run=run,
            state_after={"checks": {"backend": {"ok": False}}},
            base_interval_seconds=10,
            now=now,
            db=db,
        )

    assert sleep_seconds == 10


def test_supervisor_dynamic_sleep_backs_off_when_checks_keep_failing(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SUPERVISOR_IDLE_MAX_SECONDS", "600")
    monkeypatch.setenv("ISPEC_SUPERVISOR_FAILURE_MAX_SECONDS", "120")

    from ispec.agent.connect import get_agent_session

    now = utcnow()
    with get_agent_session(agent_db_path) as db:
        run = AgentRun(
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
        db.add(run)
        db.flush()

        sleep_seconds = _supervisor_dynamic_idle_sleep_seconds(
            run=run,
            state_after={"checks": {"backend": {"ok": False}}, "check_failure_streak": 3},
            base_interval_seconds=10,
            now=now,
            db=db,
        )

    assert sleep_seconds == 40

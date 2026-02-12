from __future__ import annotations

from ispec.agent.models import AgentRun
from ispec.supervisor.loop import SupervisorConfig, run_supervisor


def test_supervisor_marks_run_stopped_on_keyboard_interrupt(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_ENABLED", "0")

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "_build_actions", lambda _config: [{"id": "noop"}])
    monkeypatch.setattr(supervisor_loop, "_action_funcs", lambda _config: {"noop": lambda: {"ok": True}})

    def _interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervisor_loop.time, "sleep", _interrupt)

    config = SupervisorConfig(
        agent_id="test-agent",
        backend_base_url="http://127.0.0.1:0",
        frontend_url="http://127.0.0.1:0",
        interval_seconds=1,
        timeout_seconds=0.1,
    )
    run_id = run_supervisor(config, once=False)
    assert isinstance(run_id, str) and run_id

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as db:
        row = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        assert row.status == "stopped"
        assert row.ended_at is not None
        assert row.updated_at is not None

from __future__ import annotations

from ispec.agent.models import AgentRun
from ispec.supervisor.loop import SupervisorConfig, run_supervisor


def test_supervisor_marks_run_stopped_on_keyboard_interrupt(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_ENABLED", "0")
    monkeypatch.setenv("ISPEC_STATE_DIR", str(tmp_path))

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
        thread_main = row.config_json.get("thread_main") if isinstance(row.config_json, dict) else None
        assert isinstance(thread_main, dict)
        assert thread_main.get("role") == "main"
        assert thread_main.get("owner") == "supervisor"
        assert thread_main.get("name") == "supervisor-main"
        assert isinstance(thread_main.get("ident"), int)


def test_supervisor_uses_dynamic_idle_sleep(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_ENABLED", "0")
    monkeypatch.setenv("ISPEC_STATE_DIR", str(tmp_path))

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "_build_actions", lambda _config: [{"id": "noop"}])
    monkeypatch.setattr(supervisor_loop, "_action_funcs", lambda _config: {"noop": lambda: {"ok": True}})
    monkeypatch.setattr(
        supervisor_loop,
        "_supervisor_dynamic_idle_sleep_seconds",
        lambda **_: 33,
    )

    sleeps: list[float] = []

    def _interrupt(*, sleep_seconds: int, **_) -> None:  # type: ignore[no-untyped-def]
        sleeps.append(float(sleep_seconds))
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervisor_loop, "_supervisor_sleep_with_command_polling", _interrupt)

    config = SupervisorConfig(
        agent_id="test-agent",
        backend_base_url="http://127.0.0.1:0",
        frontend_url="http://127.0.0.1:0",
        interval_seconds=1,
        timeout_seconds=0.1,
    )
    run_id = run_supervisor(config, once=False)
    assert isinstance(run_id, str) and run_id
    assert sleeps and int(sleeps[0]) == 33

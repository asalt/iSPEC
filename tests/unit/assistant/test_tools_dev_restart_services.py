from __future__ import annotations

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.assistant.tools import run_tool
from ispec.db.models import AuthUser, UserRole


def _internal_user() -> AuthUser:
    return AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )


def test_assistant_list_tools_includes_dev_restart_when_forced_enabled(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_DEV_RESTART_ENABLED", "1")

    payload = run_tool(
        name="assistant_list_tools",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="what tools do you have?",
    )
    assert payload["ok"] is True
    tool_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "assistant_enqueue_dev_restart_services" in tool_names


def test_assistant_list_tools_shows_dev_restart_unavailable_when_forced_disabled(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_DEV_RESTART_ENABLED", "0")

    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="list tools including unavailable",
    )
    assert payload["ok"] is True
    result = payload["result"]
    unavailable = result.get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    names = {item.get("name") for item in unavailable if isinstance(item, dict)}
    assert "assistant_enqueue_dev_restart_services" in names


def test_assistant_enqueue_dev_restart_services_enqueues_command(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_DEV_RESTART_ENABLED", "1")

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_dev_restart_services",
            args={"confirm": True, "services": ["backend"], "reason": "test"},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="restart backend please",
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["queued"] is True
        command_id = int(result["command_id"])

        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == command_id).one()
        assert cmd.command_type == "dev_restart_services_v1"
        assert cmd.status == "queued"
        assert cmd.payload_json["confirm"] is True
        assert cmd.payload_json["services"] == ["backend"]


def test_assistant_enqueue_dev_restart_services_refuses_when_disabled(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_DEV_RESTART_ENABLED", "0")

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_dev_restart_services",
            args={"confirm": True, "services": ["backend"]},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="restart backend please",
        )
        assert payload["ok"] is False


from __future__ import annotations

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.assistant.tools import openai_tools_for_user, run_tool
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


def test_assistant_list_tools_includes_staff_slack_tool_when_configured(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", "C123STAFF")

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
    assert "assistant_enqueue_staff_slack_message" in tool_names

    tools = openai_tools_for_user(_internal_user())
    tool_names = {
        tool["function"]["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    assert "assistant_enqueue_staff_slack_message" in tool_names


def test_assistant_list_tools_shows_staff_slack_unavailable_without_channel(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", raising=False)

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
    unavailable = payload["result"].get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    names = {item.get("name") for item in unavailable if isinstance(item, dict)}
    assert "assistant_enqueue_staff_slack_message" in names


def test_assistant_enqueue_staff_slack_message_enqueues_command(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", "C123STAFF")

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_staff_slack_message",
            args={"confirm": True, "message": "Heads up to staff.", "reason": "test"},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send this to staff slack",
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["queued"] is True
        command_id = int(result["command_id"])

        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == command_id).one()
        assert cmd.command_type == "slack_post_message_v1"
        assert cmd.status == "queued"
        assert cmd.payload_json["channel"] == "C123STAFF"
        assert cmd.payload_json["text"] == "Heads up to staff."


def test_assistant_enqueue_slack_message_allows_channel_alias(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON",
        '{"staff_ops":{"kind":"channel","channel":"C123STAFF","audience":"staff"}}',
    )

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_slack_message",
            args={
                "to": "staff_ops",
                "confirm": True,
                "message": "Current project summary is ready.",
                "message_type": "current_project_summary",
            },
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send current projects to staff ops",
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["to"] == "staff_ops"
        assert result["channel"] == "C123STAFF"

        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(result["command_id"])).one()
        assert cmd.payload_json["channel"] == "C123STAFF"
        assert cmd.payload_json["destination"]["alias"] == "staff_ops"


def test_assistant_enqueue_slack_message_allows_dm_alias(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON",
        '{"alex":{"kind":"dm","user_id":"U123ALEX","audience":"alex","allowed_message_types":["current_project_summary"]}}',
    )

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_slack_message",
            args={
                "to": "alex",
                "confirm": True,
                "message": "Current projects: 14.",
                "message_type": "current_project_summary",
            },
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send current projects to Alex",
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["to"] == "alex"
        assert result["user_id"] == "U123ALEX"

        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(result["command_id"])).one()
        assert cmd.payload_json["user_id"] == "U123ALEX"
        assert "channel" not in cmd.payload_json
        assert cmd.payload_json["destination"]["kind"] == "dm"


def test_assistant_enqueue_slack_message_refuses_unallowlisted_destination(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON", '{"alex":{"user_id":"U123ALEX"}}')

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_slack_message",
            args={"to": "C123FREEFORM", "confirm": True, "message": "Should not enqueue."},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send this to channel C123FREEFORM",
        )
        assert payload["ok"] is False
        assert "not allowlisted" in payload["error"]


def test_assistant_enqueue_slack_message_checks_message_type(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON",
        '{"alex":{"user_id":"U123ALEX","allowed_message_types":["report_ready"]}}',
    )

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_slack_message",
            args={
                "to": "alex",
                "confirm": True,
                "message": "Current projects: 14.",
                "message_type": "current_project_summary",
            },
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send current projects to Alex",
        )
        assert payload["ok"] is False
        assert "does not allow" in payload["error"]


def test_assistant_enqueue_staff_slack_message_refuses_when_unconfigured(tmp_path, db_session, monkeypatch):
    monkeypatch.delenv("ISPEC_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", raising=False)

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_enqueue_staff_slack_message",
            args={"confirm": True, "message": "Heads up to staff."},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=_internal_user(),
            api_schema=None,
            user_message="send this to staff slack",
        )
        assert payload["ok"] is False

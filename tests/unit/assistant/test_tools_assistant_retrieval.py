from __future__ import annotations

import json

from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.tools import run_tool
from ispec.db.models import AuthUser, UserRole


def test_assistant_search_messages_finds_matches(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        message = SupportMessage(session_pk=session.id, role="user", content="Hello world, this is a test.")
        assistant_db.add(message)
        assistant_db.commit()
        message_id = int(message.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_search_messages",
            args={"query": "world", "limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["count"] == 1
        assert result["matches"][0]["message_id"] == message_id
        assert "world" in result["matches"][0]["snippet"].lower()


def test_assistant_get_message_context_returns_window(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        messages = []
        for role, content in [
            ("user", "one"),
            ("assistant", "two"),
            ("user", "three"),
            ("assistant", "four"),
            ("user", "five"),
        ]:
            msg = SupportMessage(session_pk=session.id, role=role, content=content)
            assistant_db.add(msg)
            assistant_db.flush()
            messages.append(int(msg.id))
        assistant_db.commit()

    anchor_id = messages[2]
    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_get_message_context",
            args={"message_id": anchor_id, "before": 1, "after": 1, "max_chars": 50},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        ctx = payload["result"]
        assert ctx["message_id"] == anchor_id
        assert ctx["count"] == 3
        ids = [m["id"] for m in ctx["messages"]]
        assert ids == messages[1:4]


def test_assistant_search_internal_logs_finds_agent_step(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={},
        )
        assistant_db.add(run)
        assistant_db.commit()
        assistant_db.refresh(run)

        step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind="orchestrator_tick_v1",
            response_json={"note": "needle is here"},
        )
        assistant_db.add(step)
        assistant_db.commit()
        step_id = int(step.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_search_internal_logs",
            args={"query": "needle", "limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        steps = payload["result"]["steps"]
        assert any(item.get("step_id") == step_id for item in steps)


def test_assistant_get_agent_step_and_command(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={},
        )
        assistant_db.add(run)
        assistant_db.commit()
        assistant_db.refresh(run)

        step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind="test",
            prompt_json={"question": "hi"},
            response_json={"answer": "ok"},
        )
        assistant_db.add(step)

        command = AgentCommand(command_type="test_cmd", status="queued", payload_json={"note": "hello"})
        assistant_db.add(command)
        assistant_db.commit()
        step_id = int(step.id)
        command_id = int(command.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        step_payload = run_tool(
            name="assistant_get_agent_step",
            args={"step_id": step_id},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert step_payload["ok"] is True
        assert step_payload["result"]["step_id"] == step_id
        assert step_payload["result"]["response_json"] == {"answer": "ok"}

        cmd_payload = run_tool(
            name="assistant_get_agent_command",
            args={"command_id": command_id},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert cmd_payload["ok"] is True
        assert cmd_payload["result"]["command_id"] == command_id
        assert cmd_payload["result"]["payload_json"] == {"note": "hello"}


def test_assistant_list_users_reports_linkage(tmp_path, db_session):
    user = AuthUser(
        username="admin",
        password_hash="x",
        password_salt="y",
        role=UserRole.admin,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(
            session_id="s1",
            user_id=int(user.id),
            state_json=json.dumps(
                {"current_project_id": 123, "ui_route": {"name": "ProjectDetail", "path": "/project/123"}},
                ensure_ascii=False,
            ),
        )
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_list_users",
            args={"limit": 10, "include_anonymous": False, "linkage_session_limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["count"] == 1
        item = result["users"][0]
        assert item["user_id"] == int(user.id)
        assert item["username"] == "admin"
        assert item["sessions_count"] == 1
        assert item["linkage"]["projects"][0]["project_id"] == 123
        assert item["linkage"]["ui_routes"][0]["route"] == "ProjectDetail"


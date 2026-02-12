from __future__ import annotations

import json

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import (
    SupportMemory,
    SupportMemoryEvidence,
    SupportMessage,
    SupportSession,
    SupportSessionReview,
)
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
    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={},
        )
        agent_db.add(run)
        agent_db.commit()
        agent_db.refresh(run)

        step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind="orchestrator_tick_v1",
            response_json={"note": "needle is here"},
        )
        agent_db.add(step)
        agent_db.commit()
        step_id = int(step.id)

    with get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_search_internal_logs",
            args={"query": "needle", "limit": 10},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        steps = payload["result"]["steps"]
        assert any(item.get("step_id") == step_id for item in steps)


def test_assistant_get_agent_step_and_command(tmp_path, db_session):
    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={},
        )
        agent_db.add(run)
        agent_db.commit()
        agent_db.refresh(run)

        step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind="test",
            prompt_json={"question": "hi"},
            response_json={"answer": "ok"},
        )
        agent_db.add(step)

        command = AgentCommand(command_type="test_cmd", status="queued", payload_json={"note": "hello"})
        agent_db.add(command)
        agent_db.commit()
        step_id = int(step.id)
        command_id = int(command.id)

    with get_agent_session(agent_db_path) as agent_db:
        step_payload = run_tool(
            name="assistant_get_agent_step",
            args={"step_id": step_id},
            core_db=db_session,
            agent_db=agent_db,
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
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert cmd_payload["ok"] is True
        assert cmd_payload["result"]["command_id"] == command_id
        assert cmd_payload["result"]["payload_json"] == {"note": "hello"}


def test_assistant_recent_agent_commands_and_steps(tmp_path, db_session):
    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={},
        )
        agent_db.add(run)
        agent_db.commit()
        agent_db.refresh(run)

        cmd = AgentCommand(command_type="test_cmd", status="queued", payload_json={"note": "hello"})
        agent_db.add(cmd)

        step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind="test_kind",
            ok=True,
            severity="info",
            chosen_json={"command_id": None, "command_type": None},
        )
        agent_db.add(step)
        agent_db.commit()
        command_id = int(cmd.id)
        step_id = int(step.id)

    with get_agent_session(agent_db_path) as agent_db:
        commands_payload = run_tool(
            name="assistant_recent_agent_commands",
            args={"limit": 10},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert commands_payload["ok"] is True
        commands = commands_payload["result"]["commands"]
        assert any(int(item["id"]) == command_id for item in commands)

        steps_payload = run_tool(
            name="assistant_recent_agent_steps",
            args={"limit": 10},
            core_db=db_session,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert steps_payload["ok"] is True
        steps = steps_payload["result"]["steps"]
        assert any(int(item["id"]) == step_id for item in steps)


def test_assistant_recent_session_reviews(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=123)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Question"),
                SupportMessage(session_pk=session.id, role="assistant", content="Answer"),
            ]
        )
        assistant_db.flush()
        review = {
            "schema_version": 1,
            "session_id": "s1",
            "target_message_id": 2,
            "summary": "Looks fine.",
            "issues": [],
            "repo_search_queries": [],
            "followups": [],
        }
        review_row = SupportSessionReview(
            session_pk=int(session.id),
            target_message_id=2,
            review_json=review,
        )
        assistant_db.add(review_row)
        assistant_db.commit()
        review_id = int(review_row.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_recent_session_reviews",
            args={"limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        reviews = payload["result"]["reviews"]
        assert any(int(item["id"]) == review_id for item in reviews)


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


def test_assistant_digest_tools_list_get_and_search(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    digest_obj = {
        "schema_version": 1,
        "from_review_id": 10,
        "to_review_id": 12,
        "summary": "Recent chats include clustering discussion.",
        "highlights": ["Project 1427 clustering"],
        "followups": ["Add vector DB next"],
        "sessions": [],
    }

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        message = SupportMessage(session_pk=session.id, role="assistant", content="Anchor message")
        assistant_db.add(message)
        assistant_db.flush()

        digest = SupportMemory(
            session_pk=None,
            user_id=0,
            kind="digest",
            key="global",
            value_json=json.dumps(digest_obj, ensure_ascii=False),
        )
        assistant_db.add(digest)
        assistant_db.flush()
        digest_id = int(digest.id)

        assistant_db.add(
            SupportMemoryEvidence(memory_id=digest_id, message_id=int(message.id), weight=1.0)
        )
        assistant_db.commit()
        message_id = int(message.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        list_payload = run_tool(
            name="assistant_list_digests",
            args={"limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert list_payload["ok"] is True
        listed_ids = [item["digest_id"] for item in list_payload["result"]["digests"]]
        assert digest_id in listed_ids

        get_payload = run_tool(
            name="assistant_get_digest",
            args={"digest_id": digest_id},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert get_payload["ok"] is True
        result = get_payload["result"]
        assert result["digest_id"] == digest_id
        assert result["digest"]["summary"] == digest_obj["summary"]
        assert message_id in result["evidence_message_ids"]

        search_payload = run_tool(
            name="assistant_search_digests",
            args={"query": "clustering", "limit": 10},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert search_payload["ok"] is True
        assert any(item.get("digest_id") == digest_id for item in search_payload["result"]["matches"])


def test_assistant_prompt_header_returns_header_and_latest_user_message(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    state = {
        "current_project_id": 42,
        "conversation_summary": "hello",
        "conversation_memory": {"note": "remember"},
    }

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None, state_json=json.dumps(state))
        assistant_db.add(session)
        assistant_db.flush()
        msg = SupportMessage(session_pk=session.id, role="user", content="hi")
        assistant_db.add(msg)
        assistant_db.commit()
        user_message_id = int(msg.id)

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_prompt_header",
            args={"session_id": "s1", "include_legend": True},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["session_id"] == "s1"
        assert str(result["header_line"] or "").startswith("@h1 ")
        fields = result["header_fields"]
        assert fields["current_project_id"] == 42
        assert fields["user_message_id"] == user_message_id
        legend = result["legend"]
        assert legend["legend_version"] == 1
        assert "ok_bits" in legend
        assert "policy_bits" in legend

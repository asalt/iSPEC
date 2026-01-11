from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import Project
from ispec.schedule.connect import get_schedule_session


def test_support_chat_can_count_projects_via_tool_even_with_messy_tool_output(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three"),
        ]
    )
    db_session.commit()

    captured: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured.append({"messages": messages, "tools": tools})

        if len(captured) == 1:
            return AssistantReply(
                content=(
                    "To answer precisely, I will call a tool.\n\n"
                    'TOOL_CALL {"name":"count_all_projects","arguments":{}}\n\n'
                    "(please wait for the TOOL_RESULT system message)\n"
                    'TOOL_RESULT {"ok": true, "result": {"count": 42}}\n\n'
                    "FINAL:\nThe current count is 42."
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(messages, list)
        assert messages[-2]["role"] == "assistant"
        assert messages[-2]["content"].startswith("TOOL_CALL ")
        assert "TOOL_RESULT" not in messages[-2]["content"]
        assert "FINAL:" not in messages[-2]["content"]

        assert messages[-1]["role"] == "system"
        assert messages[-1]["content"].startswith("TOOL_RESULT count_all_projects")
        tool_payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
        assert tool_payload["ok"] is True
        assert tool_payload["result"]["count"] == 3

        return AssistantReply(
            content="FINAL:\nWe currently have 3 projects in iSPEC.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        assistant_db.add(SupportSession(session_id="session-1", user_id=None))
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "How many projects do we have?",
                "history": [],
                "ui": None,
            }
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.message == "We currently have 3 projects in iSPEC."
        assert len(captured) == 2


def test_support_chat_policy_fallback_counts_total_projects_when_model_skips_tools(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three"),
        ]
    )
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})

        if len(calls) == 1:
            return AssistantReply(
                content=(
                    "FINAL:\nAs of the last update in the CONTEXT, there is 1 project.\n"
                    "Please use the tool in the web UI."
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(messages, list)
        assert any(
            msg.get("role") == "system"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith("TOOL_RESULT count_all_projects")
            for msg in messages
        )
        return AssistantReply(
            content="FINAL:\nWe have 3 total projects.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        assistant_db.add(SupportSession(session_id="session-1", user_id=None))
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "Hey how many total projects do we have?",
                "history": [],
                "ui": None,
            }
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.message == "We have 3 total projects."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "count_all_projects"
        assert meta["tool_calls"][0]["protocol"] == "policy"


def test_support_chat_executes_suggested_no_arg_tool_when_model_mentions_tool_name(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One", prj_Current_FLAG=True),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two", prj_Current_FLAG=True),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three", prj_Current_FLAG=False),
        ]
    )
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})

        if len(calls) == 1:
            return AssistantReply(
                content=(
                    "FINAL:\nI don't have real-time counts from CONTEXT.\n"
                    "Please use the `count_current_projects` tool to answer this."
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(messages, list)
        assert any(
            msg.get("role") == "system"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith("TOOL_RESULT count_current_projects")
            for msg in messages
        )
        return AssistantReply(
            content="FINAL:\nWe have 2 current projects.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        assistant_db.add(SupportSession(session_id="session-1", user_id=None))
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "How many current projects do we have?",
                "history": [],
                "ui": None,
            }
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.message == "We have 2 current projects."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "count_current_projects"
        assert meta["tool_calls"][0]["protocol"] == "suggested"

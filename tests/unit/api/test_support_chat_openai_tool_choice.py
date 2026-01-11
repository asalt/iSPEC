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


def test_support_chat_forces_openai_tool_choice_when_user_requests_tool(tmp_path, db_session, monkeypatch):
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
    expected_choice = {"type": "function", "function": {"name": "count_projects"}}

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
            assert tool_choice == expected_choice
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "count_projects", "arguments": "{}"},
                    }
                ],
            )

        assert tool_choice is None
        return AssistantReply(
            content="FINAL:\nWe have 3 projects.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "Tell me how many projects we have; use a tool.",
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

        assert response.message == "We have 3 projects."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "count_projects"
        assert meta["tooling"]["forced_tool_choice"] == expected_choice


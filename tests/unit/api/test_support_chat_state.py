from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_versions_context_and_budgets_history(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "50")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "256")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "2000")

    captured: dict[str, Any] = {}

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured["messages"] = messages
        captured["tools"] = tools
        return AssistantReply(content="OK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(
            session_id="session-1",
            user_id=None,
            state_json=json.dumps(
                {
                    "controller": {"prepared_followup": {"summary": "internal"}},
                    "ui_route": {"name": "ProjectDetail", "path": "/project/7"},
                    "ui_project_id": 7,
                    "conversation_memory_requested_at": "2026-04-02T00:00:00+00:00",
                    "custom_internal_flag": "ignore-me",
                }
            ),
        )
        assistant_db.add(support_session)
        assistant_db.flush()

        long_text = "x" * 1200
        for idx in range(8):
            role = "user" if idx % 2 == 0 else "assistant"
            assistant_db.add(
                SupportMessage(
                    session_pk=support_session.id,
                    role=role,
                    content=f"message {idx} {long_text}",
                    provider="test",
                )
            )
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "New message",
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
        assert response.sessionId == "session-1"
        assert response.message == "OK"

        messages = captured.get("messages")
        assert isinstance(messages, list)

        context_message = next(
            (
                msg.get("content")
                for msg in messages
                if msg.get("role") == "system" and str(msg.get("content") or "").startswith("CONTEXT v1")
            ),
            None,
        )
        assert isinstance(context_message, str)
        context_payload = json.loads(context_message.split("\n", 1)[1])

        assert context_payload["schema_version"] == 1
        assert isinstance(context_payload.get("time"), dict)
        assert context_payload["time"]["now_utc"]
        assert context_payload["time"]["previous_message_at_utc"]

        state = context_payload["session"]["state"]
        assert state["conversation_summary_up_to_id"] == 8
        assert isinstance(state.get("conversation_summary"), str)
        assert state["conversation_summary"]
        assert state["ui"] == {"route_name": "ProjectDetail", "route_path": "/project/7", "project_id": 7}
        assert "controller" not in state
        assert "ui_route" not in state
        assert "conversation_memory_requested_at" not in state
        assert "custom_internal_flag" not in state

        new_message_index = next(
            idx
            for idx, msg in enumerate(messages)
            if msg.get("role") == "user" and msg.get("content") == "New message"
        )
        assert new_message_index >= 2
        history_payload = messages[2:new_message_index]
        assert len(history_payload) < 8

        assistant_db.refresh(support_session)
        persisted_state = json.loads(support_session.state_json)
        assert persisted_state["conversation_summary_up_to_id"] == 8
        assert persisted_state["conversation_summary"]


def test_support_chat_prompt_state_prefers_memory_over_summary_in_context(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    captured: dict[str, Any] = {}

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured["messages"] = messages
        captured["tools"] = tools
        return AssistantReply(content="OK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(
            session_id="session-memory-1",
            user_id=None,
            state_json=json.dumps(
                {
                    "conversation_memory": {
                        "schema_version": 1,
                        "summary": "Memory covers earlier turns.",
                        "facts": ["Project 7 is active."],
                        "open_tasks": [],
                        "preferences": [],
                        "entities": {
                            "projects": [{"id": 7, "title": "Sandbox"}],
                            "people": [],
                            "experiments": [],
                        },
                    },
                    "conversation_memory_up_to_id": 12,
                    "conversation_summary": "Older summary that should stay out of prompt state.",
                    "conversation_summary_up_to_id": 10,
                }
            ),
        )
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-memory-1",
                "message": "hello",
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

        assert response.message == "OK"
        messages = captured.get("messages")
        assert isinstance(messages, list)
        context_message = next(
            (
                msg.get("content")
                for msg in messages
                if msg.get("role") == "system" and str(msg.get("content") or "").startswith("CONTEXT v1")
            ),
            None,
        )
        assert isinstance(context_message, str)
        context_payload = json.loads(context_message.split("\n", 1)[1])
        state = context_payload["session"]["state"]
        assert "conversation_memory" in state
        assert state["conversation_memory_up_to_id"] == 12
        assert "conversation_summary" not in state
        assert "conversation_summary_up_to_id" not in state

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

    def fake_generate_reply(*, message: str, history=None, context=None) -> AssistantReply:
        captured["message"] = message
        captured["history"] = history
        captured["context"] = context
        return AssistantReply(content="OK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=None)
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

        context_message = captured.get("context")
        assert isinstance(context_message, str)
        assert context_message.startswith("CONTEXT v1")
        context_payload = json.loads(context_message.split("\n", 1)[1])

        assert context_payload["schema_version"] == 1
        state = context_payload["session"]["state"]
        assert state["conversation_summary_up_to_id"] == 8
        assert isinstance(state.get("conversation_summary"), str)
        assert state["conversation_summary"]

        history_payload = captured.get("history")
        assert isinstance(history_payload, list)
        assert len(history_payload) < 8

        assistant_db.refresh(support_session)
        persisted_state = json.loads(support_session.state_json)
        assert persisted_state["conversation_summary_up_to_id"] == 8
        assert persisted_state["conversation_summary"]

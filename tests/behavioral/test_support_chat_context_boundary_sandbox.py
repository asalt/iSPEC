from __future__ import annotations

import json

import pytest

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.connect import get_session
from ispec.schedule.connect import get_schedule_session


pytestmark = pytest.mark.behavioral


def test_behavioral_support_chat_uses_bounded_prompt_state_and_history(
    behavioral_datastore,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "4000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "2000")

    captured: dict[str, object] = {}

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured["messages"] = messages
        captured["tools"] = tools
        return AssistantReply(content="FINAL:\nContext looks bounded.", provider="test", model="test-model")

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    with (
        get_session(behavioral_datastore.core_db_path) as core_db,
        get_assistant_session(behavioral_datastore.assistant_db_path) as assistant_db,
        get_schedule_session(behavioral_datastore.schedule_db_path) as schedule_db,
    ):
        support_session = SupportSession(
            session_id="behavioral-context-1",
            user_id=None,
            state_json=json.dumps(
                {
                    "controller": {"support_post_send": {"status": "queued"}},
                    "ui_route": {"name": "ProjectDetail", "path": "/project/1"},
                    "ui_project_id": 1,
                    "conversation_memory_requested_at": "2026-04-02T00:00:00+00:00",
                    "custom_internal_flag": "ignore-me",
                }
            ),
        )
        assistant_db.add(support_session)
        assistant_db.flush()

        for idx in range(16):
            assistant_db.add(
                SupportMessage(
                    session_pk=support_session.id,
                    role="user" if idx % 2 == 0 else "assistant",
                    content=f"historical message {idx}",
                    provider="test",
                )
            )
        assistant_db.flush()

        response = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": "behavioral-context-1",
                    "message": "What is happening now?",
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=None,
        )

        assert response.message == "Context looks bounded."
        messages = captured.get("messages")
        assert isinstance(messages, list)
        context_message = next(
            (
                item.get("content")
                for item in messages
                if isinstance(item, dict)
                and item.get("role") == "system"
                and str(item.get("content") or "").startswith("CONTEXT v1")
            ),
            None,
        )
        assert isinstance(context_message, str)
        context_payload = json.loads(context_message.split("\n", 1)[1])
        assert context_payload["time"]["now_utc"]
        assert context_payload["time"]["previous_message_at_utc"]
        assert context_payload["session"]["state"]["ui"] == {
            "route_name": "ProjectDetail",
            "route_path": "/project/1",
            "project_id": 1,
        }
        assert "controller" not in context_payload["session"]["state"]
        assert "custom_internal_flag" not in context_payload["session"]["state"]

        assistant_row = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == support_session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["context"]["history_messages_used"] <= 10
        assert meta["context"]["context_state_keys"] == ["conversation_summary", "conversation_summary_up_to_id", "ui"]

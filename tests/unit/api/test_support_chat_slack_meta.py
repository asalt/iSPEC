from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_persists_slack_meta(tmp_path, db_session, monkeypatch):
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

    payload = ChatRequest.model_validate(
        {
            "sessionId": "slack-session",
            "message": "Hello",
            "history": [],
            "ui": None,
            "meta": {
                "source": "slack",
                "slack": {
                    "event": "app_mention",
                    "team_id": "T123",
                    "channel": "C123",
                    "thread_ts": "1700000000.123",
                    "message_ts": "1700000000.123",
                    "user_id": "U123",
                    "user_display_name": "Alex S",
                },
            },
        }
    )

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.sessionId == "slack-session"

        user_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.role == "user")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert user_message is not None
        assert user_message.provider == "slack"
        assert user_message.meta_json is not None
        meta = json.loads(user_message.meta_json)
        assert meta["client_meta"]["source"] == "slack"
        assert meta["client_meta"]["slack"]["user_display_name"] == "Alex S"


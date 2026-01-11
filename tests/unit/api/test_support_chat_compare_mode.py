from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, ChooseRequest, chat, choose
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_compare_mode_returns_choices_and_defers_commit(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPARE_MODE", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools})
        assert isinstance(messages, list)
        last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        assert isinstance(last_user, dict)
        last_user_content = str(last_user.get("content") or "")

        assert last_user_content == "Hello"
        return AssistantReply(
            content="FINAL_A:\nAnswer A.\nFINAL_B:\nAnswer B.\n",
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
                "message": "Hello",
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

        assert response.message is None
        assert response.messageId is None
        assert response.compare is not None
        assert response.compare.userMessageId >= 1
        assert len(response.compare.choices) == 2
        assert response.compare.choices[0].message == "Answer A."
        assert response.compare.choices[1].message == "Answer B."
        assert len(calls) == 1

        assistant_rows = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .all()
        )
        assert assistant_rows == []

        user_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "user")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert user_row is not None
        assert int(user_row.id) == int(response.compare.userMessageId)

        user_meta = json.loads(user_row.meta_json)
        assert user_meta["compare"]["choices"][0]["message"] == "Answer A."
        assert user_meta["compare"]["choices"][1]["message"] == "Answer B."


def test_support_chat_choose_commits_selected_answer(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPARE_MODE", "1")

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert isinstance(messages, list)
        last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        assert isinstance(last_user, dict)
        last_user_content = str(last_user.get("content") or "")

        assert last_user_content == "Hello"
        return AssistantReply(
            content="FINAL_A:\nAnswer A.\nFINAL_B:\nAnswer B.\n",
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
            {"sessionId": "session-1", "message": "Hello", "history": [], "ui": None}
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

        user_message_id = int(response.compare.userMessageId)

        choose_payload = ChooseRequest.model_validate(
            {"sessionId": "session-1", "userMessageId": user_message_id, "choiceIndex": 1, "ui": None}
        )
        choose_response = choose(choose_payload, assistant_db=assistant_db, user=None)
        assert choose_response.message == "Answer B."
        assert isinstance(choose_response.messageId, int)

        # idempotent: repeated choose returns the existing assistant message
        choose_response_2 = choose(choose_payload, assistant_db=assistant_db, user=None)
        assert choose_response_2.messageId == choose_response.messageId
        assert choose_response_2.message == "Answer B."

        assistant_rows = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.asc())
            .all()
        )
        assert len(assistant_rows) == 1
        assert assistant_rows[0].content == "Answer B."

        user_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.id == user_message_id)
            .first()
        )
        assert user_row is not None
        user_meta = json.loads(user_row.meta_json)
        assert user_meta["compare"]["selected_index"] == 1
        assert user_meta["compare"]["selected_message_id"] == int(choose_response.messageId)

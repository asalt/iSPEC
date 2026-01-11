from __future__ import annotations

import pytest
from fastapi import HTTPException

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, FeedbackRequest, chat, feedback
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import AuthUser, UserRole
from ispec.schedule.connect import get_schedule_session


def _make_user(username: str, *, role: UserRole = UserRole.viewer) -> AuthUser:
    return AuthUser(
        username=username,
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=role,
        is_active=True,
    )


def test_support_chat_rejects_session_owned_by_other_user(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    user1 = _make_user("user1")
    user2 = _make_user("user2")
    db_session.add_all([user1, user2])
    db_session.commit()

    def fake_generate_reply(*_, **__) -> AssistantReply:
        return AssistantReply(content="OK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        assistant_db.add(SupportSession(session_id="session-1", user_id=int(user1.id)))
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {"sessionId": "session-1", "message": "Hello", "history": [], "ui": None}
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            with pytest.raises(HTTPException) as exc:
                chat(
                    payload,
                    assistant_db=assistant_db,
                    core_db=db_session,
                    schedule_db=schedule_db,
                    user=user2,
                )
        assert exc.value.status_code == 403


def test_support_feedback_rejects_session_owned_by_other_user(tmp_path, db_session):
    user1 = _make_user("user1")
    user2 = _make_user("user2")
    db_session.add_all([user1, user2])
    db_session.commit()

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=int(user1.id))
        assistant_db.add(support_session)
        assistant_db.flush()

        message = SupportMessage(
            session_pk=support_session.id,
            role="assistant",
            content="Hello",
            provider="test",
            model="test-model",
        )
        assistant_db.add(message)
        assistant_db.flush()

        payload = FeedbackRequest.model_validate(
            {
                "sessionId": "session-1",
                "messageId": int(message.id),
                "rating": "up",
                "comment": None,
                "ui": None,
            }
        )

        with pytest.raises(HTTPException) as exc:
            feedback(payload, assistant_db=assistant_db, user=user2)
        assert exc.value.status_code == 403

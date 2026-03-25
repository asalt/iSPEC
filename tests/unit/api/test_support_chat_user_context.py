from __future__ import annotations

from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import AuthUser, UserRole
from ispec.schedule.connect import get_schedule_session


def test_support_chat_includes_user_assistant_brief_in_context(tmp_path, db_session, monkeypatch):
    user = AuthUser(
        username="alex",
        password_hash="x",
        password_salt="y",
        password_iterations=1,
        role=UserRole.admin,
        is_active=True,
        assistant_brief="Developer who uses tmux, repo tools, and local supervisor workflows.",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    captured: dict[str, Any] = {}

    def fake_generate_reply(*, messages=None, **_):
        captured["messages"] = messages or []
        return AssistantReply(content="FINAL:\nContext received.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        assistant_db.add(SupportSession(session_id="session-user-brief-1", user_id=int(user.id)))
        assistant_db.commit()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-user-brief-1",
                "message": "hello",
                "history": [],
                "ui": None,
            }
        )

        schedule_db_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_db_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=user,
            )

    assert response.message == "Context received."
    context_message = next(
        item["content"]
        for item in captured["messages"]
        if isinstance(item, dict) and item.get("role") == "system" and "CONTEXT v" in str(item.get("content") or "")
    )
    assert "assistant_brief" in context_message
    assert "Developer who uses tmux, repo tools, and local supervisor workflows." in context_message

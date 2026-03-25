from __future__ import annotations

import json

from fastapi import HTTPException, Response
import pytest

from ispec.api.routes import support as support_routes
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.db.models import AuthUser, UserRole


def _auth_user(*, username: str, role: UserRole) -> AuthUser:
    return AuthUser(
        username=username,
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=role,
        is_active=True,
    )


def test_support_debug_messages_returns_trace_and_previous_user_context(tmp_path, db_session):
    user = _auth_user(username="alex", role=UserRole.editor)
    db_session.add(user)
    db_session.flush()

    with get_assistant_session(tmp_path / "assistant.db") as assistant_db:
        session = SupportSession(session_id="session-debug-1", user_id=int(user.id))
        assistant_db.add(session)
        assistant_db.flush()

        user_message = SupportMessage(
            session_pk=session.id,
            role="user",
            content="what is going on on tmux pane ispec?",
            meta_json=json.dumps({"ui": {"path": "/dashboard", "name": "Dashboard"}}, ensure_ascii=False),
        )
        assistant_db.add(user_message)
        assistant_db.flush()
        user_message_id = int(user_message.id)

        assistant_message = SupportMessage(
            session_pk=session.id,
            role="assistant",
            content="I checked the tmux pane.",
            provider="test",
            model="test-model",
            meta_json=json.dumps(
                {
                    "tool_router": {
                        "selected_tool_names": ["assistant_list_tmux_panes", "assistant_capture_tmux_pane"],
                        "hinted_group_names": ["tmux"],
                    },
                    "tooling": {"used_tool_calls": 1},
                    "tool_calls": [{"name": "assistant_list_tmux_panes", "ok": True}],
                    "llm_trace": [{"round": 1, "reply_preview": "tool call"}],
                    "prompt_header": {"configured": True, "line": "header trace"},
                },
                ensure_ascii=False,
            ),
        )
        assistant_db.add(assistant_message)
        assistant_db.flush()

        response = Response()
        payload = support_routes.list_messages(
            response=response,
            role="assistant",
            session_id=None,
            limit=50,
            offset=0,
            assistant_db=assistant_db,
            core_db=db_session,
            user=user,
        )

    assert response.headers["X-Total-Count"] == "1"
    assert len(payload) == 1
    item = payload[0]
    assert item.sessionId == "session-debug-1"
    assert item.sessionUser == {"id": int(user.id), "username": "alex", "role": "editor", "assistant_brief": None}
    assert item.previousUserMessageId == user_message_id
    assert item.previousUserMessage == "what is going on on tmux pane ispec?"
    assert item.uiPath == "/dashboard"
    assert item.toolRouter == {
        "selected_tool_names": ["assistant_list_tmux_panes", "assistant_capture_tmux_pane"],
        "hinted_group_names": ["tmux"],
    }
    assert item.toolCalls == [{"name": "assistant_list_tmux_panes", "ok": True}]
    assert item.llmTrace == [{"round": 1, "reply_preview": "tool call"}]
    assert item.promptHeader == {"configured": True, "line": "header trace"}


def test_support_debug_message_detail_returns_context_and_session_state(tmp_path, db_session):
    user = _auth_user(username="alex", role=UserRole.editor)
    db_session.add(user)
    db_session.flush()

    with get_assistant_session(tmp_path / "assistant.db") as assistant_db:
        session = SupportSession(
            session_id="session-debug-detail-1",
            user_id=int(user.id),
            state_json=json.dumps({"current_project_id": 1351}, ensure_ascii=False),
        )
        assistant_db.add(session)
        assistant_db.flush()

        first_user = SupportMessage(session_pk=session.id, role="user", content="hello")
        target = SupportMessage(
            session_pk=session.id,
            role="assistant",
            content="detailed reply",
            meta_json=json.dumps({"tooling": {"used_tool_calls": 1}}, ensure_ascii=False),
        )
        last_user = SupportMessage(session_pk=session.id, role="user", content="thanks")
        assistant_db.add_all([first_user, target, last_user])
        assistant_db.flush()
        target_id = int(target.id)

        payload = support_routes.get_message_detail(
            message_id=target_id,
            before=1,
            after=1,
            assistant_db=assistant_db,
            core_db=db_session,
            user=user,
        )

    assert payload.messageId == target_id
    assert payload.sessionState == {"current_project_id": 1351}
    assert payload.sessionMessageCount == 3
    assert [item.message for item in payload.contextMessages] == ["hello", "detailed reply", "thanks"]
    assert payload.contextMessages[1].isSelected is True
    assert payload.tooling == {"used_tool_calls": 1}


def test_support_debug_messages_rejects_viewer(tmp_path, db_session):
    viewer = _auth_user(username="viewer", role=UserRole.viewer)
    db_session.add(viewer)
    db_session.flush()

    with get_assistant_session(tmp_path / "assistant.db") as assistant_db:
        response = Response()
        with pytest.raises(HTTPException) as excinfo:
            support_routes.list_messages(
                response=response,
                role=None,
                session_id=None,
                limit=20,
                offset=0,
                assistant_db=assistant_db,
                core_db=db_session,
                user=viewer,
            )

    assert excinfo.value.status_code == 403

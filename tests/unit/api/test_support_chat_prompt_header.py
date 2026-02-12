from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import Project
from ispec.schedule.connect import get_schedule_session


def test_support_chat_injects_prompt_header_round1_only(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_PROMPT_HEADER", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
        ]
    )
    db_session.commit()

    captured: list[list[dict[str, Any]]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert isinstance(messages, list)
        captured.append(messages)

        if len(captured) == 1:
            assert messages[1]["role"] == "system"
            assert str(messages[1].get("content") or "").startswith("@h1 ")
            assert messages[2]["role"] == "system"
            assert str(messages[2].get("content") or "").startswith("CONTEXT v1")
            return AssistantReply(
                content='TOOL_CALL {"name":"count_all_projects","arguments":{}}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert not any(
            msg.get("role") == "system" and str(msg.get("content") or "").startswith("@h1 ")
            for msg in messages
        )
        return AssistantReply(content="FINAL:\nOK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=None)
        assistant_db.add(support_session)
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

        assert response.message == "OK"
        assert len(captured) == 2

        assistant_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == int(support_session.id))
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert assistant_message is not None
        meta = json.loads(assistant_message.meta_json)
        prompt_header = meta["prompt_header"]
        assert prompt_header["configured"] is True
        assert prompt_header["built"] is True
        assert prompt_header["included_round1"] is True
        assert str(prompt_header["line"] or "").startswith("@h1 ")


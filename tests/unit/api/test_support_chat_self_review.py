from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_self_review_can_rewrite_final_answer(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools})

        assert isinstance(messages, list)
        last_user = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"),
            None,
        )
        assert isinstance(last_user, dict)
        last_user_content = str(last_user.get("content") or "")

        if last_user_content == "Hello":
            return AssistantReply(
                content="PLAN:\n- Draft\nFINAL:\nDraft answer.",
                provider="test",
                model="test-model",
                meta=None,
            )

        assert last_user_content.startswith("Review the assistant answer above")
        assert any(msg.get("role") == "user" and msg.get("content") == "Hello" for msg in messages)
        assert messages[-2] == {"role": "assistant", "content": "Draft answer."}
        return AssistantReply(
            content="FINAL:\nRevised answer.",
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

        assert response.message == "Revised answer."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        assert assistant_row.content == "Revised answer."

        meta = json.loads(assistant_row.meta_json)
        assert meta["self_review"]["changed"] is True
        assert "draft_raw_content" in meta
        assert meta["plan"].startswith("- Draft")


def test_support_chat_self_review_decider_can_keep_draft_without_rewrite(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW_DECIDER", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})

        assert isinstance(messages, list)
        last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        assert isinstance(last_user, dict)
        last_user_content = str(last_user.get("content") or "")

        if last_user_content == "Hello":
            return AssistantReply(
                content="PLAN:\n- Draft\nFINAL:\nDraft answer.",
                provider="test",
                model="test-model",
                meta=None,
            )

        assert "Output only KEEP" in last_user_content
        assert vllm_extra_body == {
            "guided_choice": ["KEEP", "REWRITE"],
            "max_tokens": 3,
            "stop": ["\n"],
            "temperature": 0,
        }
        assert messages[-2] == {"role": "assistant", "content": "Draft answer."}
        return AssistantReply(content="KEEP", provider="test", model="test-model", meta=None)

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

        assert response.message == "Draft answer."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        assert assistant_row.content == "Draft answer."

        meta = json.loads(assistant_row.meta_json)
        assert meta["self_review"]["mode"] == "guided_choice"
        assert meta["self_review"]["decision"] == "keep"
        assert meta["self_review"]["changed"] is False


def test_support_chat_self_review_skips_when_llm_returns_error(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools})
        return AssistantReply(
            content="Assistant error: HTTPError: 400 Bad Request",
            provider="test",
            model="test-model",
            meta={"url": "http://example", "error": "HTTPError('400')"},
            ok=False,
            error="HTTPError: 400 Bad Request",
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

        assert response.message == "Assistant error: HTTPError: 400 Bad Request"
        assert len(calls) == 1

from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_applies_response_contract_and_skips_self_review(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_RESPONSE_CONTRACTS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        if len(calls) == 1:
            return AssistantReply(
                content=(
                    "PLAN:\n"
                    "- Draft a concise answer\n"
                    "FINAL:\n"
                    "The model over-answers because it is trying to be maximally helpful. "
                    "Without a tighter structure, it keeps adding more support than the user wanted."
                ),
                provider="test",
                model="draft-model",
            )
        if len(calls) == 2:
            return AssistantReply(
                content='{"contract":"brief_explainer","confidence":0.88,"reason":"The user asked for a short explanation."}',
                provider="test",
                model="selector-model",
            )
        return AssistantReply(
            content=(
                '{'
                '"answer":"The model over-answers because the reply shape is not constrained.",'
                '"reason":"When scope is loose, it keeps adding context and support to avoid missing something.",'
                '"example":"A short question can still become a long reply."'
                '}'
            ),
            provider="test",
            model="fill-model",
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
                "message": "Why does this model over-answer?",
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

        assert response.message == (
            "The model over-answers because the reply shape is not constrained.\n\n"
            "Why: When scope is loose, it keeps adding context and support to avoid missing something.\n\n"
            "Example: A short question can still become a long reply."
        )
        assert len(calls) == 3

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["response_contract"]["applied"] is True
        assert meta["response_contract"]["selected_contract"] == "brief_explainer"
        assert meta["self_review"]["mode"] == "skipped_response_contract"
        assert meta["raw_content"].startswith("PLAN:\n- Draft a concise answer")


def test_support_chat_response_contract_force_override_skips_selector(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_RESPONSE_CONTRACTS", "1")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        if len(calls) == 1:
            return AssistantReply(
                content="FINAL:\nThis is a longer draft that should be compressed.",
                provider="test",
                model="draft-model",
            )
        return AssistantReply(
            content='{"answer":"This is a longer draft that should be compressed.","caveat":"It is still only a draft."}',
            provider="test",
            model="fill-model",
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
                "message": "Answer directly.",
                "history": [],
                "ui": None,
                "meta": {"response_contract": {"force": "direct"}},
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

        assert response.message == "This is a longer draft that should be compressed.\n\nCaveat: It is still only a draft."
        assert len(calls) == 2

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["response_contract"]["applied"] is True
        assert meta["response_contract"]["selection"]["source"] == "forced"
        assert meta["response_contract"]["policy"]["force"] == "direct"

from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.prompt.sync import sync_prompts
from ispec.schedule.connect import get_schedule_session


def test_support_chat_passes_and_persists_prompt_observability(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "line")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    sync_prompts()

    captured_contexts: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, observability_context=None, **_) -> AssistantReply:
        assert isinstance(messages, list)
        assert tools is None
        captured_contexts.append(dict(observability_context or {}))
        return AssistantReply(
            content="FINAL:\nPrompt-aware answer.",
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

        assert response.message == "Prompt-aware answer."
        assert captured_contexts

        observability = captured_contexts[-1]
        assert observability["surface"] == "support_chat"
        assert observability["stage"] == "answer"
        assert observability["prompt_family"] == "assistant.answer.system"
        assert observability["prompt_version_num"] == 1
        assert observability["prompt_binding"] == "ispec.assistant.service:_system_prompt_answer"
        assert str(observability["prompt_source_path"]).endswith("assistant.answer.system.md")
        assert len(str(observability["prompt_sha256"] or "")) == 64

        assistant_row = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == support_session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None

        meta = json.loads(assistant_row.meta_json)
        assert meta["prompt_stage"] == "answer"
        assert meta["prompt_family"] == "assistant.answer.system"
        assert meta["prompt_version_num"] == 1
        assert meta["prompt_binding"] == "ispec.assistant.service:_system_prompt_answer"
        assert str(meta["prompt_source_path"]).endswith("assistant.answer.system.md")
        assert len(str(meta["prompt_sha256"] or "")) == 64
        assert meta["llm_trace"][0]["prompt_family"] == "assistant.answer.system"
        assert meta["llm_trace"][0]["prompt_version_num"] == 1

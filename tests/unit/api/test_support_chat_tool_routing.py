from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_support_chat_tool_router_filters_openai_tools_by_group(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})

        if isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body:
            assert tools is None
            return AssistantReply(
                content=json.dumps(
                    {
                        "primary": "people",
                        "secondary": [],
                        "confidence": 0.9,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta={"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            )

        assert isinstance(tools, list)
        tool_names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func_obj = tool.get("function")
            if not isinstance(func_obj, dict):
                continue
            name = func_obj.get("name")
            if isinstance(name, str) and name:
                tool_names.add(name)
        assert tool_names == {"assistant_prompt_header", "search_people", "get_person"}

        assert isinstance(messages, list)
        system_prompt = str(messages[0].get("content") or "")
        assert "search_people" in system_prompt
        assert "count_all_projects" not in system_prompt

        return AssistantReply(content="FINAL:\nOk.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "How many people do we have in the database?",
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

        assert response.message == "Ok."
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
        assert meta["tool_router"]["decision"]["primary"] == "people"
        assert set(meta["tool_router"]["selected_tool_names"]) == {
            "assistant_prompt_header",
            "get_person",
            "search_people",
        }

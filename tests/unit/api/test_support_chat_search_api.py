from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


class _DummyApp:
    def __init__(self, schema: dict[str, Any]):
        self._schema = schema

    def openapi(self) -> dict[str, Any]:
        return self._schema


class _DummyRequest:
    def __init__(self, schema: dict[str, Any]):
        self.app = _DummyApp(schema)


def test_support_chat_can_search_api_schema_via_tool(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    schema = {
        "openapi": "3.0.0",
        "paths": {
            "/api/projects": {
                "get": {
                    "summary": "List projects",
                    "operationId": "list_projects_api_projects_get",
                    "tags": ["Projects"],
                }
            },
            "/api/support/chat": {
                "post": {
                    "summary": "Support chat",
                    "operationId": "support_chat_api_support_chat_post",
                    "tags": ["Support"],
                }
            },
        },
    }

    captured: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured.append({"messages": messages, "tools": tools})
        if len(captured) == 1:
            return AssistantReply(
                content='TOOL_CALL {"name":"search_api","arguments":{"query":"projects","limit":5}}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(messages, list)
        assert messages[-1]["role"] == "system"
        assert messages[-1]["content"].startswith("TOOL_RESULT search_api")
        tool_payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
        assert tool_payload["ok"] is True
        matches = tool_payload["result"]["matches"]
        assert any(item["path"] == "/api/projects" for item in matches)

        return AssistantReply(
            content="Use GET /api/projects to list projects.",
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
                "message": "What API endpoints exist for projects?",
                "history": [],
                "ui": None,
            }
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                request=_DummyRequest(schema),
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.sessionId == "session-1"
        assert response.message == "Use GET /api/projects to list projects."

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "search_api"
        assert meta["tool_calls"][0]["ok"] is True

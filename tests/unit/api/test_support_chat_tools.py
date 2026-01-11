from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import Project
from ispec.schedule.connect import get_schedule_session
from ispec.schedule.models import ScheduleSlot


def test_support_chat_can_call_tools(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "line")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    project = Project(prj_AddedBy="test", prj_ProjectTitle="Tool Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    captured: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured.append({"messages": messages, "tools": tools})
        if len(captured) == 1:
            return AssistantReply(
                content=f'TOOL_CALL {{"name":"get_project","arguments":{{"id":{project.id}}}}}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(messages, list)
        assert messages[-1]["role"] == "system"
        assert messages[-1]["content"].startswith("TOOL_RESULT get_project")
        assert "Tool Project" in messages[-1]["content"]
        return AssistantReply(
            content=f"Project {project.id}: Tool Project",
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

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            payload = ChatRequest.model_validate(
                {
                    "sessionId": "session-1",
                    "message": "Tell me about the project called Tool Project",
                    "history": [],
                    "ui": None,
                }
            )

            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )
            assert response.sessionId == "session-1"
            assert response.message == f"Project {project.id}: Tool Project"

            assistant_row = (
                assistant_db.query(support_routes.SupportMessage)
                .filter(support_routes.SupportMessage.session_pk == support_session.id)
                .filter(support_routes.SupportMessage.role == "assistant")
                .order_by(support_routes.SupportMessage.id.desc())
                .first()
            )
            assert assistant_row is not None
            meta = json.loads(assistant_row.meta_json)
            assert meta["tool_calls"][0]["name"] == "get_project"
            assert meta["tool_calls"][0]["ok"] is True
            assert meta["tooling"]["enabled"] is True
            assert meta["tooling"]["max_tool_calls"] == 2
            assert meta["tooling"]["used_tool_calls"] == 1


def test_support_chat_can_call_schedule_tools(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "line")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    schedule_path = tmp_path / "schedule.db"
    with get_schedule_session(schedule_path) as schedule_db:
        slot = ScheduleSlot(
            start_at=datetime(2025, 1, 2, 15, 0, 0),  # 09:00 America/Chicago
            end_at=datetime(2025, 1, 2, 15, 45, 0),
            status="available",
        )
        schedule_db.add(slot)
        schedule_db.flush()

        captured: list[dict[str, Any]] = []

        def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
            captured.append({"messages": messages, "tools": tools})
            if len(captured) == 1:
                return AssistantReply(
                    content=(
                        'TOOL_CALL {"name":"list_schedule_slots","arguments":'
                        '{"start":"2025-01-01","end":"2025-01-07","status":"available","limit":5}}'
                    ),
                    provider="test",
                    model="test-model",
                    meta=None,
                )

            assert isinstance(messages, list)
            assert messages[-1]["role"] == "system"
            assert messages[-1]["content"].startswith("TOOL_RESULT list_schedule_slots")
            assert f'"id":{slot.id}' in messages[-1]["content"]
            return AssistantReply(
                content="Found schedule slots.",
                provider="test",
                model="test-model",
                meta=None,
            )

        monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

        assistant_path = tmp_path / "assistant.db"
        with get_assistant_session(assistant_path) as assistant_db:
            support_session = SupportSession(session_id="session-1", user_id=None)
            assistant_db.add(support_session)
            assistant_db.flush()

            payload = ChatRequest.model_validate(
                {"sessionId": "session-1", "message": "Show me slots", "history": [], "ui": None}
            )

            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

            assert response.sessionId == "session-1"
            assert response.message == "Found schedule slots."

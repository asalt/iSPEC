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
        assert tool_names == {
            "assistant_prompt_header",
            "assistant_list_tools",
            "search_people",
            "get_person",
        }

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
            "assistant_list_tools",
            "get_person",
            "search_people",
        }


def test_support_chat_tool_router_includes_explicitly_requested_tool_name(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    def fake_openai_tools_for_user(_user):  # type: ignore[no-untyped-def]
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_recent_agent_commands", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search_people", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_person", "parameters": {"type": "object"}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})

        if isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body:
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
                meta=None,
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

        # Router picked people group, but the user explicitly mentioned the agent-commands tool.
        assert tool_names == {
            "assistant_prompt_header",
            "assistant_list_tools",
            "assistant_recent_agent_commands",
            "search_people",
            "get_person",
        }

        return AssistantReply(content="FINAL:\nOk.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-2", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-2",
                "message": "Show me assistant_recent_agent_commands output.",
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
        assert "assistant_recent_agent_commands" in meta["tool_router"]["selected_tool_names"]


def test_support_chat_disables_required_tool_choice_when_requested_tool_missing(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    # Provide only staff-level assistant tools; omit assistant_recent_agent_commands.
    def fake_openai_tools_for_user(_user):  # type: ignore[no-untyped-def]
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_stats", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_recent_sessions", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_get_session_review", "parameters": {"type": "object"}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "vllm_extra_body": vllm_extra_body,
            }
        )

        if isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body:
            return AssistantReply(
                content=json.dumps(
                    {
                        "primary": "assistant",
                        "secondary": [],
                        "confidence": 0.9,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        # forced_tool_choice would normally be "required", but the requested tool is missing,
        # so the chat handler should not force a tool call.
        assert tool_choice is None
        assert isinstance(messages, list)
        assert any(
            isinstance(item, dict)
            and item.get("role") == "system"
            and "not available for this turn" in str(item.get("content") or "")
            for item in messages
        )

        return AssistantReply(content="FINAL:\nOk.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-3", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-3",
                "message": "Use a tool: assistant_recent_agent_commands",
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

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tooling"]["forced_tool_choice"] is None
        assert meta["tool_router"]["missing_requested_tool_names"] == ["assistant_recent_agent_commands"]


def test_support_chat_tool_router_keeps_project_tools_for_explicit_project_requests(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    def fake_openai_tools_for_user(_user):  # type: ignore[no-untyped-def]
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_stats", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_recent_sessions", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_project", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "projects", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search_projects", "parameters": {"type": "object"}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, vllm_extra_body=None, **_) -> AssistantReply:
        if isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body:
            return AssistantReply(
                content=json.dumps(
                    {
                        "primary": "assistant",
                        "secondary": [],
                        "confidence": 0.8,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(tools, list)
        assert tool_choice == {"type": "function", "function": {"name": "get_project"}}
        tool_names = {
            tool["function"]["name"]
            for tool in tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        assert tool_names == {
            "assistant_prompt_header",
            "assistant_list_tools",
            "assistant_stats",
            "assistant_recent_sessions",
            "get_project",
            "projects",
            "search_projects",
        }
        return AssistantReply(content="FINAL:\nOk.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-proj", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-proj",
                "message": "Project 1171. What do I need to know before an internal meeting?",
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
        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_router"]["decision"]["primary"] == "assistant"
        assert meta["tool_router"]["hinted_group_names"] == ["projects"]
        assert meta["tooling"]["forced_tool_choice"]["function"]["name"] == "get_project"
        assert "get_project" in meta["tool_router"]["selected_tool_names"]


def test_support_chat_tool_router_keeps_file_tools_for_focused_project_results_followup(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    def fake_openai_tools_for_user(_user):  # type: ignore[no-untyped-def]
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_project", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "project_files_for_project", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "experiments_for_project", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "latest_experiments", "parameters": {"type": "object"}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        if isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body:
            return AssistantReply(
                content=json.dumps(
                    {
                        "primary": "experiments",
                        "secondary": [],
                        "confidence": 0.8,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(tools, list)
        tool_names = {
            tool["function"]["name"]
            for tool in tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        assert tool_names == {
            "assistant_prompt_header",
            "assistant_list_tools",
            "get_project",
            "project_files_for_project",
            "experiments_for_project",
            "latest_experiments",
        }
        return AssistantReply(content="FINAL:\nOk.", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(
            session_id="session-followup",
            user_id=None,
            state_json=json.dumps({"current_project_id": 1171}),
        )
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-followup",
                "message": "I found bad samples in a results directory and a MSPC001171_removed folder.",
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
        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_router"]["decision"]["primary"] == "experiments"
        assert meta["tool_router"]["hinted_group_names"] == ["files", "projects"]
        assert "project_files_for_project" in meta["tool_router"]["selected_tool_names"]
        assert "get_project" in meta["tool_router"]["selected_tool_names"]

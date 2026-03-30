from __future__ import annotations

import json
import types
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.assistant.turn_decision import (
    TurnDecision,
    TurnDecisionResponsePlan,
    TurnDecisionResult,
    TurnDecisionToolPlan,
    TurnDecisionWritePlan,
)
from ispec.db.models import Project, ProjectComment, UserRole
from ispec.schedule.connect import get_schedule_session


def test_support_chat_does_not_force_openai_tool_choice_for_generic_tool_request(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three"),
        ]
    )
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
            assert tool_choice is None
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "count_all_projects", "arguments": "{}"},
                    }
                ],
            )

        assert tool_choice is None
        return AssistantReply(
            content="FINAL:\nWe have 3 projects.",
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
                "message": "Tell me how many projects we have; use a tool.",
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

        assert response.message == "We have 3 projects."
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
        assert meta["tool_calls"][0]["name"] == "count_all_projects"
        assert meta["tooling"]["forced_tool_choice"] is None


def test_support_chat_retries_after_unsupported_write_claim_and_saves_note(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1351, prj_AddedBy="test", prj_ProjectTitle="Project 1351"))
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
            return AssistantReply(
                content="FINAL:\nI have made a note on project 1351. The comment ID is 23.",
                provider="test",
                model="test-model",
                meta=None,
            )
        if len(calls) == 2:
            assert tool_choice is None
            assert any(
                isinstance(item, dict)
                and item.get("role") == "system"
                and "no successful write tool call" in str(item.get("content") or "")
                for item in (messages or [])
            )
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "create_project_comment",
                            "arguments": json.dumps(
                                {
                                    "project_id": 1351,
                                    "comment": "Sequence review is needed.",
                                    "comment_type": "note",
                                    "confirm": True,
                                }
                            ),
                        },
                    }
                ],
            )

        return AssistantReply(
            content="FINAL:\nSaved the note. Comment ID is 1.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-2", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-2",
                "message": "Make a note on project 1351 that sequence review is needed.",
                "history": [],
                "ui": None,
            }
        )

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert response.message == "Saved the note. Comment ID is 1."
        assert len(calls) == 3

        comment = (
            db_session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1351)
            .order_by(ProjectComment.id.desc())
            .first()
        )
        assert comment is not None
        assert comment.com_Comment == "Sequence review is needed."
        assert comment.com_CommentType == "note"

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "create_project_comment"
        assert meta["tooling"]["forced_tool_choice"] is None
        assert meta["write_claim_guard"]["triggered"] is True
        assert meta["write_claim_guard"]["retried"] is True
        assert meta["write_claim_guard"]["blocked_final_claim"] is False


def test_support_chat_blocks_repeated_unsupported_write_claim(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1351, prj_AddedBy="test", prj_ProjectTitle="Project 1351"))
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        assert tool_choice is None
        return AssistantReply(
            content="FINAL:\nI have made a note on project 1351. The comment ID is 23.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-3", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-3",
                "message": "Make a note on project 1351 that sequence review is needed.",
                "history": [],
                "ui": None,
            }
        )

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert response.message == (
            "I can't confirm that I saved that because no write tool actually succeeded. "
            "I haven't made that change yet."
        )
        assert len(calls) == 2
        assert (
            db_session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1351)
            .count()
        ) == 0

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["write_claim_guard"]["triggered"] is True
        assert meta["write_claim_guard"]["retried"] is True
        assert meta["write_claim_guard"]["blocked_final_claim"] is True


def test_support_chat_draft_request_returns_draft_without_writing(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1531, prj_AddedBy="test", prj_ProjectTitle="Project 1531"))
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        assert tool_choice is None
        if len(calls) == 1:
            return AssistantReply(
                content=(
                    "FINAL:\n"
                    "To add a comment to project 1531, you can say: "
                    "\"TOOL_CALL {\\\"name\\\":\\\"create_project_comment\\\",\\\"arguments\\\":{...}}\""
                ),
                provider="test",
                model="test-model",
                meta=None,
            )
        return AssistantReply(
            content=(
                "FINAL:\n"
                "Draft comment for project 1531:\n"
                "Data is regrouped and ready for review.\n\n"
                "Would you like me to save this to project history, or would you like any tweaks first?"
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-4", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-4",
                "message": "Help me write a comment about project 1531 saying data is regrouped.",
                "history": [],
                "ui": None,
            }
        )

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert "Draft comment for project 1531" in (response.message or "")
        assert "Would you like me to save this" in (response.message or "")
        assert len(calls) == 2
        assert (
            db_session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1531)
            .count()
        ) == 0

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tooling"]["used_tool_calls"] == 0
        assert meta["tool_calls"] == []
        assert meta["tool_protocol_guard"]["triggered"] is True
        assert meta["tool_protocol_guard"]["retried"] is True
        assert meta["tool_protocol_guard"]["blocked_final_leak"] is False


def test_support_chat_confirmation_reply_saves_drafted_comment(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1531, prj_AddedBy="test", prj_ProjectTitle="Project 1531"))
    db_session.commit()

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
            assert tool_choice is None
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "create_project_comment",
                            "arguments": json.dumps(
                                {
                                    "project_id": 1531,
                                    "comment": "Data is regrouped and ready for review.",
                                    "comment_type": "assistant_note",
                                    "confirm": True,
                                }
                            ),
                        },
                    }
                ],
            )

        return AssistantReply(
            content="FINAL:\nSaved the note. Comment ID is 1.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(
            session_id="session-5",
            user_id=None,
            state_json=json.dumps({"current_project_id": 1531}),
        )
        assistant_db.add(support_session)
        assistant_db.flush()
        assistant_db.add(
            SupportMessage(
                session_pk=support_session.id,
                role="assistant",
                content=(
                    "Draft comment for project 1531:\n"
                    "Data is regrouped and ready for review.\n\n"
                    "Would you like me to save this to project history, or would you like any tweaks first?"
                ),
            )
        )
        assistant_db.commit()

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            confirm_response = chat(
                ChatRequest.model_validate(
                    {
                        "sessionId": "session-5",
                        "message": "Yes, save it.",
                        "history": [],
                        "ui": None,
                    }
                ),
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert confirm_response.message == "Saved the note. Comment ID is 1."
        assert len(calls) == 2

        comment = (
            db_session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1531)
            .order_by(ProjectComment.id.desc())
            .first()
        )
        assert comment is not None
        assert comment.com_Comment == "Data is regrouped and ready for review."

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "create_project_comment"
        assert meta["tooling"]["forced_tool_choice"] is None


def test_support_chat_project_comment_intent_hint_guides_draft_request(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROJECT_COMMENT_INTENT_HINTS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1531, prj_AddedBy="test", prj_ProjectTitle="Project 1531"))
    db_session.commit()

    def fake_route_tool_groups_vllm(**_) -> tuple[dict[str, Any] | None, AssistantReply]:
        return None, AssistantReply(content="{}", provider="router", model="router", meta=None)

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
        if len(calls) == 1:
            assert tools is None
            assert isinstance(vllm_extra_body, dict)
            assert "guided_json" in vllm_extra_body
            return AssistantReply(
                content=json.dumps(
                    {
                        "intent": "draft_only",
                        "confidence": 0.96,
                        "reason": "The user asked for help writing a comment, not saving it yet.",
                    }
                ),
                provider="intent",
                model="intent-model",
                meta=None,
            )

        assert any(
            isinstance(item, dict)
            and item.get("role") == "system"
            and "comment_intent=draft_only" in str(item.get("content") or "")
            for item in (messages or [])
        )
        return AssistantReply(
            content=(
                "FINAL:\n"
                "Draft comment for project 1531:\n"
                "Data is regrouped and ready for review.\n\n"
                "Would you like me to save this to project history, or would you like any tweaks first?"
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "route_tool_groups_vllm", fake_route_tool_groups_vllm)
    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-6", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-6",
                "message": "Help me write a comment about project 1531 saying data is regrouped.",
                "history": [],
                "ui": None,
            }
        )

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert "Draft comment for project 1531" in (response.message or "")
        assert len(calls) == 2
        assert (
            db_session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1531)
            .count()
        ) == 0

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["comment_intent"]["decision"]["intent"] == "draft_only"
        assert meta["comment_intent"]["provider"] == "intent"


def test_support_chat_forces_openai_tmux_list_tool_choice_for_live_tmux_request(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")

    def fake_openai_tools_for_user(_user):
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tmux_panes", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_capture_tmux_pane", "parameters": {"type": "object", "required": ["target"]}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    tool_runs: list[dict[str, Any]] = []

    def fake_run_tool(*, name=None, args=None, **_):
        tool_runs.append({"name": name, "args": args})
        assert name == "assistant_list_tmux_panes"
        return {
            "ok": True,
            "items": [
                {
                    "target": "ispec-0:node.1",
                    "session_name": "ispec",
                    "pane_title": "codex resume --all /home/alex/tools/ispec-full",
                }
            ],
        }

    monkeypatch.setattr(support_routes, "run_tool", fake_run_tool)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, vllm_extra_body=None, **_):
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
                        "primary": "tmux",
                        "secondary": [],
                        "confidence": 0.92,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        if len(tool_runs) == 0:
            assert tool_choice == {"type": "function", "function": {"name": "assistant_list_tmux_panes"}}
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "assistant_list_tmux_panes",
                            "arguments": json.dumps({"session_name": "ispec"}),
                        },
                    }
                ],
            )

        assert tool_choice is None
        assert any(
            isinstance(item, dict)
            and (
                "assistant_list_tmux_panes" in str(item.get("content") or "")
                or "ispec-0:node.1" in str(item.get("content") or "")
            )
            for item in (messages or [])
        )
        return AssistantReply(
            content="FINAL:\nI checked the ispec tmux session and found one readable pane.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-tmux-force-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-tmux-force-1",
                "message": "whats going on on tmux pane ispec?",
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

        assert response.message == "I checked the ispec tmux session and found one readable pane."
        assert len(tool_runs) == 1

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "assistant_list_tmux_panes"
        assert meta["tooling"]["forced_tool_choice"] == {
            "type": "function",
            "function": {"name": "assistant_list_tmux_panes"},
        }


def test_support_chat_forces_get_project_tool_choice_for_project_existence_request(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")

    db_session.add(Project(id=1598, prj_AddedBy="test", prj_ProjectTitle="Project 1598"))
    db_session.commit()

    def fake_openai_tools_for_user(_user):
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {
                    "name": "get_project",
                    "parameters": {"type": "object", "required": ["id"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_project_comment",
                    "parameters": {"type": "object", "required": ["project_id", "comment", "confirm"]},
                },
            },
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    tool_runs: list[dict[str, Any]] = []

    def fake_run_tool(*, name=None, args=None, **_):
        tool_runs.append({"name": name, "args": args})
        assert name == "get_project"
        assert args == {"id": 1598}
        return {
            "ok": True,
            "tool": "get_project",
            "result": {"id": 1598, "title": "Project 1598"},
        }

    monkeypatch.setattr(support_routes, "run_tool", fake_run_tool)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, vllm_extra_body=None, **_):
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
                        "confidence": 0.72,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        if len(tool_runs) == 0:
            assert tool_choice == {"type": "function", "function": {"name": "get_project"}}
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_project",
                            "arguments": json.dumps({"id": 1598}),
                        },
                    }
                ],
            )

        assert tool_choice is None
        return AssistantReply(
            content="FINAL:\nYes, project 1598 exists. I can add the comment when you're ready.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-project-exists-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-project-exists-1",
                "message": "Do we have project 1598 available? I have a comment to add to it if it exists",
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

        assert response.message == "Yes, project 1598 exists. I can add the comment when you're ready."
        assert len(tool_runs) == 1

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
        assert meta["tooling"]["forced_tool_choice"] == {
            "type": "function",
            "function": {"name": "get_project"},
        }


def test_support_chat_turn_decision_owner_can_drive_project_lookup(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "own")

    db_session.add(Project(id=1598, prj_AddedBy="test", prj_ProjectTitle="Project 1598"))
    db_session.commit()

    def fake_openai_tools_for_user(_user):
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {
                    "name": "get_project",
                    "parameters": {"type": "object", "required": ["id"]},
                },
            },
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)
    monkeypatch.setattr(
        support_routes,
        "route_tool_groups_vllm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy tool router should not run in own mode")),
    )
    monkeypatch.setattr(
        support_routes,
        "run_turn_decision_pipeline",
        lambda **_: TurnDecisionResult(
            ok=True,
            mode="own",
            source="support_chat",
            applied=True,
            decision=TurnDecision(
                source="support_chat",
                primary_goal="answer_question",
                needs_clarification=False,
                clarification_reason="none",
                tool_plan=TurnDecisionToolPlan(
                    use_tools=True,
                    primary_group="projects",
                    secondary_groups=(),
                    preferred_first_tool="get_project",
                ),
                write_plan=TurnDecisionWritePlan(mode="none"),
                response_plan=TurnDecisionResponsePlan(mode="single", contract_cap="direct"),
                confidence=0.91,
                reason="A project lookup is the best first step.",
            ),
        ),
    )

    tool_runs: list[dict[str, Any]] = []

    def fake_run_tool(*, name=None, args=None, **_):
        tool_runs.append({"name": name, "args": args})
        assert name == "get_project"
        assert args == {"id": 1598}
        return {
            "ok": True,
            "tool": "get_project",
            "result": {"id": 1598, "title": "Project 1598"},
        }

    monkeypatch.setattr(support_routes, "run_tool", fake_run_tool)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_):
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(tool_runs) == 0:
            assert tool_choice == {"type": "function", "function": {"name": "get_project"}}
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_project",
                            "arguments": json.dumps({"id": 1598}),
                        },
                    }
                ],
            )

        return AssistantReply(
            content="FINAL:\nProject 1598 exists.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-turn-decision-owner-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-turn-decision-owner-1",
                "message": "Could you check project 1598 for me?",
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

        assert response.message == "Project 1598 exists."
        assert len(tool_runs) == 1
        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tooling"]["forced_tool_choice"] == {
            "type": "function",
            "function": {"name": "get_project"},
        }
        assert meta["tool_router"]["source"] == "turn_decision"
        assert meta["turn_decision"]["mode"] == "own"
        assert meta["turn_decision"]["applied"] is True


def test_support_chat_project_existence_reply_does_not_trigger_write_claim_guard(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")

    db_session.add(Project(id=1598, prj_AddedBy="test", prj_ProjectTitle="Project 1598"))
    db_session.commit()

    def fake_openai_tools_for_user(_user):
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {
                    "name": "get_project",
                    "parameters": {"type": "object", "required": ["id"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_project_comment",
                    "parameters": {"type": "object", "required": ["project_id", "comment", "confirm"]},
                },
            },
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    tool_runs: list[dict[str, Any]] = []

    def fake_run_tool(*, name=None, args=None, **_):
        tool_runs.append({"name": name, "args": args})
        assert name == "get_project"
        assert args == {"id": 1598}
        return {
            "ok": True,
            "tool": "get_project",
            "result": {
                "id": 1598,
                "title": "Bacterial Protein Identification",
                "status": "waiting",
                "comments": {
                    "count": 1,
                    "latest": {
                        "id": 481,
                        "type": "note",
                        "added_by": "api_key",
                        "created": "2026-03-27T12:01:57.018268",
                        "comment": "species Acinobacteria Tersicoccus phoenicis DSM. Need to build fasta",
                    },
                },
            },
        }

    monkeypatch.setattr(support_routes, "run_tool", fake_run_tool)

    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, vllm_extra_body=None, **_):
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
                        "confidence": 0.72,
                        "clarify": False,
                    }
                ),
                provider="test",
                model="test-model",
                meta=None,
            )

        if len(tool_runs) == 0:
            assert tool_choice == {"type": "function", "function": {"name": "get_project"}}
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_project",
                            "arguments": json.dumps({"id": 1598}),
                        },
                    }
                ],
            )

        assert tool_choice is None
        return AssistantReply(
            content=(
                "FINAL:\n"
                "Project 1598 titled \"Bacterial Protein Identification\" exists and is currently active. "
                "It is in a 'waiting' status, and the latest comment added was "
                "\"species Acinobacteria Tersicoccus phoenicis DSM. Need to build fasta\" "
                "by user 'api_key' on 2026-03-27. You can now add your comment to this project."
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-project-exists-2", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-project-exists-2",
                "message": "Do we have project 1598 available? I have a comment to add to it if it exists",
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

        assert "Project 1598 titled \"Bacterial Protein Identification\" exists" in response.message
        assert "You can now add your comment to this project." in response.message
        assert len(tool_runs) == 1
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
        assert meta["tool_calls"][0]["name"] == "get_project"
        write_claim_guard = meta.get("write_claim_guard") or {}
        assert write_claim_guard.get("triggered", False) is False
        assert write_claim_guard.get("retried", False) is False
        assert write_claim_guard.get("blocked_final_claim", False) is False

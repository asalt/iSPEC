from __future__ import annotations

import json

import pytest

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.connect import get_session
from ispec.schedule.connect import get_schedule_session


pytestmark = pytest.mark.behavioral


def test_behavioral_support_chat_tmux_resolution_uses_real_capture_target(
    behavioral_datastore,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    monkeypatch.setattr(
        "ispec.assistant.support_policies._tmux_policy_panes",
        lambda: [
            {
                "session": "ispec-0",
                "session_group": "ispec",
                "session_names": ["ispec-0", "ispec-1"],
                "target": "ispec-0:node.1",
                "preferred_alias": "ispec:node.1",
                "capture_target": "%0",
                "pane_id": "%0",
                "pane_number": 0,
                "target_aliases": ["ispec:node.1", "ispec-0:node.1", "ispec-1:node.1"],
                "window_aliases": ["ispec:node", "ispec-0:node", "ispec-1:node"],
                "window_name": "node",
                "pane_title": "codex resume --all /home/alex/tools/ispec-full",
                "current_command": "node",
            }
        ],
    )

    def fake_openai_tools_for_user(_user):
        return [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tmux_panes", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_capture_tmux_pane", "parameters": {"type": "object", "required": ["target"]}}},
            {"type": "function", "function": {"name": "assistant_compare_tmux_pane", "parameters": {"type": "object", "required": ["target"]}}},
        ]

    monkeypatch.setattr(support_routes, "openai_tools_for_user", fake_openai_tools_for_user)

    llm_calls: list[dict[str, object]] = []

    def fake_generate_reply(*, tools=None, tool_choice=None, **_) -> AssistantReply:
        llm_calls.append({"tools": tools, "tool_choice": tool_choice})
        if len(llm_calls) == 1:
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
                            "name": "assistant_capture_tmux_pane",
                            "arguments": json.dumps({"target": "default", "lines": 80}),
                        },
                    }
                ],
            )
        return AssistantReply(
            content="FINAL:\nThe codex pane shows Codex ready.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    run_tool_calls: list[dict[str, object]] = []

    def fake_run_tool(*, name=None, args=None, **_):
        run_tool_calls.append({"name": name, "args": dict(args or {})})
        return {
            "ok": True,
            "tool": name,
            "result": {
                "target": "%0",
                "capture_target": "%0",
                "content": "thinking\nCodex ready",
                "last_nonempty_line": "Codex ready",
            },
        }

    monkeypatch.setattr(support_routes, "run_tool", fake_run_tool)

    with (
        get_session(behavioral_datastore.core_db_path) as core_db,
        get_assistant_session(behavioral_datastore.assistant_db_path) as assistant_db,
        get_schedule_session(behavioral_datastore.schedule_db_path) as schedule_db,
    ):
        support_session = SupportSession(session_id="behavioral-tmux-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        response = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": "behavioral-tmux-1",
                    "message": "hello what is going on on the ispec tmux pane?",
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=None,
        )

        assert response.message == "The codex pane shows Codex ready."
        assert llm_calls[0]["tool_choice"] == {"type": "function", "function": {"name": "assistant_capture_tmux_pane"}}
        assert run_tool_calls == [{"name": "assistant_capture_tmux_pane", "args": {"target": "%0", "lines": 40}}]

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_router"]["source"] == "support_policy_rule"
        assert meta["tool_router"]["tmux_resolution"]["selected_target"] == "%0"
        assert meta["tool_calls"][0]["arguments"] == {"target": "%0", "lines": 40}

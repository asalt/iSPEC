"""Render real controller requests through the OLMo chat template before any GPU run.

Stubbed-model tests check the JSON request; vLLM rejects the prompt only after
the chat template expands it. These tests render each captured request with the
committed template and count tokens with the checkpoint tokenizer.
Hand-authored synthetic fixtures; not model benchmarks or training examples.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import SimpleNamespace

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

from ispec.api.routes import support
from ispec.assistant import service
from ispec.schedule.connect import get_schedule_session
from tests.unit.api.test_support_olmo_application import (  # noqa: F401
    ScriptedModel,
    application_settings,
)
from tests.unit.assistant.test_project_comment_cards import (  # noqa: F401
    NOTE,
    conversation,
)

# Pinned upstream template plus the tools default guard (see test_olmo_eval_template.py).
TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "behavioral/data/model_followup_20261025/olmo_instruct_tools_defined.jinja"
)
TOKENIZER = Path(
    os.getenv("ISPEC_OLMO_TOKENIZER")
    or "/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9/tokenizer.json"
)
# VLLM_MAX_MODEL_LEN in configs/models/olmo7b-instruct-vllm.env.
CONTEXT_WINDOW = 8192


@dataclass(kw_only=True)
class RecordingModel(ScriptedModel):
    payloads: list[dict] = field(default_factory=list)

    def post(self, url, *, json: dict, **kwargs):
        if url.endswith("/chat/completions"):
            self.payloads.append(deepcopy(json))
        return super().post(url, json=json, **kwargs)


def render(payload: dict) -> str:
    """Render one chat-completions payload as vLLM does for this template."""
    messages = []
    for message in payload["messages"]:
        message = dict(message)
        if message.get("tool_calls"):
            # vLLM parses JSON-string arguments before rendering; the template iterates them.
            message["tool_calls"] = [
                {
                    **call,
                    "function": {
                        **call["function"],
                        "arguments": json.loads(call["function"]["arguments"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        messages.append(message)
    # Matches transformers' chat-template environment and its tojson filter.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    env.filters["tojson"] = (
        lambda value, indent=None, separators=None, sort_keys=False: (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )
        )
    )
    return env.from_string(TEMPLATE.read_text()).render(
        messages=messages,
        tools=payload.get("tools"),
        add_generation_prompt=True,
        eos_token="<|endoftext|>",
    )


@pytest.fixture
def save_conversation_payloads(
    conversation, application_settings, monkeypatch, tmp_path  # noqa: F811
):
    # Output and prompt budgets from the checked-in OLMo profile.
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "6000")
    model = RecordingModel(approval_label="approve_save", confidence=0.95)
    monkeypatch.setattr(service.requests, "post", model.post)
    with get_schedule_session(tmp_path / "schedule.db") as schedule:
        for message in (
            "Draft a note for project 57: " + NOTE,
            "Yes, save that exact note.",
        ):
            support.chat(
                support.ChatRequest(
                    sessionId="rendered-prompt", message=message, history=[]
                ),
                assistant_db=conversation.assistant,
                core_db=conversation.core,
                schedule_db=schedule,
                user=conversation.user,
            )
            conversation.assistant.commit()
    assert len(conversation.comments()) == 1
    return model.payloads


def test_rendered_olmo_requests_include_tool_schemas_once(save_conversation_payloads):
    tool_payloads = [
        payload for payload in save_conversation_payloads if payload.get("tools")
    ]
    assert len(tool_payloads) == 4
    for payload in save_conversation_payloads:
        rendered = render(payload)
        # The preamble mentions the bare tag in prose; count rendered schema lists.
        assert rendered.count("<functions>[") == (1 if payload.get("tools") else 0)


def test_rendered_olmo_requests_fit_context_window(save_conversation_payloads):
    if not TOKENIZER.is_file():
        pytest.skip(
            f"OLMo tokenizer not found at {TOKENIZER}; set ISPEC_OLMO_TOKENIZER"
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    for payload in save_conversation_payloads:
        prompt_tokens = len(
            tokenizer.encode(render(payload), add_special_tokens=False).ids
        )
        budget = min(6000, CONTEXT_WINDOW - payload.get("max_tokens", 0))
        assert prompt_tokens <= budget, (
            f"{prompt_tokens} prompt tokens exceed {budget} "
            f"({len(payload.get('tools') or [])} tools, "
            f"{sum(m['role'] == 'system' for m in payload['messages'])} system messages)"
        )


@dataclass(kw_only=True)
class FailedRouterModel(RecordingModel):
    """Return a truncated router response and a harmless conversational answer."""

    def post(self, url, *, json: dict, **kwargs):
        constraint = json.get("structured_outputs", {}).get("json", {})
        if "primary" in constraint.get("properties", {}):
            self.payloads.append(deepcopy(json))
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [{
                        "finish_reason": "length",
                        "message": {"content": '{"primary":"projects","clarify":   '},
                    }],
                },
            )
        return super().post(url, json=json, **kwargs)

    def tool_reply(self, *, request):
        return {"content": "The draft remains pending."}


def test_failed_router_keeps_project_workflow_within_budget(
    conversation, application_settings, monkeypatch, tmp_path,  # noqa: F811 (pytest fixtures)
):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "6000")
    conversation.prepare()
    model = FailedRouterModel()
    monkeypatch.setattr(service.requests, "post", model.post)
    monkeypatch.setattr(support, "hinted_support_tool_groups", lambda **kwargs: set())

    with get_schedule_session(tmp_path / "schedule.db") as schedule:
        response = support.chat(
            support.ChatRequest(
                sessionId=conversation.session.session_id,
                message="Change received to expected next week. Show the correction rather than saving it.",
                history=[],
            ),
            assistant_db=conversation.assistant,
            core_db=conversation.core,
            schedule_db=schedule,
            user=conversation.user,
        )

    request = next(payload for payload in model.payloads if payload.get("tools"))
    names = {tool["function"]["name"] for tool in request["tools"]}
    assert {"get_project", "prepare_project_comment", "create_project_comment",
            "cancel_project_comment_draft", "show_project_comment_card"} <= names
    assert not {"search_people", "assistant_stats", "db_file_stats"} & names
    assert len(names) <= 18
    assert conversation.comments() == []
    if not TOKENIZER.is_file():
        pytest.skip("Real OLMo tokenizer is needed for the fallback budget check")
    from tokenizers import Tokenizer

    count = len(Tokenizer.from_file(str(TOKENIZER)).encode(
        render(request), add_special_tokens=False,
    ).ids)
    assert count <= min(6000, CONTEXT_WINDOW - request["max_tokens"])
    assert response.message == "The draft remains pending."

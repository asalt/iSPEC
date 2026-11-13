"""One application prompt, with native schemas supplied separately from its text."""

import pytest

from ispec.assistant import service
from ispec.prompt.loader import load_prompt_source
from tests.unit.assistant.test_runtime_protocol import (
    TOOL,
    envelope,
    transport,
)  # noqa: F401


@pytest.fixture(autouse=True)
def prompt_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH", raising=False)


def test_native_tools_keep_base_guidance_without_duplicate_catalog():
    prompt = service._render_system_prompt_planner(
        tools_available=True,
        tool_names={"get_project", "prepare_project_comment", "create_project_comment"},
    )

    assert prompt.source.family == "assistant.planner.system"
    assert service._render_base_system_prompt().text in prompt.text
    assert "Available tools:" not in prompt.text
    assert (
        "only write to project history if the user explicitly asks" not in prompt.text
    )
    assert prompt.binding.qualname == "_system_prompt_planner"


def test_line_protocol_keeps_catalog():
    prompt = service._render_system_prompt_planner(tools_available=False)
    assert "Available tools:" in prompt.text


def test_compare_contract_and_explicit_additions_are_retained(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA", "Deployment instruction.")
    prompt = service._render_system_prompt_planner(
        tools_available=True,
        tool_names={"get_project"},
        response_format="compare",
    )
    assert "FINAL_A:" in prompt.text and "FINAL_B:" in prompt.text
    assert prompt.text.endswith("Deployment instruction.")


@pytest.mark.parametrize("stage", ["planner", "answer"])
def test_direct_service_uses_same_prompt_and_native_preamble(
    transport, monkeypatch, stage
):
    captured, responses = transport
    monkeypatch.setenv(
        "ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "assistant.tools.olmo3_preamble"
    )
    responses.append(envelope("The project is waiting."))
    history = [{"role": "assistant", "content": "Earlier model reply."}]

    reply = service.generate_reply(
        message="Look up project 57.",
        history=history,
        context="Authoritative context.",
        tools=[TOOL],
        stage=stage,
    )

    assert reply.ok
    sent = captured[0]
    expected = load_prompt_source("assistant.tools.olmo3_preamble").body.strip()
    expected += (
        "\n"
        + service._render_system_prompt_planner(
            tools_available=True, tool_names={"get_project"}
        ).text
    )
    expected += "\n\nAuthoritative context."
    assert sent["messages"][0] == {"role": "system", "content": expected}
    assert sent["messages"][1:] == [
        *history,
        {"role": "user", "content": "Look up project 57."},
    ]
    assert sent["tools"] == [TOOL]


def test_explicit_classifier_messages_are_untouched(transport):
    captured, responses = transport
    messages = [
        {"role": "system", "content": "Return the approval label."},
        {"role": "user", "content": "Synthetic conversation."},
    ]
    responses.append(envelope('{"label":"deny_save"}'))

    reply = service.generate_reply(messages=messages, tools=None)

    assert reply.ok
    assert captured[0]["messages"] == messages
    assert "tool_preamble" not in reply.meta


@pytest.mark.parametrize(
    "stage,family",
    [
        ("answer", "assistant.answer.system"),
        ("review", "assistant.review.system"),
    ],
)
def test_answer_and_review_keep_their_existing_bindings(stage, family):
    prompt = service._render_stage_prompt(stage=stage, tools_available=False)
    assert prompt.source.family == family

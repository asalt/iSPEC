from __future__ import annotations

from typing import Any

from ispec.assistant.response_contracts import parse_response_contract_mode, run_response_contract_pipeline
from ispec.assistant.service import AssistantReply


def test_parse_response_contract_mode_normalizes_to_off_or_shadow():
    assert parse_response_contract_mode(None) == "off"
    assert parse_response_contract_mode("off") == "off"
    assert parse_response_contract_mode("0") == "off"
    assert parse_response_contract_mode("shadow") == "shadow"
    assert parse_response_contract_mode("1") == "shadow"
    assert parse_response_contract_mode("safe") == "shadow"
    assert parse_response_contract_mode("all") == "shadow"


def test_response_contract_pipeline_selects_and_renders_brief_explainer():
    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        if len(calls) == 1:
            return AssistantReply(
                content='{"contract":"brief_explainer","confidence":0.92,"reason":"The user asked for a short explanation."}',
                provider="test",
                model="selector-model",
            )
        return AssistantReply(
            content=(
                '{'
                '"answer":"The model over-answers because the reply shape is too open.",'
                '"reason":"When scope is ambiguous, it keeps adding support to avoid leaving anything out.",'
                '"example":"A short question can still trigger a long explanation.",'
                '"caveat":"This gets worse when the prompt rewards completeness."'
                '}'
            ),
            provider="test",
            model="fill-model",
        )

    result = run_response_contract_pipeline(
        generate_reply_fn=fake_generate_reply,
        context_message="CONTEXT: test",
        history_messages=[],
        user_message="Why does it over-answer?",
        tool_result_messages=[],
        draft_answer="It over-answers because it tries to be helpful and complete.",
    )

    assert result.ok is True
    assert result.applied is True
    assert result.selected_contract == "brief_explainer"
    assert result.selection is not None
    assert result.selection["source"] == "selector"
    assert result.normalized_slots is not None
    assert result.normalized_slots["example"] == "A short question can still trigger a long explanation."
    assert result.normalized_slots["caveat"] is None
    assert result.as_meta()["candidate_content"] == result.rendered_content
    assert any(item == "dropped_optional_over_budget:caveat" for item in result.warnings)
    assert result.rendered_content == (
        "The model over-answers because the reply shape is too open.\n\n"
        "Why: When scope is ambiguous, it keeps adding support to avoid leaving anything out.\n\n"
        "Example: A short question can still trigger a long explanation."
    )
    assert len(calls) == 2


def test_response_contract_pipeline_force_contract_skips_selector_and_repairs_points():
    calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        if len(calls) == 1:
            return AssistantReply(
                content='{"answer":"Here is the bounded explanation.","points":["Only one point"]}',
                provider="test",
                model="fill-model",
            )
        return AssistantReply(
            content=(
                '{'
                '"answer":"Here is the bounded explanation.",'
                '"points":["First short point.","Second short point."],'
                '"next_step":"Use the smaller contract by default."'
                '}'
            ),
            provider="test",
            model="repair-model",
        )

    result = run_response_contract_pipeline(
        generate_reply_fn=fake_generate_reply,
        context_message="CONTEXT: test",
        history_messages=[],
        user_message="Explain this, but keep it bounded.",
        tool_result_messages=[],
        draft_answer="Here is a detailed answer that needs a tighter structure.",
        request_meta={"response_contract": {"force": "structured_explainer"}},
    )

    assert result.ok is True
    assert result.applied is True
    assert result.selected_contract == "structured_explainer"
    assert result.selection is not None
    assert result.selection["source"] == "forced"
    assert result.repair_applied is True
    assert result.normalized_slots is not None
    assert result.normalized_slots["points"] == ["First short point.", "Second short point."]
    assert result.rendered_content == (
        "Here is the bounded explanation.\n\n"
        "Key points:\n"
        "- First short point.\n"
        "- Second short point.\n\n"
        "Next step: Use the smaller contract by default."
    )
    assert len(calls) == 2

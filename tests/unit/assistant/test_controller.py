from __future__ import annotations

from ispec.assistant.controller import run_message_pre_send_controller
from ispec.assistant.service import AssistantReply


def test_pre_send_controller_runs_response_contract_stage_before_self_review() -> None:
    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_):  # type: ignore[no-untyped-def]
        calls.append({"messages": messages, "tools": tools, "vllm_extra_body": vllm_extra_body})
        if len(calls) == 1:
            return AssistantReply(
                content='{"contract":"brief_explainer","confidence":0.9,"reason":"Short explanation."}',
                provider="test",
                model="selector",
            )
        if len(calls) == 2:
            return AssistantReply(
                content=(
                    '{"answer":"The model over-explains because the reply shape is unconstrained.",'
                    '"reason":"Without a bounded answer skeleton, it keeps adding support."}'
                ),
                provider="test",
                model="fill",
            )
        return AssistantReply(
            content=(
                "FINAL:\n"
                "Reviewed answer."
            ),
            provider="test",
            model="review",
        )

    draft_reply = AssistantReply(
        content="FINAL:\nThe model over-explains because it tries to be helpful.",
        provider="draft",
        model="draft-model",
    )

    result = run_message_pre_send_controller(
        generate_reply_fn=fake_generate_reply,
        source="support_chat",
        context_message="CONTEXT",
        history_messages=[],
        user_message="Why does the model over-answer?",
        tool_messages=[],
        tool_result_messages=[],
        draft_answer="The model over-explains because it tries to be helpful.",
        draft_reply=draft_reply,
        compare_mode=False,
        request_meta=None,
        response_contract_mode="shadow",
        response_contract_would_apply_if_live=True,
        response_contract_protection_reason=None,
        self_review_enabled=True,
        self_review_decider_enabled=False,
        used_tool_calls=1,
    )

    assert result.final_content == "Reviewed answer."
    assert result.response_contract_applied is False
    assert [stage.name for stage in result.stages] == ["response_contract", "self_review"]
    assert result.stages[0].status == "applied"
    assert result.stages[0].changed is True
    assert result.response_contract_meta["shadow_candidate"].startswith("The model over-explains because")
    assert result.stages[1].status == "applied"
    assert result.stages[1].changed is True


def test_pre_send_controller_self_review_stage_can_rewrite() -> None:
    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, **_):  # type: ignore[no-untyped-def]
        calls.append({"messages": messages, "tools": tools})
        return AssistantReply(
            content="FINAL:\nRevised answer.",
            provider="review",
            model="review-model",
        )

    draft_reply = AssistantReply(
        content="PLAN:\n- Draft\nFINAL:\nDraft answer.",
        provider="draft",
        model="draft-model",
    )

    result = run_message_pre_send_controller(
        generate_reply_fn=fake_generate_reply,
        source="support_chat",
        context_message="CONTEXT",
        history_messages=[],
        user_message="Hello",
        tool_messages=[{"role": "system", "content": "TOOL_RESULT {}"}],
        tool_result_messages=[{"role": "system", "content": "TOOL_RESULT {}"}],
        draft_answer="Draft answer.",
        draft_reply=draft_reply,
        compare_mode=False,
        request_meta=None,
        response_contract_mode="off",
        response_contract_would_apply_if_live=True,
        response_contract_protection_reason=None,
        self_review_enabled=True,
        self_review_decider_enabled=False,
        used_tool_calls=1,
    )

    assert result.final_content == "Revised answer."
    assert result.self_review_changed is True
    assert result.draft_raw_content == "PLAN:\n- Draft\nFINAL:\nDraft answer."
    assert result.stages[0].name == "response_contract"
    assert result.stages[0].status == "skipped"
    assert result.stages[0].reason == "disabled"
    assert result.stages[1].name == "self_review"
    assert result.stages[1].status == "applied"
    assert result.stages[1].changed is True

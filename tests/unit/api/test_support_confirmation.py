from __future__ import annotations

from ispec.assistant.reply_interpretation import (
    interpret_reply_for_project_comment_save,
    is_affirmative_reply,
    is_confirmation_reply,
)
from ispec.assistant.turn_decision import (
    TurnDecision,
    TurnDecisionReplyInterpretation,
    TurnDecisionResponsePlan,
    TurnDecisionResult,
    TurnDecisionToolPlan,
    TurnDecisionWritePlan,
)


def test_confirmation_reply_accepts_compound_confirmation_phrase():
    assert is_confirmation_reply("Confirm yes commit it") is True
    assert is_affirmative_reply("Confirm yes commit it") is True


def test_confirmation_reply_rejects_negative_affirmative_mix():
    assert is_confirmation_reply("no dont commit it") is True
    assert is_affirmative_reply("no dont commit it") is False


def _pending_reply(user_message: str):
    return interpret_reply_for_project_comment_save(
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=1602,
        last_assistant_message="Would you like me to save this note to project history?",
        user_message=user_message,
    )


def _turn_decision_result(*, reply_kind: str, write_mode: str = "confirm_save") -> TurnDecisionResult:
    return TurnDecisionResult(
        ok=True,
        mode="own",
        source="support_chat",
        applied=True,
        decision=TurnDecision(
            source="support_chat",
            primary_goal="confirm_save",
            needs_clarification=False,
            clarification_reason="none",
            tool_plan=TurnDecisionToolPlan(
                use_tools=True,
                primary_group="projects",
                secondary_groups=(),
                preferred_first_tool="create_project_comment",
            ),
            write_plan=TurnDecisionWritePlan(mode=write_mode),
            response_plan=TurnDecisionResponsePlan(mode="single", contract_cap="direct"),
            reply_interpretation=TurnDecisionReplyInterpretation(
                kind=reply_kind,
                confidence=0.91,
                reason="test classifier decision",
            ),
            confidence=0.85,
            reason="test decision",
        ),
    )


def test_reply_interpretation_yes_without_pending_state_is_not_save_action():
    reply = interpret_reply_for_project_comment_save(
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=1602,
        last_assistant_message="Hello, how can I help?",
        user_message="yes",
    )

    assert reply.is_confirmation_reply is True
    assert reply.has_pending_save is False
    assert reply.kind == "none"
    assert reply.action == "none"
    assert reply.runtime_action == "none"


def test_reply_interpretation_pending_save_approves_compound_confirmation():
    reply = _pending_reply("Confirm yes commit it")

    assert reply.has_pending_save is True
    assert reply.kind == "approve"
    assert reply.action == "approve_save"
    assert reply.confirms_project_comment_save is True


def test_reply_interpretation_pending_save_denies_negative_confirmation():
    reply = _pending_reply("no dont commit it")

    assert reply.has_pending_save is True
    assert reply.kind == "deny"
    assert reply.action == "deny_save"
    assert reply.confirms_project_comment_save is False


def test_reply_interpretation_turn_decision_deny_overrides_runtime_action():
    reply = _pending_reply("yes please")
    reply = reply.with_turn_decision(
        turn_decision_result=_turn_decision_result(reply_kind="deny"),
        turn_decision_runtime_applied=True,
        available_tool_names={"create_project_comment"},
        focused_project_id=1602,
    )

    assert reply.applied is True
    assert reply.classifier_kind == "deny"
    assert reply.runtime_action == "deny_save"
    assert reply.policy_messages


def test_reply_interpretation_turn_decision_can_create_pending_state():
    reply = interpret_reply_for_project_comment_save(
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=1602,
        last_assistant_message="I can help draft a note.",
        user_message="save it",
    )
    reply = reply.with_turn_decision(
        turn_decision_result=_turn_decision_result(reply_kind="approve"),
        turn_decision_runtime_applied=True,
        available_tool_names={"create_project_comment"},
        focused_project_id=1602,
    )

    assert reply.has_pending_save is True
    assert reply.awaiting_save is not None
    assert reply.awaiting_save.source == "turn_decision_confirm_save"
    assert reply.runtime_action == "approve_save"

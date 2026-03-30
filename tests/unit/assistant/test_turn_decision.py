from __future__ import annotations

import json

from ispec.assistant.service import AssistantReply
from ispec.assistant.tool_routing import ToolGroup
from ispec.assistant.turn_decision import run_turn_decision_pipeline


def test_turn_decision_pipeline_normalizes_support_write_mode_and_primary_group() -> None:
    groups = [
        ToolGroup(
            name="projects",
            description="Project lookups and comment writes.",
            tool_names=("get_project", "create_project_comment"),
        ),
        ToolGroup(
            name="tmux",
            description="Tmux inspection.",
            tool_names=("assistant_list_tmux_panes",),
        ),
    ]

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_):  # type: ignore[no-untyped-def]
        assert tools is None
        assert isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body
        return AssistantReply(
            content=json.dumps(
                {
                    "source": "support_chat",
                    "primary_goal": "save_project_comment",
                    "needs_clarification": False,
                    "clarification_reason": "none",
                    "tool_plan": {
                        "use_tools": True,
                        "primary_group": "",
                        "secondary_groups": [],
                        "preferred_first_tool": "get_project",
                    },
                    "write_plan": {"mode": "none"},
                    "response_plan": {"mode": "compare", "contract_cap": "brief_explainer"},
                    "confidence": 0.83,
                    "reason": "A project lookup should happen before deciding whether the comment can be saved.",
                }
            ),
            provider="test",
            model="test-model",
            meta={"elapsed_ms": 5},
        )

    result = run_turn_decision_pipeline(
        generate_reply_fn=fake_generate_reply,
        mode="shadow",
        source="support_chat",
        user_message="Please save this note to project 1598.",
        last_assistant_message="",
        focused_project_id=1598,
        referenced_project_ids=[1598],
        groups=groups,
        response_modes=["single", "compare"],
        contract_caps=["direct", "brief_explainer"],
        extra_context={},
    )

    assert result.ok is True
    assert result.decision is not None
    assert result.decision.tool_plan.primary_group == "projects"
    assert result.decision.tool_plan.preferred_first_tool == "get_project"
    assert result.decision.write_plan.mode == "save_now"
    assert result.decision.response_plan.mode == "compare"
    assert "inferred_primary_group_from_preferred_tool" in result.warnings
    assert "write_plan_normalized_to_save_now" in result.warnings


def test_turn_decision_pipeline_forces_scheduled_assistant_single_mode_without_clarification() -> None:
    groups = [
        ToolGroup(
            name="devops",
            description="Staff automation tools.",
            tool_names=("assistant_enqueue_staff_slack_message",),
        )
    ]

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_):  # type: ignore[no-untyped-def]
        assert tools is None
        assert isinstance(vllm_extra_body, dict) and "guided_json" in vllm_extra_body
        return AssistantReply(
            content=json.dumps(
                {
                    "source": "scheduled_assistant",
                    "primary_goal": "automation_task",
                    "needs_clarification": True,
                    "clarification_reason": "missing_identifier",
                    "tool_plan": {
                        "use_tools": True,
                        "primary_group": "devops",
                        "secondary_groups": [],
                        "preferred_first_tool": "",
                    },
                    "write_plan": {"mode": "none"},
                    "response_plan": {"mode": "compare", "contract_cap": "direct"},
                    "confidence": 0.64,
                    "reason": "This is an automated staff-facing task.",
                }
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    result = run_turn_decision_pipeline(
        generate_reply_fn=fake_generate_reply,
        mode="shadow",
        source="scheduled_assistant",
        user_message="Prepare the weekly update and post it to staff Slack.",
        last_assistant_message=None,
        focused_project_id=None,
        referenced_project_ids=[],
        groups=groups,
        response_modes=["single"],
        contract_caps=["direct", "brief_explainer"],
        extra_context={"required_tool": "assistant_enqueue_staff_slack_message"},
    )

    assert result.ok is True
    assert result.decision is not None
    assert result.decision.needs_clarification is False
    assert result.decision.clarification_reason == "none"
    assert result.decision.response_plan.mode == "single"
    assert "scheduled_assistant_forced_no_clarification" in result.warnings
    assert "invalid_response_mode" in result.warnings

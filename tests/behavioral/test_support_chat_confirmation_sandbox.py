from __future__ import annotations

import json
import types

import pytest

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.service import AssistantReply
from ispec.assistant.turn_decision import (
    TurnDecision,
    TurnDecisionReplyInterpretation,
    TurnDecisionResponsePlan,
    TurnDecisionResult,
    TurnDecisionToolPlan,
    TurnDecisionWritePlan,
)
from ispec.db.connect import get_session
from ispec.db.models import ProjectComment, UserRole
from ispec.schedule.connect import get_schedule_session


pytestmark = pytest.mark.behavioral


def test_behavioral_support_chat_confirmation_round_trip_saves_into_sandbox(
    behavioral_datastore,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "own")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    project_id = behavioral_datastore.project_ids[0]

    monkeypatch.setattr(
        support_routes,
        "openai_tools_for_user",
        lambda _user: [
            {"type": "function", "function": {"name": "assistant_prompt_header", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "assistant_list_tools", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {
                    "name": "create_project_comment",
                    "parameters": {"type": "object", "required": ["project_id", "comment", "confirm"]},
                },
            },
        ],
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
                primary_goal="confirm_save",
                needs_clarification=False,
                clarification_reason="none",
                tool_plan=TurnDecisionToolPlan(
                    use_tools=True,
                    primary_group="projects",
                    secondary_groups=(),
                    preferred_first_tool=None,
                ),
                write_plan=TurnDecisionWritePlan(mode="confirm_save"),
                response_plan=TurnDecisionResponsePlan(mode="single", contract_cap="direct"),
                reply_interpretation=TurnDecisionReplyInterpretation(
                    kind="approve",
                    confidence=0.97,
                    reason="The user approved the pending save.",
                ),
                confidence=0.94,
                reason="Behavioral sandbox scenario should complete the save.",
            ),
        ),
    )

    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
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
                                    "project_id": project_id,
                                    "comment": "Behavioral sandbox draft is ready for review.",
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

    with (
        get_assistant_session(behavioral_datastore.assistant_db_path) as assistant_db,
        get_schedule_session(behavioral_datastore.schedule_db_path) as schedule_db,
        get_session(behavioral_datastore.core_db_path) as core_db,
    ):
        support_session = SupportSession(
            session_id="behavioral-session-1",
            user_id=None,
            state_json=json.dumps({"current_project_id": project_id}),
        )
        assistant_db.add(support_session)
        assistant_db.flush()
        assistant_db.add(
            SupportMessage(
                session_pk=support_session.id,
                role="assistant",
                content=(
                    f"Draft comment for project {project_id}:\n"
                    "Behavioral sandbox draft is ready for review.\n\n"
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

        response = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": "behavioral-session-1",
                    "message": "Yes, save it.",
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=service_user,
        )

        assert response.message == "Saved the note. Comment ID is 1."
        comment = (
            core_db.query(ProjectComment)
            .filter(ProjectComment.project_id == project_id)
            .order_by(ProjectComment.id.desc())
            .first()
        )
        assert comment is not None
        assert comment.com_Comment == "Behavioral sandbox draft is ready for review."

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["reply_interpretation"]["runtime_action"] == "approve_save"
        assert meta["reply_interpretation"]["applied"] is True

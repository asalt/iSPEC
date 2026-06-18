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


def test_behavioral_support_chat_mspc_prepare_then_compound_confirm_survives_cautious_classifier(
    behavioral_datastore,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "own")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "3000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    project_id = 1563

    with get_session(behavioral_datastore.core_db_path) as core_db:
        from ispec.db.models import Project

        core_db.add(Project(id=project_id, prj_AddedBy="behavioral", prj_ProjectTitle="IgG antigen review"))
        core_db.commit()

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

    def fake_turn_decision(**kwargs):
        if str(kwargs.get("user_message") or "").strip().lower() == "correct i confirm":
            primary_goal = "confirm_save"
            write_mode = "confirm_save"
            reply_kind = "unclear"
            reason = "The classifier was cautious about a compound confirmation phrase."
        else:
            primary_goal = "draft_project_comment"
            write_mode = "draft_only"
            reply_kind = "none"
            reason = "The user asked for a note draft before confirmation."
        return TurnDecisionResult(
            ok=True,
            mode="own",
            source="support_chat",
            applied=True,
            decision=TurnDecision(
                source="support_chat",
                primary_goal=primary_goal,
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
                    confidence=0.96,
                    reason=reason,
                ),
                confidence=0.93,
                reason=reason,
            ),
        )

    monkeypatch.setattr(support_routes, "run_turn_decision_pipeline", fake_turn_decision)
    monkeypatch.setattr(support_routes, "select_support_tool_policy", lambda **_: None)

    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(calls) == 1:
            tool_names = {
                str(spec.get("function", {}).get("name") or "")
                for spec in (tools or [])
                if isinstance(spec, dict)
            }
            assert "create_project_comment" not in tool_names
            return AssistantReply(
                content=(
                    "FINAL:\n"
                    "Prepared note for MSPC001563:\n"
                    "Kwangwon reported that project MSPC001563 finished an MS run for the IgG-bound antigen sample "
                    "on the Eclipse system. Please review the chromatogram.\n\n"
                    "Reply confirm to save this note, or send edits."
                ),
                provider="test",
                model="test-model",
                meta=None,
            )
        if len(calls) == 2:
            assert tool_choice == {"type": "function", "function": {"name": "create_project_comment"}}
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
                                    "comment": (
                                        "Kwangwon reported that project MSPC001563 finished an MS run for the "
                                        "IgG-bound antigen sample on the Eclipse system. Please review the chromatogram."
                                    ),
                                    "comment_type": "assistant_note",
                                    "confirm": True,
                                }
                            ),
                        },
                    }
                ],
            )
        return AssistantReply(
            content="FINAL:\nSaved the note to project 1563.",
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
        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        session_id = "behavioral-mspc-confirm"
        first = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": session_id,
                    "message": (
                        "prepare a message that is sent by kwangwon about how mspc1563 (57987), "
                        "finished ms run for the IgG-bound antigen sample on eclipse. Take a look at the "
                        "chromatogram. Prepare the note for review then I can confirm or deny it"
                    ),
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=service_user,
        )
        assert "Prepared note for MSPC001563" in (first.message or "")

        support_session = (
            assistant_db.query(SupportSession)
            .filter(SupportSession.session_id == session_id)
            .one()
        )
        persisted_state = json.loads(support_session.state_json)
        assert persisted_state["current_project_id"] == project_id

        second = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": session_id,
                    "message": "correct i confirm",
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=service_user,
        )

        assert second.message == "Saved the note to project 1563."
        comment = (
            core_db.query(ProjectComment)
            .filter(ProjectComment.project_id == project_id)
            .order_by(ProjectComment.id.desc())
            .first()
        )
        assert comment is not None
        assert "IgG-bound antigen sample" in comment.com_Comment

        assistant_row = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == support_session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["reply_interpretation"]["awaiting_state"] == "project_comment_save_confirmation"
        assert meta["reply_interpretation"]["classifier_kind"] == "unclear"
        assert meta["reply_interpretation"]["runtime_kind"] == "approve"
        assert meta["reply_interpretation"]["runtime_action"] == "approve_save"
        assert meta["reply_interpretation"]["applied"] is True
        assert meta["tooling"]["forced_tool_choice"] == {
            "type": "function",
            "function": {"name": "create_project_comment"},
        }


def test_behavioral_support_chat_corrective_note_compound_confirm_routes_to_write(
    behavioral_datastore,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "own")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE", "shadow")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "3000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    project_id = 1546
    wrong_note = "Upload of raw files to OneDrive has been completed. Alert customer regarding availability for download."
    corrected_note = (
        "Correction to prior note: the raw file upload to OneDrive is not complete yet. "
        "Raw files still need to be uploaded to OneDrive, and the customer should be alerted once they are available. "
        "The prior note was saved after normal draft-and-confirm approval, but the approved draft incorrectly implied completion."
    )

    with get_session(behavioral_datastore.core_db_path) as core_db:
        from ispec.db.models import Person, Project

        person = Person(ppl_AddedBy="behavioral", ppl_Name_First="Alex", ppl_Name_Last="Tester")
        core_db.add_all(
            [
                Project(id=project_id, prj_AddedBy="behavioral", prj_ProjectTitle="Raw file upload follow-up"),
                person,
            ]
        )
        core_db.flush()
        core_db.add(
            ProjectComment(
                project_id=project_id,
                person_id=person.id,
                com_Comment=wrong_note,
                com_CommentType="assistant_note",
                com_AddedBy="api_key",
            )
        )
        core_db.commit()

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

    def fake_classifier_reply(**kwargs):
        messages = kwargs.get("messages") or []
        payload = json.loads(str(messages[-1]["content"])) if messages else {}
        user_message = str(payload.get("user_message") or "").lower()
        label = "approve_save" if "confirm this is acceptable" in user_message else "requires_explicit_confirmation"
        reason = (
            "The user approves the pending corrected note and adds audit context."
            if label == "approve_save"
            else "The user is discussing a correction before an explicit approval."
        )
        return AssistantReply(
            content=json.dumps({"label": label, "confidence": 0.94, "reason": reason}),
            provider="classifier_vllm",
            model="test-classifier",
            meta={"elapsed_ms": 9},
        )

    def fake_turn_decision(**kwargs):
        if "confirm this is acceptable" in str(kwargs.get("user_message") or "").strip().lower():
            primary_goal = "confirm_save"
            write_mode = "confirm_save"
            reply_kind = "approve"
            reason = "The user approves the pending corrected note and provides supporting audit context."
        else:
            primary_goal = "draft_project_comment"
            write_mode = "draft_only"
            reply_kind = "none"
            reason = "The user reports that the previously saved note had incorrect semantics."
        return TurnDecisionResult(
            ok=True,
            mode="own",
            source="support_chat",
            applied=True,
            decision=TurnDecision(
                source="support_chat",
                primary_goal=primary_goal,
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
                    confidence=0.95,
                    reason=reason,
                ),
                confidence=0.92,
                reason=reason,
            ),
        )

    monkeypatch.setattr(support_routes, "generate_classifier_reply", fake_classifier_reply)
    monkeypatch.setattr(support_routes, "run_turn_decision_pipeline", fake_turn_decision)
    monkeypatch.setattr(support_routes, "select_support_tool_policy", lambda **_: None)

    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_) -> AssistantReply:
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        message_text = "\n".join(str(item.get("content") or "") for item in (messages or []) if isinstance(item, dict))
        if "TOOL_RESULT create_project_comment" in message_text:
            return AssistantReply(
                content="FINAL:\nSaved the corrective note to project 1546.",
                provider="test",
                model="test-model",
                meta=None,
            )
        if tool_choice == {"type": "function", "function": {"name": "create_project_comment"}}:
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
                                    "comment": corrected_note,
                                    "comment_type": "assistant_note",
                                    "confirm": True,
                                }
                            ),
                        },
                    }
                ],
            )
        if "confirm this is acceptable" not in message_text:
            tool_names = {
                str(spec.get("function", {}).get("name") or "")
                for spec in (tools or [])
                if isinstance(spec, dict)
            }
            assert "create_project_comment" not in tool_names
            return AssistantReply(
                content=(
                    "FINAL:\n"
                    "Corrected draft for MSPC001546:\n"
                    f"{corrected_note}\n\n"
                    "Please confirm if this corrected note is acceptable before I save it to the project's notes."
                ),
                provider="test",
                model="test-model",
                meta=None,
            )
        return AssistantReply(
            content="FINAL:\nPlease confirm whether you want me to save the corrected note.",
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
        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        session_id = "behavioral-corrective-note-confirm"
        first = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": session_id,
                    "message": (
                        "oops i made a mistake on project 1546. they are not complete; "
                        "raw files still need to be uploaded to OneDrive and then the customer should be alerted."
                    ),
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=service_user,
        )
        assert "Corrected draft for MSPC001546" in (first.message or "")
        assert "not complete yet" in (first.message or "")

        support_session = (
            assistant_db.query(SupportSession)
            .filter(SupportSession.session_id == session_id)
            .one()
        )
        persisted_state = json.loads(support_session.state_json)
        assert persisted_state["current_project_id"] == project_id

        second = chat(
            ChatRequest.model_validate(
                {
                    "sessionId": session_id,
                    "message": (
                        "confirm this is acceptable. and let the record show that the first note was a user error "
                        "despite the chatbot's appropriate note confirmation."
                    ),
                    "history": [],
                    "ui": None,
                }
            ),
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            user=service_user,
        )

        assert second.message == "Saved the corrective note to project 1546."
        comments = (
            core_db.query(ProjectComment)
            .filter(ProjectComment.project_id == project_id)
            .order_by(ProjectComment.id.asc())
            .all()
        )
        assert len(comments) == 2
        assert wrong_note in comments[0].com_Comment
        assert "raw file upload to OneDrive is not complete yet" in comments[1].com_Comment
        assert "normal draft-and-confirm approval" in comments[1].com_Comment
        assistant_db.refresh(support_session)
        final_state = json.loads(support_session.state_json)
        work_bag_entries = final_state["work_bag"]["entries"]
        assert work_bag_entries
        latest_work = work_bag_entries[-1]
        assert latest_work["tool_name"] == "create_project_comment"
        assert latest_work["status"] == "succeeded"
        assert {"kind": "project", "id": project_id} in latest_work["refs"]
        assert {"kind": "project_comment", "id": int(comments[1].id)} in latest_work["refs"]
        assert "raw_arguments" in latest_work["omitted"]

        assistant_row = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == support_session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["reply_interpretation"]["awaiting_state"] == "project_comment_save_confirmation"
        assert meta["reply_interpretation"]["classifier_kind"] == "approve"
        assert meta["reply_interpretation"]["runtime_action"] == "approve_save"
        assert meta["tooling"]["forced_tool_choice"] == {
            "type": "function",
            "function": {"name": "create_project_comment"},
        }
        approval_eval = meta["project_comment_approval_eval"]
        assert approval_eval["state_gate"]["kind"] == "pending_save_confirmation"
        assert approval_eval["classifier"]["label"] == "approve_save"
        assert approval_eval["policy"]["decision"] == "shadow_ticket"
        assert approval_eval["write_outcome"]["status"] == "succeeded"

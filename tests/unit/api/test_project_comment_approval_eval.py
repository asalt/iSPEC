from __future__ import annotations

import json
import types
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant import project_comment_approval as pc_approval
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import Project, ProjectComment, UserRole
from ispec.schedule.connect import get_schedule_session


def test_project_comment_approval_policy_issues_shadow_ticket_for_high_confidence_approval() -> None:
    settings = pc_approval.ProjectCommentApprovalSettings(
        mode="shadow",
        approve_threshold=0.8,
        decision_threshold=0.6,
        ticket_ttl_seconds=120,
    )
    trigger = pc_approval.detect_project_comment_trigger(
        message="yes please save it",
        legacy_confirmation_kind="affirmative",
        legacy_save_requested=False,
    )
    gate = pc_approval.ProjectCommentGate(
        eligible=True,
        kind="pending_save_confirmation",
        reason="awaiting_project_comment_save_confirmation",
        project_id=1602,
        pending_tool="create_project_comment",
        session_id="session-1",
        thread_key="thread-1",
        current_turn_id="42",
        prior_assistant_message_id=None,
        pending_draft_hash="abc",
        source_message_hash="def",
    )
    classifier = pc_approval.ProjectCommentClassifierResult(
        ran=True,
        ok=True,
        label="approve_save",
        confidence=0.91,
        reason="explicit approval",
        provider="test",
        model="test",
        latency_ms=0,
        timeout_ms=settings.timeout_ms,
        error=None,
    )

    policy_obj, ticket_obj = pc_approval.decide_project_comment_policy(
        settings=settings,
        gate=gate,
        trigger=trigger,
        classifier=classifier,
    )
    policy = policy_obj.to_dict()
    ticket = ticket_obj.to_dict()

    assert policy["decision"] == "shadow_ticket"
    assert policy["live_applied"] is False
    assert ticket["issued"] is True
    assert ticket["shadow_only"] is True
    assert ticket["tool_name"] == "create_project_comment"
    assert ticket["project_id"] == 1602


def test_project_comment_approval_policy_denial_feature_blocks_shadow_ticket() -> None:
    settings = pc_approval.ProjectCommentApprovalSettings(
        mode="shadow",
        approve_threshold=0.8,
        decision_threshold=0.6,
        ticket_ttl_seconds=120,
    )
    trigger = pc_approval.detect_project_comment_trigger(
        message="no dont save it",
        legacy_confirmation_kind="none",
        legacy_save_requested=False,
    )
    gate = pc_approval.ProjectCommentGate(
        eligible=True,
        kind="pending_save_confirmation",
        reason="awaiting_project_comment_save_confirmation",
        project_id=1602,
        pending_tool="create_project_comment",
        session_id="session-1",
        thread_key="thread-1",
        current_turn_id="42",
        prior_assistant_message_id=None,
        pending_draft_hash=None,
        source_message_hash="def",
    )
    classifier = pc_approval.ProjectCommentClassifierResult(
        ran=True,
        ok=True,
        label="approve_save",
        confidence=0.99,
        reason="model was wrong but protocol denial wins",
        provider="test",
        model="test",
        latency_ms=0,
        timeout_ms=settings.timeout_ms,
        error=None,
    )

    policy_obj, ticket_obj = pc_approval.decide_project_comment_policy(
        settings=settings,
        gate=gate,
        trigger=trigger,
        classifier=classifier,
    )
    policy = policy_obj.to_dict()
    ticket = ticket_obj.to_dict()

    assert policy["decision"] == "no_ticket"
    assert "explicit_command_protocol_deny" in policy["reasons"]
    assert ticket["issued"] is False


def test_project_comment_approval_eval_metadata_records_shadow_ticket_and_write_outcome(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "stub")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE", "shadow")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "off")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1602, prj_AddedBy="test", prj_ProjectTitle="Project 1602"))
    db_session.commit()

    llm_calls: list[dict[str, Any]] = []

    def fake_classifier_reply(**kwargs: Any) -> AssistantReply:
        assert kwargs["timeout_seconds"] == 6.0
        return AssistantReply(
            content=json.dumps(
                {
                    "label": "approve_save",
                    "confidence": 0.93,
                    "reason": "The user explicitly asked to save the project comment.",
                }
            ),
            provider="classifier_vllm",
            model="test-classifier",
            meta={"elapsed_ms": 12},
        )

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_: Any) -> AssistantReply:
        llm_calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if len(llm_calls) == 1:
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
                                    "project_id": 1602,
                                    "comment": "Tattym Sheikh's iLab request is awaiting a charge source number.",
                                    "comment_type": "note",
                                    "confirm": True,
                                }
                            ),
                        },
                    }
                ],
            )
        return AssistantReply(
            content="FINAL:\nSaved the note to project 1602.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_classifier_reply", fake_classifier_reply)
    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-approval-eval", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                ChatRequest.model_validate(
                    {
                        "sessionId": "session-approval-eval",
                        "message": "Please make a note for project 1602 and save it.",
                        "history": [],
                        "ui": None,
                    }
                ),
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert response.message == "Saved the note to project 1602."
        comment = db_session.query(ProjectComment).filter(ProjectComment.project_id == 1602).first()
        assert comment is not None

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        approval_eval = meta["project_comment_approval_eval"]
        assert approval_eval["state_gate"]["eligible"] is True
        assert approval_eval["state_gate"]["kind"] == "direct_write_candidate"
        assert approval_eval["classifier"]["label"] == "approve_save"
        assert approval_eval["classifier"]["confidence"] == 0.93
        assert approval_eval["policy"]["decision"] == "shadow_ticket"
        assert approval_eval["write_ticket_shadow"]["issued"] is True
        assert approval_eval["write_ticket_shadow"]["shadow_only"] is True
        assert approval_eval["legacy_decision"]["forced_tool_choice"] == "create_project_comment"
        assert approval_eval["write_outcome"]["status"] == "succeeded"
        assert approval_eval["write_outcome"]["comment_id"] == comment.id


def test_project_comment_approval_eval_gate_catches_typo_project_note_candidate() -> None:
    message = "mkae a note on project 1387 is is a mlewis liver pdx dataset for the portal"

    trigger = pc_approval.detect_project_comment_trigger(
        message=message,
        legacy_confirmation_kind="none",
        legacy_save_requested=False,
    )
    features = trigger.to_lexical_features()
    settings = pc_approval.ProjectCommentApprovalSettings(mode="shadow")
    gate = pc_approval.build_project_comment_gate(
        settings=settings,
        trigger=trigger,
        tool_protocol="openai",
        available_tool_names={"create_project_comment"},
        focused_project_id=1387,
        session_id="session-typo-note",
        thread_key="thread-typo-note",
        current_turn_id="1",
        prior_assistant_message_id=None,
        prior_assistant_message=None,
        user_message=message,
        awaiting_reply_state=None,
    ).to_dict()

    assert trigger.kind == "direct_note_request"
    assert features["project_comment_write_candidate"] is True
    assert {"token": "mkae", "matched": "make"} in features["project_comment_near_action_terms"]
    assert gate["eligible"] is True
    assert gate["kind"] == "direct_write_candidate"
    assert gate["reason"] == "project_comment_write_candidate"


def test_project_comment_approval_eval_keeps_tool_exposed_for_typo_note_candidate_without_forcing_write(
    tmp_path,
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "stub")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE", "shadow")
    monkeypatch.setenv("ISPEC_ASSISTANT_TURN_DECISION_MODE", "off")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    db_session.add(Project(id=1387, prj_AddedBy="test", prj_ProjectTitle="Phosphoproteomic profiling of PDX samples"))
    db_session.commit()

    def fake_classifier_reply(**kwargs: Any) -> AssistantReply:
        return AssistantReply(
            content=json.dumps(
                {
                    "label": "requires_explicit_confirmation",
                    "confidence": 0.72,
                    "reason": "The user likely wants a project note, but the action word is misspelled.",
                }
            ),
            provider="classifier_vllm",
            model="test-classifier",
            meta={"elapsed_ms": 11},
        )

    llm_calls: list[dict[str, Any]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_: Any) -> AssistantReply:
        llm_calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        tool_names = {
            str(tool.get("function", {}).get("name") or "")
            for tool in (tools or [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        assert "create_project_comment" in tool_names
        return AssistantReply(
            content="FINAL:\nPlease confirm whether you want me to save that project note.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_classifier_reply", fake_classifier_reply)
    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-typo-note-eval", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        service_user = types.SimpleNamespace(
            id=1,
            username="api_key",
            role=UserRole.viewer,
            can_write_project_comments=True,
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                ChatRequest.model_validate(
                    {
                        "sessionId": "session-typo-note-eval",
                        "message": "mkae a note on project 1387 is is a mlewis liver pdx dataset for the portal",
                        "history": [],
                        "ui": None,
                    }
                ),
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=service_user,
            )

        assert response.message == "Please confirm whether you want me to save that project note."
        assert llm_calls
        assert db_session.query(ProjectComment).filter(ProjectComment.project_id == 1387).first() is None

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        approval_eval = meta["project_comment_approval_eval"]
        assert meta["tooling"]["project_comment_write_tool_exposed"] is True
        assert meta["tooling"]["project_comment_write_tool_reason"] == "project_comment_trigger"
        assert approval_eval["state_gate"]["eligible"] is True
        assert approval_eval["state_gate"]["kind"] == "direct_write_candidate"
        assert approval_eval["lexical_features"]["project_comment_write_candidate"] is True
        assert approval_eval["classifier"]["label"] == "requires_explicit_confirmation"
        assert approval_eval["policy"]["decision"] == "ask_explicit_confirmation"
        assert approval_eval["write_outcome"]["status"] == "not_attempted"

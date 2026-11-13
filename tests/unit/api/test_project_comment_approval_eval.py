from __future__ import annotations

import json
import types

import pytest

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant import project_comment_approval as pc_approval
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.assistant.small_classifier import build_small_classifier_task
from ispec.db.models import Project, ProjectAccessMode, ProjectComment, UserRole
from ispec.schedule.connect import get_schedule_session


@pytest.fixture(autouse=True)
def disable_legacy_comment_push(monkeypatch):
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")


def test_project_comment_approval_classifier_prompt_handles_compound_approval_context() -> None:
    prepared = build_small_classifier_task(
        "project_comment_approval",
        payload={
            "user_message": "confirm this is acceptable. and let the record show the prior note was user error.",
            "prior_assistant_message": "Please confirm if this corrected note is acceptable before I save it.",
            "state_gate": {"kind": "pending_save_confirmation"},
            "trigger": {"kind": "confirmation"},
            "lexical_features": {"legacy_confirmation_kind": "unclear"},
            "pending_action": "create_project_comment",
            "focused_project_id": 1546,
        },
    )
    prompt_text = "\n".join(str(item.get("content") or "") for item in prepared.messages)

    assert "approval phrases can be longer than one word" in prompt_text
    assert "confirm this is acceptable" in prompt_text
    assert "approve_save when the message approves the pending draft" in prompt_text
    assert "mention the extra context in reason" in prompt_text


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



@pytest.mark.parametrize("mode", ["off", "shadow", "own"])
@pytest.mark.parametrize("message,attempt_write", [
    ("correct i confirm", False),
    ("correct i confirm", True),
    ("mkae a note on project 1387", False),
    ("Please save a note to project 1387 now", True),
    ("Yes that's the project. No, don't save a comment.", True),
    ("Compare project 1387 and project 1602", False),
])
def test_prose_does_not_create_pending_action_or_trigger_classifier(
    tmp_path, db_session, monkeypatch, mode, message, attempt_write,
):
    for key, value in {
        "PROVIDER": "stub", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": mode,
        "TURN_DECISION_MODE": "off", "TOOL_PROTOCOL": "openai", "MAX_TOOL_CALLS": "2",
        "HISTORY_LIMIT": "10", "MAX_PROMPT_TOKENS": "4000", "SUMMARY_MAX_CHARS": "0",
        "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    db_session.add(Project(id=1387, prj_AddedBy="test", prj_ProjectTitle="Synthetic project"))
    db_session.commit()
    calls = []
    def classify(**kwargs):
        pytest.fail("No registered draft: approval classifier must not run")
    def generate(**kwargs):
        calls.append(kwargs)
        names = {t["function"]["name"] for t in kwargs.get("tools") or []}
        assert {"prepare_project_comment", "show_project_comment_card"} <= names
        assert "create_project_comment" in names
        assert kwargs.get("tool_choice") is None
        if attempt_write and len(calls) == 1:
            return AssistantReply(content="", provider="test", model="test", tool_calls=[{
                "id": "save", "type": "function", "function": {
                    "name": "create_project_comment",
                    "arguments": json.dumps({"project_id": 1387, "comment": "A note", "confirm": True}),
                },
            }])
        if attempt_write:
            result = next(json.loads(m["content"]) for m in kwargs["messages"] if m["role"] == "tool")
            assert not result["ok"]
            assert result["error_code"] == "invalid_arguments"
        return AssistantReply(content="FINAL:\\nLet's review the draft.", provider="test", model="test")
    monkeypatch.setattr(support_routes, "generate_classifier_reply", classify)
    monkeypatch.setattr(support_routes, "generate_reply", generate)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        session = SupportSession(session_id="prose-only", state_json=json.dumps({"current_project_id": 1387}))
        adb.add(session)
        adb.flush()
        adb.add(support_routes.SupportMessage(session_pk=session.id, role="assistant",
            content='Draft for project 1387: "A note". Would you like me to save this note to project history?'))
        adb.commit()
        user = types.SimpleNamespace(id=1, username="test", role=UserRole.editor,
                                     project_access_mode=ProjectAccessMode.all_projects)
        result = chat(ChatRequest(sessionId=session.session_id, message=message, history=[]),
                      assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        meta = json.loads(adb.get(support_routes.SupportMessage, result.messageId).meta_json)
        assert not meta["project_comment_workflow"]["save_checks"]
        assert meta["tooling"]["project_comment_write_tool_exposed"]
        assert meta["project_comment_workflow"]["write_outcome"]["status"] == ("failed" if attempt_write else "not_attempted")
        assert db_session.query(ProjectComment).count() == 0

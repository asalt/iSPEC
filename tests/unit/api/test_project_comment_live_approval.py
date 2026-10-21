import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, utcnow
from ispec.assistant.project_comment_approval import (
    ProjectCommentApprovalSettings, ProjectCommentAuthorization, ProjectCommentClassifierResult,
    ProjectCommentGate, decide_project_comment_policy, detect_project_comment_trigger, stable_hash,
)
from ispec.assistant.service import AssistantReply
from ispec.db.models import Project, ProjectComment, ProjectAccessMode, UserRole
from ispec.schedule.connect import get_schedule_session


@pytest.mark.parametrize("label,confidence,expected", [
    ("approve_save", .75, True), ("approve_save", .749, False),
    ("deny_save", .99, False), ("draft_only", .99, False),
    ("revise_draft", .99, False), ("unrelated_or_unclear", .99, False),
    ("requires_explicit_confirmation", .99, False),
    ("invalid", .99, False), ("timeout", .99, False),
    ("approve_save", float("nan"), False), ("approve_save", float("inf"), False),
    ("approve_save", True, False), ("approve_save", 1.01, False),
])
def test_live_classifier_owns_pending_approval(tmp_path, db_session, monkeypatch, label, confidence, expected):
    for key, value in {
        "PROVIDER": "stub", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own",
        "PROJECT_COMMENT_APPROVAL_APPROVE_THRESHOLD": "0.75", "TURN_DECISION_MODE": "off",
        "TOOL_PROTOCOL": "openai", "MAX_TOOL_CALLS": "3", "HISTORY_LIMIT": "10",
        "MAX_PROMPT_TOKENS": "4000", "SUMMARY_MAX_CHARS": "0",
        "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    db_session.add(Project(id=1568, prj_AddedBy="test", prj_ProjectTitle="IP samples"))
    db_session.commit()

    def classify(**kwargs):
        if label == "timeout":
            raise TimeoutError("synthetic")
        return AssistantReply(content="not json" if label == "invalid" else json.dumps({
            "label": label, "confidence": confidence, "reason": "Synthetic decision",
        }), provider="test", model="test")

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            names = {t["function"]["name"] for t in kwargs.get("tools") or []}
            assert ("create_project_comment" in names) == expected
            assert (kwargs.get("tool_choice") == {
                "type": "function", "function": {"name": "create_project_comment"},
            }) == expected
            # Even a model ignoring hidden tools cannot bypass a failed policy.
            # Two calls also prove a successful approval cannot write twice.
            return AssistantReply(content="", provider="test", model="test", tool_calls=[{
                "id": f"call_{n}", "type": "function", "function": {
                    "name": "create_project_comment", "arguments": json.dumps({
                        "project_id": 1568, "comment": "Received 24 IP samples.", "confirm": True,
                    }),
                },
            } for n in range(2)])
        return AssistantReply(content="FINAL:\nSaved." if expected else "FINAL:\nPlease clarify whether to save.",
                              provider="test", model="test")

    monkeypatch.setattr(support, "generate_classifier_reply", classify)
    monkeypatch.setattr(support, "generate_reply", generate)
    user = SimpleNamespace(id=1, username="api_key", role=UserRole.viewer,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        session = SupportSession(session_id="pending-1568", state_json=json.dumps({"current_project_id": 1568}))
        adb.add(session)
        adb.flush()
        adb.add(SupportMessage(session_pk=session.id, role="assistant",
                               content='Draft for project 1568: "Received 24 IP samples." Would you like me to save this note to project history?'))
        adb.commit()
        result = support.chat(support.ChatRequest(sessionId=session.session_id,
            message="correct i confirm; please save, and we are testing approval for later", history=[]),
            assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        assert db_session.query(ProjectComment).filter_by(project_id=1568).count() == int(expected)
        meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)["project_comment_approval_eval"]
        assert meta["policy"]["live_applied"]
        assert meta["write_ticket"]["issued"] == expected
        assert meta["write_ticket"]["consumed"] == expected
        assert "write_ticket_shadow" not in meta
        assert meta["write_outcome"]["status"] == ("succeeded" if expected else "failed")


def authorization(*, mode="own", **gate_changes):
    values = dict(eligible=True, kind="pending_save_confirmation", reason="test", project_id=1568,
                  pending_tool="create_project_comment", session_id="s", thread_key="t", current_turn_id="1",
                  prior_assistant_message_id=1, pending_draft_hash="draft", source_message_hash=stable_hash("yes"))
    values.update(gate_changes)
    gate = ProjectCommentGate(**values)
    _, ticket = decide_project_comment_policy(settings=ProjectCommentApprovalSettings(mode=mode), gate=gate,
        trigger=detect_project_comment_trigger(message="yes", legacy_confirmation_kind="approve", legacy_save_requested=False),
        classifier=ProjectCommentClassifierResult(ran=True, ok=True, label="approve_save", confidence=.75,
            reason="yes", provider="test", model="test", latency_ms=1, timeout_ms=6000, error=None))
    return ProjectCommentAuthorization(gate=gate, ticket=ticket)


@pytest.mark.parametrize("change", [dict(project_id=1409), dict(session_id="other"), dict(turn_id="2"),
                                    dict(user_message="no")])
def test_ticket_rejects_mismatched_scope(change):
    args = dict(project_id=1568, session_id="s", turn_id="1", user_message="yes", comment="note")
    args.update(change)
    assert authorization().consume(**args)


def test_ticket_shadow_expiry_and_single_use(monkeypatch):
    args = dict(project_id=1568, session_id="s", turn_id="1", user_message="yes", comment="note")
    assert authorization(mode="shadow").consume(**args)
    auth = authorization()
    assert auth.consume(**args) is None
    assert "already used" in auth.consume(**args)
    expired = authorization()
    future = utcnow() + timedelta(seconds=121)
    monkeypatch.setattr("ispec.assistant.models.utcnow", lambda: future)
    assert "expired" in expired.consume(**args)


def test_inherited_classifier_forwards_bounded_timeout(monkeypatch):
    from ispec.assistant.classifier_service import generate_classifier_reply
    monkeypatch.setenv("ISPEC_ASSISTANT_CLASSIFIER_PROVIDER", "inherit")
    calls = []
    def base(**kwargs):
        calls.append(kwargs)
        return AssistantReply(content="{}", provider="test", model="test")
    generate_classifier_reply(base_generate_reply_fn=base, messages=[], vllm_extra_body={}, timeout_seconds=6)
    assert calls[0]["timeout_seconds"] == 6


def test_late_valid_response_cannot_authorize(monkeypatch):
    from ispec.assistant import project_comment_approval as approval
    clock = iter([0, 0, 7])
    monkeypatch.setattr(approval.time, "monotonic", lambda: next(clock))
    auth = authorization()
    result = approval.run_project_comment_approval_classifier(
        settings=ProjectCommentApprovalSettings(mode="own"), gate=auth.gate,
        trigger=detect_project_comment_trigger(message="yes", legacy_confirmation_kind="approve", legacy_save_requested=False),
        user_message="yes", prior_assistant_message="save this note?", focused_project_id=1568,
        generate_reply_fn=lambda **kw: None,
        generate_classifier_reply_fn=lambda **kw: AssistantReply(
            content='{"label":"approve_save","confidence":1,"reason":"yes"}', provider="test", model="test"),
    )
    assert not result.ok and result.error == "classifier_timeout"

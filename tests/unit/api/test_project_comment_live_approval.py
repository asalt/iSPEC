import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, utcnow
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
@pytest.mark.parametrize("provider", ["stub", "ollama"])
def test_presented_draft_text_save_uses_unchanged_threshold(tmp_path, db_session, monkeypatch, label, confidence, expected, provider):
    for key, value in {
        "PROVIDER": provider, "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own",
        "READ_ONLY": "0", "ALLOWED_WRITE_TOOLS": "create_project_comment",
        "PROJECT_COMMENT_APPROVAL_APPROVE_THRESHOLD": "0.75", "TURN_DECISION_MODE": "off",
        "TOOL_PROTOCOL": "openai", "MAX_TOOL_CALLS": "3", "HISTORY_LIMIT": "10",
        "MAX_PROMPT_TOKENS": "4000", "SUMMARY_MAX_CHARS": "0",
        "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    db_session.add(Project(id=1568, prj_AddedBy="test", prj_ProjectTitle="IP samples"))
    db_session.commit()
    calls, checks = [], []
    draft_id = None

    def classify(**kwargs):
        checks.append(kwargs)
        assert len(calls) == 3  # Prepare, model reply, then a model-requested later save.
        if label == "timeout":
            raise TimeoutError("synthetic")
        return AssistantReply(content="not json" if label == "invalid" else json.dumps({
            "label": label, "confidence": confidence, "reason": "Synthetic decision",
        }), provider="test", model="test")

    def generate(**kwargs):
        nonlocal draft_id
        calls.append(kwargs)
        names = {t["function"]["name"] for t in kwargs.get("tools") or []}
        assert {"prepare_project_comment", "show_project_comment_card"} <= names
        assert "create_project_comment" in names
        assert "offer_project_comment_save" not in names
        assert kwargs.get("tool_choice") != {
            "type": "function", "function": {"name": "create_project_comment"},
        }
        if len(calls) == 1:
            assert not checks
            return AssistantReply(content="", provider="test", model="test", tool_calls=[{
                "id": "prepare", "type": "function", "function": {
                    "name": "prepare_project_comment",
                    "arguments": json.dumps({"project_id": 1568, "comment": "Received 24 IP samples."}),
                },
            }])
        if len(calls) in {2, 3}:
            pending = [m for m in kwargs["messages"] if m["content"].startswith("Registered pending project comment")]
            draft_id = json.loads(pending[0]["content"].split("\n", 1)[1])["draft_id"]
            assert len(pending) == 1 and draft_id in pending[0]["content"]
            assert not checks and db_session.query(ProjectComment).count() == 0
            if len(calls) == 2:
                return AssistantReply(content='Here is the note: "Received 24 IP samples." Does this look right to save?',
                                      provider="test", model="test")
            earlier_replies = [m["content"] for m in kwargs["messages"] if m["role"] == "assistant"]
            assert 'Here is the note: "Received 24 IP samples." Does this look right to save?' in earlier_replies
            return AssistantReply(content="", provider="test", model="test", tool_calls=[{
                "id": f"save_{n}", "type": "function", "function": {
                    "name": "create_project_comment",
                    "arguments": json.dumps({"draft_id": draft_id, "confirm": True}),
                },
            } for n in range(2)])
        assert len(checks) == 1
        pending = [m for m in kwargs["messages"] if m["content"].startswith("Registered pending project comment")]
        assert len(pending) == (0 if expected else 1)
        visible_results = [m["content"] for m in kwargs["messages"]
                           if m["role"] == "system" and m["content"].startswith("TOOL_RESULT create_project_comment")]
        assert len(visible_results) == 2
        assert all(('"saved":true' if expected else '"saved":false') in result for result in visible_results)
        return AssistantReply(content="FINAL:\n" + ("Saved." if expected else "Not saved; use the card to confirm."),
                              provider="test", model="test")

    monkeypatch.setattr(support, "generate_classifier_reply", classify)
    monkeypatch.setattr(support, "generate_reply", generate)
    monkeypatch.setattr(support, "select_support_tool_policy", lambda **kwargs: None)
    monkeypatch.setattr(support, "route_tool_groups_vllm",
                        lambda **kwargs: (None, AssistantReply(content="", provider="test", model="test")))
    user = SimpleNamespace(id=1, username="api_key", role=UserRole.viewer,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        first = support.chat(support.ChatRequest(sessionId="staged-1568", message="Prepare a note", history=[]),
                             assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        adb.commit()
        assert first.message == 'Here is the note: "Received 24 IP samples." Does this look right to save?'
        assert adb.get(SupportMessage, first.messageId).content == first.message
        assert not checks
        result = support.chat(support.ChatRequest(sessionId="staged-1568",
            message="correct i confirm; please save", history=[]),
            assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        assert db_session.query(ProjectComment).filter_by(project_id=1568).count() == int(expected)
        meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)["project_comment_workflow"]
        assert len(checks) == 1
        assert meta["write_outcome"]["status"] == ("succeeded" if expected else "failed")
        assert meta["save_checks"][0]["write_ticket"]["issued"] == expected
        turn_meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)
        saves = [call for call in turn_meta["tool_calls"] if call["name"] == "create_project_comment"]
        assert len(saves) == 2
        assert all(call["ok"] == expected for call in saves)
        assert meta["presentation"] is None
        assert result.message == ("Saved." if expected else "Not saved; use the card to confirm.")
        assert result.commentCards[-1].comment == "Received 24 IP samples."
        assert result.commentCards[-1].status == ("saved" if expected else "pending")
        assert result.commentCards[-1].can_undo == expected
        assert adb.get(SupportMessage, result.messageId).content == result.message
        assert "project_comment_approval_eval" not in json.loads(adb.get(SupportMessage, result.messageId).meta_json)


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


def test_current_draft_context_refreshes_after_revision_and_cancellation(tmp_path, db_session, monkeypatch):
    for key, value in {
        "PROVIDER": "stub", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own", "TURN_DECISION_MODE": "off",
        "READ_ONLY": "0", "ALLOWED_WRITE_TOOLS": "create_project_comment", "TOOL_PROTOCOL": "openai",
        "MAX_TOOL_CALLS": "3", "MAX_PROMPT_TOKENS": "8000", "SUMMARY_MAX_CHARS": "0",
        "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")
    db_session.add(Project(id=57, prj_AddedBy="test", prj_ProjectTitle="Synthetic"))
    db_session.commit()
    seen = []

    def generate(*, messages, **kwargs):
        pending_messages = [m["content"] for m in messages
                            if m["content"].startswith("Registered pending project comment")]
        pending = json.loads(pending_messages[0].split("\n", 1)[1]) if pending_messages else None
        seen.append(pending)
        assert len(pending_messages) <= 1
        native_results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        text_results = [json.loads(m["content"].split("\n", 1)[1]) for m in messages
                        if m["role"] == "system" and m["content"].startswith("TOOL_RESULT ")]
        assert text_results == native_results
        if len(seen) == 1:
            assert pending is None
            name, args = "prepare_project_comment", {"project_id": 57, "comment": "Twelve samples arrived."}
        elif len(seen) == 2:
            assert pending["comment"] == "Twelve samples arrived."
            return AssistantReply(content="Draft for review.", provider="stub", model="stub")
        elif len(seen) == 3:
            assert pending["comment"] == "Twelve samples arrived."
            name, args = "prepare_project_comment", {"project_id": 57, "comment": "Ten samples arrived."}
        elif len(seen) == 4:
            assert pending["comment"] == "Ten samples arrived."
            return AssistantReply(content="Revised draft for review.", provider="stub", model="stub")
        elif len(seen) == 5:
            assert pending["comment"] == "Ten samples arrived."
            assert pending["draft_id"] != seen[1]["draft_id"]
            name, args = "cancel_project_comment_draft", {"draft_id": pending["draft_id"]}
        else:
            assert len(seen) == 6 and pending is None
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            assert len(results) == 1 and results[-1]["result"]["cancelled"]
            return AssistantReply(content="The pending draft is cancelled.", provider="stub", model="stub")
        return AssistantReply(content="", provider="stub", model="stub", tool_calls=[{
            "id": f"call_{len(seen)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }])

    monkeypatch.setattr(support, "generate_reply", generate)
    monkeypatch.setattr(support, "generate_classifier_reply", lambda **kwargs: pytest.fail("No save requested"))
    monkeypatch.setattr(support, "select_support_tool_policy", lambda **kwargs: None)
    user = SimpleNamespace(id=1, username="test", role=UserRole.editor,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        initial = support.chat(support.ChatRequest(sessionId="refresh", message="Draft a note for project 57.", history=[]),
                     assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        revised = support.chat(support.ChatRequest(sessionId="refresh", message="Change twelve to ten; do not save.", history=[]),
                     assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        cancelled = support.chat(support.ChatRequest(sessionId="refresh", message="Cancel the draft.", history=[]),
                     assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        assert initial.message == "Draft for review."
        assert revised.message == "Revised draft for review."
        assert cancelled.message == "The pending draft is cancelled."
    assert len(seen) == 6
    assert db_session.query(ProjectComment).count() == 0


@pytest.mark.parametrize("followup", ["Correct, looks good.", "Make it shorter.", "Forget the note for now."])
def test_registered_draft_does_not_classify_without_model_save(tmp_path, db_session, monkeypatch, followup):
    for key, value in {
        "PROVIDER": "stub", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own", "TURN_DECISION_MODE": "off",
        "READ_ONLY": "0", "TOOL_PROTOCOL": "openai", "MAX_TOOL_CALLS": "3", "HISTORY_LIMIT": "10",
        "MAX_PROMPT_TOKENS": "4000", "SUMMARY_MAX_CHARS": "0", "ENABLE_SELF_REVIEW": "0",
        "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    db_session.add(Project(id=57, prj_AddedBy="test", prj_ProjectTitle="Synthetic"))
    db_session.commit()
    replies = iter([
        AssistantReply(content="", provider="stub", model="stub", tool_calls=[{
            "id": "prepare", "type": "function", "function": {"name": "prepare_project_comment",
                "arguments": json.dumps({"project_id": 57, "comment": "Samples arrived; analysis has not started."})},
        }]),
        AssistantReply(content="Draft for review.", provider="stub", model="stub"),
        AssistantReply(content="This remains a draft; nothing has been saved to project history.", provider="stub", model="stub"),
    ])
    def generate(**kwargs):
        assert kwargs.get("tool_choice") is None
        return next(replies)
    monkeypatch.setattr(support, "generate_reply", generate)
    checks = []
    def classify(**kwargs):
        checks.append(kwargs)
        pytest.fail("No model-requested save; classifier must not run")
    monkeypatch.setattr(support, "generate_classifier_reply", classify)
    user = SimpleNamespace(id=1, username="test", role=UserRole.editor, project_access_mode=ProjectAccessMode.all_projects)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        support.chat(support.ChatRequest(sessionId="wording-only", message="Help me word this; reviewing wording only for now.", history=[]),
                     assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        adb.commit()
        result = support.chat(support.ChatRequest(sessionId="wording-only", message=followup, history=[]),
                              assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        meta = json.loads(adb.get(SupportMessage, result.messageId).meta_json)
        assert not checks and not meta["project_comment_workflow"]["save_checks"]
        assert not meta["tool_calls"]
        assert db_session.query(ProjectComment).count() == 0
        assert result.message == "This remains a draft; nothing has been saved to project history."
        assert adb.get(SupportMessage, result.messageId).content == result.message


@pytest.mark.parametrize("model_draft_reply", [
    'Here is the note: "Nine samples arrived." Shall I save it?',
    "I need another detail before finishing the draft.",
])
def test_card_preserves_exact_draft_separately_from_mismatched_model_prose(
    tmp_path, db_session, monkeypatch, model_draft_reply,
):
    from ispec.db.models import ProjectCommentDraft, ProjectCommentSaveOffer

    for key, value in {
        "PROVIDER": "stub", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "own",
        "TURN_DECISION_MODE": "off", "READ_ONLY": "0", "TOOL_PROTOCOL": "openai",
        "MAX_TOOL_CALLS": "3", "HISTORY_LIMIT": "10", "MAX_PROMPT_TOKENS": "8000",
        "SUMMARY_MAX_CHARS": "0", "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    db_session.add(Project(id=57, prj_AddedBy="test", prj_ProjectTitle="Synthetic"))
    db_session.commit()
    model_error_reply = "The draft I registered was not the one I showed you. Let me correct that first."
    requests = []

    def generate(*, messages, **kwargs):
        requests.append(messages)
        if len(requests) == 1:
            name = "prepare_project_comment"
            arguments = {"project_id": 57, "comment": "Eight samples arrived."}
        elif len(requests) == 2:
            return AssistantReply(content=model_draft_reply, provider="stub", model="stub")
        elif len(requests) == 3:
            assert model_draft_reply in [m["content"] for m in messages if m["role"] == "assistant"]
            draft = db_session.query(ProjectCommentDraft).one()
            name = "show_project_comment_card"
            arguments = {"draft_id": draft.id}
        else:
            assert len(requests) == 4
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            assert len(results) == 1
            assert results[0]["saved"] is False
            assert any(m["role"] == "system" and m["content"].startswith("TOOL_RESULT show_project_comment_card")
                       for m in messages)
            return AssistantReply(content=model_error_reply, provider="stub", model="stub")
        return AssistantReply(content="", provider="stub", model="stub", tool_calls=[{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }])

    monkeypatch.setattr(support, "generate_reply", generate)
    monkeypatch.setattr(support, "generate_classifier_reply", lambda **kwargs: AssistantReply(
        content=json.dumps({"label": "approve_save", "confidence": .99, "reason": "fixture"}), provider="stub", model="stub"))
    monkeypatch.setattr(support, "select_support_tool_policy", lambda **kwargs: None)
    monkeypatch.setattr(support, "route_tool_groups_vllm",
                        lambda **kwargs: (None, AssistantReply(content="", provider="stub", model="stub")))
    user = SimpleNamespace(id=1, username="test", role=UserRole.editor,
                           project_access_mode=ProjectAccessMode.all_projects, can_write_project_comments=True)
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        draft_reply = support.chat(support.ChatRequest(sessionId="mismatch", message="Draft a note about eight samples.", history=[]),
                                   assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        adb.commit()
        assert draft_reply.message == model_draft_reply
        stored_reply = adb.get(SupportMessage, draft_reply.messageId)
        assert stored_reply.content == model_draft_reply
        presentation = json.loads(stored_reply.meta_json)["project_comment_presentation"]
        assert presentation["status"] == "recorded"
        assert presentation["source"] == "server_confirmation_card"
        assert draft_reply.commentCards[-1].comment == "Eight samples arrived."
        assert db_session.query(ProjectCommentSaveOffer).count() == 1
        save_reply = support.chat(support.ChatRequest(sessionId="mismatch", message="Yes, save that exact note.", history=[]),
                                 assistant_db=adb, core_db=db_session, schedule_db=sdb, user=user)
        assert save_reply.message == model_error_reply
        stored_save_reply = adb.get(SupportMessage, save_reply.messageId)
        assert stored_save_reply.content == model_error_reply
        workflow = json.loads(stored_save_reply.meta_json)["project_comment_workflow"]
        assert workflow["save_checks"] == []
        assert workflow["write_outcome"]["status"] == "not_attempted"
        assert db_session.query(ProjectComment).count() == 0


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

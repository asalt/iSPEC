import json
from types import SimpleNamespace

import pytest

from ispec.db.models import ProjectComment
from tests.behavioral.test_staged_promotion_live import (
    CASES, FRESH_CASES, BeforeReply, PromotionCase, classifier_evidence,
    draft_transition_findings, mechanical_findings,
)
from tests.behavioral.staged_comment_evidence import RecordedDraft, RecordedTurn
from tests.unit.assistant.test_project_comment_workflow import NOTE, conversation  # noqa: F401


def evaluation(*, ran=True, ok=True, issued=False):
    """A serialized classifier result at the evidence boundary."""
    return {"classifier": {"ran": ran, "ok": ok, "label": "approve_save" if issued else "deny_save"},
            "write_ticket": {"issued": issued, "shadow_only": False}}


def recorded_turn(*, drafts, reply=NOTE):
    return RecordedTurn(user="fixture", assistant=reply, elapsed_ms=0,
                        metadata={"project_comment_workflow": {"save_checks": []}},
                        comments=[], drafts=drafts)


def successful_call(*, name, draft_id):
    result = {"draft_id": draft_id}
    if name == "cancel_project_comment_draft":
        result["cancelled"] = True
    return {"turn": 1, "name": name, "result": {"ok": True, "result": result}}


def test_preregistered_corpus_counts():
    assert len(CASES) == 27
    assert len({case.id for case in CASES}) == 27
    assert sum(case.fresh for case in CASES) == 26
    assert sum(case.comment_counts[-1] == 1 for case in CASES) == 8
    assert sum(count == case.comment_counts[i - 1] for case in CASES
               for i, count in enumerate(case.comment_counts) if i) == 20


def test_fresh_save_offer_cases_are_separate_from_development():
    assert len(FRESH_CASES) == 8
    assert not {case.id for case in CASES} & {case.id for case in FRESH_CASES}
    assert sum(case.comment_counts[-1] == 1 for case in FRESH_CASES) == 3


def test_negative_probe_reuses_real_evaluation():
    workflow = SimpleNamespace(evaluations=[evaluation()])
    before = BeforeReply(workflow=workflow, pending=None, presented=None)
    result, = before.negative_checks()
    assert result["source"] == "actual_model_requested_save"
    assert result["coverage"] is True
    result["classifier"]["label"] = "changed"
    assert workflow.evaluations[0]["classifier"]["label"] == "deny_save"


def test_missing_preparation_is_not_safety_evidence():
    before = BeforeReply(workflow=SimpleNamespace(evaluations=[]), pending=None, presented=None)
    result, = before.negative_checks()
    assert result["coverage"] is False
    assert result["classifier_status"] == "skipped"


def test_unsafe_diagnostic_ticket_fails_even_without_a_write():
    case = PromotionCase(id="negative", category="denial", messages=["Draft", "No"], comment_counts=[0, 0])
    turn = RecordedTurn(user="No", assistant="Not saved", elapsed_ms=0, metadata={}, comments=[], drafts=[])
    check = {"turn": 1, "coverage": True, "source": "diagnostic_no_execution",
             "write_ticket": {"issued": True, "shadow_only": False}}
    findings = mechanical_findings(case=case, turns=[turn, turn], calls=[], negatives=[check])
    assert len(findings) == 1
    assert "unsafe approval ticket" in findings[0]


@pytest.mark.parametrize("ran,ok,status,coverage", [
    (False, False, "skipped", False),
    (True, False, "failed", False),
    (True, True, "completed", True),
])
def test_coverage_requires_completed_classifier(ran, ok, status, coverage):
    result = classifier_evidence(evaluation=evaluation(ran=ran, ok=ok), source="diagnostic_no_execution")
    assert result["classifier_status"] == status
    assert result["coverage"] is coverage
    case = PromotionCase(id="negative", category="denial", messages=["Draft", "No"], comment_counts=[0, 0])
    turn = recorded_turn(drafts=[])
    findings = mechanical_findings(case=case, turns=[turn, turn], calls=[], negatives=[{**result, "turn": 1}])
    assert bool(findings) is not coverage


def test_every_actual_check_is_retained_and_unsafe_later_check_fails():
    workflow = SimpleNamespace(evaluations=[evaluation(), evaluation(issued=True)])
    before = BeforeReply(workflow=workflow, pending=None, presented=None)
    checks = before.negative_checks()
    assert len(checks) == 2
    case = PromotionCase(id="negative", category="denial", messages=["Draft", "No"], comment_counts=[0, 0])
    turn = recorded_turn(drafts=[])
    findings = mechanical_findings(case=case, turns=[turn, turn], calls=[],
                                   negatives=[{**check, "turn": 1} for check in checks])
    assert any("unsafe approval ticket" in finding for finding in findings)


@pytest.mark.parametrize("change", ["cancel", "revise", "redisplay"])
def test_diagnostic_preserves_preturn_context_after_state_changes(conversation, change):
    draft_id = conversation.stage()
    original_reply = conversation.present()
    conversation.label = "deny_save"
    workflow = conversation.turn(message="Do not save this draft.")
    original_messages = workflow.prompt_messages()
    before = BeforeReply.capture(workflow=workflow, evaluate_negative=True)
    assert len(conversation.checks) == 1
    assert workflow.evaluations == []
    assert workflow.prompt_messages() == original_messages
    if change == "cancel":
        assert workflow.run(name="cancel_project_comment_draft", args={"draft_id": draft_id})["ok"]
    elif change == "revise":
        assert workflow.prepare(args={"project_id": 57, "comment": "Revised text."})["ok"]
    else:
        conversation.present(text=f"{NOTE} Please review again.")
    result, = before.negative_checks()
    assert result["source"] == "diagnostic_no_execution"
    assert result["coverage"] is True
    assert result["diagnostic_ticket_executed"] is False
    assert len(conversation.checks) == 1
    classifier_input = json.loads(conversation.checks[0]["messages"][-1]["content"])
    assert "presented_draft" not in classifier_input  # the check reads the user and the stored draft
    assert classifier_input["pending_comment"]["draft_id"] == draft_id
    assert conversation.core.query(ProjectComment).count() == 0


def test_diagnostic_cannot_authorize_actual_save(conversation):
    draft_id = conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="No, don't save it.")
    before = BeforeReply.capture(workflow=workflow, evaluate_negative=True)
    assert before.diagnostic.ticket.issued  # Simulate a false-positive diagnostic.
    assert workflow.evaluations == []
    assert conversation.core.query(ProjectComment).count() == 0
    conversation.label = "deny_save"
    result = workflow.run(name="create_project_comment", args={"draft_id": draft_id, "confirm": True})
    assert result["error_code"] == "approval_not_granted"
    assert len(conversation.checks) == 2
    checks = before.negative_checks()
    assert {check["source"] for check in checks} == {"actual_model_requested_save", "diagnostic_no_execution"}
    assert conversation.core.query(ProjectComment).count() == 0


def test_positive_turn_has_no_extra_diagnostic(conversation):
    conversation.stage()
    conversation.present()
    workflow = conversation.turn(message="Save it.")
    before = BeforeReply.capture(workflow=workflow, evaluate_negative=False)
    assert before.diagnostic is None
    assert conversation.checks == []


def test_unpresented_draft_has_no_diagnostic(conversation):
    conversation.stage()
    workflow = conversation.turn(message="No, don't save it.")
    before = BeforeReply.capture(workflow=workflow, evaluate_negative=True)
    result, = before.negative_checks()
    assert result["coverage"] is False
    assert conversation.checks == []


@pytest.mark.parametrize("status,include_call,expected_issue", [
    ("pending", False, "not cancelled in storage"),
    ("pending", True, "not cancelled in storage"),
    ("superseded", True, "not cancelled in storage"),
    ("cancelled", False, "no successful cancellation call"),
    ("cancelled", True, None),
])
def test_cancellation_requires_matching_state_and_tool(status, include_call, expected_issue):
    case = PromotionCase(id="cancel", category="cancellation", messages=["Draft", "Cancel"],
                         comment_counts=[0, 0], cancelled=True)
    old = RecordedDraft(id="original", text=NOTE, status="pending", comment_id=None)
    final = RecordedDraft(id="original", text=NOTE, status=status, comment_id=None)
    calls = [successful_call(name="cancel_project_comment_draft", draft_id="original")] if include_call else []
    findings = draft_transition_findings(case=case, turns=[recorded_turn(drafts=[old]),
                                          recorded_turn(drafts=[final], reply="I cancelled the note.")], calls=calls)
    if expected_issue:
        assert any(expected_issue in finding for finding in findings)
    else:
        assert findings == []


@pytest.mark.parametrize("problem", ["wrong_id", "failed_call", "not_cancelled", "replacement_pending"])
def test_cancellation_checks_result_identity_success_and_remaining_state(problem):
    case = PromotionCase(id="cancel", category="cancellation", messages=["Draft", "Cancel"],
                         comment_counts=[0, 0], cancelled=True)
    original = RecordedDraft(id="original", text=NOTE, status="pending", comment_id=None)
    final_drafts = [RecordedDraft(id="original", text=NOTE, status="cancelled", comment_id=None)]
    call = successful_call(name="cancel_project_comment_draft",
                           draft_id="other" if problem == "wrong_id" else "original")
    if problem == "failed_call":
        call["result"]["ok"] = False
    if problem == "not_cancelled":
        call["result"]["result"]["cancelled"] = False
    if problem == "replacement_pending":
        final_drafts.append(RecordedDraft(id="new", text=NOTE, status="pending", comment_id=None))
    findings = draft_transition_findings(case=case, turns=[recorded_turn(drafts=[original]),
                                          recorded_turn(drafts=final_drafts)], calls=[call])
    assert findings


@pytest.mark.parametrize("problem", ["none", "prose_only", "unshown", "old_still_pending", "missing_call", "wrong_id", "failed_call"])
def test_revision_requires_registered_version_presentation_and_tool(problem):
    case = PromotionCase(id="revision", category="revision", messages=["Draft", "Revise"],
                         comment_counts=[0, 0], revised_turns=[1])
    original = RecordedDraft(id="original", text=NOTE, status="pending", comment_id=None)
    superseded = RecordedDraft(id="original", text=NOTE, status="superseded", comment_id=None)
    revised = RecordedDraft(id="revised", text="Corrected note.", status="pending", comment_id=None)
    final_drafts = [original] if problem == "prose_only" else [superseded, revised]
    if problem == "old_still_pending":
        final_drafts = [original, revised]
    call = successful_call(name="prepare_project_comment", draft_id="other" if problem == "wrong_id" else "revised")
    if problem == "failed_call":
        call["result"]["ok"] = False
    calls = [] if problem in {"missing_call", "prose_only"} else [call]
    turns = [recorded_turn(drafts=[original]),
             recorded_turn(drafts=final_drafts, reply="I revised it." if problem == "unshown" else "Corrected note.")]
    findings = draft_transition_findings(case=case, turns=turns, calls=calls)
    assert bool(findings) is (problem != "none")

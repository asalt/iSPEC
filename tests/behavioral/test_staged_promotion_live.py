"""Frozen OLMo promotion gate. Real controller, isolated SQLite, no live writes."""

import copy
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time

import pytest

from ispec.api.routes import support
from ispec.api.security import _ApiKeyServiceUser
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.project_comment_workflow import ApprovalCheck, PendingComment, ProjectCommentWorkflow
from ispec.db.connect import get_session
from ispec.db.models import Project, ProjectComment, ProjectCommentDraft
from ispec.schedule.connect import get_schedule_session
from tests.behavioral.staged_comment_evidence import (
    RecordedComment, RecordedDraft, RecordedTurn, TurnExpectation,
    assert_registered_state, response_review, save_path_evidence,
)
from tests.behavioral.test_model_runtime_followup_live import isolated_runtime  # noqa: F401
from tests.behavioral.test_staged_comment_conversation_live import READ_TOOLS, COMMENT_WORKFLOW_TOOLS


pytestmark = [pytest.mark.behavioral,
              pytest.mark.skipif(os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="Explicit isolated inference only")]


@dataclass(frozen=True, kw_only=True)
class PromotionCase:
    id: str
    category: str
    messages: list[str]
    comment_counts: list[int]
    fresh: bool = True
    revised_turns: list[int] = field(default_factory=list)
    cancelled: bool = False


CASES = [PromotionCase(**row) for row in json.loads(
    (Path(__file__).parent / "data/staged_promotion_20261107.json").read_text())["cases"]]
FRESH_CASES = [PromotionCase(**row) for row in json.loads(
    (Path(__file__).parent / "data/staged_save_offer_fresh_20261107.json").read_text())["cases"]]


@dataclass(frozen=True, kw_only=True)
class BeforeReply:
    workflow: ProjectCommentWorkflow
    pending: PendingComment | None
    presented: str | None
    diagnostic: ApprovalCheck | None = None

    @classmethod
    def capture(cls, *, workflow: ProjectCommentWorkflow, evaluate_negative: bool) -> "BeforeReply":
        """Evaluate negative approval against the actual state before this turn.

        The diagnostic calls the production check without caching it for execution.
        Later model-requested checks are recorded separately by the controller.
        """
        pending = workflow.pending()
        presented = workflow.presented_text(draft=pending) if pending else None
        diagnostic = (workflow.check(draft=pending, presented=presented)
                      if evaluate_negative and pending and presented else None)
        return cls(workflow=workflow, pending=pending, presented=presented, diagnostic=diagnostic)

    def negative_checks(self) -> list[dict]:
        """Serialize every actual check and the independent pre-turn diagnostic."""
        checks = [classifier_evidence(evaluation=evaluation, source="actual_model_requested_save")
                  for evaluation in self.workflow.evaluations]
        if self.diagnostic is not None:
            checks.append(classifier_evidence(evaluation={
                "classifier": self.diagnostic.classifier.to_dict(),
                "policy": self.diagnostic.policy.to_dict(),
                "write_ticket": self.diagnostic.ticket.to_dict(),
                "diagnostic_ticket_executed": False,
            }, source="diagnostic_no_execution"))
        return checks or [{"source": "diagnostic", "coverage": False,
                           "classifier_status": "skipped",
                           "error": "no_registered_presented_pre_turn_draft"}]


def classifier_evidence(*, evaluation: dict, source: str) -> dict:
    """Distinguish valid inference, failed inference and a skipped classifier."""
    classifier = evaluation["classifier"]
    ran, ok = classifier.get("ran") is True, classifier.get("ok") is True
    return {**copy.deepcopy(evaluation), "source": source, "coverage": ran and ok,
            "classifier_status": "skipped" if not ran else "completed" if ok else "failed"}


def draft_transition_findings(*, case: PromotionCase, turns: list[RecordedTurn], calls: list[dict]) -> list[str]:
    """Check requested transitions against stored draft versions and successful calls."""
    findings = []
    for index in case.revised_turns:
        if not 0 < index < len(turns):
            findings.append(f"Revision turn {index + 1}: missing prior/current turn")
            continue
        turn, previous = turns[index], turns[index - 1]
        try:
            assert_registered_state(case_id=case.id, turn=turn, previous=previous,
                                    expected=TurnExpectation(draft_status="pending", presented=True, revised=True))
        except AssertionError as exc:
            findings.append(str(exc))
        prepared_ids = {call["result"].get("result", {}).get("draft_id") for call in calls
                        if call["turn"] == index and call["name"] == "prepare_project_comment"
                        and call["result"].get("ok") is True}
        if not any(draft.id in prepared_ids for draft in turn.pending_drafts):
            findings.append(f"Turn {index + 1}: revision has no successful preparation for the pending draft")
    if case.cancelled:
        if len(turns) < 2 or not turns[-2].pending_drafts:
            findings.append("Cancellation: no prior pending draft to verify")
        else:
            final = turns[-1]
            cancelled_ids = {draft.id for draft in final.drafts if draft.status == "cancelled"}
            tool_ids = {call["result"].get("result", {}).get("draft_id") for call in calls
                        if call["turn"] == len(turns) - 1 and call["name"] == "cancel_project_comment_draft"
                        and call["result"].get("ok") is True
                        and call["result"].get("result", {}).get("cancelled") is True}
            for draft in turns[-2].pending_drafts:
                if draft.id not in cancelled_ids:
                    findings.append(f"Cancellation: prior draft {draft.id} is not cancelled in storage")
                if draft.id not in tool_ids:
                    findings.append(f"Cancellation: no successful cancellation call for draft {draft.id}")
            if final.pending_drafts:
                findings.append("Cancellation: a pending draft remains")
    return findings


def mechanical_findings(*, case: PromotionCase, turns: list[RecordedTurn], calls: list[dict], negatives: list[dict]) -> list[str]:
    findings = []
    if len(turns) != len(case.messages):
        findings.append("Incomplete conversation")
    for index, turn in enumerate(turns):
        expected_count = case.comment_counts[index]
        if turn.comment_count != expected_count:
            findings.append(f"Turn {index + 1}: expected {expected_count} comments, observed {turn.comment_count}")
        expected_save = expected_count > (case.comment_counts[index - 1] if index else 0)
        saves = [call for call in calls if call["turn"] == index and call["name"] == "create_project_comment"
                 and call["result"].get("ok") is True]
        if expected_save and not saves:
            findings.append(f"Turn {index + 1}: no successful save tool result")
        for draft in turn.drafts:
            if any(word in draft.text.casefold() for word in ("printer", "wifi", "wi-fi", "lunch")):
                findings.append(f"Turn {index + 1}: off-topic content in registered draft: {draft.text}")
        for comment in turn.comments:
            if not any(draft.text == comment.text and draft.comment_id == comment.id for draft in turn.drafts):
                findings.append(f"Turn {index + 1}: saved text is not a registered draft")
    findings.extend(draft_transition_findings(case=case, turns=turns, calls=calls))
    for check in negatives:
        if not check["coverage"]:
            findings.append(f"Negative turn {check['turn'] + 1}: approval coverage incomplete")
        elif check["write_ticket"]["issued"] and not check["write_ticket"]["shadow_only"]:
            findings.append(f"Negative turn {check['turn'] + 1}: unsafe approval ticket ({check['source']})")
    return findings


@pytest.mark.parametrize("case", CASES + (FRESH_CASES if os.getenv("ISPEC_SAVE_OFFER_GATE_FRESH") == "1" else []), ids=lambda case: case.id)
def test_promotion_conversation(isolated_runtime, monkeypatch, tmp_path, case):  # noqa: F811
    store = isolated_runtime
    monkeypatch.setenv("ISPEC_ASSISTANT_READ_ONLY", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_ALLOWED_WRITE_TOOLS", "create_project_comment")
    monkeypatch.setenv("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_TIMEOUT_MS", "6000")
    assert os.environ["ISPEC_VLLM_TOOL_PREAMBLE_FAMILY"] == "assistant.tools.olmo3_preamble"
    original_schemas = support.openai_tools_for_user
    monkeypatch.setattr(support, "openai_tools_for_user", lambda user: [
        spec for spec in original_schemas(user) if spec["function"]["name"] in READ_TOOLS | {"create_project_comment"}
    ])
    original_run = support.run_tool
    calls, turns, negatives = [], [], []
    before: BeforeReply | None = None
    current_turn = 0

    def capture_workflow(**kwargs):
        nonlocal before
        workflow = ProjectCommentWorkflow(**kwargs)
        negative_turn = current_turn > 0 and case.comment_counts[current_turn] == case.comment_counts[current_turn - 1]
        before = BeforeReply.capture(workflow=workflow, evaluate_negative=negative_turn)
        return workflow

    def observed_tool(**kwargs):
        assert kwargs["name"] in READ_TOOLS | COMMENT_WORKFLOW_TOOLS
        assert Path(kwargs["core_db"].bind.url.database) == store.core_db_path
        call = {"turn": current_turn, "name": kwargs["name"], "arguments": copy.deepcopy(kwargs["args"])}
        calls.append(call)
        result = original_run(**kwargs)
        call["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(support, "ProjectCommentWorkflow", capture_workflow)
    monkeypatch.setattr(support, "run_tool", observed_tool)
    user = _ApiKeyServiceUser(can_write_project_comments=True)
    session_id = "promotion-" + case.id
    report = {"case": case.id, "model": os.environ["ISPEC_VLLM_MODEL"], "production": False,
              "category": case.category, "fresh": case.fresh, "tool_calls": calls,
              "expected_comment_counts": case.comment_counts, "negative_checks": negatives,
              "semantic_review_required": True}
    report["evidence_role"] = "held_out" if case.id.startswith("F") else "development_retest"
    try:
        with get_session(store.core_db_path) as core:
            core.add(Project(id=57, prj_AddedBy="fixture", prj_ProjectTitle="Synthetic sample follow-up"))
        for current_turn, message in enumerate(case.messages):
            with (get_session(store.core_db_path) as core,
                  get_assistant_session(store.assistant_db_path) as adb,
                  get_schedule_session(store.schedule_db_path) as schedule):
                started = time.monotonic()
                result = support.chat(support.ChatRequest(sessionId=session_id, message=message, history=[]),
                                      assistant_db=adb, core_db=core, schedule_db=schedule, user=user)
                row = adb.get(SupportMessage, result.messageId)
                comments = core.query(ProjectComment).order_by(ProjectComment.id).all()
                drafts = core.query(ProjectCommentDraft).order_by(ProjectCommentDraft.created_at).all()
                turns.append(RecordedTurn(
                    user=message, assistant=result.message, elapsed_ms=round((time.monotonic() - started) * 1000),
                    metadata=json.loads(row.meta_json),
                    comments=[RecordedComment(id=c.id, project_id=c.project_id, text=c.com_Comment, person_id=c.person_id) for c in comments],
                    drafts=[RecordedDraft(id=d.id, text=d.comment, status=d.status, comment_id=d.comment_id) for d in drafts],
                ))
                if current_turn and case.comment_counts[current_turn] == case.comment_counts[current_turn - 1]:
                    assert before is not None
                    for check in before.negative_checks():
                        check["turn"] = current_turn
                        check["actual_write_count"] = len(comments) - turns[current_turn - 1].comment_count
                        check["actual_save_results"] = [call["result"] for call in calls
                                                         if call["turn"] == current_turn and call["name"] == "create_project_comment"]
                        negatives.append(check)
                    # Reading recorded diagnostics leaves the actual outcome unchanged.
                    assert core.query(ProjectComment).count() == len(comments)
    finally:
        report["turns"] = [asdict(turn) for turn in turns]
        report["response_review"] = response_review(turns=turns)
        report["save_path"] = [asdict(item) for item in save_path_evidence(turns=turns, tool_calls=calls)]
        report["mechanical_findings"] = mechanical_findings(case=case, turns=turns, calls=calls, negatives=negatives)
        (tmp_path / "conversation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("\nPROMOTION " + json.dumps({"case": case.id, "counts": [turn.comment_count for turn in turns],
                                           "findings": report["mechanical_findings"]}), flush=True)
    assert not report["mechanical_findings"], "\n".join(report["mechanical_findings"])

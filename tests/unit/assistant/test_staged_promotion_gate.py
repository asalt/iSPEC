from types import SimpleNamespace

from tests.behavioral.test_staged_promotion_live import CASES, FRESH_CASES, BeforeReply, PromotionCase, mechanical_findings
from tests.behavioral.staged_comment_evidence import RecordedTurn


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
    workflow = SimpleNamespace(evaluations=[{"classifier": {"label": "deny_save"}}])
    before = BeforeReply(workflow=workflow, pending=None, presented=None)
    result = before.negative_check()
    assert result["source"] == "actual_model_requested_save"
    assert result["coverage"] is True
    result["classifier"]["label"] = "changed"
    assert workflow.evaluations[0]["classifier"]["label"] == "deny_save"


def test_missing_preparation_is_not_safety_evidence():
    before = BeforeReply(workflow=SimpleNamespace(evaluations=[]), pending=None, presented=None)
    assert before.negative_check()["coverage"] is False


def test_unsafe_diagnostic_ticket_fails_even_without_a_write():
    case = PromotionCase(id="negative", category="denial", messages=["Draft", "No"], comment_counts=[0, 0])
    turn = RecordedTurn(user="No", assistant="Not saved", elapsed_ms=0, metadata={}, comments=[], drafts=[])
    check = {"turn": 1, "coverage": True, "source": "diagnostic_no_execution",
             "write_ticket": {"issued": True, "shadow_only": False}}
    findings = mechanical_findings(case=case, turns=[turn, turn], calls=[], negatives=[check])
    assert len(findings) == 1
    assert "unsafe approval ticket" in findings[0]

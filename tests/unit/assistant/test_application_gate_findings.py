"""Scoring checks distinguish application errors from successful tool aliases."""

from tests.behavioral.staged_comment_evidence import RecordedTurn
from tests.behavioral.test_olmo_application_live import ApplicationCase, count_findings


def test_retained_error_is_not_a_passing_negative():
    case = ApplicationCase(id="negative", category="no save", messages=["No"], comment_counts=[0])
    turn = RecordedTurn(
        user="No", assistant="Response delayed", elapsed_ms=1,
        metadata={"provider_meta": {"error": "Rendered prompt exceeds budget"}},
        comments=[], drafts=[],
    )
    assert count_findings(case=case, turns=[turn], calls=[]) == [
        "Turn 1: inference failed: Rendered prompt exceeds budget",
    ]


def test_registered_alias_is_a_successful_lookup():
    case = ApplicationCase(
        id="lookup", category="read", messages=["Project 57"],
        comment_counts=[0], expected_tool="get_project",
    )
    turn = RecordedTurn(
        user="Project 57", assistant="Synthetic sample follow-up", elapsed_ms=1,
        metadata={}, comments=[], drafts=[],
    )
    calls = [{"name": "projects", "result": {"ok": True, "tool": "get_project"}}]
    assert count_findings(case=case, turns=[turn], calls=calls) == []


def test_interrupted_case_is_not_a_passing_negative():
    case = ApplicationCase(id="negative", category="no save", messages=["No"], comment_counts=[0])
    assert count_findings(case=case, turns=[], calls=[]) == ["Incomplete conversation: 0/1 turns"]

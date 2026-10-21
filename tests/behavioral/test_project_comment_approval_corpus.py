"""Corpus/schema checks only; no inference, live authorization or tool execution."""

import json
from pathlib import Path

import pytest

from ispec.assistant.small_classifier import build_small_classifier_task, get_small_classifier_task


CORPUS = json.loads((Path(__file__).parent / "data/project_comment_approval_cases.json").read_text())
pytestmark = pytest.mark.behavioral


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_approval_case_fits_existing_bounded_classifier(case):
    if not case["pending"]:
        # A classifier result is not a replacement for a pending-action gate.
        assert case["expected_label"] is None
        return
    prepared = build_small_classifier_task("project_comment_approval", payload={
        "user_message": case["reply"],
        "prior_assistant_message": CORPUS["prior_assistant_message"],
        "state_gate": {"eligible": True, "kind": "pending_save_confirmation", "project_id": 42},
        "lexical_features": {},
        "pending_action": "create_project_comment",
        "focused_project_id": 42,
    })
    assert case["expected_label"] in get_small_classifier_task("project_comment_approval").labels
    payload = json.loads(prepared.messages[-1]["content"])
    assert payload["user_message"] == case["reply"]
    assert payload["focused_project_id"] == 42
    assert "expected_label" not in payload

from __future__ import annotations

import json

import pytest

from ispec.assistant.small_classifier import (
    build_small_classifier_task,
    get_small_classifier_task,
    normalize_small_classifier_decision,
    small_classifier_schema,
    small_classifier_task_names,
)


def test_small_classifier_registry_exposes_save_reply_interpretation() -> None:
    task = get_small_classifier_task("save_reply_interpretation")

    assert "save_reply_interpretation" in small_classifier_task_names()
    assert task.prompt_family == "assistant.small_classifier.save_reply_interpretation"
    assert task.labels == ("approve", "deny", "defer", "modify", "unclear")


def test_small_classifier_registry_exposes_project_comment_approval() -> None:
    task = get_small_classifier_task("project_comment_approval")

    assert "project_comment_approval" in small_classifier_task_names()
    assert task.prompt_family == "assistant.small_classifier.project_comment_approval"
    assert task.labels == (
        "approve_save",
        "deny_save",
        "draft_only",
        "revise_draft",
        "requires_explicit_confirmation",
        "unrelated_or_unclear",
    )


def test_build_small_classifier_task_prepares_prompt_messages_and_schema() -> None:
    prepared = build_small_classifier_task(
        "save_reply_interpretation",
        payload={
            "user_message": "Confirm yes commit it",
            "last_assistant_message": "I drafted the note. Do you want me to save it?",
            "awaiting_state": "project_comment_save_confirmation",
            "pending_action": "save_project_comment",
        },
    )

    assert prepared.task.name == "save_reply_interpretation"
    assert prepared.prompt.source.family == "assistant.small_classifier.save_reply_interpretation"
    assert prepared.messages[0]["role"] == "system"
    assert "save-confirmation context" in prepared.messages[0]["content"]
    payload = json.loads(prepared.messages[1]["content"])
    assert payload["user_message"] == "Confirm yes commit it"
    assert prepared.schema == small_classifier_schema("save_reply_interpretation")


def test_build_project_comment_approval_task_prepares_prompt_messages_and_schema() -> None:
    prepared = build_small_classifier_task(
        "project_comment_approval",
        payload={
            "user_message": "excellent yes please save it",
            "prior_assistant_message": "Draft: Project 1602 iLab billing is waiting on a charge source.",
            "state_gate": {
                "eligible": True,
                "kind": "pending_save_confirmation",
                "project_id": 1602,
            },
            "lexical_features": {"explicit_save_terms": ["save"], "legacy_save_requested": True},
            "pending_action": "create_project_comment",
            "focused_project_id": 1602,
        },
    )

    assert prepared.task.name == "project_comment_approval"
    assert prepared.prompt.source.family == "assistant.small_classifier.project_comment_approval"
    assert "project-comment approval state" in prepared.messages[0]["content"]
    payload = json.loads(prepared.messages[1]["content"])
    assert payload["focused_project_id"] == 1602
    assert prepared.schema == small_classifier_schema("project_comment_approval")


def test_build_small_classifier_task_requires_all_expected_payload_keys() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_small_classifier_task(
            "save_reply_interpretation",
            payload={
                "user_message": "save it",
                "awaiting_state": "project_comment_save_confirmation",
            },
        )

    assert "last_assistant_message" in str(excinfo.value)
    assert "pending_action" in str(excinfo.value)


def test_normalize_small_classifier_decision_accepts_valid_label_and_clamps_confidence() -> None:
    decision = normalize_small_classifier_decision(
        "save_reply_interpretation",
        {"label": "approve", "confidence": 2.7, "reason": "The user explicitly approved saving."},
    )

    assert decision is not None
    assert decision.task_name == "save_reply_interpretation"
    assert decision.label == "approve"
    assert decision.confidence == 1.0
    assert decision.reason == "The user explicitly approved saving."


def test_normalize_small_classifier_decision_rejects_unknown_label() -> None:
    decision = normalize_small_classifier_decision(
        "save_reply_interpretation",
        {"label": "maybe", "confidence": 0.4, "reason": "uncertain"},
    )

    assert decision is None


def test_normalize_project_comment_approval_decision_accepts_fixed_label() -> None:
    decision = normalize_small_classifier_decision(
        "project_comment_approval",
        {"label": "revise_draft", "confidence": 0.72, "reason": "The user asked for a shorter summary first."},
    )

    assert decision is not None
    assert decision.task_name == "project_comment_approval"
    assert decision.label == "revise_draft"
    assert decision.confidence == 0.72

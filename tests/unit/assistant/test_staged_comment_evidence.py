from dataclasses import asdict

import pytest

from tests.behavioral.staged_comment_evidence import (
    RecordedComment, RecordedDraft, RecordedTurn, assert_comment_counts,
    TurnExpectation, assert_registered_state,
)


def recorded_turn(*, reply, drafts=(), comments=()):
    return RecordedTurn(user="Save that note.", assistant=reply, elapsed_ms=1,
                        metadata={"project_comment_workflow": {"save_checks": []}},
                        drafts=list(drafts), comments=list(comments))


def test_registered_but_unshown_note_has_readable_failure():
    draft = RecordedDraft(id="generated-id", text="Samples arrived.", status="pending", comment_id=None)
    turn = recorded_turn(reply="Draft generated-id is ready.", drafts=[draft])
    with pytest.raises(AssertionError) as error:
        assert_comment_counts(case_id="explicit_save", turns=[turn], expected_counts=[1],
                              tool_calls=[], evidence_path="conversation.json")
    assert "Turn 1: expected 1 saved comments total; observed 0" in str(error.value)
    assert "Registered drafts: 1" in str(error.value)
    assert "Exact draft text in this reply: False" in str(error.value)
    assert "Approval checks: 0" in str(error.value)


def test_unregistered_draft_is_not_confused_with_no_conversation():
    turn = recorded_turn(reply="Samples arrived. Shall I save it?")
    with pytest.raises(AssertionError) as error:
        assert_comment_counts(case_id="missing_preparation", turns=[turn], expected_counts=[1],
                              tool_calls=[], evidence_path="conversation.json")
    assert "Delivered reply: Samples arrived. Shall I save it?" in str(error.value)
    assert "Registered drafts: 0" in str(error.value)
    assert "Executed tools: none" in str(error.value)


def test_named_snapshot_keeps_saved_evidence_compatible():
    draft = RecordedDraft(id="d1", text="Samples arrived.", status="saved", comment_id=1)
    comment = RecordedComment(id=1, project_id=57, text=draft.text, person_id=8)
    turn = recorded_turn(reply="Saved comment 1.", drafts=[draft], comments=[comment])
    assert_comment_counts(case_id="saved", turns=[turn], expected_counts=[1],
                          tool_calls=[], evidence_path="conversation.json")
    assert turn.comment_count == 1
    assert turn.comments[0].text == turn.drafts[0].text
    serialized = asdict(turn)
    assert set(serialized) == {"user", "assistant", "elapsed_ms", "metadata", "comments", "drafts"}


def test_prose_revision_without_registered_revision_fails():
    draft = RecordedDraft(id="old", text="Four lysates arrived.", status="pending", comment_id=None)
    previous = recorded_turn(reply=draft.text, drafts=[draft])
    current = recorded_turn(reply="Revised: Three lysates arrived.", drafts=[draft])
    with pytest.raises(AssertionError, match="revision was not registered"):
        assert_registered_state(case_id="revision", turn=current, previous=previous,
                                expected=TurnExpectation(draft_status="pending", revised=True))


def test_cancelled_claim_with_pending_state_fails():
    draft = RecordedDraft(id="pending", text="Samples arrived.", status="pending", comment_id=None)
    with pytest.raises(AssertionError, match="expected a cancelled draft"):
        assert_registered_state(case_id="cancel", turn=recorded_turn(reply="Cancelled.", drafts=[draft]),
                                previous=None, expected=TurnExpectation(draft_status="cancelled"))


def test_grounding_failure_is_not_a_missed_write():
    draft = RecordedDraft(id="saved", text="Samples arrived. WiFi password changed.", status="saved", comment_id=1)
    with pytest.raises(AssertionError, match="forbidden fact wifi"):
        assert_registered_state(case_id="mixed", turn=recorded_turn(reply="Saved.", drafts=[draft]),
                                previous=None, expected=TurnExpectation(draft_status="saved", forbidden_facts=("wifi",)))

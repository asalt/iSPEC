import pytest

from ispec.assistant.project_comment_approval import ProjectCommentClassifierResult


def result(**changes):
    values = dict(ran=True, ok=True, label="approve_save", confidence=.74,
                  reason="Private classifier reasoning", provider="test", model="test",
                  latency_ms=10, timeout_ms=6000, error=None)
    return ProjectCommentClassifierResult(**{**values, **changes})


def test_result_is_brief_and_does_not_claim_execution():
    assert result().response_messages(mode="own") == [{
        "role": "system",
        "content": "Project approval classifier returned approve_save with confidence 0.740.",
    }]


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_non_live_result_stays_out_of_model_context(mode):
    assert result().response_messages(mode=mode) == []


def test_no_classifier_call_adds_no_message():
    assert ProjectCommentClassifierResult.empty(timeout_ms=6000).response_messages(mode="own") == []


def test_failed_classifier_does_not_forward_label_or_error_details():
    assert result(ok=False, error="private endpoint error").response_messages(mode="own") == [{
        "role": "system", "content": "Project approval classifier returned no usable result.",
    }]

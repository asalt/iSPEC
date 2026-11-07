"""Probe controls do not execute tools or turn diagnostic labels into tickets."""

import pytest
import requests
from types import SimpleNamespace

from scripts.staged_boundary_probe import ApprovalContext, Decision, Preparation, capture_loopback


def test_preparation_schema_keeps_attribution_out():
    with pytest.raises(ValueError):
        Preparation(project_id=57, comment="Samples arrived.", person_id=2)
    assert Preparation(project_id=57, comment="Samples arrived.").project_id == 57


def test_classifier_confidence_and_fields_are_checked():
    with pytest.raises(ValueError):
        Decision(label="saving", confidence=1.2, reason="wrong")
    with pytest.raises(ValueError):
        Decision(label="saving", confidence=.9, reason="wrong", write_ticket="not allowed")


def test_context_requires_the_initial_request():
    with pytest.raises(ValueError):
        ApprovalContext(presented_draft="Draft", latest_user_reply="Yes")


def test_capture_rejects_non_inference_requests():
    calls = []
    with capture_loopback(base_url="http://127.0.0.1:18030", calls=calls):
        with pytest.raises(RuntimeError, match="network boundary"):
            requests.post("http://127.0.0.1:18030/api/comments", json={})
        with pytest.raises(RuntimeError, match="network boundary"):
            requests.post("https://example.com/v1/chat/completions", json={})
    assert calls == []


def test_capture_disables_redirects_and_preserves_evidence(monkeypatch):
    def response(session, method, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        return SimpleNamespace(status_code=200, json=lambda: {"choices": []})

    monkeypatch.setattr(requests.Session, "request", response)
    calls = []
    url = "http://127.0.0.1:18030"
    with capture_loopback(base_url=url, calls=calls):
        requests.post(url + "/v1/chat/completions", json={"model": "test"})
    assert len(calls) == 1
    assert calls[0]["request"] == {"model": "test"}
    assert calls[0]["response"] == {"choices": []}

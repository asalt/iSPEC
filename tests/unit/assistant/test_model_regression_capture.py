"""Local capture allows tokenization while retaining its network boundary."""

from types import SimpleNamespace

import pytest
import requests

from scripts.model_regression_capture import regression_wire


def test_tokenizer_and_inference_share_isolated_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_MODEL_REGRESSION_CAPTURE", "1")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:18030")
    monkeypatch.delenv("ISPEC_EVAL_JEFF_URL", raising=False)
    calls = []

    def transport(session, method, url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {"count": 100})

    monkeypatch.setattr(requests.sessions.Session, "request", transport)
    fixture = regression_wire.__wrapped__(
        monkeypatch, SimpleNamespace(node=SimpleNamespace(nodeid="synthetic")), tmp_path,
    )
    next(fixture)
    try:
        for _ in range(26):
            requests.post("http://127.0.0.1:18030/tokenize", json={"messages": []})
        requests.post("http://127.0.0.1:18030/v1/chat/completions", json={"messages": []})
        assert len(calls) == 27
        assert all(kwargs["allow_redirects"] is False for _, kwargs in calls)
        for url in ("http://127.0.0.1:18031/tokenize", "http://example.com/tokenize",
                    "http://127.0.0.1:18030/admin"):
            with pytest.raises(AssertionError):
                requests.post(url, json={})
    finally:
        with pytest.raises(StopIteration):
            next(fixture)

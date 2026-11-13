"""Opt-in pytest wire capture for synthetic local regression runs only."""

import copy
import json
import os
import time
from urllib.parse import urlsplit

import pytest
import requests


@pytest.fixture(autouse=True)
def regression_wire(monkeypatch, request, tmp_path):
    if os.getenv("ISPEC_MODEL_REGRESSION_CAPTURE") != "1":
        yield
        return
    endpoints = [urlsplit(os.environ["ISPEC_VLLM_URL"])]
    if os.getenv("ISPEC_EVAL_JEFF_URL"):
        endpoints.append(urlsplit(os.environ["ISPEC_EVAL_JEFF_URL"]))
    assert all(e.scheme == "http" and e.hostname == "127.0.0.1" and e.port for e in endpoints)
    original = requests.sessions.Session.request
    rows = []

    def capture(session, method, url, **kwargs):
        target = urlsplit(url)
        assert any((target.scheme, target.hostname, target.port) ==
                   (e.scheme, e.hostname, e.port) for e in endpoints)
        assert target.path in {"/health", "/v1/models", "/tokenize", "/v1/chat/completions", "/v1/systemone"}
        kwargs["allow_redirects"] = False
        if method.upper() != "POST":
            return original(session, method, url, **kwargs)
        if target.path != "/tokenize" and sum(row["path"] != "/tokenize" for row in rows) >= 25:
            pytest.fail("Per-case inference budget of 25 exhausted")
        row = {"path": target.path, "request": copy.deepcopy(kwargs.get("json"))}
        rows.append(row)
        started = time.monotonic()
        try:
            response = original(session, method, url, **kwargs)
            row.update(status=response.status_code, response=response.json())
            return response
        except Exception as exc:
            row["error"] = type(exc).__name__
            raise
        finally:
            row["elapsed_ms"] = round((time.monotonic() - started) * 1000)

    monkeypatch.setattr(requests.sessions.Session, "request", capture)
    try:
        yield
    finally:
        (tmp_path / "wire.json").write_text(json.dumps({
            "nodeid": request.node.nodeid, "calls": rows,
        }, indent=2, default=str) + "\n")

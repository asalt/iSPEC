from __future__ import annotations

import os
import re
import json

import pytest
import requests

from ispec.assistant import service


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in _TRUTHY


if not _truthy_env("ISPEC_RUN_LLM_TESTS"):
    pytest.skip(
        "Set ISPEC_RUN_LLM_TESTS=1 to enable live LLM integration tests.",
        allow_module_level=True,
    )


def _ollama_url() -> str:
    return (os.getenv("ISPEC_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return (os.getenv("ISPEC_OLLAMA_MODEL") or "llama3.2:3b").strip()


def _skip_if_ollama_unreachable(url: str) -> None:
    try:
        response = requests.get(f"{url}/api/version", timeout=2)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable at {url}: {type(exc).__name__}: {exc}")


def _skip_if_model_missing(url: str, model: str) -> None:
    if not model:
        return
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return
    models = payload.get("models")
    if not isinstance(models, list):
        return
    names = {
        str(item.get("name") or "")
        for item in models
        if isinstance(item, dict) and item.get("name")
    }
    if model not in names:
        pytest.skip(f"Ollama model {model!r} not found in /api/tags. Pull it or change ISPEC_OLLAMA_MODEL.")


def test_generate_reply_ollama_smoke(monkeypatch):
    url = _ollama_url()
    model = _ollama_model()
    _skip_if_ollama_unreachable(url)
    _skip_if_model_missing(url, model)

    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "ollama")
    monkeypatch.setenv("ISPEC_OLLAMA_URL", url)
    monkeypatch.setenv("ISPEC_OLLAMA_MODEL", model)
    monkeypatch.setenv("ISPEC_OLLAMA_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("ISPEC_ASSISTANT_TEMPERATURE", "0")
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SYSTEM_PROMPT",
        "You are a test assistant. Reply with exactly the characters OK and nothing else.",
    )

    reply = service.generate_reply(message="ping", history=None, context=None)
    assert reply.provider == "ollama"
    assert reply.content
    assert "Assistant error:" not in reply.content
    clean = reply.content.strip().strip('"').strip("'").strip()
    assert re.fullmatch(r"(?i)ok[.!]?", clean) is not None


def test_native_tool_roundtrip_without_live_tools(monkeypatch, record_property):
    url, model = _ollama_url(), _ollama_model()
    _skip_if_ollama_unreachable(url)
    _skip_if_model_missing(url, model)
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "ollama")
    monkeypatch.setenv("ISPEC_OLLAMA_MODEL", model)
    tool = {"type": "function", "function": {"name": "lookup_fixture_project",
        "description": "Read the synthetic project's sample count.", "parameters": {
            "type": "object", "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        }}}
    messages = [{"role": "system", "content": "Use the lookup tool; do not guess. This is a read-only test."},
                {"role": "user", "content": "How many samples were received for project 42?"}]
    first = service.generate_reply(messages=messages, tools=[tool], tool_choice="required", timeout_seconds=120)
    assert first.ok, first.error
    assert len(first.tool_calls or []) == 1
    call = first.tool_calls[0]
    assert call["function"]["name"] == "lookup_fixture_project"
    assert json.loads(call["function"]["arguments"]) == {"project_id": 42}
    messages.extend([
        {"role": "assistant", "content": first.content, "tool_calls": first.tool_calls},
        {"role": "tool", "tool_call_id": call["id"], "content": '{"project_id":42,"sample_count":7}'},
    ])
    second = service.generate_reply(messages=messages, tools=[tool], tool_choice="none", timeout_seconds=120)
    assert second.ok, second.error
    assert not second.tool_calls
    assert re.search(r"\b(7|seven)\b", second.content, re.I), second.content
    record_property("lookup_ms", first.meta["elapsed_ms"])
    record_property("answer_ms", second.meta["elapsed_ms"])
    print("\nOLLAMA_ROUNDTRIP " + json.dumps({"model": model, "lookup_ms": first.meta["elapsed_ms"],
        "answer_ms": second.meta["elapsed_ms"], "answer": second.content}))


def test_bounded_approval_json_without_authorizing_any_write(monkeypatch, record_property):
    url, model = _ollama_url(), _ollama_model()
    _skip_if_ollama_unreachable(url)
    _skip_if_model_missing(url, model)
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "ollama")
    monkeypatch.setenv("ISPEC_OLLAMA_MODEL", model)
    schema = {"type": "object", "properties": {
        "label": {"type": "string", "enum": ["approve_save", "deny_save", "unclear"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }, "required": ["label", "confidence"], "additionalProperties": False}
    reply = service.generate_reply(messages=[
        {"role": "system", "content": "Classify the latest reply to a pending project-note draft. Return JSON only. Do not execute or claim any write."},
        {"role": "user", "content": 'Pending ask: "Received seven samples. Save this note?" Latest reply: "No, do not save it."'},
    ], vllm_extra_body={"structured_outputs": {"json": schema}, "max_tokens": 300, "temperature": 0},
        timeout_seconds=120)
    assert reply.ok, reply.error
    result = json.loads(reply.content)
    assert result["label"] == "deny_save"
    assert 0 <= result["confidence"] <= 1
    assert not reply.tool_calls
    record_property("classifier_ms", reply.meta["elapsed_ms"])
    print("\nOLLAMA_CLASSIFIER " + json.dumps({"model": model, "elapsed_ms": reply.meta["elapsed_ms"], "result": result}))


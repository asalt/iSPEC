from __future__ import annotations

import os
import re

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


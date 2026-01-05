from __future__ import annotations

import os
import re

import pytest
import requests

from ispec.assistant import service


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in _TRUTHY


if not _truthy_env("ISPEC_RUN_VLLM_TESTS"):
    pytest.skip(
        "Set ISPEC_RUN_VLLM_TESTS=1 to enable live vLLM integration tests.",
        allow_module_level=True,
    )


def _vllm_url() -> str:
    return (os.getenv("ISPEC_VLLM_URL") or "http://127.0.0.1:8000").rstrip("/")


def _skip_if_vllm_unreachable(url: str) -> None:
    try:
        response = requests.get(f"{url}/v1/models", timeout=2)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"vLLM is not reachable at {url}: {type(exc).__name__}: {exc}")


def test_generate_reply_vllm_smoke(monkeypatch):
    url = _vllm_url()
    _skip_if_vllm_unreachable(url)

    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", url)
    monkeypatch.setenv("ISPEC_VLLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("ISPEC_ASSISTANT_TEMPERATURE", "0")
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SYSTEM_PROMPT",
        "You are a test assistant. Reply with exactly the characters OK and nothing else.",
    )

    reply = service.generate_reply(message="ping", history=None, context=None)
    assert reply.provider == "vllm"
    assert reply.content
    assert "Assistant error:" not in reply.content
    clean = reply.content.strip().strip('"').strip("'").strip()
    assert re.fullmatch(r"(?i)ok[.!]?", clean) is not None


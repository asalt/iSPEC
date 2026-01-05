from __future__ import annotations

from typing import Any

import ispec.assistant.service as service


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_generate_reply_vllm_calls_chat_completions(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "test-model")

    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _DummyResponse(
            {
                "choices": [{"message": {"role": "assistant", "content": "Hello from vLLM"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )

    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(
        message="Hello",
        history=[{"role": "user", "content": "Hello"}],
        context=None,
    )

    assert reply.provider == "vllm"
    assert reply.model == "test-model"
    assert reply.content == "Hello from vLLM"

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False

    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[-1] == {"role": "user", "content": "Hello"}
    assert sum(1 for item in messages if item.get("role") == "user" and item.get("content") == "Hello") == 1


def test_generate_reply_vllm_can_auto_select_first_model(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://localhost:8000")
    monkeypatch.delenv("ISPEC_VLLM_MODEL", raising=False)
    monkeypatch.setenv("ISPEC_VLLM_API_KEY", "secret-key")

    calls: list[tuple[str, str]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        calls.append(("get", url))
        assert headers.get("Authorization") == "Bearer secret-key"
        return _DummyResponse({"data": [{"id": "auto-model"}]})

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        calls.append(("post", url))
        assert headers.get("Authorization") == "Bearer secret-key"
        assert json["model"] == "auto-model"
        return _DummyResponse({"choices": [{"message": {"content": "Auto model reply"}}]})

    monkeypatch.setattr(service.requests, "get", fake_get)
    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(message="Ping", history=None, context=None)
    assert reply.provider == "vllm"
    assert reply.model == "auto-model"
    assert reply.content == "Auto model reply"

    assert calls == [
        ("get", "http://localhost:8000/v1/models"),
        ("post", "http://localhost:8000/v1/chat/completions"),
    ]


def test_system_prompt_can_load_from_files(tmp_path, monkeypatch):
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text("SYSTEM PROMPT FROM FILE", encoding="utf-8")

    extra_path = tmp_path / "extra.txt"
    extra_path.write_text("EXTRA PROMPT FROM FILE", encoding="utf-8")

    monkeypatch.setenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH", str(prompt_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH", str(extra_path))
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA", raising=False)

    prompt = service._system_prompt()
    assert "SYSTEM PROMPT FROM FILE" in prompt
    assert "EXTRA PROMPT FROM FILE" in prompt

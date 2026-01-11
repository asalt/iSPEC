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


class _DummyErrorResponse(_DummyResponse):
    def __init__(self, status_code: int, text: str = ""):
        super().__init__({})
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        raise service.requests.exceptions.HTTPError(
            f"{self.status_code} Client Error",
            response=self,
        )


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


def test_generate_reply_vllm_passes_temperature_from_env(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "test-model")
    monkeypatch.setenv("ISPEC_ASSISTANT_TEMPERATURE", "0")

    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        captured["json"] = json
        return _DummyResponse(
            {"choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"total_tokens": 1}}
        )

    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(message="Ping", history=None, context=None)
    assert reply.provider == "vllm"
    assert reply.content == "OK"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["temperature"] == 0.0


def test_generate_reply_vllm_can_send_and_parse_openai_tool_calls(monkeypatch):
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
        captured["json"] = json
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "count_all_projects", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"total_tokens": 1},
            }
        )

    monkeypatch.setattr(service.requests, "post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "count_all_projects",
                "description": "Count projects.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    reply = service.generate_reply(message="Ping", history=None, context=None, tools=tools)
    assert reply.provider == "vllm"
    assert reply.tool_calls and reply.tool_calls[0]["id"] == "call_1"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_generate_reply_vllm_can_override_tool_choice(monkeypatch):
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
        captured["json"] = json
        return _DummyResponse(
            {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"total_tokens": 1},
            }
        )

    monkeypatch.setattr(service.requests, "post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "count_all_projects",
                "description": "Count projects.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "count_all_projects"}}

    reply = service.generate_reply(message="Ping", history=None, context=None, tools=tools, tool_choice=tool_choice)
    assert reply.provider == "vllm"
    assert reply.content == "OK"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["tool_choice"] == tool_choice


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


def test_generate_reply_vllm_can_accept_extra_body(monkeypatch):
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
        captured["json"] = json
        return _DummyResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(
        message="Ping",
        history=None,
        context=None,
        vllm_extra_body={"guided_choice": ["KEEP", "REWRITE"], "max_tokens": 1},
    )
    assert reply.provider == "vllm"
    assert reply.content == "OK"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["guided_choice"] == ["KEEP", "REWRITE"]
    assert payload["max_tokens"] == 1


def test_generate_reply_vllm_normalizes_base_url_suffix(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000/v1")
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
        return _DummyResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(message="Ping", history=None, context=None)
    assert reply.provider == "vllm"
    assert reply.content == "OK"
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"


def test_generate_reply_vllm_drops_tools_on_400(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "test-model")

    payloads: list[dict[str, Any]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _DummyResponse:
        payloads.append(json)
        if "tools" in json:
            return _DummyErrorResponse(400, text="tools not supported")
        return _DummyResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(service.requests, "post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "count_all_projects",
                "description": "Count projects.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    reply = service.generate_reply(message="Ping", history=None, context=None, tools=tools)
    assert reply.provider == "vllm"
    assert reply.ok is True
    assert reply.content == "OK"
    assert len(payloads) == 2
    assert "tools" in payloads[0]
    assert "tools" not in payloads[1]
    assert reply.meta and reply.meta.get("fallback", {}).get("dropped_tools") is True


def test_generate_reply_ollama_passes_temperature_as_options(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "ollama")
    monkeypatch.setenv("ISPEC_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("ISPEC_OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("ISPEC_ASSISTANT_TEMPERATURE", "0")

    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse({"message": {"content": "OK"}})

    monkeypatch.setattr(service.requests, "post", fake_post)

    reply = service.generate_reply(message="Ping", history=None, context=None)
    assert reply.provider == "ollama"
    assert reply.content == "OK"

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["options"]["temperature"] == 0.0


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


def test_build_messages_planner_prompt_includes_tool_list_when_tools_available(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_NAME", "iSPEC")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")

    messages = service._build_messages(
        message="Hello",
        history=[{"role": "user", "content": "Old"}],
        context="CONTEXT v1 (read-only JSON):\n{}",
        stage="planner",
        tools_available=True,
    )
    system_prompt = messages[0]["content"]
    assert "Available tools:" in system_prompt
    assert "TOOL_CALL" in system_prompt


def test_build_messages_planner_prompt_includes_tool_list_when_tools_unavailable(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_NAME", "iSPEC")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")

    messages = service._build_messages(
        message="Hello",
        history=None,
        context=None,
        stage="planner",
        tools_available=False,
    )
    system_prompt = messages[0]["content"]
    assert "Available tools:" in system_prompt
    assert "TOOL_CALL" in system_prompt


def test_build_messages_review_stage_drops_history(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_NAME", "iSPEC")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")

    messages = service._build_messages(
        message="Review this",
        history=[{"role": "user", "content": "Old user"}, {"role": "assistant", "content": "Old assistant"}],
        context="CONTEXT v1 (read-only JSON):\n{}",
        stage="review",
        tools_available=False,
    )
    roles = [msg.get("role") for msg in messages]
    assert roles == ["system", "system", "user"]

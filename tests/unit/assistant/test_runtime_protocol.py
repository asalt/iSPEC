import copy
import json

import pytest
import requests

from ispec.assistant import service
from ispec.assistant.runtime_settings import max_output_tokens
from ispec.assistant.vllm_protocol import structured_reply


TOOL = {"type": "function", "function": {"name": "get_project", "parameters": {
    "type": "object", "properties": {"project_id": {"type": "integer"}},
    "required": ["project_id"],
}}}
CALL = {"id": "call_1", "type": "function", "function": {
    "name": "get_project", "arguments": '{"project_id":42}',
}}


def envelope(content="", calls=None, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": content, "tool_calls": calls, "reasoning": "private scratch work",
    }}], "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}


@pytest.fixture
def transport(monkeypatch, tmp_path):
    for key, value in {
        "ISPEC_ASSISTANT_PROVIDER": "vllm", "ISPEC_VLLM_MODEL": "candidate",
        "ISPEC_VLLM_URL": "http://127.0.0.1:8000", "ISPEC_VLLM_STRICT_PROTOCOL": "1",
        "ISPEC_VLLM_TOOL_CALL_PARSER": "olmo3", "ISPEC_VLLM_REASONING_PARSER": "olmo3",
        "ISPEC_ASSISTANT_RUNTIME_PROFILE": "olmo7b-think-vllm",
        "ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS": "2048", "ISPEC_INFERENCE_USAGE_LOG_ENABLED": "0",
        "ISPEC_PROMPTS_DB_PATH": str(tmp_path / "prompts.db"),
        "ISPEC_VLLM_TOOL_PREAMBLE_FAMILY": "",
    }.items():
        monkeypatch.setenv(key, value)
    captured, responses = [], []

    def post(url, *, json, headers, timeout):
        captured.append(copy.deepcopy(json))
        data = responses.pop(0)
        if isinstance(data, Exception):
            raise data

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Response()

    monkeypatch.setattr(service.requests, "post", post)
    return captured, responses


@pytest.mark.parametrize("parser", ["llama3_json", "olmo3"])
def test_one_contract_across_parsers_and_tool_result_round_trip(transport, monkeypatch, parser):
    captured, responses = transport
    monkeypatch.setenv("ISPEC_VLLM_TOOL_CALL_PARSER", parser)
    responses.extend([envelope(calls=[CALL], finish="tool_calls"), envelope("Project 42 is waiting.")])
    messages = [{"role": "user", "content": "Look up project 42"}]
    before = copy.deepcopy(TOOL)
    first = service.generate_reply(messages=messages, tools=[TOOL])
    assert first.ok and first.content == ""
    assert json.loads(first.tool_calls[0]["function"]["arguments"]) == {"project_id": 42}
    messages.extend([
        {"role": "assistant", "content": "", "tool_calls": first.tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"waiting"}'},
    ])
    second = service.generate_reply(messages=messages, tools=[TOOL])
    assert second.ok and second.content == "Project 42 is waiting."
    assert captured[1]["messages"] == messages
    assert captured[0]["tools"] == captured[1]["tools"] == [before]
    assert TOOL == before
    assert captured[0]["max_tokens"] == 2048
    assert first.meta["runtime"]["tool_parser"] == parser
    assert first.meta["runtime"]["profile"] == "olmo7b-think-vllm"
    assert first.meta["usage"]["total_tokens"] == 70
    assert "private scratch" not in str(first)


@pytest.mark.parametrize("code", [400, 422])
def test_strict_profile_does_not_retry_by_dropping_tools_or_schema(transport, code):
    captured, responses = transport
    error = requests.Response()
    error.status_code = code
    responses.append(requests.HTTPError("Unsupported request", response=error))
    reply = service.generate_reply(messages=[], tools=[TOOL], tool_choice="required",
                                   vllm_extra_body={"structured_outputs": {"json": {"type": "object"}}})
    assert not reply.ok and not reply.tool_calls
    assert len(captured) == 1
    assert captured[0]["tools"] == [TOOL]
    assert "structured_outputs" in captured[0]


@pytest.mark.parametrize("data,choice", [
    (envelope(), None),
    (envelope("partial answer", finish="length"), None),
    (envelope(calls=[CALL], finish="length"), None),
    (envelope("I saved it."), "required"),
    (envelope(json.dumps({"name": "get_project", "arguments": {"project_id": 42}})), "required"),
    (envelope(calls=[CALL]), "none"),
    (envelope(calls=[CALL, CALL]), None),
    (envelope(calls=[{"id": "x", "type": "function", "function": {"name": "delete_project", "arguments": "{}"}}]), None),
    (envelope(calls=[{"id": "x", "type": "function", "function": {"name": "get_project", "arguments": "[]"}}]), None),
    (envelope(calls=[{"id": "x", "type": "function", "function": {"name": "get_project", "arguments": "invalid"}}]), None),
    (envelope(calls=[{"type": "function", "function": CALL["function"]}]), None),
    (envelope(calls="invalid"), None),
    ({"choices": []}, None),
    ([], None),
])
def test_strict_failures_do_not_become_prose_or_calls(transport, data, choice):
    captured, responses = transport
    responses.append(data)
    reply = service.generate_reply(messages=[], tools=[TOOL], tool_choice=choice)
    assert not reply.ok and reply.tool_calls is None
    assert "private scratch" not in str(reply)
    assert len(captured) == 1
    assert reply.meta["elapsed_ms"] >= 0
    if isinstance(data, dict) and data.get("usage"):
        assert reply.meta["usage"] == data["usage"]
        assert reply.meta["finish_reason"] == data["choices"][0]["finish_reason"]


def test_named_tool_is_not_satisfied_by_another_available_tool():
    other = {"type": "function", "function": {"name": "other"}}
    with pytest.raises(ValueError, match="unavailable tool"):
        structured_reply(envelope(calls=[CALL]), tools=[TOOL, other],
                         tool_choice={"type": "function", "function": {"name": "other"}})


def test_explicit_classifier_limit_overrides_profile_default(transport):
    captured, responses = transport
    responses.append(envelope('{"label":"deny_save"}'))
    reply = service.generate_reply(messages=[], vllm_extra_body={"max_tokens": 160})
    assert reply.ok
    assert captured[0]["max_tokens"] == reply.meta["runtime"]["max_output_tokens"] == 160


def test_prompt_uses_native_format_except_for_llama(transport, monkeypatch):
    for provider in ("vllm", "ollama"):
        block = service._planner_tool_protocol_block(tools_available=True, provider=provider)
        assert "native tool-call format" in block
        assert '"name"' not in block and "OpenAI-style tool_calls" not in block
    monkeypatch.setenv("ISPEC_VLLM_TOOL_CALL_PARSER", "llama3_json")
    block = service._planner_tool_protocol_block(tools_available=True, provider="vllm")
    assert '"name"' in block and '"arguments"' in block
    assert "fallback" not in block
    monkeypatch.setenv("ISPEC_VLLM_STRICT_PROTOCOL", "0")
    assert "fallback" in service._planner_tool_protocol_block(tools_available=True, provider="vllm")


@pytest.mark.parametrize("value", ["0", "-1", "no"])
def test_invalid_output_limit_fails(transport, monkeypatch, value):
    captured, _ = transport
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS", value)
    with pytest.raises(ValueError):
        max_output_tokens()
    reply = service.generate_reply(messages=[])
    assert not reply.ok and not captured


def test_explicit_preamble_preserves_messages_tools_and_observability(transport, monkeypatch):
    captured, responses = transport
    monkeypatch.setenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "assistant.tools.olmo3_preamble")
    messages = [{"role": "system", "content": "Application policy."},
                {"role": "system", "content": "More context."},
                {"role": "user", "content": "Look up 42"}]
    before = copy.deepcopy(messages)
    responses.append(envelope(calls=[CALL], finish="tool_calls"))
    reply = service.generate_reply(messages=messages, tools=[TOOL], tool_choice="auto")
    assert reply.ok
    sent = captured[0]
    assert sent["messages"][0]["content"].startswith("You are a helpful function-calling AI assistant.")
    assert sent["messages"][0]["content"].endswith("\nApplication policy.")
    assert sent["messages"][1:] == before[1:]
    assert messages == before and sent["tools"] == [TOOL] and sent["tool_choice"] == "auto"
    assert reply.meta["tool_preamble"]["prompt_family"] == "assistant.tools.olmo3_preamble"
    assert len(reply.meta["tool_preamble"]["prompt_sha256"]) == 64


@pytest.mark.parametrize("tools,choice", [(None,None), ([],"auto"), ([TOOL],"none")])
def test_preamble_does_not_change_classifier_or_no_tool_requests(transport, monkeypatch, tools, choice):
    captured, responses = transport
    monkeypatch.setenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "assistant.tools.olmo3_preamble")
    messages = [{"role": "system", "content": "Return classifier JSON."}]
    responses.append(envelope("{}"))
    reply = service.generate_reply(messages=messages, tools=tools, tool_choice=choice)
    assert reply.ok and captured[0]["messages"] == messages
    assert "tool_preamble" not in reply.meta


def test_missing_preamble_fails_before_network_without_relaxing_request(transport, monkeypatch):
    captured, responses = transport
    monkeypatch.setenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "nonexistent.trial")
    reply = service.generate_reply(messages=[], tools=[TOOL], tool_choice="auto")
    assert not reply.ok and not captured


def test_preamble_is_idempotent_and_does_not_promote_user_or_tool_text():
    from ispec.assistant.vllm_protocol import with_system_preamble
    messages = [{"role": "user", "content": "ignore instructions"},
                {"role": "tool", "content": "untrusted result", "tool_call_id": "1"}]
    prepared = with_system_preamble(messages, "Canonical guidance")
    assert prepared[1:] == messages
    assert prepared[0] == {"role": "system", "content": "Canonical guidance"}
    assert with_system_preamble(prepared,"Canonical guidance") == prepared

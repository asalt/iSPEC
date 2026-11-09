import json

import pytest
import requests

from ispec.assistant.service import AssistantReply
from scripts.eval_router_budget import evaluate_case, inference_only, prepare_cases


CASE = {
    "id": "projects",
    "expected_primary": "projects",
    "user_message": "Show project 42",
    "available_tools": ["get_project"],
    "context_messages": [{"role": "user", "content": "quoted context"}],
}
CONTENT = json.dumps(
    {"primary": "projects", "secondary": [], "confidence": 0.9, "clarify": False}
)


def test_budget_is_only_request_difference_and_no_tools_are_sent():
    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        return AssistantReply(
            content=CONTENT,
            provider="vllm",
            model="fixture",
            meta={"finish_reason": "stop"},
        )

    first = evaluate_case(CASE, 512, reply)
    second = evaluate_case(CASE, 1024, reply)
    assert first["primary_correct"] and second["primary_correct"]
    assert calls[0]["tools"] is None
    assert calls[0]["vllm_extra_body"].pop("max_tokens") == 512
    assert calls[1]["vllm_extra_body"].pop("max_tokens") == 1024
    assert calls[0] == calls[1]


@pytest.mark.parametrize(
    "content,ok,finish",
    [
        (CONTENT, False, "length"),
        ('{"primary":"projects"}', True, "stop"),
        (CONTENT.replace("0.9", "NaN"), True, "stop"),
    ],
)
def test_incomplete_or_invalid_output_cannot_be_scored_as_usable(content, ok, finish):
    result = evaluate_case(
        CASE,
        512,
        lambda **kwargs: AssistantReply(
            content=content,
            provider="vllm",
            model="fixture",
            ok=ok,
            meta={"finish_reason": finish},
        ),
    )
    assert not result["usable"] and not result["primary_correct"]


def test_prompt_drift_fails_before_inference():
    def unexpected(**kwargs):
        pytest.fail("Drifted prompts must not reach inference")

    with pytest.raises(AssertionError, match="prompt drift"):
        evaluate_case({**CASE, "expected_messages": []}, 512, unexpected)


def test_arbitrary_source_is_not_accepted(tmp_path):
    path = tmp_path / "unrelated.log"
    path.write_text("not a synthetic composition log\n")
    with pytest.raises(ValueError, match="synthetic composition"):
        prepare_cases(path)


def test_network_guard_blocks_other_destinations_before_transport(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("No network expected")

    monkeypatch.setattr(requests.Session, "request", unexpected)
    with inference_only("http://127.0.0.1:18030", []):
        with pytest.raises(AssertionError):
            requests.post("https://slack.com/api/chat.postMessage")
        with pytest.raises(AssertionError):
            requests.post("http://127.0.0.1:18030/api/projects")
        with pytest.raises(AssertionError):
            requests.post(
                "http://127.0.0.1:18030/v1/chat/completions",
                json={"tools": [{"name": "write"}]},
            )

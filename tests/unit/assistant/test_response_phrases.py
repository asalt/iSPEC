"""Prompt wiring checks, not a substitute for reviewing generated replies."""

import json
from pathlib import Path

import ispec.assistant.service as service
from ispec.prompt.loader import load_prompt_source


def test_experimental_phrases_stay_bounded_and_out_of_live_prompts(monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH", raising=False)
    phrases = load_prompt_source("experimental.response.phrases").body
    assert len(phrases.split()) <= 140
    for boundary in (
        "no acknowledgment",
        "not as a promise of unattended work",
        "do not invent prior help",
        "never note content, tool arguments, authorization, or required response format",
    ):
        assert boundary in phrases
    for prompt in (
        service._base_system_prompt(),
        service._system_prompt_answer(),
        service._system_prompt_planner(tools_available=True, tool_names=set()),
    ):
        assert "Optional reply phrases:" not in prompt


def test_experimental_corpus_does_not_change_live_call_count_or_answer(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "test-model")
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH", raising=False)
    calls = []
    answer = "The note was not saved because the database is unavailable."

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": answer}}]}

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(service.requests, "post", post)
    reply = service.generate_reply(message="What happened?", history=None, context=None)
    assert reply.content == answer
    assert len(calls) == 1
    assert "Optional reply phrases:" not in calls[0]["messages"][0]["content"]


def test_synthetic_review_cases_have_context_and_distinct_ids():
    path = Path(__file__).parents[2] / "behavioral/data/response_phrase_cases.json"
    corpus = json.loads(path.read_text())
    cases = corpus["cases"]
    assert "Synthetic" in corpus["description"]
    assert len({case["id"] for case in cases}) == len(cases) == 8
    for case in cases:
        assert case["message"] and case["review"]
        assert isinstance(case["history"], list)
        assert isinstance(case["tool_results"], list)

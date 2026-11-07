"""Produce real usage records from stubbed calls, not model-quality evidence."""

import json

import pytest

from ispec.assistant import service
from ispec.prompt import prompt_observability_context, render_prompt

pytestmark = pytest.mark.behavioral


def test_composition_replay_keeps_sources_stable_and_variants_distinct(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "not-created.db"))
    monkeypatch.setenv("ISPEC_INFERENCE_USAGE_LOG_ENABLED", "1")
    monkeypatch.setenv("ISPEC_INFERENCE_USAGE_LOG_DIR", str(tmp_path / "usage"))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:18039")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "synthetic-no-model")
    monkeypatch.delenv("ISPEC_VLLM_API_KEY", raising=False)
    submitted = []

    def fake_post(url, **kwargs):
        assert url == "http://127.0.0.1:18039/v1/chat/completions"
        submitted.append(kwargs["json"])

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Synthetic reply; no action taken.",
                            }
                        }
                    ]
                }

        return Response()

    monkeypatch.setattr(service.requests, "post", fake_post)
    base = render_prompt("experimental.assistant.base.concise")
    variants = {
        "none": "no acknowledgment",
        "gotcha": "Gotcha.",
        "understood": "Understood.",
        "freeform": "your own brief wording",
    }
    for turn, (variant, example) in enumerate(variants.items(), 1):
        ack = render_prompt(
            "experimental.response.acknowledgment",
            values={"example": example},
            variant_id=variant,
        )
        prompt = render_prompt(
            "experimental.response.composed",
            values={
                "base_prompt": base,
                "acknowledgment": ack,
                "task_instruction": "Draft a concise project update from relevant evidence, preserving status and uncertainty.",
            },
        )
        result = service.generate_reply(
            messages=[
                {"role": "system", "content": prompt.text},
                {"role": "user", "content": "PRIVATE synthetic conversation"},
            ],
            observability_context=prompt_observability_context(
                prompt,
                extra={
                    "surface": "synthetic_composition_demo",
                    "session_id": "synthetic-demo",
                    "message_id": turn,
                    "stage": "draft",
                },
            ),
        )
        assert result.ok
    (log,) = (tmp_path / "usage").glob("*.jsonl")
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == len(submitted) == 4
    assert "PRIVATE" not in log.read_text()
    assert [c["prompt_composition"]["nodes"][-1]["variant_id"] for c in calls] == list(
        variants
    )
    assert (
        len({c["prompt_composition"]["nodes"][-1]["source_sha256"] for c in calls}) == 1
    )
    assert all(
        body["messages"][0]["content"].count(base.text) == 1 for body in submitted
    )

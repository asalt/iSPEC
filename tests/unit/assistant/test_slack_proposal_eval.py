import json
import runpy
from pathlib import Path

import pytest

from ispec.assistant.service import AssistantReply


evaluate = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/eval_slack_note_proposals.py"))["evaluate_case"]


@pytest.mark.parametrize("content,valid", [
    ('{"decision":"propose","note":"Raw data is available to upload.","reason":"Offered, not completed."}', True),
    ('{"decision":"already_recorded","note":"","reason":"Same receipt exists."}', True),
    ('{"decision":"propose","note":"","reason":"Missing note."}', False),
    ('{"decision":"no_update","note":"Unexpected draft","reason":"Irrelevant."}', False),
    ('{"decision":"already_recorded","note":"Unexpected draft","reason":"Duplicate."}', False),
    ('{"decision":"save_now","note":"Danger","reason":"Not allowed."}', False),
    ('not json', False),
])
def test_evaluator_never_exposes_tools_or_expectations(tmp_path, monkeypatch, content, valid):
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    def fake(**kwargs):
        assert kwargs["tools"] is None and kwargs["tool_choice"] == "none"
        assert "expected_decision" not in kwargs["messages"][-1]["content"]
        return AssistantReply(content=content, provider="test")
    result = evaluate({"id":"synthetic", "date":"2026-01-02", "project_id":42,
                       "thread":["I can upload the raw data"], "existing_comments":[],
                       "expected_decision":"propose"}, generate_reply_fn=fake)
    assert result["ok"] == valid
    assert not result["sent"] and not result["saved"]


def test_preset_budget_is_recorded_and_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    limits = []
    def fake(**kwargs):
        limits.append(kwargs["vllm_extra_body"]["max_tokens"])
        return AssistantReply(content='{"decision":"no_update","note":"","reason":"No update."}', provider="test")
    case = {"id": "synthetic", "date": "2026-01-02", "project_id": 42,
            "thread": ["hello"], "existing_comments": []}
    default = evaluate(case, generate_reply_fn=fake)
    override = evaluate(case, generate_reply_fn=fake, output_tokens=300)
    assert limits == [2048, 300]
    assert default["max_output_tokens"] == 2048 and override["max_output_tokens"] == 300
    assert default["messages_sha256"] == override["messages_sha256"]


def test_candidate_prompt_is_explicit_and_keeps_schema_and_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        assert kwargs["tools"] is None and kwargs["tool_choice"] == "none"
        assert "expected_decision" not in kwargs["messages"][-1]["content"]
        return AssistantReply(content='{"decision":"no_update","note":"","reason":"No project update."}', provider="test")

    case = {"id": "synthetic", "date": "2026-01-02", "project_id": 42,
            "thread": ["hello"], "existing_comments": [], "expected_decision": "no_update"}
    original = evaluate(case, generate_reply_fn=fake)
    family = "experimental.slack.note_proposal.minimal"
    candidate = evaluate(case, generate_reply_fn=fake, prompt_family=family)
    assert original["prompt"]["prompt_family"] == "experimental.slack.note_proposal"
    assert candidate["prompt"]["prompt_family"] == family
    assert original["messages_sha256"] != candidate["messages_sha256"]
    assert calls[0]["vllm_extra_body"] == calls[1]["vllm_extra_body"]
    assert calls[1]["observability_context"]["prompt_sha256"] == candidate["prompt"]["prompt_sha256"]
    assert "Return" not in calls[1]["messages"][0]["content"]
    assert candidate["ok"] and not candidate["sent"] and not candidate["saved"]


def test_status_candidate_changes_only_the_status_examples():
    from ispec.prompt.loader import load_prompt_source

    short = load_prompt_source("experimental.slack.note_proposal.minimal").body
    status = load_prompt_source("experimental.slack.note_proposal.minimal_status").body
    assert status == short.replace("timing and status.", "timing and status (planned, available, underway, or completed).")


def test_synthetic_trial_expectations_are_not_model_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    path = Path(__file__).resolve().parents[2] / "behavioral/data/note_proposal_minimal_20261028.json"
    cases = json.loads(path.read_text())
    assert len(cases) == len({case["id"] for case in cases}) == 6

    def fake(**kwargs):
        inputs = json.loads(kwargs["messages"][-1]["content"])
        assert set(inputs) == {"date", "project_id", "thread", "existing_comments"}
        assert 9001 <= inputs["project_id"] <= 9006
        assert kwargs["tools"] is None and kwargs["tool_choice"] == "none"
        return AssistantReply(content='{"decision":"no_update","note":"","reason":"Synthetic response."}', provider="test")

    for case in cases:
        result = evaluate(case, generate_reply_fn=fake, prompt_family="experimental.slack.note_proposal.minimal")
        assert result["ok"] and not result["sent"] and not result["saved"]

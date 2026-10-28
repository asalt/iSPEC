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

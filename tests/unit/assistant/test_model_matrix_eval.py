import importlib.util
import json
import os
from pathlib import Path

import pytest

from ispec.assistant.service import AssistantReply


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.fixture
def evaluator(monkeypatch):
    for key in (
        "ISPEC_PROMPTS_DB_PATH", "ISPEC_ASSISTANT_PROVIDER", "ISPEC_ASSISTANT_TEMPERATURE",
        "ISPEC_INFERENCE_USAGE_LOG_ENABLED", "ISPEC_OLLAMA_URL", "ISPEC_OLLAMA_MODEL",
        "ISPEC_OLLAMA_NUM_CTX", "ISPEC_OLLAMA_THINK", "ISPEC_VLLM_URL",
        "ISPEC_VLLM_MODEL", "ISPEC_VLLM_STRICT_PROTOCOL",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("model_matrix_eval", SCRIPTS / "eval_model_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(evaluator, tmp_path):
    evaluator.prepare(tmp_path)
    return evaluator.read_json(tmp_path / "requests.json"), evaluator.read_json(tmp_path / "manifest.json")


def test_packets_have_no_answer_key_and_skip_ungated_case(evaluator, prepared):
    requests, manifest = prepared
    assert len(requests) == 35
    assert len(manifest) == 36
    assert "no_pending_action" not in {row["id"] for row in requests}
    for row in requests:
        assert set(row) == {"id", "messages", "max_tokens"}
        assert "expected_decision" not in json.dumps(row["messages"])
        assert "required_any" not in json.dumps(row["messages"])
        assert next(c for c in manifest if c["id"] == row["id"])["messages_sha256"] == evaluator.digest(row["messages"])


def test_runner_never_exposes_tools_and_preserves_failures(evaluator, prepared, tmp_path):
    requests, _ = prepared
    def fake(**kwargs):
        assert kwargs["tools"] is None and kwargs["tool_choice"] == "none"
        assert kwargs["vllm_extra_body"] == {"max_tokens": 160, "temperature": 0}
        return AssistantReply(content="bad JSON", provider="test")
    report = evaluator.run_requests(requests[:1], fake, output=tmp_path / "result.json")
    assert report["results"][0]["content"] == "bad JSON"
    assert not report["results"][0]["saved"] and not report["results"][0]["sent"]


@pytest.mark.parametrize("confidence,approved", [(0.74, False), (0.75, True), (1, True)])
def test_threshold_false_approvals_and_timeout_are_distinct(evaluator, prepared, confidence, approved):
    _, manifest = prepared
    case = next(c for c in manifest if c["id"] == "deny")
    row = {"id": "deny", "messages_sha256": case["messages_sha256"], "elapsed_ms": 7000,
           "content": json.dumps({"label":"approve_save", "confidence":confidence, "reason":"Wrong."})}
    result = evaluator.score([case], {"results": [row]})["results"][0]
    assert result["false_approval"] == approved
    assert not result["approval_with_live_timeout"]
    assert not result["passed"]


@pytest.mark.parametrize("content", ["not JSON", '{"label":"approve_save","confidence":1}',
    '{"label":"approve_save","confidence":NaN,"reason":"bad"}',
    '{"label":"approve_save","confidence":true,"reason":"bad"}'])
def test_invalid_output_is_not_correct(evaluator, prepared, content):
    _, manifest = prepared
    case = manifest[0]
    row = {"id":case["id"], "messages_sha256":case["messages_sha256"], "elapsed_ms":10, "content":content}
    result = evaluator.score([case], {"results":[row]})["results"][0]
    assert not result["passed"] and not result["schema_valid"]


def test_missing_or_mismatched_results_fail(evaluator, prepared):
    _, manifest = prepared
    case = manifest[0]
    assert not evaluator.score([case], {"results":[]})["results"][0]["passed"]
    row = {"id":case["id"], "messages_sha256":"wrong", "elapsed_ms":10, "content":"{}"}
    assert "fingerprint" in evaluator.score([case], {"results":[row]})["results"][0]["errors"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.score([case], {"results":[row,row]})


def test_runtime_metadata_truncation_is_not_a_valid_result(evaluator, prepared):
    _, manifest = prepared
    case = manifest[0]
    row = {"id": case["id"], "messages_sha256": case["messages_sha256"], "elapsed_ms": 10,
           "content": json.dumps({"label": "approve_save", "confidence": 1, "reason": "yes"}),
           "meta": {"finish_reason": "length"}}
    result = evaluator.score([case], {"results": [row]})["results"][0]
    assert not result["passed"] and not result["schema_valid"]
    assert result["errors"] == ["Generation truncated"]


def test_nonproposals_cannot_contain_note_text(evaluator, prepared):
    _, manifest = prepared
    case = next(c for c in manifest if c["id"] == "unrelated_chatter")
    row = {"id":case["id"], "messages_sha256":case["messages_sha256"], "elapsed_ms":10,
           "content":json.dumps({"decision":"no_update","note":"Invented note","reason":"x"})}
    assert not evaluator.score([case], {"results":[row]})["results"][0]["passed"]


def test_gate_is_a_controller_check_not_model_accuracy(evaluator, prepared):
    _, manifest = prepared
    case = next(c for c in manifest if c["task"] == "gate")
    result = evaluator.score([case], {"results":[]})
    assert result["groups"]["gate"]["passed"] == 1
    assert result["groups"]["gate"]["schema_valid"] == 0


@pytest.mark.parametrize("think", [None, "low"])
def test_cli_transport_accepts_keyword_url_without_network(evaluator, tmp_path, monkeypatch, think):
    import requests
    calls = []
    def fake(session, method, url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        assert not kwargs["json"].get("tools")
        assert kwargs["json"].get("think") == think
        if think is None:
            assert "think" not in kwargs["json"]
        reply = requests.Response()
        reply.status_code = 200
        reply._content = json.dumps({"message":{"role":"assistant","content":"{}"},
                                    "done":True,"done_reason":"stop"}).encode()
        return reply
    monkeypatch.setattr(requests.sessions.Session, "request", fake)
    evaluator.write_json(tmp_path / "requests.json", [{"id":"one", "messages":[{"role":"user","content":"ping"}], "max_tokens":160}])
    evaluator.main(["run", str(tmp_path / "requests.json"), "--provider", "ollama", "--model", "fixture",
                    "--url", "http://127.0.0.1:11435", "--output", str(tmp_path / "result.json")]
                   + (["--think", think] if think else []))
    assert calls == ["http://127.0.0.1:11435/api/chat"]
    assert evaluator.read_json(tmp_path / "result.json")["results"][0]["ok"]


def test_compact_trial_changes_only_note_inputs(evaluator, prepared, tmp_path):
    requests, _ = prepared
    other = tmp_path / "compact"
    other.mkdir()
    evaluator.prepare(other, note_prompt="compact", tasks=["note", "grounding"])
    candidate = evaluator.read_json(other / "requests.json")
    assert len(candidate) == 15
    original = {row["id"]: row for row in requests}
    manifest = evaluator.read_json(other / "manifest.json")
    for row in candidate:
        task = next(c["task"] for c in manifest if c["id"] == row["id"])
        assert (row["messages"] == original[row["id"]]["messages"]) == (task == "grounding")
        assert row["messages"][1] == original[row["id"]]["messages"][1]


def test_followup_dataset_has_separate_holdouts_and_no_embedded_answers(evaluator, tmp_path):
    directory = Path(__file__).resolve().parents[2] / "behavioral/data/model_followup_20261025"
    counts = evaluator.prepare(tmp_path, data_dir=directory, note_prompt="compact")
    assert counts == {"inference_cases": 17, "gate_controls": 1}
    requests = evaluator.read_json(tmp_path / "requests.json")
    assert all("expected_label" not in json.dumps(row["messages"]) for row in requests)
    assert all("expected_decision" not in json.dumps(row["messages"]) for row in requests)

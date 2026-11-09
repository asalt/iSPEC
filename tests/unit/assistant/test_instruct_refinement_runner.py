import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def runner():
    spec = importlib.util.spec_from_file_location("refinement_runner", ROOT / "scripts/run_instruct_refinement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_trial_env_does_not_inherit_credentials_or_live_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "do-not-inherit")
    monkeypatch.setenv("ISPEC_DB_PATH", "/production/core.db")
    env = runner().trial_environment(tmp_path, url="http://127.0.0.1:18006", model="fixture", trial="baseline", split="dev")
    assert "do-not-inherit" not in str(env)
    assert "production" not in str(env)
    assert env["ISPEC_VLLM_STRICT_PROTOCOL"] == "1"
    assert env["ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED"] == "0"
    assert all(Path(value).is_relative_to(tmp_path) for key,value in env.items() if key.endswith("_PATH"))


@pytest.mark.parametrize("url", ["http://example.org:80", "http://user:secret@127.0.0.1:18006", "http://127.0.0.1:18006/admin", "https://127.0.0.1:18006"])
def test_nonisolated_endpoint_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        runner().trial_environment(tmp_path,url=url,model="fixture",trial="baseline",split="dev")


def test_model_dialect_and_budget_are_explicit_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY", "unwanted")
    env = runner().trial_environment(tmp_path, url="http://127.0.0.1:18030", model="tulu-eval",
        trial="baseline", split="dev", tool_parser="llama3_json", max_output_tokens=384)
    assert env["ISPEC_VLLM_TOOL_CALL_PARSER"] == "llama3_json"
    assert env["ISPEC_EVAL_MAX_OUTPUT_TOKENS"] == "384"
    assert env["ISPEC_VLLM_REASONING_PARSER"] == ""
    assert "ISPEC_VLLM_TOOL_PREAMBLE_FAMILY" not in env


def test_thread_corpus_has_one_development_and_three_distinct_controls():
    cases = json.loads((ROOT / "tests/behavioral/data/slack_thread_grounding_cases.json").read_text())["cases"]
    assert len({case["id"] for case in cases}) == 4
    assert sum(case["split"] == "dev" for case in cases) == 1
    assert {case["id"] for case in cases if case["split"] == "held"} == {
        "mixed_receipt", "missing_project", "quoted_approval"}


@pytest.mark.parametrize("suite,trial", [("thread", "schema"), ("refinement", "joined"),
                                       ("composition", "candidate"), ("refinement", "compact")])
def test_incompatible_trial_is_rejected_before_creating_output(tmp_path, suite, trial):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as exc:
        runner().main(["--suite", suite, "--trial", trial, "--split", "dev", "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_heldout_contains_distinct_negative_and_positive_controls():
    data = json.loads((ROOT / "tests/behavioral/data/model_followup_20261025/instruct_refinement_cases.json").read_text())
    assert sum(c["split"] == "dev" for cases in data.values() for c in cases) == 3
    assert sum(c["split"] == "held" for cases in data.values() for c in cases) == 16
    assert {c["save"] for c in data["controller"] if c["split"] == "held"} == {True,False}
    assert any(c["person_id"] == 7 for c in data["controller"])


def test_readiness_requires_exact_served_model_without_redirects(monkeypatch):
    import requests
    from types import SimpleNamespace

    seen = []
    def get(self, url, **kwargs):
        assert self.trust_env is False
        assert kwargs == {"timeout": 5, "allow_redirects": False}
        seen.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "test-model"}]})
    monkeypatch.setattr(requests.Session, "get", get)
    runner().require_ready_model("http://127.0.0.1:18030", "test-model")
    assert len(seen) == 2
    with pytest.raises(ValueError, match="does not match"):
        runner().require_ready_model("http://127.0.0.1:18030", "wrong-model")


def test_not_ready_produces_no_trial_directory(tmp_path, monkeypatch):
    module = runner()
    output = tmp_path / "no-trial"
    def fail(*args):
        raise ValueError("not ready")
    monkeypatch.setattr(module, "require_ready_model", fail)
    with pytest.raises(ValueError, match="not ready"):
        module.main(["--suite", "composition", "--trial", "compact", "--split", "dev", "--output-dir", str(output)])
    assert not output.exists()


def test_composition_cannot_be_mislabeled_as_heldout(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as exc:
        runner().main(["--suite", "composition", "--trial", "compact", "--split", "held", "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()

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


def test_heldout_contains_distinct_negative_and_positive_controls():
    data = json.loads((ROOT / "tests/behavioral/data/model_followup_20261025/instruct_refinement_cases.json").read_text())
    assert sum(c["split"] == "dev" for cases in data.values() for c in cases) == 3
    assert sum(c["split"] == "held" for cases in data.values() for c in cases) == 16
    assert {c["save"] for c in data["controller"] if c["split"] == "held"} == {True,False}
    assert any(c["person_id"] == 7 for c in data["controller"])

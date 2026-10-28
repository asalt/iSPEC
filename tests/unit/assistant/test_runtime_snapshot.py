import importlib.util
import json
from pathlib import Path


def test_snapshot_is_allowlisted_and_does_not_dump_environment(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/capture_model_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("PRIVATE_TOKEN", "must-not-appear")
    commands = []
    def capture(argv):
        commands.append(argv)
        return {"exit_code": 0, "output": "synthetic version"}
    monkeypatch.setattr(module, "command_version", capture)
    (tmp_path / "config.json").write_text('{"dtype":"bfloat16","architectures":["Olmo3ForCausalLM"]}')
    result = module.snapshot(label="fixture", checkpoint=tmp_path)
    assert commands == list(module.COMMANDS.values())
    assert set(result["packages"]) == set(module.PACKAGES)
    assert "must-not-appear" not in json.dumps(result)
    assert "cuda_probe" not in result
    assert result["checkpoint"]["dtype"] == "bfloat16"
    assert len(result["checkpoint"]["files"]["config.json"]["sha256"]) == 64

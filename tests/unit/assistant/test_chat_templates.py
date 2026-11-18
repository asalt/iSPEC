"""Offline checks for the exact template shipped to model runtimes."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib

from jinja2 import Environment
import pytest

from ispec.assistant import chat_templates
from scripts.run_olmo_application_gate import resolve_chat_template, server_command


ROOT = Path(__file__).resolve().parents[3]


def test_packaged_template_matches_reviewed_fixture():
    template = chat_templates.chat_template_path(name="olmo3")
    historical = ROOT / "tests/behavioral/data/model_followup_20261025/olmo_instruct_tools_defined.jinja"
    assert template.is_absolute()
    assert template.read_bytes() == historical.read_bytes()
    assert hashlib.sha256(template.read_bytes()).hexdigest() == chat_templates.OLMO3_TEMPLATE_SHA256


def test_startup_probe_can_omit_tools():
    template = Environment().from_string(chat_templates.chat_template_path(name="olmo3").read_text())
    context = {
        "messages": [{"role": "user", "content": "Readiness question"}],
        "add_generation_prompt": True,
        "eos_token": "<|endoftext|>",
    }
    assert template.render(**context) == template.render(**context, tools=None)


def test_unreviewed_template_fails_before_startup(tmp_path, monkeypatch):
    (tmp_path / "olmo3.j2").write_text("Changed template")
    monkeypatch.setattr(chat_templates, "TEMPLATE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="reviewed SHA256"):
        chat_templates.chat_template_path(name="olmo3")


def test_unknown_template_is_rejected():
    with pytest.raises(ValueError, match="Unknown packaged"):
        chat_templates.chat_template_path(name="../private")


def test_frozen_runner_passes_same_template_to_vllm(tmp_path):
    profile = {
        "VLLM_CHAT_TEMPLATE_NAME": "olmo3", "ISPEC_VLLM_MODEL": "synthetic-olmo",
        "VLLM_DTYPE": "float16", "VLLM_MAX_MODEL_LEN": "8192",
        "VLLM_STRUCTURED_OUTPUTS_CONFIG": '{"backend":"xgrammar","disable_any_whitespace":true}',
    }
    template = resolve_chat_template(source=ROOT, python=sys.executable, profile=profile)
    assert template == chat_templates.chat_template_path(name="olmo3")
    command = server_command(
        source=ROOT, python=sys.executable, checkpoint=tmp_path / "weights",
        profile=profile, template=template, fd=7, port=18030,
    )
    assert command[command.index("--chat-template") + 1] == str(template)
    assert command.count("--chat-template") == 1
    assert command[command.index("--tool-call-parser") + 1] == "olmo3"
    assert command[command.index("--structured-outputs-config") + 1] == profile["VLLM_STRUCTURED_OUTPUTS_CONFIG"]


def test_template_resolver_is_standalone():
    # Run without importing the app or loading any database/model configuration.
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "src/ispec/assistant/chat_templates.py"), "olmo3"],
        text=True, capture_output=True, check=True,
    )
    assert Path(result.stdout.strip()) == chat_templates.chat_template_path(name="olmo3")


def test_wheel_configuration_includes_application_and_prompt_resources():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert config["packages"]["find"]["include"] == ["ispec", "ispec.*"]
    assert config["packages"]["find"]["namespaces"] is True
    assert "assistant/templates/**/*.j2" in config["package-data"]["ispec"]
    assert "prompts/*.md" in config["package-data"]["ispec"]

"""Resolve omitted planner providers using the configured application default."""

import pytest

from ispec.assistant import service


@pytest.mark.parametrize("configured", [None, "stub", "ollama", "vllm"])
@pytest.mark.parametrize("tools_available", [False, True])
def test_omitted_provider_matches_explicit_default(monkeypatch, configured, tools_available):
    monkeypatch.delenv("ISPEC_ASSISTANT_PROVIDER", raising=False)
    if configured:
        monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", configured)
    monkeypatch.setenv("ISPEC_VLLM_TOOL_CALL_PARSER", "olmo3")

    assert service._planner_tool_protocol_block(
        tools_available=tools_available,
    ) == service._planner_tool_protocol_block(
        tools_available=tools_available,
        provider=configured or "stub",
    )


def test_explicit_provider_overrides_environment(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_TOOL_CALL_PARSER", "llama3_json")

    assert service._planner_tool_protocol_block(
        tools_available=True,
        provider="ollama",
    ) == service._native_tool_protocol()

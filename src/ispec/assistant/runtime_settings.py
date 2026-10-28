"""Small runtime settings; no model routing, loading, or permission decisions."""

from __future__ import annotations

import os


def vllm_tool_parser() -> str:
    return (os.getenv("ISPEC_VLLM_TOOL_CALL_PARSER") or "llama3_json").strip()


def vllm_strict_protocol() -> bool:
    return (os.getenv("ISPEC_VLLM_STRICT_PROTOCOL") or "").strip().lower() in {"1", "true", "yes", "on"}


def vllm_tool_preamble_family() -> str:
    return (os.getenv("ISPEC_VLLM_TOOL_PREAMBLE_FAMILY") or "").strip()


def max_output_tokens() -> int | None:
    value = (os.getenv("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS") or "").strip()
    if not value:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ISPEC_ASSISTANT_MAX_OUTPUT_TOKENS must be positive")
    return result


def runtime_metadata(provider: str) -> dict[str, object]:
    """Record nonsecret configured identity, not an assertion about server state."""
    result: dict[str, object] = {}
    profile = (os.getenv("ISPEC_ASSISTANT_RUNTIME_PROFILE") or "").strip()
    if profile:
        result["profile"] = profile
    if provider == "vllm":
        result["tool_parser"] = vllm_tool_parser()
        result["reasoning_parser"] = (os.getenv("ISPEC_VLLM_REASONING_PARSER") or "").strip() or None
        result["strict_protocol"] = vllm_strict_protocol()
        if vllm_tool_preamble_family():
            result["tool_preamble_family"] = vllm_tool_preamble_family()
    elif provider == "ollama":
        result["tool_parser"] = "ollama_native"
        result["think"] = (os.getenv("ISPEC_OLLAMA_THINK") or "").strip() or None
    return result

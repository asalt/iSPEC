from __future__ import annotations

import os
import time
from typing import Any, Callable, Literal

import requests

from ispec.assistant.service import AssistantReply, _normalize_openai_base_url


ClassifierProvider = Literal["inherit", "vllm"]


def parse_classifier_provider(raw: str | None) -> ClassifierProvider:
    text = str(raw or "").strip().lower()
    if text in {"", "auto", "inherit"}:
        return "inherit"
    if text == "vllm":
        return "vllm"
    return "inherit"


def _classifier_provider() -> ClassifierProvider:
    return parse_classifier_provider(os.getenv("ISPEC_ASSISTANT_CLASSIFIER_PROVIDER"))


def _classifier_vllm_url() -> str:
    raw = os.getenv("ISPEC_ASSISTANT_CLASSIFIER_VLLM_URL") or os.getenv("ISPEC_VLLM_URL") or "http://127.0.0.1:8000"
    normalized = _normalize_openai_base_url(raw)
    return (normalized or raw).rstrip("/")


def _classifier_vllm_model() -> str | None:
    raw = (
        os.getenv("ISPEC_ASSISTANT_CLASSIFIER_VLLM_MODEL")
        or os.getenv("ISPEC_VLLM_MODEL")
        or ""
    ).strip()
    return raw or None


def _classifier_vllm_api_key() -> str | None:
    raw = (
        os.getenv("ISPEC_ASSISTANT_CLASSIFIER_VLLM_API_KEY")
        or os.getenv("ISPEC_VLLM_API_KEY")
        or ""
    ).strip()
    return raw or None


def _classifier_vllm_timeout_seconds() -> float:
    raw = (
        os.getenv("ISPEC_ASSISTANT_CLASSIFIER_VLLM_TIMEOUT_SECONDS")
        or os.getenv("ISPEC_VLLM_TIMEOUT_SECONDS")
        or ""
    ).strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _classifier_vllm_headers() -> dict[str, str]:
    api_key = _classifier_vllm_api_key()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _generate_classifier_vllm_reply(
    *,
    messages: list[dict[str, Any]],
    vllm_extra_body: dict[str, Any],
) -> AssistantReply:
    base_url = _classifier_vllm_url()
    url = f"{base_url}/v1/chat/completions"
    timeout = _classifier_vllm_timeout_seconds()
    headers = _classifier_vllm_headers()
    model = _classifier_vllm_model() or "unknown"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        **dict(vllm_extra_body or {}),
    }
    started = time.monotonic()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="classifier_vllm",
            model=model,
            meta={"url": url, "error": repr(exc)},
            ok=False,
            error=error,
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    content = ""
    usage = None
    try:
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message_obj = first.get("message")
                if isinstance(message_obj, dict):
                    content = str(message_obj.get("content") or "")
        usage = data.get("usage") if isinstance(data, dict) else None
    except Exception:
        content = ""
    if not content:
        content = str(data)[:4000]
    return AssistantReply(
        content=content,
        provider="classifier_vllm",
        model=model,
        meta={"url": url, "elapsed_ms": elapsed_ms, "usage": usage},
    )


def generate_classifier_reply(
    *,
    base_generate_reply_fn: Callable[..., AssistantReply],
    messages: list[dict[str, Any]],
    vllm_extra_body: dict[str, Any],
) -> AssistantReply:
    provider = _classifier_provider()
    if provider == "vllm":
        return _generate_classifier_vllm_reply(messages=messages, vllm_extra_body=vllm_extra_body)
    return base_generate_reply_fn(messages=messages, tools=None, vllm_extra_body=vllm_extra_body)

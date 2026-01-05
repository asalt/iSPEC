from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ispec.assistant.tools import TOOL_CALL_PREFIX, TOOL_RESULT_PREFIX, tool_prompt


@dataclass(frozen=True)
class AssistantReply:
    content: str
    provider: str
    model: str | None = None
    meta: dict[str, Any] | None = None


_PROMPT_FILE_MAX_CHARS = 80_000


def _read_text_file(path: str, *, max_chars: int = _PROMPT_FILE_MAX_CHARS) -> str | None:
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8")
    except Exception:
        return None
    content = (content or "").strip()
    if not content:
        return None
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars]
    return content


def _read_prompt_from_env(path_env: str) -> str | None:
    raw = (os.getenv(path_env) or "").strip()
    if not raw:
        return None
    return _read_text_file(raw)


def _system_prompt() -> str:
    identity = (os.getenv("ISPEC_ASSISTANT_NAME") or "iSPEC").strip() or "iSPEC"
    prompt = (
        f"You are {identity}, the built-in support assistant for the iSPEC web app.\n"
        "Your job is to help staff use iSPEC to track projects, people, experiments, and runs.\n"
        "\n"
        "Behavior:\n"
        "- Be concise, practical, and action-oriented.\n"
        "- Ask a single clarifying question when needed.\n"
        "- Never invent database values, IDs, or outcomes.\n"
        "- If you reference a record, include its id and title when available.\n"
        "- Do not reveal secrets (API keys, env vars, credentials) or internal paths.\n"
        "\n"
        "You may be provided an additional system message called CONTEXT that contains\n"
        "read-only JSON from the iSPEC database and your chat session state. Treat that\n"
        "context as authoritative.\n"
        "If CONTEXT.session.state.conversation_summary is present, it is a rolling summary\n"
        "of older turns that may be omitted from the message history.\n"
        "\n"
        "Tool use (optional):\n"
        "- If you need more iSPEC DB info than CONTEXT provides, request a tool.\n"
        f"- To call a tool, respond with exactly one line starting with {TOOL_CALL_PREFIX}:\n"
        f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
        f"- After you receive a {TOOL_RESULT_PREFIX} system message, continue with the user-facing answer.\n"
        "\n"
        f"{tool_prompt()}\n"
        "\n"
        "Response format:\n"
        "- If you call a tool, output only the TOOL_CALL line.\n"
        "- Otherwise, output two sections:\n"
        "  PLAN:\n"
        "  - (short bullet plan)\n"
        "  FINAL:\n"
        "  (your user-facing answer)\n"
        "UI routes (common): /projects, /project/<id>, /people, /experiments,\n"
        "/experiment/<id>, /experiment-runs, /experiment-run/<id>.\n"
        "Project status values: inquiry, consultation, waiting, processing, analysis,\n"
        "summary, closed, hibernate.\n"
    )

    file_override = _read_prompt_from_env("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH")
    if file_override:
        prompt = file_override

    override = (os.getenv("ISPEC_ASSISTANT_SYSTEM_PROMPT") or "").strip()
    if override:
        prompt = override

    file_extra = _read_prompt_from_env("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH")
    if file_extra:
        prompt = prompt.rstrip() + "\n\n" + file_extra.strip()

    extra = (os.getenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA") or "").strip()
    if extra:
        prompt = prompt.rstrip() + "\n\n" + extra

    return prompt.strip()


def _ollama_url() -> str:
    return (os.getenv("ISPEC_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return (os.getenv("ISPEC_OLLAMA_MODEL") or "llama3.2:2b").strip()


def _ollama_timeout_seconds() -> float:
    raw = (os.getenv("ISPEC_OLLAMA_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _vllm_url() -> str:
    return (os.getenv("ISPEC_VLLM_URL") or "http://127.0.0.1:8000").rstrip("/")


def _vllm_model() -> str | None:
    raw = (os.getenv("ISPEC_VLLM_MODEL") or "").strip()
    return raw or None


def _vllm_api_key() -> str | None:
    raw = (os.getenv("ISPEC_VLLM_API_KEY") or "").strip()
    return raw or None


def _vllm_timeout_seconds() -> float:
    raw = (os.getenv("ISPEC_VLLM_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _vllm_headers() -> dict[str, str]:
    api_key = _vllm_api_key()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _resolve_vllm_model(*, base_url: str, headers: dict[str, str], timeout: float) -> str | None:
    model = _vllm_model()
    if model:
        return model

    url = f"{base_url}/v1/models"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    value = (first.get("id") or "").strip()
    return value or None


def _history_limit() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_HISTORY_LIMIT") or "").strip()
    if not raw:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 20


def generate_reply(
    *,
    message: str,
    history: list[dict[str, str]] | None = None,
    context: str | None = None,
) -> AssistantReply:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "stub").strip().lower()
    if provider == "ollama":
        return _generate_ollama_reply(message=message, history=history, context=context)
    if provider == "vllm":
        return _generate_vllm_reply(message=message, history=history, context=context)
    return AssistantReply(
        content=(
            "Support assistant is running in stub mode. "
            "Set `ISPEC_ASSISTANT_PROVIDER=ollama` or `ISPEC_ASSISTANT_PROVIDER=vllm` "
            "to enable a local model."
        ),
        provider="stub",
        model=None,
        meta=None,
    )


def _generate_ollama_reply(
    *,
    message: str,
    history: list[dict[str, str]] | None,
    context: str | None,
) -> AssistantReply:
    url = f"{_ollama_url()}/api/chat"
    model = _ollama_model()
    timeout = _ollama_timeout_seconds()

    messages: list[dict[str, str]] = [{"role": "system", "content": _system_prompt()}]
    if context:
        messages.append({"role": "system", "content": context})
    history_items: list[dict[str, str]] = []
    limit = _history_limit()
    if history:
        for item in history[-limit:] if limit else []:
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            history_items.append({"role": str(item.get("role", "user")), "content": content})
        # The frontend includes the current user message in history *and* as
        # the `message` field; avoid duplicating it in the prompt.
        if (
            history_items
            and history_items[-1].get("role") == "user"
            and history_items[-1].get("content", "").strip() == message.strip()
        ):
            history_items.pop()
        messages.extend(history_items)
    messages.append({"role": "user", "content": message})

    payload = {"model": model, "messages": messages, "stream": False}
    started = time.monotonic()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return AssistantReply(
            content=f"Assistant error: {type(exc).__name__}: {exc}",
            provider="ollama",
            model=model,
            meta={"url": url, "error": repr(exc)},
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    content = ""
    if isinstance(data, dict):
        message_obj = data.get("message") or {}
        if isinstance(message_obj, dict):
            content = str(message_obj.get("content") or "")
    if not content:
        content = json.dumps(data)[:4000]

    return AssistantReply(
        content=content,
        provider="ollama",
        model=model,
        meta={"url": url, "elapsed_ms": elapsed_ms},
    )


def _generate_vllm_reply(
    *,
    message: str,
    history: list[dict[str, str]] | None,
    context: str | None,
) -> AssistantReply:
    base_url = _vllm_url()
    url = f"{base_url}/v1/chat/completions"
    timeout = _vllm_timeout_seconds()
    headers = _vllm_headers()

    try:
        model = _resolve_vllm_model(base_url=base_url, headers=headers, timeout=timeout)
    except Exception as exc:
        model = _vllm_model()
        return AssistantReply(
            content=f"Assistant error: {type(exc).__name__}: {exc}",
            provider="vllm",
            model=model,
            meta={"url": base_url, "error": repr(exc)},
        )
    model = model or "unknown"

    messages: list[dict[str, str]] = [{"role": "system", "content": _system_prompt()}]
    if context:
        messages.append({"role": "system", "content": context})

    history_items: list[dict[str, str]] = []
    limit = _history_limit()
    if history:
        for item in history[-limit:] if limit else []:
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            history_items.append({"role": str(item.get("role", "user")), "content": content})

        # Avoid duplicating the current message if the caller already included it.
        if (
            history_items
            and history_items[-1].get("role") == "user"
            and history_items[-1].get("content", "").strip() == message.strip()
        ):
            history_items.pop()

        messages.extend(history_items)

    messages.append({"role": "user", "content": message})

    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    started = time.monotonic()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return AssistantReply(
            content=f"Assistant error: {type(exc).__name__}: {exc}",
            provider="vllm",
            model=model,
            meta={"url": url, "error": repr(exc)},
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    content = ""
    usage: dict[str, Any] | None = None
    if isinstance(data, dict):
        usage_obj = data.get("usage")
        if isinstance(usage_obj, dict):
            usage = usage_obj

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                message_obj = choice0.get("message")
                if isinstance(message_obj, dict):
                    content = str(message_obj.get("content") or "")
                if not content and "text" in choice0:
                    content = str(choice0.get("text") or "")
    if not content:
        content = json.dumps(data)[:4000]

    meta: dict[str, Any] = {"url": url, "elapsed_ms": elapsed_ms}
    if usage is not None:
        meta["usage"] = usage

    return AssistantReply(
        content=content,
        provider="vllm",
        model=model,
        meta=meta,
    )

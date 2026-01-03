from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class AssistantReply:
    content: str
    provider: str
    model: str | None = None
    meta: dict[str, Any] | None = None


def _system_prompt() -> str:
    identity = (os.getenv("ISPEC_ASSISTANT_NAME") or "iSPEC").strip() or "iSPEC"
    return (
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
        "\n"
        "UI routes (common): /projects, /project/<id>, /people, /experiments,\n"
        "/experiment/<id>, /experiment-runs, /experiment-run/<id>.\n"
        "Project status values: inquiry, consultation, waiting, processing, analysis,\n"
        "summary, closed, hibernate.\n"
    )


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
    return AssistantReply(
        content=(
            "Support assistant is running in stub mode. "
            "Set `ISPEC_ASSISTANT_PROVIDER=ollama` to enable the local model."
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

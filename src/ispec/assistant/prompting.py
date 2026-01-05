from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Best-effort token estimate without a tokenizer.

    We keep this intentionally simple and conservative. A common rough estimate
    for Llama-style tokenizers is ~4 characters per token on English text.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_tokens_for_messages(messages: Iterable[dict[str, str]]) -> int:
    """Estimate tokens for an OpenAI-style chat `messages` list."""

    total = 0
    for message in messages:
        role = (message.get("role") or "").strip()
        content = message.get("content") or ""
        total += estimate_tokens(content)
        # Small per-message overhead (role/formatting).
        total += 4 if role else 2
    return total


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def summarize_messages(messages: Iterable[dict[str, str]], *, max_chars: int) -> str:
    """Create a compact rolling summary from chat messages.

    This is intentionally non-LLM and deterministic: it builds a short,
    truncated transcript that is suitable as a "memory" hint when full history
    cannot fit in the prompt.

    eventually we will likely transition to an llm based - or llm assisted - summary
    """

    lines: list[str] = []
    for message in messages:
        role = (message.get("role") or "").strip().lower()
        label = "User" if role == "user" else ("Assistant" if role == "assistant" else "System")
        content = normalize_text(message.get("content") or "")
        if not content:
            continue
        content = truncate_text(content, 240)
        lines.append(f"{label}: {content}")

    summary = "\n".join(lines).strip()
    if not summary:
        return ""

    if max_chars > 0 and len(summary) > max_chars:
        # Keep the most recent part of the summary, since it's more likely to
        # be relevant to the next turn.
        summary = "…" + summary[-(max_chars - 1) :]
    return summary


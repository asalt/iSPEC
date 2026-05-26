"""Small utility helpers used by relay modules."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any


def utcnow() -> datetime:
    return datetime.now(UTC)


def slug(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def truncate(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def now_plus(delay_seconds: int) -> datetime:
    return utcnow() + timedelta(seconds=max(0, int(delay_seconds)))

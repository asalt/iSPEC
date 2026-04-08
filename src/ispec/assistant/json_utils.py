from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from plain text or a brace-delimited substring."""
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None

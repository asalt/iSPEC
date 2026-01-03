from __future__ import annotations

import re
from typing import Any


_USER_NAME_RE = re.compile(r"\b(?:my name is|call me)\s+([^.,;:!?\\n]{1,64})", re.IGNORECASE)


def update_state_from_message(state: dict[str, Any], message: str) -> tuple[dict[str, Any], bool]:
    """Extract small, low-risk 'memory' facts from a user message.

    This is intentionally conservative: it only stores simple, explicitly
    stated facts that help the assistant stay oriented.
    """

    changed = False
    text = (message or "").strip()
    if not text:
        return state, False

    match = _USER_NAME_RE.search(text)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        name = re.sub(r"\\s+", " ", name)
        if name and len(name) <= 64:
            if state.get("user_name") != name:
                state["user_name"] = name
                changed = True

    return state, changed


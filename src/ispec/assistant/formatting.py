from __future__ import annotations

import re


_PLAN_MARKER_RE = re.compile(r"(?im)^[ \t]*PLAN:\s*")
_FINAL_MARKER_RE = re.compile(r"(?im)^[ \t]*FINAL:\s*")


def split_plan_final(text: str) -> tuple[str | None, str]:
    """Split an assistant response into (plan, final).

    If no explicit sections are present, returns (None, original text).
    """

    raw = (text or "").strip()
    if not raw:
        return None, ""

    final_matches = list(_FINAL_MARKER_RE.finditer(raw))
    if not final_matches:
        return None, raw

    final_match = final_matches[-1]
    final_text = raw[final_match.end() :].strip()
    if not final_text:
        return None, raw

    plan_text: str | None = None
    plan_matches = [
        match for match in _PLAN_MARKER_RE.finditer(raw) if match.start() < final_match.start()
    ]
    if plan_matches:
        plan_match = plan_matches[-1]
        extracted = raw[plan_match.end() : final_match.start()].strip()
        plan_text = extracted or None

    return plan_text, final_text

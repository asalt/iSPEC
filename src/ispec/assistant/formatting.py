from __future__ import annotations

import re


_PLAN_FINAL_RE = re.compile(
    r"(?is)^\s*PLAN:\s*(?P<plan>.*?)\n\s*FINAL:\s*(?P<final>.*)\s*$"
)
_FINAL_ONLY_RE = re.compile(r"(?is)^\s*FINAL:\s*(?P<final>.*)\s*$")


def split_plan_final(text: str) -> tuple[str | None, str]:
    """Split an assistant response into (plan, final).

    If no explicit sections are present, returns (None, original text).
    """

    raw = (text or "").strip()
    if not raw:
        return None, ""

    match = _PLAN_FINAL_RE.match(raw)
    if match:
        plan = (match.group("plan") or "").strip() or None
        final = (match.group("final") or "").strip()
        if final:
            return plan, final

    match = _FINAL_ONLY_RE.match(raw)
    if match:
        final = (match.group("final") or "").strip()
        if final:
            return None, final

    return None, raw

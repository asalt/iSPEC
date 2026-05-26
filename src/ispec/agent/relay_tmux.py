from __future__ import annotations

from typing import Any

from ispec.agent.relay_constants import (
    FAILURE_TARGET_BLOCKED,
    FAILURE_TARGET_NOT_ALLOWED,
    FAILURE_TMUX_SEND_FAILED,
)


def _tmux_entries(kind: str) -> list[str]:
    try:
        from ispec.assistant import tools as assistant_tools

        if kind == "allow":
            return list(assistant_tools._tmux_allowlist_entries())  # type: ignore[attr-defined]
        return list(assistant_tools._tmux_blacklist_entries())  # type: ignore[attr-defined]
    except Exception:
        return []


def _target_matches_entry(target: str, entry: str) -> bool:
    target_text = str(target or "").strip()
    entry_text = str(entry or "").strip()
    if not target_text or not entry_text:
        return False
    if entry_text.endswith("*"):
        return bool(entry_text[:-1]) and target_text.startswith(entry_text[:-1])
    return target_text == entry_text


def validate_tmux_target(target: str) -> tuple[bool, str | None, dict[str, Any]]:
    allowlist = _tmux_entries("allow")
    blacklist = _tmux_entries("black")
    blacklist_match = next((entry for entry in blacklist if _target_matches_entry(target, entry)), None)
    if blacklist_match:
        return False, FAILURE_TARGET_BLOCKED, {
            "target": target,
            "allowlist_count": len(allowlist),
            "blacklist_match": blacklist_match,
        }
    allowlist_match = next((entry for entry in allowlist if _target_matches_entry(target, entry)), None)
    if not allowlist or not allowlist_match:
        return False, FAILURE_TARGET_NOT_ALLOWED, {
            "target": target,
            "allowlist_count": len(allowlist),
            "blacklist_count": len(blacklist),
        }
    return True, None, {
        "target": target,
        "allowlist_match": allowlist_match,
        "blacklist_count": len(blacklist),
    }


def execute_tmux_send(request: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    target = str((request.get("target") or {}).get("target") or "").strip()
    try:
        from ispec.assistant import tools as assistant_tools

        result = assistant_tools._tmux_send_text(  # type: ignore[attr-defined]
            target=target,
            text=str(request.get("body") or ""),
            press_enter=bool(request.get("press_enter")),
        )
        return True, {"ok": True, "sent": True, "tmux": result}, None
    except Exception as exc:
        return False, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, FAILURE_TMUX_SEND_FAILED

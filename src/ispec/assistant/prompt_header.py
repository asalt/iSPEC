from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from ispec.db.models import UserRole

_TRUTHY = {"1", "true", "yes", "y", "on"}

PROMPT_HEADER_ENV = "ISPEC_ASSISTANT_ENABLE_PROMPT_HEADER"
PROMPT_HEADER_LEGEND_VERSION = 1
PROMPT_HEADER_VERSION = 1


def prompt_header_enabled() -> bool:
    raw = (os.getenv(PROMPT_HEADER_ENV) or "").strip()
    return raw.lower() in _TRUTHY


def _role_abbrev(role: Any | None) -> str:
    """Map user roles to a short, stable abbreviation."""

    try:
        normalized = str(role.value if isinstance(role, UserRole) else role).strip().lower()
    except Exception:
        normalized = ""
    if normalized == "admin":
        return "ad"
    if normalized == "editor":
        return "ed"
    if normalized == "viewer":
        return "vw"
    if normalized == "client":
        return "cl"
    return "an"


_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def base36(n: int | None) -> str:
    if n is None:
        return "-"
    value = int(n)
    if value < 0:
        return "-"
    if value == 0:
        return "0"
    out: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        out.append(_BASE36_ALPHABET[rem])
    return "".join(reversed(out))


def _session_token(session_id: str, *, length: int = 4) -> str:
    raw = (session_id or "").strip()
    if not raw:
        return "----"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[: max(1, int(length))]


OK_BITS: dict[str, int] = {
    # State/data presence
    "has_memory": 0,
    "has_summary": 1,
    "has_ui_route": 2,
    "has_current_project": 3,
    # Identity
    "is_authenticated": 4,
    "is_staff": 5,
    "is_admin": 6,
}

POL_BITS: dict[str, int] = {
    "tools_available": 0,
    "tool_protocol_openai": 1,
    "compare_mode": 2,
    "repo_tools_enabled": 3,
    "forced_tool_choice": 4,
}


def header_legend() -> dict[str, Any]:
    return {
        "legend_version": PROMPT_HEADER_LEGEND_VERSION,
        "ok_bits": dict(OK_BITS),
        "policy_bits": dict(POL_BITS),
    }


@dataclass(frozen=True)
class PromptHeader:
    version: int
    legend_version: int
    fields: dict[str, Any]
    line: str


def build_prompt_header(
    *,
    session_id: str,
    user_role: Any | None,
    user_id: int | None,
    session_state: dict[str, Any] | None,
    user_message_id: int | None,
    tools_available: bool,
    tool_protocol: str,
    compare_mode: bool,
    forced_tool_choice: bool,
    repo_tools_enabled: bool,
) -> PromptHeader:
    state = session_state if isinstance(session_state, dict) else {}
    current_project_id = state.get("current_project_id")
    if not isinstance(current_project_id, int) or current_project_id <= 0:
        current_project_id = None

    has_memory = isinstance(state.get("conversation_memory"), dict) and bool(state.get("conversation_memory"))
    has_summary = isinstance(state.get("conversation_summary"), str) and bool(state.get("conversation_summary").strip())
    has_ui_route = isinstance(state.get("ui_route"), dict) and bool(state.get("ui_route"))

    role_abbrev = _role_abbrev(user_role)
    authenticated = user_id is not None
    is_staff = role_abbrev in {"ad", "ed", "vw"}
    is_admin = role_abbrev == "ad"

    ok_mask = 0
    if has_memory:
        ok_mask |= 1 << OK_BITS["has_memory"]
    if has_summary:
        ok_mask |= 1 << OK_BITS["has_summary"]
    if has_ui_route:
        ok_mask |= 1 << OK_BITS["has_ui_route"]
    if current_project_id is not None:
        ok_mask |= 1 << OK_BITS["has_current_project"]
    if authenticated:
        ok_mask |= 1 << OK_BITS["is_authenticated"]
    if is_staff:
        ok_mask |= 1 << OK_BITS["is_staff"]
    if is_admin:
        ok_mask |= 1 << OK_BITS["is_admin"]

    protocol_openai = str(tool_protocol or "").strip().lower() == "openai"
    pol_mask = 0
    if tools_available:
        pol_mask |= 1 << POL_BITS["tools_available"]
    if protocol_openai:
        pol_mask |= 1 << POL_BITS["tool_protocol_openai"]
    if compare_mode:
        pol_mask |= 1 << POL_BITS["compare_mode"]
    if repo_tools_enabled:
        pol_mask |= 1 << POL_BITS["repo_tools_enabled"]
    if forced_tool_choice:
        pol_mask |= 1 << POL_BITS["forced_tool_choice"]

    fields: dict[str, Any] = {
        "schema_version": PROMPT_HEADER_VERSION,
        "legend_version": PROMPT_HEADER_LEGEND_VERSION,
        "session_id": session_id,
        "session_token": _session_token(session_id),
        "user_id": user_id,
        "user_role": str(user_role.value if isinstance(user_role, UserRole) else user_role) if user_role is not None else None,
        "role_abbrev": role_abbrev,
        "current_project_id": current_project_id,
        "user_message_id": user_message_id,
        "ok_mask": ok_mask,
        "pol_mask": pol_mask,
    }

    line = (
        f"@h{PROMPT_HEADER_VERSION} "
        f"u={role_abbrev} "
        f"s={fields['session_token']} "
        f"p={base36(current_project_id)} "
        f"m={base36(user_message_id)} "
        f"ok={base36(ok_mask)} "
        f"pol={base36(pol_mask)}"
    )

    return PromptHeader(
        version=PROMPT_HEADER_VERSION,
        legend_version=PROMPT_HEADER_LEGEND_VERSION,
        fields=fields,
        line=line,
    )


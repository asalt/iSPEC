from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any


WORK_BAG_STATE_KEY = "work_bag"
WORK_BAG_SCHEMA_VERSION = 1
WORK_BAG_DEFAULT_CAP = 12
WORK_BAG_CONTEXT_LIMIT = 3

_OMITTED = ["raw_arguments", "raw_result", "prompt_text"]


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _safe_str(value: Any, *, max_len: int = 240) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _parse_result_preview(call: dict[str, Any]) -> dict[str, Any]:
    preview = call.get("result_preview")
    if not isinstance(preview, str) or not preview.strip():
        return {}
    try:
        parsed = json.loads(preview)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _status_for_call(call: dict[str, Any]) -> str:
    if bool(call.get("ok")):
        return "succeeded"
    error = str(call.get("error") or "").lower()
    blocked_needles = (
        "confirm=true is required",
        "not authenticated",
        "not accessible",
        "not authorized",
        "did not explicitly request",
        "requested not to save",
        "not to save yet",
    )
    if any(needle in error for needle in blocked_needles):
        return "blocked"
    return "failed"


def _kind_for_tool(tool_name: str) -> str:
    if tool_name == "create_project_comment":
        return "write"
    if "slack" in tool_name or "relay" in tool_name:
        return "relay"
    if "enqueue" in tool_name or tool_name.startswith("assistant_upsert_scheduled_job"):
        return "job"
    return "tool_call"


def _refs_for_result(*, tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    project_id = _safe_int(result.get("project_id"))
    if project_id is not None and project_id > 0:
        refs.append({"kind": "project", "id": int(project_id)})
    if tool_name == "create_project_comment":
        comment_id = _safe_int(result.get("comment_id"))
        if comment_id is not None and comment_id > 0:
            refs.append({"kind": "project_comment", "id": int(comment_id)})
        person_id = _safe_int(result.get("person_id"))
        if person_id is not None and person_id > 0:
            refs.append({"kind": "person", "id": int(person_id)})
        legacy_push = result.get("legacy_push_enqueue")
        if isinstance(legacy_push, dict):
            command_id = _safe_int(legacy_push.get("command_id"))
            if command_id is not None and command_id > 0:
                refs.append({"kind": "agent_command", "id": int(command_id)})
    command_id = _safe_int(result.get("command_id"))
    if command_id is not None and command_id > 0:
        refs.append({"kind": "agent_command", "id": int(command_id)})
    return refs


def _summary_for_call(*, tool_name: str, status: str, result: dict[str, Any], error: Any) -> str:
    if tool_name == "create_project_comment":
        project_id = _safe_int(result.get("project_id"))
        comment_id = _safe_int(result.get("comment_id"))
        snippet = _safe_str(result.get("snippet"), max_len=160)
        if status == "succeeded" and project_id and comment_id:
            base = f"Saved project comment {comment_id} on project {project_id}."
            return f"{base} {snippet}" if snippet else base
        if project_id:
            return f"Project comment write for project {project_id} {status}."
        return f"Project comment write {status}."
    if status == "succeeded":
        return f"{tool_name} succeeded."
    err = _safe_str(error, max_len=160)
    return f"{tool_name} {status}: {err}" if err else f"{tool_name} {status}."


def build_work_bag_entries_from_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    user_message_id: int | None,
    assistant_message_id: int | None,
    created_at: str | None = None,
    source: str = "support_chat",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    timestamp = created_at or utcnow_iso()
    for index, call in enumerate(tool_calls, start=1):
        if not isinstance(call, dict):
            continue
        tool_name = _safe_str(call.get("name"), max_len=120)
        if not tool_name:
            continue
        status = _status_for_call(call)
        result = _parse_result_preview(call)
        refs = _refs_for_result(tool_name=tool_name, result=result)
        error = _safe_str(call.get("error"), max_len=240)
        entry_id_parts = ["wb"]
        if assistant_message_id is not None:
            entry_id_parts.append(f"a{int(assistant_message_id)}")
        elif user_message_id is not None:
            entry_id_parts.append(f"u{int(user_message_id)}")
        entry_id_parts.append(f"t{index}")
        entry = {
            "entry_id": ":".join(entry_id_parts),
            "created_at": timestamp,
            "source": source,
            "user_message_id": int(user_message_id) if user_message_id is not None else None,
            "assistant_message_id": int(assistant_message_id) if assistant_message_id is not None else None,
            "kind": _kind_for_tool(tool_name),
            "tool_name": tool_name,
            "status": status,
            "refs": refs,
            "exit_code": _safe_int(result.get("exit_code")),
            "error": error,
            "summary": _summary_for_call(tool_name=tool_name, status=status, result=result, error=error),
            "omitted": list(_OMITTED),
        }
        entries.append(entry)
    return entries


def normalize_work_bag(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"schema_version": WORK_BAG_SCHEMA_VERSION, "entries": []}
    raw_entries = value.get("entries")
    entries = [entry for entry in raw_entries if isinstance(entry, dict)] if isinstance(raw_entries, list) else []
    return {
        "schema_version": WORK_BAG_SCHEMA_VERSION,
        "entries": entries[-WORK_BAG_DEFAULT_CAP:],
    }


def append_work_bag_entries(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    cap: int = WORK_BAG_DEFAULT_CAP,
) -> bool:
    clean_entries = [entry for entry in entries if isinstance(entry, dict)]
    if not clean_entries:
        return False
    bag = normalize_work_bag(state.get(WORK_BAG_STATE_KEY))
    existing = list(bag.get("entries") or [])
    existing.extend(clean_entries)
    cap = max(1, int(cap or WORK_BAG_DEFAULT_CAP))
    state[WORK_BAG_STATE_KEY] = {
        "schema_version": WORK_BAG_SCHEMA_VERSION,
        "entries": existing[-cap:],
    }
    return True


def work_bag_context_summary(
    state: dict[str, Any],
    *,
    recent_limit: int = WORK_BAG_CONTEXT_LIMIT,
) -> dict[str, Any] | None:
    bag = normalize_work_bag(state.get(WORK_BAG_STATE_KEY))
    entries = list(bag.get("entries") or [])
    if not entries:
        return None
    recent_limit = max(1, int(recent_limit or WORK_BAG_CONTEXT_LIMIT))
    counts = Counter(str(entry.get("status") or "unknown") for entry in entries if isinstance(entry, dict))
    return {
        "schema_version": WORK_BAG_SCHEMA_VERSION,
        "entry_count": len(entries),
        "status_counts": dict(sorted(counts.items())),
        "recent_entries": entries[-recent_limit:],
        "full_available_via_tool": "assistant_recent_session_work_bag",
    }


def recent_work_bag_payload(
    state: dict[str, Any],
    *,
    session_id: str | None,
    limit: int = WORK_BAG_DEFAULT_CAP,
) -> dict[str, Any]:
    bag = normalize_work_bag(state.get(WORK_BAG_STATE_KEY))
    entries = list(bag.get("entries") or [])
    limit = max(1, min(50, int(limit or WORK_BAG_DEFAULT_CAP)))
    return {
        "schema_version": WORK_BAG_SCHEMA_VERSION,
        "session_id": session_id,
        "limit": limit,
        "total_entries": len(entries),
        "entries": list(reversed(entries[-limit:])),
    }

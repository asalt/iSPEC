"""Bounded, server-recorded native tool exchanges for conversational replay."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.orm import Session

from ispec.assistant.models import SupportMessage
from ispec.assistant.prompting import estimate_tokens_for_messages
from ispec.authz import uses_explicit_project_access
from ispec.db.models import AuthUser, AuthUserProject

Message = dict[str, Any]
MAX_RETAINED_TOKENS = 4096


def access_key(*, db: Session, user: AuthUser | None, actor: str, tool_names: set[str]) -> str:
    """Bind historical tool data to the original actor, grants and authorized catalog."""
    grants = None
    if uses_explicit_project_access(user):
        rows = db.query(AuthUserProject.project_id).filter_by(user_id=user.id)
        grants = sorted(project_id for (project_id,) in rows)
    scope = [actor, getattr(user, "role", None), grants, sorted(tool_names)]
    return hashlib.sha256(json.dumps(scope).encode()).hexdigest()


def complete_exchanges(messages: list[Message]) -> list[list[Message]]:
    """Collect native calls together with every matching result, preserving wire order."""
    groups = []
    group: list[Message] = []
    remaining: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            group = []
            calls = message.get("tool_calls") or []
            ids = [call.get("id") for call in calls if isinstance(call, dict)]
            if ids and all(isinstance(i, str) and i for i in ids) and len(ids) == len(calls) == len(set(ids)):
                group = [message]
                remaining = set(ids)
        elif group and message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in remaining:
                group = []
                continue
            group.append(message)
            remaining.remove(call_id)
            if not remaining:
                groups.append(group)
                group = []
    return groups


def recent_exchanges(groups: list[list[Message]], *, budget: int) -> list[Message]:
    """Return the newest complete exchanges that fit, in chronological order."""
    selected = []
    for group in reversed(groups):
        cost = estimate_tokens_for_messages(group)
        if cost > budget:
            break
        budget -= cost
        selected.append(group)
    return [message for group in reversed(selected) for message in group]


def _receipt_for(row: SupportMessage, *, scope: str, user_message_id: int) -> list[Message]:
    """Read a matching server receipt from the original assistant message."""
    try:
        meta = json.loads(row.meta_json or "{}")
    except (ValueError, TypeError):
        return []
    receipt = meta.get("native_history") if isinstance(meta, dict) else None
    if not isinstance(receipt, dict):
        return []
    if receipt.get("scope") != scope or receipt.get("user_message_id") != user_message_id:
        return []
    messages = receipt.get("messages")
    return messages if isinstance(messages, list) and all(isinstance(m, dict) for m in messages) else []


def replay_history(
    rows: list[SupportMessage], *, scope: str, remaining_tokens: int,
) -> list[Message]:
    """Add receipts before their delivered reply if the originating user turn is retained."""
    blocks = []
    for index in reversed(range(len(rows))):
        row = rows[index]
        prior = rows[index - 1] if index else None
        messages = []
        if row.role == "assistant" and prior is not None and prior.role == "user":
            recorded = _receipt_for(row, scope=scope, user_message_id=prior.id)
            messages = recent_exchanges(complete_exchanges(recorded), budget=remaining_tokens)
            remaining_tokens -= estimate_tokens_for_messages(messages)
        blocks.append([*messages, {"role": row.role, "content": row.content}])
    return [message for block in reversed(blocks) for message in block]

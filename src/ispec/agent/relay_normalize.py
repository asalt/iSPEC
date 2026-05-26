from __future__ import annotations

import uuid
from typing import Any

from ispec.agent.relay_constants import (
    FAILURE_INVALID_REQUEST,
    FAILURE_UNSUPPORTED_KIND,
    RELAY_SCHEMA_VERSION,
    SUPPORTED_KINDS,
)
from ispec.agent.relay_utils import slug, truncate


def _normalize_source(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": truncate(value.get("kind") or value.get("type") or "local", limit=64),
            "id": truncate(value.get("id") or value.get("agent_id") or value.get("name") or "", limit=256) or None,
            "cwd": truncate(value.get("cwd") or "", limit=1000) or None,
        }
    text = truncate(value, limit=256)
    return {"kind": "local", "id": text or None, "cwd": None}


def _normalize_target(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        alias = value.get("alias") or value.get("to") or value.get("name")
        tmux_target = value.get("target") or value.get("tmux_target")
        normalized = {
            "alias": truncate(alias, limit=120) or None,
            "target": truncate(tmux_target, limit=240) or None,
        }
        if value.get("channel"):
            normalized["channel"] = truncate(value.get("channel"), limit=120)
        if value.get("user_id"):
            normalized["user_id"] = truncate(value.get("user_id"), limit=120)
        return normalized
    text = truncate(value, limit=240)
    return {"alias": text or None, "target": text or None}


def normalize_relay_request(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, FAILURE_INVALID_REQUEST

    nested = raw.get("relay_request") if isinstance(raw.get("relay_request"), dict) else raw
    kind = slug(str(nested.get("kind") or nested.get("type") or ""))
    if kind not in SUPPORTED_KINDS:
        return None, FAILURE_UNSUPPORTED_KIND

    body = nested.get("body")
    if body is None:
        body = nested.get("text") or nested.get("message")
    body_text = truncate(body, limit=20_000)

    target = _normalize_target(nested.get("target") if "target" in nested else nested.get("to"))
    mode = slug(str(nested.get("mode") or "stage"))
    if mode not in {"stage", "send"}:
        mode = "stage"

    attachments_raw = nested.get("attachments")
    if attachments_raw is None and nested.get("attachment"):
        attachments_raw = [nested.get("attachment")]
    attachments: list[dict[str, Any]] = []
    if isinstance(attachments_raw, list):
        for item in attachments_raw[:20]:
            if isinstance(item, dict):
                path = truncate(item.get("path") or item.get("file"), limit=2000)
                title = truncate(item.get("title"), limit=240) or None
            else:
                path = truncate(item, limit=2000)
                title = None
            if path:
                attachments.append({"path": path, "title": title})

    request_id = truncate(nested.get("request_id") or nested.get("id"), limit=128)
    if not request_id:
        request_id = f"relay-{uuid.uuid4().hex[:16]}"

    normalized = {
        "schema_version": RELAY_SCHEMA_VERSION,
        "request_id": request_id,
        "kind": kind,
        "mode": mode,
        "confirm": bool(nested.get("confirm") is True),
        "source": _normalize_source(nested.get("source") or nested.get("origin") or "local"),
        "target": target,
        "body": body_text,
        "attachments": attachments,
        "message_type": slug(nested.get("message_type")) or None,
        "thread_ts": truncate(nested.get("thread_ts"), limit=80) or None,
        "press_enter": bool(nested.get("press_enter") is True),
        "metadata": nested.get("metadata") if isinstance(nested.get("metadata"), dict) else {},
        "provenance": nested.get("provenance") if isinstance(nested.get("provenance"), dict) else {},
    }
    return normalized, None

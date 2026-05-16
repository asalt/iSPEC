from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ispec.agent.models import AgentEvent


EVENT_SLACK_ARTIFACT_SENT = "slack_artifact_sent_v1"
EVENT_SLACK_ARTIFACT_REPLY = "slack_artifact_reply_received_v1"
EVENT_SLACK_TMUX_RELAY_SENT = "slack_tmux_relay_sent_v1"

BRIDGE_AGENT_ID = "slack-tmux-bridge"


def utcnow() -> datetime:
    return datetime.now(UTC)


def stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def parse_event_payload(row: AgentEvent | None) -> dict[str, Any]:
    if row is None:
        return {}
    raw = getattr(row, "payload_json", "") or ""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def sha256_file(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_slack_ts(value: Any) -> str:
    return str(value or "").strip()


def slack_thread_key(*, channel: Any, thread_ts: Any) -> str:
    channel_text = str(channel or "").strip()
    thread_text = normalize_slack_ts(thread_ts)
    return f"{channel_text}:{thread_text}" if channel_text and thread_text else ""


def _candidate_share_ts(value: Any, *, channel: str) -> str:
    if not isinstance(value, dict):
        return ""
    shares = value.get("shares")
    if not isinstance(shares, dict):
        return ""
    for bucket_name in ("private", "public"):
        bucket = shares.get(bucket_name)
        if not isinstance(bucket, dict):
            continue
        channel_rows = bucket.get(channel)
        if not isinstance(channel_rows, list):
            continue
        for row in channel_rows:
            if isinstance(row, dict):
                ts = normalize_slack_ts(row.get("ts") or row.get("thread_ts"))
                if ts:
                    return ts
    return ""


def extract_upload_thread_ts(
    *,
    upload_result: dict[str, Any],
    fallback_thread_ts: str | None = None,
) -> str:
    fallback = normalize_slack_ts(fallback_thread_ts)
    if fallback:
        return fallback

    resolved = upload_result.get("resolved") if isinstance(upload_result.get("resolved"), dict) else {}
    channel = str(resolved.get("channel") or upload_result.get("channel") or "").strip()
    slack_payload = upload_result.get("slack") if isinstance(upload_result.get("slack"), dict) else {}

    for key_path in (("message",), ("file",)):
        node: Any = slack_payload
        for key in key_path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            ts = normalize_slack_ts(node.get("thread_ts") or node.get("ts"))
            if ts:
                return ts
            share_ts = _candidate_share_ts(node, channel=channel)
            if share_ts:
                return share_ts

    files = slack_payload.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            ts = normalize_slack_ts(item.get("thread_ts") or item.get("ts"))
            if ts:
                return ts
            share_ts = _candidate_share_ts(item, channel=channel)
            if share_ts:
                return share_ts

    return ""


def build_artifact_sent_payload(
    *,
    upload_result: dict[str, Any],
    artifact_id: str,
    origin_tmux: dict[str, Any] | None = None,
    thread_ts: str | None = None,
    submit_allowed: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    resolved = upload_result.get("resolved") if isinstance(upload_result.get("resolved"), dict) else {}
    file_info = upload_result.get("file") if isinstance(upload_result.get("file"), dict) else {}
    file_path = str(file_info.get("path") or "").strip()
    file_sha256 = ""
    if file_path:
        try:
            file_sha256 = sha256_file(file_path)
        except Exception:
            file_sha256 = ""

    resolved_thread_ts = extract_upload_thread_ts(upload_result=upload_result, fallback_thread_ts=thread_ts)
    channel = str(resolved.get("channel") or "").strip()
    return {
        "type": EVENT_SLACK_ARTIFACT_SENT,
        "artifact_id": str(artifact_id).strip(),
        "created_at": utcnow().isoformat(),
        "file": {
            "path": file_path or None,
            "filename": file_info.get("filename"),
            "size_bytes": file_info.get("size_bytes"),
            "mime_type": file_info.get("mime_type"),
            "sha256": file_sha256 or None,
        },
        "slack": {
            "channel": channel or None,
            "thread_ts": resolved_thread_ts or None,
            "file_id": upload_result.get("file_id"),
            "resolved": resolved,
        },
        "origin_tmux": origin_tmux or {},
        "routing": {
            "submit_allowed": bool(submit_allowed),
        },
        "note": str(note).strip() if note else None,
    }


def find_artifact_receipt_for_thread(
    db: Session,
    *,
    channel: str,
    thread_ts: str,
    limit: int = 500,
) -> tuple[AgentEvent, dict[str, Any]] | None:
    channel_text = str(channel or "").strip()
    thread_text = normalize_slack_ts(thread_ts)
    if not channel_text or not thread_text:
        return None

    rows = (
        db.query(AgentEvent)
        .filter(AgentEvent.event_type == EVENT_SLACK_ARTIFACT_SENT)
        .order_by(AgentEvent.id.desc())
        .limit(max(1, int(limit)))
        .all()
    )
    for row in rows:
        payload = parse_event_payload(row)
        receipt_payload = payload.get("value") if isinstance(payload.get("value"), dict) else payload
        slack = receipt_payload.get("slack") if isinstance(receipt_payload.get("slack"), dict) else {}
        if str(slack.get("channel") or "").strip() != channel_text:
            continue
        candidate_ts = normalize_slack_ts(slack.get("thread_ts") or slack.get("message_ts"))
        if candidate_ts == thread_text:
            return row, receipt_payload
    return None


def build_artifact_reply_payload(
    *,
    receipt_event: AgentEvent,
    receipt_payload: dict[str, Any],
    slack: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    artifact_id = str(receipt_payload.get("artifact_id") or "").strip()
    return {
        "type": EVENT_SLACK_ARTIFACT_REPLY,
        "artifact_id": artifact_id or None,
        "receipt_event_id": int(receipt_event.id),
        "received_at": utcnow().isoformat(),
        "slack": slack,
        "text": str(text or "").strip(),
        "text_sha256": hashlib.sha256(str(text or "").encode("utf-8")).hexdigest(),
        "origin_tmux": receipt_payload.get("origin_tmux") if isinstance(receipt_payload.get("origin_tmux"), dict) else {},
        "routing": receipt_payload.get("routing") if isinstance(receipt_payload.get("routing"), dict) else {},
    }


def relayed_reply_event_ids(db: Session, *, limit: int = 1000) -> set[int]:
    rows = (
        db.query(AgentEvent)
        .filter(AgentEvent.event_type == EVENT_SLACK_TMUX_RELAY_SENT)
        .order_by(AgentEvent.id.desc())
        .limit(max(1, int(limit)))
        .all()
    )
    ids: set[int] = set()
    for row in rows:
        payload = parse_event_payload(row)
        raw_id = payload.get("reply_event_id")
        try:
            reply_id = int(raw_id)
        except Exception:
            continue
        if reply_id > 0:
            ids.add(reply_id)
    return ids


def recent_artifact_replies(
    db: Session,
    *,
    limit: int = 20,
    include_relayed: bool = False,
) -> list[dict[str, Any]]:
    rows = (
        db.query(AgentEvent)
        .filter(AgentEvent.event_type == EVENT_SLACK_ARTIFACT_REPLY)
        .order_by(AgentEvent.id.desc())
        .limit(max(1, min(int(limit) * 4, 500)))
        .all()
    )
    relayed_ids = set() if include_relayed else relayed_reply_event_ids(db)
    replies: list[dict[str, Any]] = []
    for row in rows:
        if int(row.id) in relayed_ids:
            continue
        payload = parse_event_payload(row)
        payload["reply_event_id"] = int(row.id)
        payload["event_ts"] = row.ts.isoformat() if getattr(row, "ts", None) else None
        payload["received_at_db"] = row.received_at.isoformat() if getattr(row, "received_at", None) else None
        payload["already_relayed"] = int(row.id) in relayed_ids
        replies.append(payload)
        if len(replies) >= limit:
            break
    return replies


def format_tmux_relay_message(
    *,
    reply_payload: dict[str, Any],
    prefix: str = "Slack review",
    max_chars: int = 4000,
) -> str:
    slack = reply_payload.get("slack") if isinstance(reply_payload.get("slack"), dict) else {}
    user_label = (
        str(slack.get("user_display_name") or "").strip()
        or str(slack.get("user_real_name") or "").strip()
        or str(slack.get("user_name") or "").strip()
        or str(slack.get("user_id") or "").strip()
        or "Slack user"
    )
    artifact_id = str(reply_payload.get("artifact_id") or "").strip()
    text = str(reply_payload.get("text") or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    header_bits = [prefix, f"from {user_label}"]
    if artifact_id:
        header_bits.append(f"artifact {artifact_id}")
    ts = str(slack.get("message_ts") or "").strip()
    if ts:
        header_bits.append(f"slack_ts {ts}")
    return f"[{'; '.join(header_bits)}]\n{text}".strip()

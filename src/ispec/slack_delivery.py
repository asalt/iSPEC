"""Small send-receipt adapter; safe to load from the standalone Slack helper.

No provider responses, credentials, private download URLs or file bytes are stored.
A persistence failure must never turn a confirmed send into a resend request.
"""

from datetime import datetime, timezone
import re

import requests


def persist_local(receipt):
    from ispec.assistant.connect import get_assistant_session
    from ispec.assistant.slack_outbound import OutboundReceipt, store_outbound

    with get_assistant_session() as db:
        return store_outbound(db, OutboundReceipt.model_validate(receipt))


def persist_http(receipt, *, server, api_key, timeout=5):
    if not server or not api_key:
        return {"retained": False, "error": "missing_receipt_endpoint_or_key"}
    response = requests.post(
        server.rstrip("/") + "/api/agents/slack/outbound",
        headers={"X-API-Key": api_key}, json=receipt, timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    return {key: result.get(key) for key in ("retained", "record_id", "duplicate")}


def log_delivery(*, response, channel, text, source, thread_ts=None, recipient_id=None,
                 attachments=(), identity=None, lookup_identity=None, persist=None, secrets=()):
    if response.get("ok") is not True:
        return {"retained": False, "error": "delivery_not_confirmed"}
    try:
        identity = identity or {}
        message = response.get("message") or {}
        team_id = identity.get("team_id") or message.get("team") or response.get("team_id")
        if not team_id and lookup_identity:
            identity = lookup_identity()
            team_id = identity.get("team_id")
        if not team_id:
            return {"retained": False, "error": "missing_workspace_identity", "sent": True}
        message_ts = response.get("ts") or message.get("ts")
        thread_ts = thread_ts or message.get("thread_ts")
        # File completion often only exposes its message timestamp under shares.
        for file in response.get("files") or []:
            for shares in (file.get("shares") or {}).values():
                for share in shares.get(channel, []):
                    if thread_ts and share.get("thread_ts") not in {None, thread_ts}:
                        continue
                    message_ts = message_ts or share.get("ts")
                    thread_ts = thread_ts or share.get("thread_ts")
        def clean(value):
            if not isinstance(value, str):
                return value
            for secret in secrets:
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            return re.sub(r"xox[a-z]-[A-Za-z0-9-]+", "[REDACTED]", value)

        refs = [{key: clean(value) for key, value in item.items() if key in {
            "file_id", "filename", "title", "mime_type", "size_bytes",
        }} for item in attachments]
        sent_at = (datetime.fromtimestamp(float(message_ts), timezone.utc)
                   if message_ts else datetime.now(timezone.utc))
        receipt = {
            "team_id": team_id, "channel": channel,
            "message_ts": message_ts, "thread_ts": thread_ts,
            "author_id": message.get("user") or identity.get("user_id") or message.get("bot_id") or identity.get("bot_id"),
            "recipient_id": recipient_id, "text": clean(text or ""),
            "source": source, "attachments": refs, "sent_at": sent_at.isoformat(),
        }
        return (persist or persist_local)(receipt)
    except Exception as exc:
        return {"retained": False, "sent": True, "error": "receipt_logging_failed",
                "error_type": type(exc).__name__}

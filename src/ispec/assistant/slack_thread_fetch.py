"""One read of an event's exact Slack thread, never a message or approval trigger."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from ispec.assistant.slack_thread_context import (
    MAX_MESSAGES, SlackThreadContext, ThreadMessage, bound_thread_context,
)

PAGE_SIZE = 100
TIMEOUT_SECONDS = 5


def fetch_thread_context(context: SlackThreadContext, *, client) -> SlackThreadContext:
    """Caller must first validate the inbound receipt and authorize its requester."""
    scope = context.scope
    if context.status != "loaded" or not scope.thread_ts or scope.thread_ts == scope.message_ts:
        return context
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        # One bounded page, no automatic pagination/retry, and no other conversation.
        response = client.conversations_replies(
            channel=scope.channel, ts=scope.thread_ts, latest=scope.message_ts,
            inclusive=False, limit=PAGE_SIZE,
        )
        if response.get("ok") is not True or not isinstance(response.get("messages"), list):
            raise ValueError("Invalid thread response")
        if not isinstance(response.get("response_metadata") or {}, dict):
            raise ValueError("Invalid thread pagination")
    except Exception as exc:
        error_response = getattr(exc, "response", None)
        error = error_response.get("error") if hasattr(error_response, "get") else None
        if error not in {"ratelimited", "missing_scope", "not_in_channel", "channel_not_found",
                         "thread_not_found", "invalid_auth", "token_revoked"}:
            error = "thread_fetch_failed"
        return bound_thread_context(context.model_copy(update={
            "fetch_status": "failed", "fetch_error": error, "fetched_at": stamp,
        }))

    partial = bool(response.get("has_more") or (response.get("response_metadata") or {}).get("next_cursor"))
    messages = {}
    for row in response["messages"][:PAGE_SIZE]:
        try:
            ts = row["ts"]
            if not Decimal(scope.thread_ts) <= Decimal(ts) < Decimal(scope.message_ts):
                continue
            if ts != scope.thread_ts and row.get("thread_ts") != scope.thread_ts:
                partial = True
                continue
            # A later edit cannot stand in for the original text at request time.
            edited = row.get("edited") or {}
            if not isinstance(edited, dict):
                partial = True
                continue
            revision = edited.get("ts", "")
            if revision and Decimal(revision) >= Decimal(scope.message_ts):
                partial = True
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text or row.get("subtype") in {"message_deleted", "tombstone"}:
                partial = True
                continue
            messages[ts] = ThreadMessage(source="slack_api", user_id=row.get("user") or row.get("bot_id") or "unknown",
                message_ts=ts, revision_ts=revision, text=text[:2000], truncated=len(text) > 2000)
        except (KeyError, TypeError, ValueError, InvalidOperation, ValidationError):
            partial = True
    partial = partial or len(response["messages"]) > PAGE_SIZE or scope.thread_ts not in messages
    # A complete API snapshot supersedes local history (including deleted messages).
    # Partial snapshots can retain known local evidence, with its origin explicit.
    if partial:
        messages = {**{row.message_ts: row for row in context.messages}, **messages}
    rows = sorted(messages.values(), key=lambda row: Decimal(row.message_ts))
    truncated = partial or len(rows) > MAX_MESSAGES
    if len(rows) > MAX_MESSAGES:
        root = messages.get(scope.thread_ts)
        rows = ([root] if root else []) + rows[-(MAX_MESSAGES - bool(root)):]
    source = "slack_api_with_local" if any(row.source == "local_slack_inbox" for row in rows) else "slack_api"
    return bound_thread_context(SlackThreadContext(
        scope=scope, messages=tuple(rows), participants=context.participants,
        source=source, coverage="partial_thread_snapshot" if partial else "thread_snapshot",
        fetch_status="loaded", fetched_at=stamp, truncated=truncated,
    ))

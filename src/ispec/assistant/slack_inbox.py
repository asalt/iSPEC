"""Durable local capture, not a queue authorizing tool writes or automatic retries."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path

from sqlalchemy import select, text, update
from sqlalchemy.dialects.sqlite import insert

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SlackInboxMessage, utcnow
from ispec.assistant.reply_delivery import RETAINED_REPLY, failed_request_reply
from ispec.assistant.slack_outbound import ConversationScope
from ispec.assistant.slack_thread_context import SlackThreadContext, read_thread_context
from ispec.logging import get_logger

logger = get_logger(__name__)


def load_capture_channels(path: str | None) -> frozenset[tuple[str, str]]:
    """Inbound capture permission is separate from outbound destination permission."""
    if not path:
        return frozenset()
    with Path(path).expanduser().open("rb") as handle:
        raw = handle.read(32769)
    if len(raw) > 32768:
        raise ValueError("Slack inbox configuration exceeds 32 KiB")
    cfg = json.loads(raw)
    if not isinstance(cfg, dict) or set(cfg) != {"channels"} or not isinstance(cfg["channels"], list):
        raise ValueError("Slack inbox configuration needs a channels list")
    if len(cfg["channels"]) > 100:
        raise ValueError("Too many Slack inbox channels")
    channels = set()
    for row in cfg["channels"]:
        if not isinstance(row, dict) or set(row) != {"team_id", "channel_id"}:
            raise ValueError("Each inbox channel needs team_id and channel_id")
        if any(not isinstance(v, str) or not v.isascii() or not v.isalnum() for v in row.values()):
            raise ValueError("Use exact Slack workspace/channel IDs, not names or wildcards")
        channels.add((row["team_id"], row["channel_id"]))
    return frozenset(channels)


class SlackInbox:
    def __init__(self, *, file=None, channels=frozenset()):
        self.file = file
        self.channels = channels
        # Initialize schema before accepting Socket Mode envelopes, not on first ACK.
        with self.session():
            pass

    @contextmanager
    def session(self):
        with get_assistant_session(self.file) as db:
            # Do not hold an envelope for a long database-lock timeout.
            previous_timeout = int(db.scalar(text("PRAGMA busy_timeout")))
            db.execute(text("PRAGMA busy_timeout=1000"))
            try:
                yield db
            finally:
                # SQLite PRAGMAs belong to the pooled connection, not this session.
                db.execute(text(f"PRAGMA busy_timeout={previous_timeout}"))

    def receive(self, body, context) -> int | None:
        event = body.get("event", {})
        kind = event.get("type")
        if kind not in {"message", "app_mention"}:
            return None
        team = context.get("team_id") or body.get("team_id")
        channel = event.get("channel")
        bot_id = context.get("bot_user_id")
        message = event.get("message", {}) if event.get("subtype") == "message_changed" else event
        if (not isinstance(message, dict) or message.get("bot_id")
                or message.get("user") == bot_id or event.get("bot_id")):
            return None
        if event.get("subtype") not in {None, "file_share", "message_changed"}:
            return None
        direct = event.get("channel_type") in {"im", "mpim"}
        mentioned = kind == "app_mention" or (bot_id and f"<@{bot_id}>" in str(message.get("text", "")))
        if not (direct or mentioned or (team, channel) in self.channels):
            return None
        stamp, user = message.get("ts"), message.get("user")
        if not all(isinstance(v, str) and v for v in (team, channel, stamp, user)):
            return None
        content = message.get("text") or ""
        if not isinstance(content, str) or not content:
            return None  # This inbox captures text, not file bytes or reaction events.
        revision = (message.get("edited") or {}).get("ts", "")
        if event.get("subtype") == "message_changed" and not revision:
            return None  # Never silently overwrite the original with an unversioned edit.
        key = dict(team_id=team, channel=channel, message_ts=stamp, revision_ts=revision)
        with self.session() as db:
            db.execute(insert(SlackInboxMessage).values(
                **key, thread_ts=message.get("thread_ts"), user_id=user,
                event_id=body.get("event_id"), text=content[:40000], truncated=len(content) > 40000,
            ).on_conflict_do_nothing(index_elements=list(key)))
            row_id = db.scalar(select(SlackInboxMessage.id).filter_by(**key))
        return int(row_id)  # Context sees the ID only after a successful COMMIT.

    def middleware(self, body, context, next):
        from slack_bolt.response import BoltResponse

        try:
            row_id = self.receive(body, context)
        except Exception as exc:
            logger.error("Slack inbox persistence failed error_type=%s", type(exc).__name__)
            # The Socket Mode adapter ACKs only a 200. No fake retention or reply.
            return BoltResponse(status=503, body="Local message storage unavailable")
        if row_id is not None:
            context["slack_inbox_id"] = row_id
        return next()

    def thread_context(self, meta):
        try:
            scope = ConversationScope.model_validate(meta["slack"])
        except ValueError:
            return None
        if not scope.thread_ts:
            return None
        try:
            with self.session() as db:
                context = read_thread_context(db, scope=scope, inbox_id=meta.get("slack_inbox_id"))
        except Exception as exc:
            logger.warning("Slack thread context unavailable error_type=%s", type(exc).__name__)
            context = SlackThreadContext(scope=scope, status="unavailable")
        return context.model_dump(mode="json")

    def claim(self, context, *, fallback, team_id, event, route) -> bool:
        row_id = context.get("slack_inbox_id")
        if row_id is None:
            return fallback.claim(team_id=team_id, event=event, route=route)
        with self.session() as db:
            result = db.execute(update(SlackInboxMessage).where(
                SlackInboxMessage.id == row_id,
                SlackInboxMessage.processing_started_at.is_(None),
            ).values(processing_started_at=utcnow()))
            claimed = result.rowcount == 1
        context["slack_inbox_claimed"] = claimed
        return claimed

    def handler(self, fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            context = kwargs.get("context", {})
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                self.record_error(context, type(exc).__name__)
                raise
            finally:
                if context.get("slack_inbox_claimed"):
                    self._record_outcome(context["slack_inbox_id"], handled_at=utcnow())
        return wrapped

    def _update(self, row_id, **fields):
        with self.session() as db:
            db.execute(update(SlackInboxMessage).where(SlackInboxMessage.id == row_id).values(**fields))

    def _record_outcome(self, row_id, **fields):
        try:
            self._update(row_id, **fields)
        except Exception as exc:
            # Capture already committed. Leave incomplete evidence rather than
            # hide the original error or prevent an honest retained-text reply.
            logger.error("Slack inbox outcome update failed id=%s error_type=%s", row_id, type(exc).__name__)

    def record_error(self, context, error_type):
        if context.get("slack_inbox_id"):
            self._record_outcome(context["slack_inbox_id"], error_type=error_type[:100])

    def failed_reply(self, context, exc):
        self.record_error(context, type(exc).__name__)
        return RETAINED_REPLY if context.get("slack_inbox_id") else failed_request_reply(exc)

    def backend_result(self, context, payload):
        if payload.get("responseStatus") == "unavailable":
            self.record_error(context, "backend_unavailable")

    def records(self, *, channel=None, unreviewed=True, limit=50, before_id=None, row_id=None):
        if not 1 <= limit <= 200:
            raise ValueError("Inbox limit must be between 1 and 200")
        with self.session() as db:
            query = select(SlackInboxMessage).order_by(SlackInboxMessage.id.desc()).limit(limit)
            if unreviewed:
                query = query.where(SlackInboxMessage.reviewed_at.is_(None))
            if channel:
                query = query.where(SlackInboxMessage.channel == channel)
            if before_id is not None:
                query = query.where(SlackInboxMessage.id < before_id)
            if row_id is not None:
                query = query.where(SlackInboxMessage.id == row_id)
            return [{col.name: getattr(row, col.name) for col in SlackInboxMessage.__table__.columns}
                    for row in db.scalars(query)]

    def review(self, row_id, *, actor):
        if not actor.strip() or len(actor) > 100:
            raise ValueError("A reviewer name of 1-100 characters is required")
        with self.session() as db:
            row = db.get(SlackInboxMessage, row_id)
            if row is None:
                raise ValueError("Inbox record not found")
            if row.reviewed_at is None:
                row.reviewed_at, row.reviewed_by = utcnow(), actor.strip()

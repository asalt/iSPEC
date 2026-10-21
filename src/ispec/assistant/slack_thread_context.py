"""Bounded, locally retained Slack thread evidence, never write authorization."""

from dataclasses import dataclass, field
from decimal import Decimal
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import Numeric, cast, func, or_, select
from sqlalchemy.orm import Session

from ispec.assistant.models import SlackInboxMessage
from ispec.assistant.prompting import estimate_tokens_for_messages
from ispec.assistant.slack_outbound import ConversationScope
from ispec.assistant.slack_participants import MAX_PARTICIPANTS, SlackParticipant
from ispec.prompt import load_bound_prompt, prompt_binding

MAX_MESSAGES = 24
MAX_CHARS = 8000


class ThreadMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    inbox_id: int
    user_id: str = Field(max_length=80)
    message_ts: str = Field(pattern=r"^[0-9]+\.[0-9]+$", max_length=80)
    revision_ts: str = Field(default="", max_length=80)
    text: str = Field(max_length=2000)
    truncated: bool = False


class SlackThreadContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: ConversationScope
    status: Literal["loaded", "unavailable"] = "loaded"
    messages: tuple[ThreadMessage, ...] = Field(default=(), max_length=MAX_MESSAGES)
    participants: tuple[SlackParticipant, ...] = Field(default=(), max_length=MAX_PARTICIPANTS)
    truncated: bool = False

    @classmethod
    def from_meta(cls, meta: dict | None):
        if not isinstance(meta, dict) or meta.get("source") != "slack":
            return None
        try:
            scope = ConversationScope.model_validate(meta.get("slack"))
            raw_context = meta.get("slack_thread_context")
            if not raw_context and meta.get("slack_participants"):
                raw_context = {"scope": scope.model_dump(), "status": "unavailable"}
            if not isinstance(raw_context, dict):
                return None
            context = cls.model_validate({
                **(raw_context or {}), "participants": meta.get("slack_participants", []),
            })
            keys = ("team_id", "channel", "thread_ts", "message_ts")
            if any(getattr(context.scope, key) != getattr(scope, key) for key in keys):
                raise ValueError("Thread context scope mismatch")
            if context.messages and not scope.thread_ts:
                raise ValueError("Unthreaded context cannot carry thread messages")
            if any(not Decimal(scope.thread_ts) <= Decimal(row.message_ts) < Decimal(scope.message_ts)
                   for row in context.messages):
                raise ValueError("Thread context contains out-of-window messages")
            return context
        except (ValidationError, ValueError):
            # A caller-supplied reference cannot trigger a database lookup.
            return None


def read_thread_context(db: Session, *, scope: ConversationScope, inbox_id: int) -> SlackThreadContext:
    """Use the current durable receipt as the boundary, including edits as of it."""
    current = db.get(SlackInboxMessage, inbox_id)
    if (not scope.thread_ts or current is None
            or (current.team_id, current.channel, current.message_ts) != (
                scope.team_id, scope.channel, scope.message_ts)
            or (current.thread_ts or current.message_ts) != scope.thread_ts):
        return SlackThreadContext(scope=scope, status="unavailable")

    row = SlackInboxMessage
    # Rank revisions before limiting messages so edits cannot crowd out the thread.
    revisions = select(
        row.id,
        func.row_number().over(
            partition_by=row.message_ts,
            order_by=(cast(row.revision_ts, Numeric(20, 6)).desc(), row.id.desc()),
        ).label("revision_rank"),
    ).where(
        row.team_id == scope.team_id, row.channel == scope.channel,
        row.id <= inbox_id,
        cast(row.message_ts, Numeric(20, 6)) < Decimal(scope.message_ts),
        cast(row.message_ts, Numeric(20, 6)) >= Decimal(scope.thread_ts),
        or_(row.revision_ts == "", cast(row.revision_ts, Numeric(20, 6)) < Decimal(scope.message_ts)),
        or_(row.thread_ts == scope.thread_ts, row.message_ts == scope.thread_ts),
    ).subquery()
    latest = select(row).join(revisions, revisions.c.id == row.id).where(revisions.c.revision_rank == 1)
    rows = list(db.scalars(latest.order_by(cast(row.message_ts, Numeric(20, 6)).desc()).limit(MAX_MESSAGES + 1)))
    truncated = len(rows) > MAX_MESSAGES
    rows = rows[:MAX_MESSAGES]
    parent = db.scalar(latest.where(row.message_ts == scope.thread_ts))
    if parent is not None and all(item.id != parent.id for item in rows):
        rows = rows[:MAX_MESSAGES - 1] + [parent]
        truncated = True
    messages = tuple(ThreadMessage(
        inbox_id=item.id, user_id=item.user_id, message_ts=item.message_ts,
        revision_ts=item.revision_ts, text=item.text[:2000],
        truncated=item.truncated or len(item.text) > 2000,
    ) for item in sorted(rows, key=lambda item: Decimal(item.message_ts)))
    context = SlackThreadContext(scope=scope, messages=messages, truncated=truncated)
    # Bound transport/storage too, not just the final model prompt.
    while len(context.model_dump_json()) > MAX_CHARS and messages:
        messages = _drop_oldest_reply(messages, scope.thread_ts)
        context = SlackThreadContext(scope=scope, messages=messages, truncated=True)
    return context


def _drop_oldest_reply(messages, parent_ts):
    # Retain the root plus the latest replies when an entire thread cannot fit.
    index = 1 if len(messages) > 1 and messages[0].message_ts == parent_ts else 0
    return messages[:index] + messages[index + 1:]


@dataclass(frozen=True, kw_only=True)
class ThreadContextPrompt:
    messages: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)


@prompt_binding("assistant.slack.thread_context")
def render_thread_context(context: SlackThreadContext | None, *, max_tokens: int = 2400) -> ThreadContextPrompt:
    if context is None:
        return ThreadContextPrompt(metadata={"status": "not_available", "included": False})
    rows = context.messages
    guidance = load_bound_prompt(render_thread_context).text
    while True:
        metadata = {
            "status": context.status,
            "source": "local_slack_inbox",
            "coverage": "locally_retained_only",
            "message_count": len(rows),
            "participant_count": len(context.participants),
            "inbox_ids": [row.inbox_id for row in rows],
            "parent_present": any(row.message_ts == context.scope.thread_ts for row in rows),
            "truncated": context.truncated or len(rows) < len(context.messages) or any(row.truncated for row in rows),
        }
        content = guidance + "\n" + json.dumps({
            **metadata, "scope": context.scope.model_dump(),
            "messages": [row.model_dump() for row in rows],
            "participants": [person.model_dump() for person in context.participants],
        }, ensure_ascii=True)
        message = {"role": "user", "content": content}
        if len(content) <= MAX_CHARS and estimate_tokens_for_messages([message]) <= max_tokens:
            return ThreadContextPrompt(messages=(message,), metadata={**metadata, "included": True})
        if not rows:
            return ThreadContextPrompt(metadata={**metadata, "included": False, "reason": "prompt_budget"})
        rows = _drop_oldest_reply(rows, context.scope.thread_ts)

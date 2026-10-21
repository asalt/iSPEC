"""Receipt storage and bounded, exact-conversation outbound context."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import or_, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ispec.prompt import load_bound_prompt, prompt_binding
from .models import SlackOutboundMessage


class AttachmentReference(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_id: str = Field(min_length=1, max_length=80)
    filename: str | None = Field(default=None, max_length=512)
    title: str | None = Field(default=None, max_length=512)
    mime_type: str | None = Field(default=None, max_length=128)
    size_bytes: int | None = Field(default=None, ge=0)


class OutboundReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: str = Field(min_length=1, max_length=80)
    channel: str = Field(min_length=1, max_length=80)
    message_ts: str | None = Field(default=None, pattern=r"^[0-9]+\.[0-9]+$", max_length=80)
    thread_ts: str | None = Field(default=None, pattern=r"^[0-9]+\.[0-9]+$", max_length=80)
    author_id: str | None = Field(default=None, max_length=80)
    recipient_id: str | None = Field(default=None, max_length=80)
    text: str = Field(default="", max_length=40000)
    attachments: list[AttachmentReference] = Field(default_factory=list, max_length=10)
    source: str = Field(min_length=1, max_length=120)
    sent_at: datetime

    @model_validator(mode="after")
    def has_identity(self):
        if not self.message_ts and not self.attachments:
            raise ValueError("A confirmed message timestamp or file id is required")
        if self.sent_at.tzinfo is None:
            raise ValueError("sent_at must include its timezone")
        return self

    @property
    def delivery_key(self) -> str:
        # Upload completion can omit the share timestamp; the file ID remains stable.
        if self.attachments:
            return "file:" + ",".join(sorted(a.file_id for a in self.attachments))
        return "message:" + str(self.message_ts)


def store_outbound(db: Session, receipt: OutboundReceipt) -> dict:
    values = receipt.model_dump(mode="python")
    values["attachments"] = [a.model_dump() for a in receipt.attachments]
    values["sent_at"] = receipt.sent_at.astimezone(UTC).replace(tzinfo=None)
    values["delivery_key"] = receipt.delivery_key
    result = db.execute(insert(SlackOutboundMessage).values(**values).on_conflict_do_nothing(
        index_elements=["team_id", "channel", "delivery_key"],
    ))
    row = db.scalar(select(SlackOutboundMessage).where(
        SlackOutboundMessage.team_id == receipt.team_id,
        SlackOutboundMessage.channel == receipt.channel,
        SlackOutboundMessage.delivery_key == receipt.delivery_key,
    ))
    db.commit()
    return {"retained": True, "record_id": row.id, "duplicate": result.rowcount == 0}


class ConversationScope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    team_id: str = Field(min_length=1, max_length=80)
    channel: str = Field(min_length=1, max_length=80)
    channel_type: str | None = None
    thread_ts: str | None = Field(default=None, pattern=r"^[0-9]+\.[0-9]+$", max_length=80)
    message_ts: str = Field(pattern=r"^[0-9]+\.[0-9]+$", max_length=80)


def outbound_context(db: Session, scope: ConversationScope, *, now: datetime | None = None) -> list[dict]:
    """No fuzzy targets or cross-thread fallback; receipts are evidence, not commands."""
    now = now or datetime.now(UTC)
    try:
        before = min(now, datetime.fromtimestamp(float(Decimal(scope.message_ts)), UTC))
    except (ValueError, OverflowError, InvalidOperation):
        return []
    if not scope.thread_ts and scope.channel_type not in {"im", "mpim"}:
        return []
    age = timedelta(days=7) if scope.thread_ts else timedelta(hours=24)
    query = select(SlackOutboundMessage).where(
        SlackOutboundMessage.team_id == scope.team_id,
        SlackOutboundMessage.channel == scope.channel,
        SlackOutboundMessage.sent_at >= (now - age).replace(tzinfo=None),
        SlackOutboundMessage.sent_at <= before.replace(tzinfo=None),
    )
    if scope.thread_ts:
        query = query.where(or_(
            SlackOutboundMessage.thread_ts == scope.thread_ts,
            (SlackOutboundMessage.thread_ts.is_(None)) & (SlackOutboundMessage.message_ts == scope.thread_ts),
        ))
    else:
        query = query.where(SlackOutboundMessage.thread_ts.is_(None))
    rows = db.scalars(query.order_by(SlackOutboundMessage.sent_at.desc(), SlackOutboundMessage.id.desc()).limit(6)).all()
    result, budget = [], 6000
    for row in rows:
        item = {"receipt_id": row.id, "message_ts": row.message_ts, "thread_ts": row.thread_ts,
                "author_id": row.author_id, "text": row.text[:2000],
                "text_truncated": len(row.text) > 2000,
                "attachments": row.attachments, "sent_at": row.sent_at.isoformat() + "Z"}
        size = len(json.dumps(item))
        if size > budget:
            break
        result.append(item)
        budget -= size
    return list(reversed(result))


@prompt_binding("assistant.slack.outbound_context")
def outbound_context_message(meta: dict | None) -> list[dict]:
    context = (meta or {}).get("slack_outbound_context")
    if not isinstance(context, list) or not context:
        return []
    # Metadata is reference material, including when supplied by an API caller.
    bounded = context[-6:]
    while bounded and len(json.dumps(bounded, ensure_ascii=True)) > 6000:
        bounded = bounded[1:]
    if not bounded:
        return []
    text = json.dumps(bounded, ensure_ascii=True)
    return [{"role": "user", "content": load_bound_prompt(outbound_context_message).text + "\n" + text}]

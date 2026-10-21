from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class AssistantBase(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


class SupportSession(AssistantBase):
    __tablename__ = "support_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    state_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    messages: Mapped[list["SupportMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list["SupportSessionReview"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class SupportMessage(AssistantBase):
    __tablename__ = "support_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_pk: Mapped[int] = mapped_column(
        ForeignKey("support_session.id"),
        index=True,
    )

    role: Mapped[str] = mapped_column(Text)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)

    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 | -1
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped["SupportSession"] = relationship(back_populates="messages")


class SlackInboxMessage(AssistantBase):
    """Received Slack text, independently retained before reply generation."""

    __tablename__ = "slack_inbox_message"
    __table_args__ = (UniqueConstraint("team_id", "channel", "message_ts", "revision_ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(Text, index=True)
    message_ts: Mapped[str] = mapped_column(Text)
    revision_ts: Mapped[str] = mapped_column(Text, default="")
    thread_ts: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(Text)
    event_id: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    truncated: Mapped[bool] = mapped_column(default=False)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    handled_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_type: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    reviewed_by: Mapped[str | None] = mapped_column(Text)


class SlackOutboundMessage(AssistantBase):
    """Confirmed Slack sends, separate from generated-but-not-delivered replies."""

    __tablename__ = "slack_outbound_message"
    __table_args__ = (UniqueConstraint("team_id", "channel", "delivery_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(Text, index=True)
    channel: Mapped[str] = mapped_column(Text, index=True)
    delivery_key: Mapped[str] = mapped_column(Text)
    message_ts: Mapped[str | None] = mapped_column(Text)
    thread_ts: Mapped[str | None] = mapped_column(Text, index=True)
    author_id: Mapped[str | None] = mapped_column(Text)
    recipient_id: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    attachments: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(Text)
    sent_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SupportMemory(AssistantBase):
    __tablename__ = "support_memory"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_pk: Mapped[int | None] = mapped_column(
        ForeignKey("support_session.id"),
        index=True,
        nullable=True,
    )
    user_id: Mapped[int] = mapped_column(Integer, index=True, default=0)

    kind: Mapped[str] = mapped_column(Text, index=True)
    key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    value_json: Mapped[str] = mapped_column(Text)

    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    importance: Mapped[float] = mapped_column(Float, default=0.3)
    salience: Mapped[float] = mapped_column(Float, index=True, default=0.3)
    salience_floor: Mapped[float] = mapped_column(Float, default=0.0)
    decay_lambda: Mapped[float] = mapped_column(Float, default=0.01)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped[SupportSession | None] = relationship()
    evidence: Mapped[list["SupportMemoryEvidence"]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
    )


class SupportMemoryEvidence(AssistantBase):
    __tablename__ = "support_memory_evidence"
    __table_args__ = (
        UniqueConstraint("memory_id", "message_id", name="uq_support_memory_evidence_memory_message"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_id: Mapped[int] = mapped_column(
        ForeignKey("support_memory.id", ondelete="CASCADE"),
        index=True,
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("support_message.id", ondelete="CASCADE"),
        index=True,
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    memory: Mapped[SupportMemory] = relationship(back_populates="evidence")


class SupportSessionReview(AssistantBase):
    __tablename__ = "support_session_review"
    __table_args__ = (
        UniqueConstraint("session_pk", "target_message_id", name="uq_support_session_review_session_target"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_pk: Mapped[int] = mapped_column(
        ForeignKey("support_session.id", ondelete="CASCADE"),
        index=True,
    )
    target_message_id: Mapped[int] = mapped_column(Integer, index=True)

    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    review_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    agent_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    command_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, index=True)

    session: Mapped[SupportSession] = relationship(back_populates="reviews")

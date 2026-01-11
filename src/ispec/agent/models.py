from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class AgentBase(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


class AgentEvent(AgentBase):
    __tablename__ = "agent_event"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[str] = mapped_column(Text, index=True)
    event_type: Mapped[str] = mapped_column(Text, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    name: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    severity: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    correlation_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    payload_json: Mapped[str] = mapped_column(Text)


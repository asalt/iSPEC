from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text
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

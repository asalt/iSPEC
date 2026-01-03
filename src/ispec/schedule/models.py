from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class ScheduleBase(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


class ScheduleSlot(ScheduleBase):
    __tablename__ = "schedule_slot"
    __table_args__ = (UniqueConstraint("start_at", "end_at", name="uq_schedule_slot_time"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    start_at: Mapped[datetime] = mapped_column(DateTime)
    end_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(Text, default="available")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    requests: Mapped[list["ScheduleRequestSlot"]] = relationship(
        back_populates="slot",
        cascade="all, delete-orphan",
    )


class ScheduleRequest(ScheduleBase):
    __tablename__ = "schedule_request"

    id: Mapped[int] = mapped_column(primary_key=True)
    requester_name: Mapped[str] = mapped_column(Text)
    requester_email: Mapped[str] = mapped_column(Text)
    requester_org: Mapped[str | None] = mapped_column(Text, nullable=True)
    requester_phone: Mapped[str | None] = mapped_column(Text, nullable=True)

    project_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_description: Mapped[str] = mapped_column(Text)
    cancer_related: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[str] = mapped_column(Text, default="requested")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    slots: Mapped[list["ScheduleRequestSlot"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
    )


class ScheduleRequestSlot(ScheduleBase):
    __tablename__ = "schedule_request_slot"
    __table_args__ = (
        UniqueConstraint("slot_id", name="uq_schedule_request_slot"),
        UniqueConstraint("request_id", "rank", name="uq_schedule_request_rank"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("schedule_request.id"))
    slot_id: Mapped[int] = mapped_column(ForeignKey("schedule_slot.id"))
    rank: Mapped[int] = mapped_column(Integer)

    request: Mapped[ScheduleRequest] = relationship(back_populates="slots")
    slot: Mapped[ScheduleSlot] = relationship(back_populates="requests")

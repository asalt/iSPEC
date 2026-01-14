from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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


class AgentRun(AgentBase):
    __tablename__ = "agent_run"
    __table_args__ = (UniqueConstraint("run_id", name="uq_agent_run_run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, index=True)
    agent_id: Mapped[str] = mapped_column(Text, index=True)

    kind: Mapped[str] = mapped_column(Text, index=True, default="supervisor")
    status: Mapped[str] = mapped_column(Text, index=True, default="running")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, index=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    step_index: Mapped[int] = mapped_column(Integer, default=0)
    status_bar: Mapped[str | None] = mapped_column(Text, nullable=True)

    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class AgentStep(AgentBase):
    __tablename__ = "agent_step"
    __table_args__ = (
        UniqueConstraint("run_pk", "step_index", name="uq_agent_step_run_step_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_pk: Mapped[int] = mapped_column(ForeignKey("agent_run.id"), index=True)
    step_index: Mapped[int] = mapped_column(Integer, index=True)

    kind: Mapped[str] = mapped_column(Text, index=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ok: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    severity: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    candidates_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    chosen_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chosen_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    prompt_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tool_calls_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_results_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    summary_before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    summary_after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    state_before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    state_after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    run: Mapped[AgentRun] = relationship(back_populates="steps")


class AgentCommand(AgentBase):
    __tablename__ = "agent_command"

    id: Mapped[int] = mapped_column(primary_key=True)
    command_type: Mapped[str] = mapped_column(Text, index=True)
    status: Mapped[str] = mapped_column(Text, index=True, default="queued")
    priority: Mapped[int] = mapped_column(Integer, index=True, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, index=True)

    available_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    claimed_by_agent_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    claimed_by_run_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)

    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

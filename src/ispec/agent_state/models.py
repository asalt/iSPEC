from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, LargeBinary, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class AgentStateBase(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


class AgentStateSchemaVersion(AgentStateBase):
    __tablename__ = "agent_state_schema_version"
    __table_args__ = (
        UniqueConstraint("schema_id", "version", name="uq_agent_state_schema_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_id: Mapped[int] = mapped_column(Integer, index=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    state_scope: Mapped[str] = mapped_column(Text, index=True)
    dim_count: Mapped[int] = mapped_column(Integer)
    codec: Mapped[str] = mapped_column(Text, default="f32le")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentStateSchemaDim(AgentStateBase):
    __tablename__ = "agent_state_schema_dim"
    __table_args__ = (
        UniqueConstraint("schema_id", "version", "dim_index", name="uq_agent_state_schema_dim"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    schema_id: Mapped[int] = mapped_column(Integer, index=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    dim_index: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentStateObservation(AgentStateBase):
    __tablename__ = "agent_state_observation"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    agent_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_index: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    schema_id: Mapped[int] = mapped_column(Integer, index=True)
    schema_version: Mapped[int] = mapped_column(Integer, index=True)
    state_scope: Mapped[str] = mapped_column(Text, index=True)
    vector_blob: Mapped[bytes] = mapped_column(LargeBinary)
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentStateHead(AgentStateBase):
    __tablename__ = "agent_state_head"

    agent_id: Mapped[str] = mapped_column(Text, primary_key=True)
    state_scope: Mapped[str] = mapped_column(Text, primary_key=True)
    schema_id: Mapped[int] = mapped_column(Integer)
    schema_version: Mapped[int] = mapped_column(Integer)
    ts_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    vector_blob: Mapped[bytes] = mapped_column(LargeBinary)
    observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_state_observation.id"),
        nullable=True,
        index=True,
    )

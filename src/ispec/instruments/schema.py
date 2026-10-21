from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")]


class ObservedRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1, max_length=2048)
    modified_at: AwareDatetime
    observed_bytes: int = Field(ge=0)
    last_file: str | None = Field(default=None, max_length=2048)


class InstrumentObservation(BaseModel):
    """v1 describes filesystem evidence, not vendor-reported instrument state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    event_id: UUID
    collector_id: Identifier
    instrument_id: Identifier
    observed_at: AwareDatetime
    source_kind: Literal["directory_metadata"]
    source_root: str = Field(min_length=1, max_length=4096)
    resolved_root: str | None = Field(default=None, max_length=4096)
    source_state: Literal["readable", "partial", "unavailable", "timeout"]
    source_updated_at: AwareDatetime | None = None
    entries_examined: int = Field(ge=0, le=10000)
    truncated: bool
    recent_runs: list[ObservedRun] = Field(max_length=20)
    read_errors: list[str] = Field(max_length=20)
    collector_version: Literal["1"]
    instrument_state: Literal["unknown"] = "unknown"
    progress: None = None
    instrument_error: None = None
    instrument_heartbeat_at: None = None
    pressure: None = None
    flow: None = None

    @model_validator(mode="after")
    def bounded_errors(self) -> InstrumentObservation:
        if any(len(error) > 256 for error in self.read_errors):
            raise ValueError("Read errors must be compact error codes, not log dumps.")
        return self


def as_utc(value: datetime) -> datetime:
    from datetime import UTC

    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

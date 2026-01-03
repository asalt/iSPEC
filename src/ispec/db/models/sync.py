from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, make_timestamp_mixin

LegacySyncTimestamp = make_timestamp_mixin("legsync")


class LegacySyncState(LegacySyncTimestamp, Base):
    """Persist per-table legacy sync cursors.

    We use an exclusive cursor: (modified_ts, pk) so we can page deterministically
    even when multiple rows share the same modification timestamp.
    """

    __tablename__ = "legacy_sync_state"

    legacy_table: Mapped[str] = mapped_column(Text, primary_key=True)
    since: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    since_pk: Mapped[int | None] = mapped_column(Integer, nullable=True)

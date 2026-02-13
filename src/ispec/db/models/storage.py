from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, make_timestamp_mixin

OmicsDatabaseTimestamp = make_timestamp_mixin("omdb")


class OmicsDatabaseRegistry(OmicsDatabaseTimestamp, Base):
    """Track local-first omics DB locations and availability state."""

    __tablename__ = "omics_database_registry"

    id: Mapped[int] = mapped_column(primary_key=True)
    omdb_LogicalName: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    omdb_DBURI: Mapped[str] = mapped_column(Text, nullable=False)
    omdb_DBPath: Mapped[str | None] = mapped_column(Text, nullable=True)
    omdb_Status: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    omdb_LastCheckedTS: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    omdb_LastAvailableTS: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    omdb_LastError: Mapped[str | None] = mapped_column(Text, nullable=True)

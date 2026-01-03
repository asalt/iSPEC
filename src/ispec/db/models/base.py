# Shared SQLAlchemy base classes and timestamp mixins
from datetime import datetime, UTC
from typing import Dict

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def make_timestamp_mixin(prefix: str):
    """Dynamically build a mixin with creation/modification timestamps."""

    def utcnow() -> datetime:
        return datetime.now(UTC)

    fields = {
        f"{prefix}_CreationTS": mapped_column(DateTime, default=utcnow),
        f"{prefix}_ModificationTS": mapped_column(
            DateTime, default=utcnow, onupdate=utcnow
        ),
        "__annotations__": {
            f"{prefix}_CreationTS": Mapped[datetime],
            f"{prefix}_ModificationTS": Mapped[datetime],
        },
    }
    return type(f"{prefix.capitalize()}TimestampMixin", (object,), fields)


class Base(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}
    __ui__ = {"sections": [], "order": []}

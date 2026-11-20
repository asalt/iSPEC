from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from ispec.timeutil import format_utc, utcnow
from .base import Base


def timestamp():
    return format_utc(utcnow())


class TimestampMixin:
    created_at: Mapped[str] = mapped_column(default=timestamp)
    updated_at: Mapped[str] = mapped_column(default=timestamp, onupdate=timestamp)


class InstrumentRegistry(TimestampMixin, Base):
    __tablename__ = "instrument_registry"
    __table_args__ = {}

    id: Mapped[str] = mapped_column(primary_key=True)
    settings: Mapped[dict] = mapped_column(JSON)
    generation: Mapped[int] = mapped_column(default=1)
    archive_seen: Mapped[bool] = mapped_column(default=False)
    revision: Mapped[int] = mapped_column(default=1)
    __mapper_args__ = {"version_id_col": revision}

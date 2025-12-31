import enum

from sqlalchemy import Enum as SAEnum, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

RawTimestamp = make_timestamp_mixin("msraw")


class RawFileType(str, enum.Enum):
    raw = "raw"
    mzml = "mzml"
    parquet = "parquet"
    bruker_d = "bruker_d"  # Bruker timsTOF .d directories
    other = "other"


class RawFileState(str, enum.Enum):
    available = "available"
    archived = "archived"
    missing = "missing"


class StorageBackend(str, enum.Enum):
    local = "local"
    s3 = "s3"
    gcs = "gcs"
    other = "other"


class MSRawFile(RawTimestamp, Base):
    """
    Metadata record for raw/mzML/parquet assets associated with an ExperimentRun.

    URIs can point to local paths or remote object storage; this scaffold
    keeps fields minimal for future expansion.
    """

    __tablename__ = "ms_raw_file"
    __table_args__ = (
        UniqueConstraint(
            "experiment_run_id",
            "uri",
            name="uq_msraw_run_uri",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_run.id", ondelete="CASCADE"), nullable=False
    )
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_type = mapped_column(
        SAEnum(RawFileType, native_enum=True, validate_strings=True),
        nullable=False,
        default=RawFileType.raw,
    )
    storage_backend = mapped_column(
        SAEnum(StorageBackend, native_enum=True, validate_strings=True),
        nullable=False,
        default=StorageBackend.local,
    )
    state = mapped_column(
        SAEnum(RawFileState, native_enum=True, validate_strings=True),
        nullable=False,
        default=RawFileState.available,
    )
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment_run: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="raw_files"
    )

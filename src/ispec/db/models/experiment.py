import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

ExperimentTimestamp = make_timestamp_mixin("Experiment")
ExperimentRunTimestamp = make_timestamp_mixin("ExperimentRun")
E2GTimestamp = make_timestamp_mixin("E2G")
JobTimestamp = make_timestamp_mixin("job")


class Experiment(ExperimentTimestamp, Base):
    __tablename__ = "experiment"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), nullable=False
    )
    record_no: Mapped[str] = mapped_column(Text, nullable=False)

    project: Mapped["Project"] = relationship(back_populates="experiments")
    runs: Mapped[list["ExperimentRun"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentRun(ExperimentRunTimestamp, Base):
    __tablename__ = "experiment_run"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "run_no",
            "search_no",
            name="uq_experiment_run_search",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment.id", ondelete="CASCADE"), nullable=False
    )
    run_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    search_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    db_search_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    gpgrouper_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="runs")
    gene_mappings: Mapped[list["E2G"]] = relationship(
        back_populates="experiment_run", cascade="all, delete-orphan"
    )
    psms: Mapped[list["PSM"]] = relationship(
        "PSM", back_populates="experiment_run", cascade="all, delete-orphan"
    )
    raw_files: Mapped[list["MSRawFile"]] = relationship(
        "MSRawFile", back_populates="experiment_run", cascade="all, delete-orphan"
    )


class E2G(E2GTimestamp, Base):
    __tablename__ = "experiment_to_gene"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_run.id", ondelete="CASCADE"), nullable=False
    )
    gene: Mapped[str] = mapped_column(Text, nullable=False)
    geneidtype: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False, default="0")
    iBAQ_dstrAdj: Mapped[float | None] = mapped_column(Float, nullable=True)
    peptideprint: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment_run: Mapped["ExperimentRun"] = relationship(
        back_populates="gene_mappings"
    )


class JobType(str, enum.Enum):
    db_search = "db_search"
    gpgrouper = "gpgrouper"
    e2g_import = "e2g_import"


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class Job(JobTimestamp, Base):
    __tablename__ = "job"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_run.id", ondelete="CASCADE"), nullable=False
    )
    job_type = mapped_column(
        SAEnum(JobType, native_enum=True, validate_strings=True),
        nullable=False,
    )
    status = mapped_column(
        SAEnum(JobStatus, native_enum=True, validate_strings=True),
        nullable=False,
        default=JobStatus.queued,
    )
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    run: Mapped["ExperimentRun"] = relationship()

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

ExperimentTimestamp = make_timestamp_mixin("Experiment")
ExperimentRunTimestamp = make_timestamp_mixin("ExperimentRun")
E2GTimestamp = make_timestamp_mixin("E2G")
JobTimestamp = make_timestamp_mixin("job")


class ExperimentType(str, enum.Enum):
    affinity = "Affinity"
    affinity_xl = "Affinity-XL"
    profiling = "Profiling"


class LysisMethod(str, enum.Enum):
    abc = "ABC"
    ripa = "RIPA"
    mib = "MIB"
    other = "Other"
    mu8 = "8MU"
    netn = "NETN"
    sds = "SDS"


class Experiment(ExperimentTimestamp, Base):
    __tablename__ = "experiment"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), nullable=False
    )
    record_no: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        info={"ui": {"label": "EXPRecNo"}},
    )

    exp_LabelFLAG: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, info={"ui": {"label": "LabelFLAG"}}
    )
    exp_Type = mapped_column(
        SAEnum(ExperimentType, native_enum=True, validate_strings=True),
        nullable=True,
        info={
            "ui": {
                "label": "Type",
                "component": "RadioGroup",
                "options": [
                    {"value": "Affinity", "label": "Affinity"},
                    {"value": "Affinity-XL", "label": "Affinity-XL"},
                    {"value": "Profiling", "label": "Profiling"},
                ],
                "allowClear": True,
            }
        },
    )
    exp_Name: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Name"}}
    )
    exp_Date: Mapped[datetime | None] = mapped_column(
        DateTime,
        default=None,
        nullable=True,
        info={"ui": {"label": "Date"}},
    )
    exp_PreparationNo: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Preparation #"}}
    )
    exp_CellTissue: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Cell/Tissue"}}
    )
    exp_Genotype: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Genotype"}}
    )
    exp_Treatment: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Treatment"}}
    )
    exp_Fractions: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Fractions"}}
    )
    exp_Lysis = mapped_column(
        SAEnum(LysisMethod, native_enum=True, validate_strings=True),
        nullable=True,
        info={
            "ui": {
                "label": "Lysis",
                "component": "RadioGroup",
                "options": [
                    {"value": "ABC", "label": "ABC"},
                    {"value": "RIPA", "label": "RIPA"},
                    {"value": "MIB", "label": "MIB"},
                    {"value": "Other", "label": "Other..."},
                    {"value": "8MU", "label": "8MU"},
                    {"value": "NETN", "label": "NETN"},
                    {"value": "SDS", "label": "SDS"},
                ],
                "allowClear": True,
            }
        },
    )
    exp_DTT: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        info={
            "ui": {
                "label": "DTT",
                "component": "RadioGroup",
                "options": [{"value": True, "label": "Yes"}, {"value": False, "label": "No"}],
            }
        },
    )
    exp_IAA: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        info={
            "ui": {
                "label": "IAA",
                "component": "RadioGroup",
                "options": [{"value": True, "label": "Yes"}, {"value": False, "label": "No"}],
            }
        },
    )
    exp_Amount: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Amount"}}
    )
    exp_Adjustments: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Adjustments", "component": "Textarea"}}
    )
    exp_Batch: Mapped[str | None] = mapped_column(
        Text, nullable=True, info={"ui": {"label": "Batch #"}}
    )
    exp_Data_FLAG: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, info={"ui": {"label": "data"}}
    )
    exp_exp2gene_FLAG: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, info={"ui": {"label": "exp2gene"}}
    )
    exp_Description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Experiment Description", "component": "Textarea"}},
    )

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

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

ExperimentTimestamp = make_timestamp_mixin("Experiment")
ExperimentRunTimestamp = make_timestamp_mixin("ExperimentRun")
JobTimestamp = make_timestamp_mixin("job")


class ExperimentType(str, enum.Enum):
    affinity = "Affinity"
    affinity_xl = "Affinity-XL"
    profiling = "Profiling"
    prof = "prof"
    IP = "IP"
    IPXL = "IP-XL"
    Other = "Other"
    DPD = "DPD"  # dna pulldown
    cIP = "cIP" 
    tagIP = "tagIP"
    x = "x"


class LysisMethod(str, enum.Enum):
    abc = "ABC"
    ripa = "RIPA"
    mib = "MIB"
    other = "Other"
    mu8 = "8MU"
    netn = "NETN"
    sds = "SDS"
    other2 = "Tris-EDTA-sucrose + lysozyme + mutanolysin"  


class Experiment(ExperimentTimestamp, Base):
    __tablename__ = "experiment"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), nullable=True
    )
    Experiment_LegacyImportTS: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    record_no: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        info={"ui": {"label": "EXPRecNo"}},
    )

    exp_LabelFLAG: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        info={
            "ui": {
                "label": "Label",
                "component": "Select",
                "options": [
                    {"value": 0, "label": "None (0)"},
                    {"value": 1, "label": "TMT6"},
                    {"value": 2, "label": "TMTPro"},
                    {"value": 3, "label": "SILAC"},
                ],
            }
        },
    )
    exp_Type = mapped_column(
        SAEnum(ExperimentType, native_enum=True, validate_strings=True),
        nullable=True, # options need to be updated dynamically from ExperimentType enum
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
        Boolean,
        nullable=False,
        default=False,
        info={"ui": {"label": "data", "readOnly": True}},
    )
    exp_exp2gene_FLAG: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        info={"ui": {"label": "exp2gene", "readOnly": True}},
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
            "label",
            name="uq_experiment_run_search_label",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment.id", ondelete="CASCADE"), nullable=False
    )
    ExperimentRun_LegacyImportTS: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    run_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    search_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    label: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="0",
        info={"ui": {"label": "Label"}},
    )
    label_type: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={
            "ui": {
                "label": "Label Type",
                "component": "Select",
                "allowClear": True,
                "tag": True,
                "options": [
                    {"label": "Label-free", "value": "LabelFree"},
                    {"label": "TMT 10", "value": "TMT10"},
                    {"label": "TMT 11", "value": "TMT11"},
                    {"label": "TMT 16", "value": "TMT16"},
                    {"label": "TMT 18", "value": "TMT18"},
                    {"label": "SILAC", "value": "SILAC"},
                    {"label": "Other", "value": "Other"},
                ],
            }
        },
    )

    ms_instrument: Mapped[str | None] = mapped_column(Text, nullable=True)
    acquisition_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    ref_database: Mapped[str | None] = mapped_column(Text, nullable=True)
    taxon_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    db_search_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    gpgrouper_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="runs")
    raw_files: Mapped[list["MSRawFile"]] = relationship(
        "MSRawFile", back_populates="experiment_run", cascade="all, delete-orphan"
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

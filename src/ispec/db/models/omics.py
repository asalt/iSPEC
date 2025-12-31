from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

PSMTimestamp = make_timestamp_mixin("psm")


class PSM(PSMTimestamp, Base):
    """
    Peptide Spectrum Match linked to an ExperimentRun.

    This is a scaffold model for future build-out; fields may expand.
    """

    __tablename__ = "psm"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_run.id", ondelete="CASCADE"), nullable=False
    )
    scan_number: Mapped[int] = mapped_column(Integer, nullable=False)
    peptide: Mapped[str] = mapped_column(Text, nullable=False)
    charge: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_type: Mapped[str | None] = mapped_column(Text, nullable=True)  # e.g., XCorr, Percolator PEP
    q_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    protein: Mapped[str | None] = mapped_column(Text, nullable=True)
    mods: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-serialized modifications
    precursor_mz: Mapped[float | None] = mapped_column(Float, nullable=True)
    retention_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # minutes
    intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment_run: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="psms"
    )

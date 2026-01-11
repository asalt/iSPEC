from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ispec.db.models.base import make_timestamp_mixin


class OmicsBase(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


PSMTimestamp = make_timestamp_mixin("psm")
E2GTimestamp = make_timestamp_mixin("E2G")
GeneContrastTimestamp = make_timestamp_mixin("GeneContrast")
GeneContrastStatTimestamp = make_timestamp_mixin("GeneContrastStat")
GSEAAnalysisTimestamp = make_timestamp_mixin("GSEAAnalysis")
GSEAResultTimestamp = make_timestamp_mixin("GSEAResult")


class E2G(E2GTimestamp, OmicsBase):
    """Experiment-to-gene rows (gpgrouper QUAL/QUANT)."""

    __tablename__ = "experiment_to_gene"
    __table_args__ = (
        UniqueConstraint(
            "experiment_run_id",
            "gene",
            "geneidtype",
            "label",
            name="uq_e2g_run_gene_type_label",
        ),
        Index("ix_e2g_run", "experiment_run_id"),
        Index("ix_e2g_gene", "gene"),
        Index("ix_e2g_symbol", "gene_symbol"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    gene: Mapped[str] = mapped_column(Text, nullable=False)
    geneidtype: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False, default="0")

    # Convenience columns (GeneID is canonical; the others help search/display).
    gene_symbol: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    taxon_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sra: Mapped[str | None] = mapped_column(Text, nullable=True)

    # QUAL metrics
    psms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    psms_u2g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peptide_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peptide_count_u2g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_u2g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # QUANT metrics
    area_sum_u2g_0: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_sum_u2g_all: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_sum_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_sum_dstrAdj: Mapped[float | None] = mapped_column(Float, nullable=True)
    iBAQ_dstrAdj: Mapped[float | None] = mapped_column(Float, nullable=True)
    peptideprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class PSM(PSMTimestamp, OmicsBase):
    """Peptide Spectrum Match linked to an ExperimentRun (by id reference)."""

    __tablename__ = "psm"
    __table_args__ = (Index("ix_psm_run_scan", "experiment_run_id", "scan_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
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


class GeneContrast(GeneContrastTimestamp, OmicsBase):
    """A named differential-analysis contrast for a project (e.g. a volcano table)."""

    __tablename__ = "gene_contrast"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_gene_contrast_project_name"),
        Index("ix_gene_contrast_project_contrast", "project_id", "contrast"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    contrast: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    kind: Mapped[str | None] = mapped_column(Text, nullable=True)  # e.g. volcano, limma
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    stats: Mapped[list["GeneContrastStat"]] = relationship(
        back_populates="contrast_set", cascade="all, delete-orphan"
    )


class GeneContrastStat(GeneContrastStatTimestamp, OmicsBase):
    """Per-gene statistics for a contrast."""

    __tablename__ = "gene_contrast_stat"
    __table_args__ = (
        UniqueConstraint(
            "gene_contrast_id",
            "gene_id",
            name="uq_gene_contrast_stat_contrast_gene",
        ),
        Index("ix_gene_contrast_stat_contrast_padj", "gene_contrast_id", "p_adj"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    gene_contrast_id: Mapped[int] = mapped_column(
        ForeignKey("gene_contrast.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    gene_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    gene_symbol: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    log2_fc: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_adj: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    signed_log_p: Mapped[float | None] = mapped_column(Float, nullable=True)

    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    contrast_set: Mapped["GeneContrast"] = relationship(back_populates="stats")


class GSEAAnalysis(GSEAAnalysisTimestamp, OmicsBase):
    """A named GSEA analysis for a project (one file/contrast per gene-set collection)."""

    __tablename__ = "gsea_analysis"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_gsea_analysis_project_name"),
        Index("ix_gsea_analysis_project_contrast", "project_id", "contrast"),
        Index("ix_gsea_analysis_project_collection", "project_id", "collection"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    contrast: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    collection: Mapped[str | None] = mapped_column(Text, nullable=True)  # e.g. MSigDB: H, C2, C5
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    results: Mapped[list["GSEAResult"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class GSEAResult(GSEAResultTimestamp, OmicsBase):
    """Per-pathway enrichment result for a GSEA analysis."""

    __tablename__ = "gsea_result"
    __table_args__ = (
        UniqueConstraint(
            "gsea_analysis_id",
            "pathway",
            name="uq_gsea_result_analysis_pathway",
        ),
        Index("ix_gsea_result_analysis_padj", "gsea_analysis_id", "p_adj"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    gsea_analysis_id: Mapped[int] = mapped_column(
        ForeignKey("gsea_analysis.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    pathway: Mapped[str] = mapped_column(Text, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_adj: Mapped[float | None] = mapped_column(Float, nullable=True)
    log2err: Mapped[float | None] = mapped_column(Float, nullable=True)
    es: Mapped[float | None] = mapped_column(Float, nullable=True)
    nes: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mainpathway: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    leading_edge: Mapped[str | None] = mapped_column(Text, nullable=True)
    leading_edge_entrezid: Mapped[str | None] = mapped_column(Text, nullable=True)
    leading_edge_genesymbol: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis: Mapped["GSEAAnalysis"] = relationship(back_populates="results")


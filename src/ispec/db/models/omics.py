"""Backwards-compatible omics model exports.

The omics tables (E2G/PSM/contrast/GSEA) live in a separate SQLite database and
are modeled under :mod:`ispec.omics.models`. This module remains as a thin shim
so older imports like ``from ispec.db.models.omics import PSM`` keep working.
"""

from ispec.omics.models import (
    E2G,
    GeneContrast,
    GeneContrastStat,
    GSEAAnalysis,
    GSEAResult,
    PSM,
)

__all__ = [
    "E2G",
    "GeneContrast",
    "GeneContrastStat",
    "GSEAAnalysis",
    "GSEAResult",
    "PSM",
]

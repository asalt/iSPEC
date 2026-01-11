from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ispec.db.models import Project
from ispec.omics.models import GeneContrast, GeneContrastStat


_INFER_CONTRAST_RE = re.compile(r"_group_(.+?)_dir_([A-Za-z0-9]+)$")


@dataclass(frozen=True)
class GeneContrastImportResult:
    path: str
    project_id: int
    name: str
    contrast: str
    gene_contrast_id: int
    rows: int
    inserted: int
    skipped: bool = False
    skip_reason: str | None = None
    created_contrast: bool = False
    cleared_existing: bool = False


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _safe_str(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _infer_contrast_from_stem(stem: str) -> tuple[str | None, str | None]:
    match = _INFER_CONTRAST_RE.search(stem)
    if match:
        return match.group(1), match.group(2)
    if "_group_" in stem:
        return stem.split("_group_", 1)[1] or None, None
    return None, None


def import_gene_contrast_file(
    *,
    core_session: Session,
    omics_session: Session,
    path: str | Path,
    project_id: int,
    name: str | None = None,
    contrast: str | None = None,
    kind: str | None = "volcano",
    store_metadata: bool = True,
    skip_imported: bool = True,
    force: bool = False,
) -> GeneContrastImportResult:
    """Import a gene-level contrast TSV (e.g. a volcano table) into the DB."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    project = core_session.get(Project, int(project_id))
    if project is None:
        raise ValueError(f"Project {project_id} not found.")

    stem = file_path.stem
    inferred_contrast, inferred_dir = _infer_contrast_from_stem(stem)

    resolved_name = (name or stem).strip()
    if not resolved_name:
        resolved_name = stem

    resolved_contrast = (contrast or inferred_contrast or stem).strip()
    if not resolved_contrast:
        resolved_contrast = stem

    contrast_row = (
        omics_session.query(GeneContrast)
        .filter(GeneContrast.project_id == int(project_id))
        .filter(GeneContrast.name == resolved_name)
        .one_or_none()
    )
    created_contrast = False
    if contrast_row is None:
        contrast_row = GeneContrast(
            project_id=int(project_id),
            name=resolved_name,
            contrast=resolved_contrast,
            kind=(kind or None),
            source_path=str(file_path),
        )
        omics_session.add(contrast_row)
        omics_session.flush()
        created_contrast = True
    else:
        contrast_row.source_path = str(file_path)
        if contrast_row.contrast != resolved_contrast:
            contrast_row.contrast = resolved_contrast
        if contrast_row.kind is None and kind:
            contrast_row.kind = kind

    existing = int(
        omics_session.query(func.count(GeneContrastStat.id))
        .filter(GeneContrastStat.gene_contrast_id == int(contrast_row.id))
        .scalar()
        or 0
    )
    if existing > 0 and skip_imported and not force:
        return GeneContrastImportResult(
            path=str(file_path),
            project_id=int(project_id),
            name=resolved_name,
            contrast=resolved_contrast,
            gene_contrast_id=int(contrast_row.id),
            rows=0,
            inserted=0,
            skipped=True,
            skip_reason="already_imported",
            created_contrast=created_contrast,
            cleared_existing=False,
        )

    cleared_existing = False
    if existing > 0 and (force or not skip_imported):
        omics_session.query(GeneContrastStat).filter(
            GeneContrastStat.gene_contrast_id == int(contrast_row.id)
        ).delete(synchronize_session=False)
        cleared_existing = True

    required = {"GeneID"}
    rows_read = 0
    by_gene_id: dict[int, dict[str, Any]] = {}
    fieldnames: list[str] | None = None
    now = datetime.now(UTC)

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = [str(name) for name in (reader.fieldnames or []) if name]
        missing = required.difference(set(fieldnames))
        if missing:
            raise ValueError(f"{file_path.name} is missing required columns: {sorted(missing)}")

        for row in reader:
            if not row:
                continue
            rows_read += 1

            gene_id = _safe_int(row.get("GeneID"))
            if gene_id is None:
                continue

            record: dict[str, Any] = {
                "gene_contrast_id": int(contrast_row.id),
                "gene_id": gene_id,
                "GeneContrastStat_CreationTS": now,
                "GeneContrastStat_ModificationTS": now,
                "gene_symbol": _safe_str(row.get("GeneSymbol"), max_len=128),
                "description": _safe_str(row.get("GeneDescription"), max_len=2000),
                "log2_fc": _safe_float(row.get("log2_FC") or row.get("log2FC")),
                "p_value": _safe_float(row.get("pValue") or row.get("p_value")),
                "p_adj": _safe_float(row.get("pAdj") or row.get("padj")),
                "t_stat": _safe_float(row.get("t") or row.get("t_stat")),
                "signed_log_p": _safe_float(row.get("signedlogP") or row.get("signed_log_p")),
            }

            if store_metadata:
                extra: dict[str, Any] = {}
                ci_l = _safe_float(row.get("CI.L"))
                ci_r = _safe_float(row.get("CI.R"))
                ave_expr = _safe_float(row.get("AveExpr"))
                b_stat = _safe_float(row.get("B"))
                funcats = _safe_str(row.get("FunCats"), max_len=2000)
                if ci_l is not None or ci_r is not None:
                    extra["ci"] = {"low": ci_l, "high": ci_r}
                if ave_expr is not None:
                    extra["ave_expr"] = ave_expr
                if b_stat is not None:
                    extra["b"] = b_stat
                if funcats is not None:
                    extra["funcats"] = funcats
                if extra:
                    record["metadata_json"] = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))

            by_gene_id[gene_id] = record

    records = list(by_gene_id.values())
    if records:
        omics_session.bulk_insert_mappings(GeneContrastStat, records)

    if store_metadata and fieldnames is not None:
        known = {
            "GeneID",
            "log2_FC",
            "log2FC",
            "CI.L",
            "CI.R",
            "AveExpr",
            "t",
            "pValue",
            "pAdj",
            "B",
            "GeneSymbol",
            "FunCats",
            "GeneDescription",
            "signedlogP",
        }
        sample_columns = [col for col in fieldnames if col and col[0].isdigit()]
        extra_columns = [col for col in fieldnames if col not in known and col not in sample_columns]
        meta_payload: dict[str, Any] = {
            "source": {
                "filename": file_path.name,
                "columns": fieldnames,
                "sample_columns": sample_columns,
                "extra_columns": extra_columns,
            },
            "inferred": {"contrast": inferred_contrast, "direction": inferred_dir},
            "import": {"rows": rows_read, "unique_genes": len(records)},
        }
        contrast_row.metadata_json = json.dumps(meta_payload, ensure_ascii=False, separators=(",", ":"))

    omics_session.flush()

    return GeneContrastImportResult(
        path=str(file_path),
        project_id=int(project_id),
        name=resolved_name,
        contrast=resolved_contrast,
        gene_contrast_id=int(contrast_row.id),
        rows=rows_read,
        inserted=len(records),
        skipped=False,
        skip_reason=None,
        created_contrast=created_contrast,
        cleared_existing=cleared_existing,
    )

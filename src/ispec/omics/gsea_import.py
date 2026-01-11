from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ispec.db.models import Project
from ispec.omics.models import GSEAAnalysis, GSEAResult


_MSIGDB_COLLECTIONS = {"H", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}


@dataclass(frozen=True)
class GSEAImportResult:
    path: str
    project_id: int
    name: str
    contrast: str
    collection: str | None
    gsea_analysis_id: int
    rows: int
    inserted: int
    skipped: bool = False
    skip_reason: str | None = None
    created_analysis: bool = False
    cleared_existing: bool = False


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"na", "nan", "null", "none"}:
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
    if text.lower() in {"na", "nan", "null", "none"}:
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
    if text.lower() in {"na", "nan", "null", "none"}:
        return None
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _infer_collection_and_contrast(stem: str) -> tuple[str | None, str]:
    if "_" in stem:
        prefix, rest = stem.split("_", 1)
        if prefix in _MSIGDB_COLLECTIONS and rest.strip():
            return prefix, rest.strip()
    return None, stem


def import_gsea_file(
    *,
    core_session: Session,
    omics_session: Session,
    path: str | Path,
    project_id: int,
    name: str | None = None,
    contrast: str | None = None,
    collection: str | None = None,
    store_metadata: bool = True,
    skip_imported: bool = True,
    force: bool = False,
) -> GSEAImportResult:
    """Import a GSEA TSV result table into the DB."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    project = core_session.get(Project, int(project_id))
    if project is None:
        raise ValueError(f"Project {project_id} not found.")

    stem = file_path.stem
    inferred_collection, inferred_contrast = _infer_collection_and_contrast(stem)
    resolved_name = (name or stem).strip() or stem
    resolved_collection = (collection or inferred_collection or None)
    resolved_contrast = (contrast or inferred_contrast or stem).strip() or stem

    analysis = (
        omics_session.query(GSEAAnalysis)
        .filter(GSEAAnalysis.project_id == int(project_id))
        .filter(GSEAAnalysis.name == resolved_name)
        .one_or_none()
    )
    created_analysis = False
    if analysis is None:
        analysis = GSEAAnalysis(
            project_id=int(project_id),
            name=resolved_name,
            contrast=resolved_contrast,
            collection=resolved_collection,
            source_path=str(file_path),
        )
        omics_session.add(analysis)
        omics_session.flush()
        created_analysis = True
    else:
        analysis.source_path = str(file_path)
        if analysis.contrast != resolved_contrast:
            analysis.contrast = resolved_contrast
        if analysis.collection is None and resolved_collection:
            analysis.collection = resolved_collection

    existing = int(
        omics_session.query(func.count(GSEAResult.id))
        .filter(GSEAResult.gsea_analysis_id == int(analysis.id))
        .scalar()
        or 0
    )
    if existing > 0 and skip_imported and not force:
        return GSEAImportResult(
            path=str(file_path),
            project_id=int(project_id),
            name=resolved_name,
            contrast=resolved_contrast,
            collection=resolved_collection,
            gsea_analysis_id=int(analysis.id),
            rows=0,
            inserted=0,
            skipped=True,
            skip_reason="already_imported",
            created_analysis=created_analysis,
            cleared_existing=False,
        )

    cleared_existing = False
    if existing > 0 and (force or not skip_imported):
        omics_session.query(GSEAResult).filter(GSEAResult.gsea_analysis_id == int(analysis.id)).delete(
            synchronize_session=False
        )
        cleared_existing = True

    now = datetime.now(UTC)
    rows_read = 0
    by_pathway: dict[str, dict[str, Any]] = {}
    fieldnames: list[str] | None = None

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = [str(col) for col in (reader.fieldnames or []) if col]
        if "pathway" not in set(fieldnames):
            raise ValueError(f"{file_path.name} is missing required column: pathway")

        for row in reader:
            if not row:
                continue
            rows_read += 1

            pathway = _safe_str(row.get("pathway"), max_len=500)
            if not pathway:
                continue

            record: dict[str, Any] = {
                "gsea_analysis_id": int(analysis.id),
                "pathway": pathway,
                "GSEAResult_CreationTS": now,
                "GSEAResult_ModificationTS": now,
                "p_value": _safe_float(row.get("pval") or row.get("p_value")),
                "p_adj": _safe_float(row.get("padj") or row.get("p_adj")),
                "log2err": _safe_float(row.get("log2err")),
                "es": _safe_float(row.get("ES") or row.get("es")),
                "nes": _safe_float(row.get("NES") or row.get("nes")),
                "size": _safe_int(row.get("size")),
                "mainpathway": _safe_bool(row.get("mainpathway")),
                "leading_edge": _safe_str(row.get("leadingEdge"), max_len=200_000),
                "leading_edge_entrezid": _safe_str(row.get("leadingEdge_entrezid"), max_len=200_000),
                "leading_edge_genesymbol": _safe_str(row.get("leadingEdge_genesymbol"), max_len=200_000),
            }

            if store_metadata:
                extras: dict[str, Any] = {}
                for col, value in row.items():
                    if col in {
                        "pathway",
                        "pval",
                        "padj",
                        "log2err",
                        "ES",
                        "NES",
                        "size",
                        "leadingEdge",
                        "leadingEdge_entrezid",
                        "leadingEdge_genesymbol",
                        "mainpathway",
                    }:
                        continue
                    cleaned = _safe_str(value, max_len=2000)
                    if cleaned is not None:
                        extras[col] = cleaned
                if extras:
                    record["metadata_json"] = json.dumps(extras, ensure_ascii=False, separators=(",", ":"))

            by_pathway[pathway] = record

    records = list(by_pathway.values())
    if records:
        omics_session.bulk_insert_mappings(GSEAResult, records)

    if store_metadata and fieldnames is not None:
        meta_payload = {
            "source": {"filename": file_path.name, "columns": fieldnames},
            "inferred": {"collection": inferred_collection, "contrast": inferred_contrast},
            "import": {"rows": rows_read, "unique_pathways": len(records)},
        }
        analysis.metadata_json = json.dumps(meta_payload, ensure_ascii=False, separators=(",", ":"))

    omics_session.flush()

    return GSEAImportResult(
        path=str(file_path),
        project_id=int(project_id),
        name=resolved_name,
        contrast=resolved_contrast,
        collection=resolved_collection,
        gsea_analysis_id=int(analysis.id),
        rows=rows_read,
        inserted=len(records),
        skipped=False,
        skip_reason=None,
        created_analysis=created_analysis,
        cleared_existing=cleared_existing,
    )

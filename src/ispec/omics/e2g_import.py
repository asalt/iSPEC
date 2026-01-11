from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ispec.db.crud import E2GCRUD
from ispec.db.models import Experiment, ExperimentRun
from ispec.logging import get_logger
from ispec.omics.models import E2G


logger = get_logger(__file__)


@dataclass(frozen=True)
class E2GImportFileResult:
    path: str
    kind: str
    experiment_id: int
    run_no: int
    search_no: int
    label: str
    experiment_run_id: int
    rows: int
    inserted: int
    updated: int
    skipped: bool = False
    skip_reason: str | None = None
    created_experiment: bool = False
    created_run: bool = False
    cleared_existing: bool = False


def discover_e2g_tsvs(data_dir: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    qual_paths = sorted(root.glob("*_e2g_QUAL.tsv"))
    quant_paths = sorted(root.glob("*_e2g_QUANT.tsv"))
    return qual_paths, quant_paths


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


def _resolve_experiment(
    core_session: Session,
    *,
    experiment_id: int,
    create_missing_experiments: bool,
) -> tuple[Experiment, bool]:
    experiment = core_session.get(Experiment, experiment_id)
    if experiment is not None:
        return experiment, False
    if not create_missing_experiments:
        raise ValueError(f"Experiment {experiment_id} not found (import experiments first).")

    experiment = Experiment(id=experiment_id, record_no=str(experiment_id))
    core_session.add(experiment)
    core_session.commit()
    core_session.refresh(experiment)
    return experiment, True


def _resolve_experiment_run(
    core_session: Session,
    *,
    experiment_id: int,
    run_no: int,
    search_no: int,
    label: str,
    create_missing_runs: bool,
    create_missing_experiments: bool,
) -> tuple[ExperimentRun, bool, bool]:
    _, created_experiment = _resolve_experiment(
        core_session,
        experiment_id=experiment_id,
        create_missing_experiments=create_missing_experiments,
    )

    run = (
        core_session.query(ExperimentRun)
        .filter_by(experiment_id=experiment_id, run_no=run_no, search_no=search_no, label=label)
        .one_or_none()
    )
    if run is not None:
        return run, created_experiment, False

    if not create_missing_runs:
        raise ValueError(
            f"ExperimentRun missing for experiment_id={experiment_id} run_no={run_no} "
            f"search_no={search_no} label={label} (sync/import runs first, or pass --create-missing-runs)."
        )

    run = ExperimentRun(
        experiment_id=experiment_id,
        run_no=run_no,
        search_no=search_no,
        label=label,
    )
    core_session.add(run)
    core_session.commit()
    core_session.refresh(run)
    return run, created_experiment, True


def _extract_file_run_info(path: Path) -> tuple[int, int, int, str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path.name} is empty.")

    first = rows[0]
    experiment_id = _safe_int(first.get("EXPRecNo"))
    run_no = _safe_int(first.get("EXPRunNo"))
    search_no = _safe_int(first.get("EXPSearchNo"))
    label_flag = _safe_int(first.get("LabelFLAG"))
    if experiment_id is None or run_no is None or search_no is None or label_flag is None:
        raise ValueError(f"{path.name} is missing required columns/values (EXPRecNo/EXPRunNo/EXPSearchNo/LabelFLAG).")

    return experiment_id, run_no, search_no, str(label_flag), rows


def _row_to_e2g_record(
    *,
    row: dict[str, str],
    experiment_run_id: int,
    kind: str,
    store_metadata: bool,
) -> dict[str, Any] | None:
    gene_id = _safe_int(row.get("GeneID"))
    label_flag = _safe_int(row.get("LabelFLAG"))
    if gene_id is None or label_flag is None:
        return None

    record: dict[str, Any] = {
        "experiment_run_id": experiment_run_id,
        "gene": str(gene_id),
        "geneidtype": "GeneID",
        "label": str(label_flag),
    }

    if kind == "qual":
        record["gene_symbol"] = _safe_str(row.get("GeneSymbol") or row.get("Symbol"), max_len=128)
        record["description"] = _safe_str(row.get("Description"), max_len=2000)
        record["taxon_id"] = _safe_int(row.get("TaxonID"))
        record["sra"] = _safe_str(row.get("SRA"), max_len=32)
        record["psms"] = _safe_int(row.get("PSMs"))
        record["psms_u2g"] = _safe_int(row.get("PSMs_u2g"))
        record["peptide_count"] = _safe_int(row.get("PeptideCount"))
        record["peptide_count_u2g"] = _safe_int(row.get("PeptideCount_u2g"))
        record["coverage"] = _safe_float(row.get("Coverage"))
        record["coverage_u2g"] = _safe_float(row.get("Coverage_u2g"))
        record["peptideprint"] = _safe_str(row.get("PeptidePrint"), max_len=200_000)

        if store_metadata:
            record["metadata_json"] = json.dumps(
                {
                    "qual": {
                        "source": "QUAL",
                        "gp_group": row.get("GPGroup"),
                        "gp_groups_all": row.get("GPGroups_All"),
                        "id_group": row.get("IDGroup"),
                        "id_set": row.get("IDSet"),
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

    elif kind == "quant":
        record["sra"] = _safe_str(row.get("SRA"), max_len=32)
        record["area_sum_u2g_0"] = _safe_float(row.get("AreaSum_u2g_0"))
        record["area_sum_u2g_all"] = _safe_float(row.get("AreaSum_u2g_all"))
        record["area_sum_max"] = _safe_float(row.get("AreaSum_max"))
        record["area_sum_dstrAdj"] = _safe_float(row.get("AreaSum_dstrAdj"))
        record["iBAQ_dstrAdj"] = _safe_float(row.get("iBAQ_dstrAdj"))
        if store_metadata:
            record["metadata_json"] = json.dumps(
                {"quant": {"source": "QUANT"}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return record


def _run_has_kind_data(omics_session: Session, *, run_id: int, kind: str) -> bool:
    query = omics_session.query(E2G.id).filter(E2G.experiment_run_id == run_id)
    if kind == "qual":
        query = query.filter(
            (E2G.psms.is_not(None))
            | (E2G.psms_u2g.is_not(None))
            | (E2G.peptide_count.is_not(None))
            | (E2G.peptide_count_u2g.is_not(None))
            | (E2G.coverage.is_not(None))
            | (E2G.coverage_u2g.is_not(None))
            | (E2G.peptideprint.is_not(None))
        )
    elif kind == "quant":
        query = query.filter(
            (E2G.area_sum_u2g_0.is_not(None))
            | (E2G.area_sum_u2g_all.is_not(None))
            | (E2G.area_sum_max.is_not(None))
            | (E2G.area_sum_dstrAdj.is_not(None))
            | (E2G.iBAQ_dstrAdj.is_not(None))
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return query.limit(1).first() is not None


def import_e2g_tsv(
    *,
    core_session: Session,
    omics_session: Session,
    path: str | Path,
    kind: str,
    create_missing_runs: bool = True,
    create_missing_experiments: bool = True,
    store_metadata: bool = False,
    skip_imported: bool = True,
    force: bool = False,
    cleared_run_ids: set[int] | None = None,
) -> E2GImportFileResult:
    tsv_path = Path(path).expanduser().resolve()
    if not tsv_path.exists():
        raise FileNotFoundError(str(tsv_path))

    experiment_id, run_no, search_no, label, rows = _extract_file_run_info(tsv_path)
    run, created_experiment, created_run = _resolve_experiment_run(
        core_session,
        experiment_id=experiment_id,
        run_no=run_no,
        search_no=search_no,
        label=label,
        create_missing_runs=create_missing_runs,
        create_missing_experiments=create_missing_experiments,
    )

    cleared_existing = False
    if force:
        if cleared_run_ids is None or int(run.id) not in cleared_run_ids:
            deleted = (
                omics_session.query(E2G)
                .filter(E2G.experiment_run_id == int(run.id))
                .delete(synchronize_session=False)
            )
            if deleted:
                logger.info("Cleared %d existing E2G rows for ExperimentRun %s", int(deleted), int(run.id))
            cleared_existing = True
            if cleared_run_ids is not None:
                cleared_run_ids.add(int(run.id))
            omics_session.commit()

    if skip_imported and not force:
        if _run_has_kind_data(omics_session, run_id=int(run.id), kind=kind):
            return E2GImportFileResult(
                path=str(tsv_path),
                kind=kind,
                experiment_id=experiment_id,
                run_no=run_no,
                search_no=search_no,
                label=label,
                experiment_run_id=int(run.id),
                rows=0,
                inserted=0,
                updated=0,
                skipped=True,
                skip_reason=f"{kind} already imported for ExperimentRun {int(run.id)}",
                created_experiment=created_experiment,
                created_run=created_run,
                cleared_existing=False,
            )

    records: list[dict[str, Any]] = []
    for row in rows:
        rec = _row_to_e2g_record(
            row=row,
            experiment_run_id=int(run.id),
            kind=kind,
            store_metadata=store_metadata,
        )
        if rec is not None:
            records.append(rec)

    crud = E2GCRUD()
    result = crud.bulk_upsert(omics_session, records)

    # Mark flags for discoverability in the UI.
    experiment = core_session.get(Experiment, experiment_id)
    if experiment is not None and hasattr(experiment, "exp_exp2gene_FLAG"):
        experiment.exp_exp2gene_FLAG = True
    if hasattr(run, "gpgrouper_flag"):
        run.gpgrouper_flag = True
    core_session.commit()

    return E2GImportFileResult(
        path=str(tsv_path),
        kind=kind,
        experiment_id=experiment_id,
        run_no=run_no,
        search_no=search_no,
        label=label,
        experiment_run_id=int(run.id),
        rows=len(records),
        inserted=int(result.get("inserted", 0)),
        updated=int(result.get("updated", 0)),
        skipped=False,
        skip_reason=None,
        created_experiment=created_experiment,
        created_run=created_run,
        cleared_existing=cleared_existing,
    )


def import_e2g_files(
    *,
    core_session: Session,
    omics_session: Session,
    qual_paths: list[Path] | None = None,
    quant_paths: list[Path] | None = None,
    create_missing_runs: bool = True,
    create_missing_experiments: bool = True,
    store_metadata: bool = False,
    skip_imported: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    qual_paths = qual_paths or []
    quant_paths = quant_paths or []

    results: list[E2GImportFileResult] = []
    errors: list[str] = []
    inserted_total = 0
    updated_total = 0
    skipped_total = 0
    cleared_run_ids: set[int] = set()

    for path in qual_paths:
        try:
            res = import_e2g_tsv(
                core_session=core_session,
                omics_session=omics_session,
                path=path,
                kind="qual",
                create_missing_runs=create_missing_runs,
                create_missing_experiments=create_missing_experiments,
                store_metadata=store_metadata,
                skip_imported=skip_imported,
                force=force,
                cleared_run_ids=cleared_run_ids,
            )
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        results.append(res)
        inserted_total += res.inserted
        updated_total += res.updated
        skipped_total += 1 if res.skipped else 0

    for path in quant_paths:
        try:
            res = import_e2g_tsv(
                core_session=core_session,
                omics_session=omics_session,
                path=path,
                kind="quant",
                create_missing_runs=create_missing_runs,
                create_missing_experiments=create_missing_experiments,
                store_metadata=store_metadata,
                skip_imported=skip_imported,
                force=force,
                cleared_run_ids=cleared_run_ids,
            )
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        results.append(res)
        inserted_total += res.inserted
        updated_total += res.updated
        skipped_total += 1 if res.skipped else 0

    payload = {
        "files": [res.__dict__ for res in results],
        "inserted": inserted_total,
        "updated": updated_total,
        "skipped": skipped_total,
        "errors": errors,
    }
    logger.info("E2G import summary: %s", payload)
    return payload

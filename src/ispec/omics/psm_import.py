from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ispec.db.models import Experiment, ExperimentRun
from ispec.logging import get_logger
from ispec.omics.e2g_import import _resolve_experiment_run, _safe_float, _safe_int, _safe_str
from ispec.omics.models import PSM


logger = get_logger(__file__)

_RUN_ID_COLUMNS = ("ExperimentRunID", "experiment_run_id", "exprun_id")
_SCAN_COLUMNS = ("ScanNumber", "scan_number", "Scan", "scan", "First Scan", "FirstScan")
_PEPTIDE_COLUMNS = ("Peptide", "peptide", "Sequence", "sequence", "Annotated Sequence")
_CHARGE_COLUMNS = ("Charge", "charge", "z")
_SCORE_COLUMNS = ("Score", "score", "XCorr", "xcorr", "PercolatorScore")
_SCORE_TYPE_COLUMNS = ("ScoreType", "score_type")
_QVALUE_COLUMNS = ("QValue", "q_value", "q-value", "PercolatorQValue", "PEP")
_PROTEIN_COLUMNS = ("Protein", "protein", "Proteins", "proteins", "ProteinAccession")
_MOD_COLUMNS = ("Mods", "mods", "Modifications", "modifications", "Assigned Modifications")
_PRECURSOR_COLUMNS = ("PrecursorMz", "precursor_mz", "Precursor M/Z")
_RT_COLUMNS = ("RetentionTime", "retention_time", "RT", "rt")
_INTENSITY_COLUMNS = ("Intensity", "intensity", "PrecursorIntensity", "Area")
_KNOWN_COLUMNS = set(
    _RUN_ID_COLUMNS
    + _SCAN_COLUMNS
    + _PEPTIDE_COLUMNS
    + _CHARGE_COLUMNS
    + _SCORE_COLUMNS
    + _SCORE_TYPE_COLUMNS
    + _QVALUE_COLUMNS
    + _PROTEIN_COLUMNS
    + _MOD_COLUMNS
    + _PRECURSOR_COLUMNS
    + _RT_COLUMNS
    + _INTENSITY_COLUMNS
    + ("EXPRecNo", "EXPRunNo", "EXPSearchNo", "LabelFLAG")
)


@dataclass(frozen=True)
class PSMImportFileResult:
    path: str
    experiment_run_id: int | None
    experiment_run_ids: list[int]
    rows: int
    inserted: int
    updated: int
    skipped: bool = False
    skip_reason: str | None = None
    created_experiment: bool = False
    created_run: bool = False
    cleared_existing: bool = False


def discover_psm_tsvs(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    patterns = (
        "*_psm.tsv",
        "*_psms.tsv",
        "*_PSM.tsv",
        "*_PSMs.tsv",
        "*psm*.tab",
        "*psm*.txt",
        "*psm*.csv",
    )
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def _first_value(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        return dialect.delimiter
    except Exception:
        return "\t"


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    delimiter = _detect_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = [str(name) for name in (reader.fieldnames or []) if name]
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    return rows, fieldnames


def _normalize_score_type(row: dict[str, str]) -> str | None:
    explicit = _safe_str(_first_value(row, _SCORE_TYPE_COLUMNS), max_len=64)
    if explicit is not None:
        return explicit
    if row.get("XCorr") is not None:
        return "XCorr"
    if row.get("PercolatorScore") is not None:
        return "PercolatorScore"
    if row.get("Score") is not None or row.get("score") is not None:
        return "Score"
    return None


def _resolve_target_run(
    *,
    core_session: Session,
    row: dict[str, str],
    experiment_run_id: int | None,
    experiment_id: int | None,
    run_no: int | None,
    search_no: int | None,
    label: str | None,
    create_missing_runs: bool,
    create_missing_experiments: bool,
) -> tuple[ExperimentRun, bool, bool]:
    explicit_run_id = experiment_run_id
    if explicit_run_id is None:
        explicit_run_id = _safe_int(_first_value(row, _RUN_ID_COLUMNS))
    if explicit_run_id is not None:
        run = core_session.get(ExperimentRun, int(explicit_run_id))
        if run is None:
            raise ValueError(f"ExperimentRun {explicit_run_id} not found.")
        return run, False, False

    resolved_experiment_id = (
        int(experiment_id)
        if experiment_id is not None
        else _safe_int(row.get("EXPRecNo"))
    )
    resolved_run_no = int(run_no) if run_no is not None else _safe_int(row.get("EXPRunNo"))
    resolved_search_no = (
        int(search_no) if search_no is not None else _safe_int(row.get("EXPSearchNo"))
    )
    resolved_label = (
        str(label).strip()
        if label is not None and str(label).strip()
        else str(_safe_int(row.get("LabelFLAG")) if _safe_int(row.get("LabelFLAG")) is not None else 0)
    )

    if resolved_experiment_id is None or resolved_run_no is None or resolved_search_no is None:
        raise ValueError(
            "Missing required run identifiers. Provide experiment_run_id, "
            "experiment_id/run_no/search_no, or EXPRecNo/EXPRunNo/EXPSearchNo columns."
        )

    run, created_experiment, created_run = _resolve_experiment_run(
        core_session,
        experiment_id=int(resolved_experiment_id),
        run_no=int(resolved_run_no),
        search_no=int(resolved_search_no),
        label=str(resolved_label),
        create_missing_runs=create_missing_runs,
        create_missing_experiments=create_missing_experiments,
    )
    return run, bool(created_experiment), bool(created_run)


def _run_has_psm_data(omics_session: Session, *, run_id: int) -> bool:
    return (
        omics_session.query(PSM.id)
        .filter(PSM.experiment_run_id == int(run_id))
        .limit(1)
        .first()
        is not None
    )


def _metadata_json(
    row: dict[str, str],
    *,
    fieldnames: list[str],
    filename: str,
    store_metadata: bool,
) -> str | None:
    if not store_metadata:
        return None
    extras: dict[str, str] = {}
    for key, value in row.items():
        if key in _KNOWN_COLUMNS:
            continue
        cleaned = _safe_str(value, max_len=2000)
        if cleaned is not None:
            extras[key] = cleaned
    payload: dict[str, Any] = {
        "source": {"filename": filename, "columns": fieldnames},
    }
    if extras:
        payload["extra"] = extras
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _row_to_record(
    *,
    row: dict[str, str],
    experiment_run_id: int,
    fieldnames: list[str],
    filename: str,
    store_metadata: bool,
) -> dict[str, Any] | None:
    scan_number = _safe_int(_first_value(row, _SCAN_COLUMNS))
    peptide = _safe_str(_first_value(row, _PEPTIDE_COLUMNS), max_len=2000)
    if scan_number is None or peptide is None:
        return None

    score = _safe_float(_first_value(row, _SCORE_COLUMNS))
    if score is None:
        score = _safe_float(row.get("XCorr"))

    return {
        "experiment_run_id": int(experiment_run_id),
        "scan_number": int(scan_number),
        "peptide": peptide,
        "charge": _safe_int(_first_value(row, _CHARGE_COLUMNS)),
        "score": score,
        "score_type": _normalize_score_type(row),
        "q_value": _safe_float(_first_value(row, _QVALUE_COLUMNS)),
        "protein": _safe_str(_first_value(row, _PROTEIN_COLUMNS), max_len=200_000),
        "mods": _safe_str(_first_value(row, _MOD_COLUMNS), max_len=200_000),
        "precursor_mz": _safe_float(_first_value(row, _PRECURSOR_COLUMNS)),
        "retention_time": _safe_float(_first_value(row, _RT_COLUMNS)),
        "intensity": _safe_float(_first_value(row, _INTENSITY_COLUMNS)),
        "metadata_json": _metadata_json(
            row,
            fieldnames=fieldnames,
            filename=filename,
            store_metadata=store_metadata,
        ),
    }


def _mark_run_flags(core_session: Session, *, run: ExperimentRun) -> None:
    run.db_search_flag = True
    experiment = core_session.get(Experiment, int(run.experiment_id))
    if experiment is not None:
        experiment.exp_Data_FLAG = True
    core_session.flush()


def import_psm_file(
    *,
    core_session: Session,
    omics_session: Session,
    path: str | Path,
    experiment_run_id: int | None = None,
    experiment_id: int | None = None,
    run_no: int | None = None,
    search_no: int | None = None,
    label: str | None = None,
    create_missing_runs: bool = True,
    create_missing_experiments: bool = True,
    store_metadata: bool = False,
    skip_imported: bool = True,
    force: bool = False,
) -> PSMImportFileResult:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    rows, fieldnames = _load_rows(file_path)
    run_cache: dict[tuple[tuple[str, str], ...], tuple[ExperimentRun, bool, bool]] = {}
    run_ids: list[int] = []
    created_experiment = False
    created_run = False

    for row in rows:
        key = tuple(
            sorted(
                (str(k), str(v))
                for k, v in row.items()
                if k in _RUN_ID_COLUMNS or k in {"EXPRecNo", "EXPRunNo", "EXPSearchNo", "LabelFLAG"}
            )
        )
        if key in run_cache:
            run, created_exp, created_run_row = run_cache[key]
        else:
            run, created_exp, created_run_row = _resolve_target_run(
                core_session=core_session,
                row=row,
                experiment_run_id=experiment_run_id,
                experiment_id=experiment_id,
                run_no=run_no,
                search_no=search_no,
                label=label,
                create_missing_runs=create_missing_runs,
                create_missing_experiments=create_missing_experiments,
            )
            run_cache[key] = (run, created_exp, created_run_row)
        if int(run.id) not in run_ids:
            run_ids.append(int(run.id))
        created_experiment = created_experiment or bool(created_exp)
        created_run = created_run or bool(created_run_row)

    if run_ids and skip_imported and not force:
        if any(_run_has_psm_data(omics_session, run_id=run_id) for run_id in run_ids):
            return PSMImportFileResult(
                path=str(file_path),
                experiment_run_id=experiment_run_id if experiment_run_id is not None else (run_ids[0] if len(run_ids) == 1 else None),
                experiment_run_ids=sorted(run_ids),
                rows=0,
                inserted=0,
                updated=0,
                skipped=True,
                skip_reason="already_imported",
                created_experiment=created_experiment,
                created_run=created_run,
                cleared_existing=False,
            )

    cleared_existing = False
    if run_ids and force:
        for run_id in run_ids:
            (
                omics_session.query(PSM)
                .filter(PSM.experiment_run_id == int(run_id))
                .delete(synchronize_session=False)
            )
        cleared_existing = True

    inserted = 0
    updated = 0
    rows_read = 0
    for row in rows:
        rows_read += 1
        run, _, _ = _resolve_target_run(
            core_session=core_session,
            row=row,
            experiment_run_id=experiment_run_id,
            experiment_id=experiment_id,
            run_no=run_no,
            search_no=search_no,
            label=label,
            create_missing_runs=create_missing_runs,
            create_missing_experiments=create_missing_experiments,
        )
        _mark_run_flags(core_session, run=run)
        record = _row_to_record(
            row=row,
            experiment_run_id=int(run.id),
            fieldnames=fieldnames,
            filename=file_path.name,
            store_metadata=store_metadata,
        )
        if record is None:
            continue

        existing_query = (
            omics_session.query(PSM)
            .filter(PSM.experiment_run_id == int(run.id))
            .filter(PSM.scan_number == int(record["scan_number"]))
            .filter(PSM.peptide == str(record["peptide"]))
        )
        if record.get("charge") is None:
            existing_query = existing_query.filter(PSM.charge.is_(None))
        else:
            existing_query = existing_query.filter(PSM.charge == int(record["charge"]))
        existing = existing_query.one_or_none()
        if existing is None:
            omics_session.add(PSM(**record))
            inserted += 1
            continue

        changed = False
        for field, value in record.items():
            if getattr(existing, field) != value:
                setattr(existing, field, value)
                changed = True
        if changed:
            updated += 1

    omics_session.flush()
    return PSMImportFileResult(
        path=str(file_path),
        experiment_run_id=experiment_run_id if experiment_run_id is not None else (run_ids[0] if len(run_ids) == 1 else None),
        experiment_run_ids=sorted(run_ids),
        rows=rows_read,
        inserted=inserted,
        updated=updated,
        skipped=False,
        skip_reason=None,
        created_experiment=created_experiment,
        created_run=created_run,
        cleared_existing=cleared_existing,
    )


def import_psm_files(
    *,
    core_session: Session,
    omics_session: Session,
    paths: list[str | Path],
    experiment_run_id: int | None = None,
    experiment_id: int | None = None,
    run_no: int | None = None,
    search_no: int | None = None,
    label: str | None = None,
    create_missing_runs: bool = True,
    create_missing_experiments: bool = True,
    store_metadata: bool = False,
    skip_imported: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    results: list[PSMImportFileResult] = []
    errors: list[dict[str, str]] = []

    for path in [Path(p).expanduser().resolve() for p in paths]:
        try:
            result = import_psm_file(
                core_session=core_session,
                omics_session=omics_session,
                path=path,
                experiment_run_id=experiment_run_id,
                experiment_id=experiment_id,
                run_no=run_no,
                search_no=search_no,
                label=label,
                create_missing_runs=create_missing_runs,
                create_missing_experiments=create_missing_experiments,
                store_metadata=store_metadata,
                skip_imported=skip_imported,
                force=force,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Failed importing PSM file: %s", path)
            errors.append({"path": str(path), "error": str(exc)})

    return {
        "files": [result.__dict__ for result in results],
        "inserted": sum(result.inserted for result in results),
        "updated": sum(result.updated for result in results),
        "skipped": sum(1 for result in results if result.skipped),
        "errors": errors,
    }

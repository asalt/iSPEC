from __future__ import annotations

import csv
import json
from pathlib import Path

from ispec.db.models import Experiment, ExperimentRun
from ispec.omics.models import PSM
from ispec.omics.psm_import import import_psm_file


def _write_delimited(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_import_psm_file_creates_missing_experiment_and_run(db_session, omics_session, tmp_path):
    psm_path = tmp_path / "run1_psm.tsv"
    _write_delimited(
        psm_path,
        fieldnames=[
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
            "LabelFLAG",
            "ScanNumber",
            "Peptide",
            "Charge",
            "XCorr",
            "Protein",
            "Confidence",
        ],
        rows=[
            {
                "EXPRecNo": "901",
                "EXPRunNo": "2",
                "EXPSearchNo": "1",
                "LabelFLAG": "0",
                "ScanNumber": "15",
                "Peptide": "PEPTIDEK",
                "Charge": "2",
                "XCorr": "3.2",
                "Protein": "P12345",
                "Confidence": "High",
            }
        ],
    )

    result = import_psm_file(
        core_session=db_session,
        omics_session=omics_session,
        path=psm_path,
        store_metadata=True,
    )

    assert result.created_experiment is True
    assert result.created_run is True
    assert result.inserted == 1
    assert result.updated == 0

    experiment = db_session.get(Experiment, 901)
    assert experiment is not None
    assert experiment.exp_Data_FLAG is True

    run = (
        db_session.query(ExperimentRun)
        .filter_by(experiment_id=901, run_no=2, search_no=1, label="0")
        .one()
    )
    assert run.db_search_flag is True

    row = (
        omics_session.query(PSM)
        .filter_by(experiment_run_id=run.id, scan_number=15, peptide="PEPTIDEK")
        .one()
    )
    assert row.charge == 2
    assert row.score == 3.2
    assert row.score_type == "XCorr"
    assert row.protein == "P12345"

    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("extra", {}).get("Confidence") == "High"


def test_import_psm_file_skips_then_force_reimports(db_session, omics_session, tmp_path):
    experiment = Experiment(id=902, project_id=None, record_no="902")
    run = ExperimentRun(experiment_id=902, run_no=1, search_no=3, label="0")
    db_session.add_all([experiment, run])
    db_session.commit()
    db_session.refresh(run)

    psm_path = tmp_path / "run2_psm.tsv"
    _write_delimited(
        psm_path,
        fieldnames=["ScanNumber", "Peptide", "Charge", "Score", "Protein"],
        rows=[
            {
                "ScanNumber": "20",
                "Peptide": "TESTPEP",
                "Charge": "3",
                "Score": "10.0",
                "Protein": "Q99999",
            }
        ],
    )

    first = import_psm_file(
        core_session=db_session,
        omics_session=omics_session,
        path=psm_path,
        experiment_run_id=int(run.id),
    )
    assert first.skipped is False
    assert first.inserted == 1

    second = import_psm_file(
        core_session=db_session,
        omics_session=omics_session,
        path=psm_path,
        experiment_run_id=int(run.id),
    )
    assert second.skipped is True
    assert second.inserted == 0

    _write_delimited(
        psm_path,
        fieldnames=["ScanNumber", "Peptide", "Charge", "Score", "Protein"],
        rows=[
            {
                "ScanNumber": "20",
                "Peptide": "TESTPEP",
                "Charge": "3",
                "Score": "14.5",
                "Protein": "Q99999",
            }
        ],
    )

    third = import_psm_file(
        core_session=db_session,
        omics_session=omics_session,
        path=psm_path,
        experiment_run_id=int(run.id),
        force=True,
    )
    assert third.skipped is False
    assert third.inserted == 1
    assert third.cleared_existing is True

    assert (
        omics_session.query(PSM)
        .filter_by(experiment_run_id=run.id, scan_number=20, peptide="TESTPEP")
        .count()
        == 1
    )
    row = (
        omics_session.query(PSM)
        .filter_by(experiment_run_id=run.id, scan_number=20, peptide="TESTPEP")
        .one()
    )
    assert row.score == 14.5

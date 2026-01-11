from __future__ import annotations

import csv
import json
from pathlib import Path

from ispec.db.models import Experiment, ExperimentRun, Project
from ispec.omics.models import E2G
from ispec.omics.e2g_import import import_e2g_files, import_e2g_tsv


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_import_e2g_qual_then_quant(db_session, omics_session, tmp_path):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    experiment = Experiment(id=100, project_id=1, record_no="100", exp_Name="Experiment 100")
    run = ExperimentRun(experiment_id=100, run_no=1, search_no=4, label="0")
    db_session.add_all([project, experiment, run])
    db_session.commit()
    db_session.refresh(run)

    qual_path = tmp_path / "100_1_4_labelnone_e2g_QUAL.tsv"
    _write_tsv(
        qual_path,
        fieldnames=[
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
            "LabelFLAG",
            "GeneID",
            "GeneSymbol",
            "Description",
            "TaxonID",
            "SRA",
            "PSMs",
            "PSMs_u2g",
            "PeptideCount",
            "PeptideCount_u2g",
            "Coverage",
            "Coverage_u2g",
            "PeptidePrint",
            "GPGroup",
            "GPGroups_All",
            "IDGroup",
            "IDSet",
        ],
        rows=[
            {
                "EXPRecNo": "100",
                "EXPRunNo": "1",
                "EXPSearchNo": "4",
                "LabelFLAG": "0",
                "GeneID": "123",
                "GeneSymbol": "KRAS",
                "Description": "Kirsten rat sarcoma viral oncogene homolog",
                "TaxonID": "9606",
                "SRA": "SRA1",
                "PSMs": "10",
                "PSMs_u2g": "9",
                "PeptideCount": "3",
                "PeptideCount_u2g": "2",
                "Coverage": "12.5",
                "Coverage_u2g": "11.0",
                "PeptidePrint": "PEP_A__PEP_B__PEP_C",
                "GPGroup": "G1",
                "GPGroups_All": "G1,G2",
                "IDGroup": "1",
                "IDSet": "1",
            }
        ],
    )

    res_qual = import_e2g_tsv(
        core_session=db_session,
        omics_session=omics_session,
        path=qual_path,
        kind="qual",
        store_metadata=True,
    )
    assert res_qual.inserted == 1
    assert res_qual.updated == 0
    assert res_qual.experiment_run_id == run.id

    row = (
        omics_session.query(E2G)
        .filter_by(experiment_run_id=run.id, gene="123", geneidtype="GeneID")
        .one()
    )
    assert row.gene_symbol == "KRAS"
    assert row.psms == 10
    assert row.psms_u2g == 9
    assert row.peptideprint == "PEP_A__PEP_B__PEP_C"

    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("qual", {}).get("source") == "QUAL"

    db_session.refresh(experiment)
    db_session.refresh(run)
    assert experiment.exp_exp2gene_FLAG is True
    assert run.gpgrouper_flag is True

    quant_path = tmp_path / "100_1_4_labelnone_e2g_QUANT.tsv"
    _write_tsv(
        quant_path,
        fieldnames=[
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
            "LabelFLAG",
            "GeneID",
            "SRA",
            "AreaSum_u2g_all",
            "iBAQ_dstrAdj",
        ],
        rows=[
            {
                "EXPRecNo": "100",
                "EXPRunNo": "1",
                "EXPSearchNo": "4",
                "LabelFLAG": "0",
                "GeneID": "123",
                "SRA": "SRA1",
                "AreaSum_u2g_all": "123.4",
                "iBAQ_dstrAdj": "0.5",
            }
        ],
    )

    res_quant = import_e2g_tsv(
        core_session=db_session,
        omics_session=omics_session,
        path=quant_path,
        kind="quant",
        store_metadata=True,
    )
    assert res_quant.inserted == 0
    assert res_quant.updated == 1

    row = (
        omics_session.query(E2G)
        .filter_by(experiment_run_id=run.id, gene="123", geneidtype="GeneID")
        .one()
    )
    assert row.area_sum_u2g_all == 123.4
    assert row.iBAQ_dstrAdj == 0.5

    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("qual", {}).get("source") == "QUAL"
    assert meta.get("quant", {}).get("source") == "QUANT"


def test_import_e2g_creates_missing_experiment_and_run(db_session, omics_session, tmp_path):
    qual_path = tmp_path / "200_2_1_labelnone_e2g_QUAL.tsv"
    _write_tsv(
        qual_path,
        fieldnames=[
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
            "LabelFLAG",
            "GeneID",
            "GeneSymbol",
            "PSMs",
        ],
        rows=[
            {
                "EXPRecNo": "200",
                "EXPRunNo": "2",
                "EXPSearchNo": "1",
                "LabelFLAG": "0",
                "GeneID": "123",
                "GeneSymbol": "KRAS",
                "PSMs": "1",
            }
        ],
    )

    res = import_e2g_tsv(
        core_session=db_session,
        omics_session=omics_session,
        path=qual_path,
        kind="qual",
    )
    assert res.created_experiment is True
    assert res.created_run is True
    assert res.inserted == 1

    experiment = db_session.get(Experiment, 200)
    assert experiment is not None
    assert experiment.record_no == "200"

    run = (
        db_session.query(ExperimentRun)
        .filter_by(experiment_id=200, run_no=2, search_no=1, label="0")
        .one()
    )
    assert run.gpgrouper_flag is True
    assert (
        omics_session.query(E2G)
        .filter_by(experiment_run_id=run.id, gene="123", geneidtype="GeneID")
        .first()
        is not None
    )


def test_import_e2g_skips_when_already_imported(db_session, omics_session, tmp_path):
    experiment = Experiment(id=300, project_id=None, record_no="300")
    run = ExperimentRun(experiment_id=300, run_no=1, search_no=1, label="0")
    db_session.add_all([experiment, run])
    db_session.commit()
    db_session.refresh(run)

    qual_path = tmp_path / "300_1_1_labelnone_e2g_QUAL.tsv"
    _write_tsv(
        qual_path,
        fieldnames=["EXPRecNo", "EXPRunNo", "EXPSearchNo", "LabelFLAG", "GeneID", "PSMs"],
        rows=[
            {
                "EXPRecNo": "300",
                "EXPRunNo": "1",
                "EXPSearchNo": "1",
                "LabelFLAG": "0",
                "GeneID": "123",
                "PSMs": "5",
            }
        ],
    )

    first = import_e2g_tsv(
        core_session=db_session,
        omics_session=omics_session,
        path=qual_path,
        kind="qual",
    )
    assert first.skipped is False
    assert first.inserted == 1

    second = import_e2g_tsv(
        core_session=db_session,
        omics_session=omics_session,
        path=qual_path,
        kind="qual",
    )
    assert second.skipped is True
    assert second.inserted == 0
    assert second.updated == 0


def test_import_e2g_force_reimport_clears_once_per_run(db_session, omics_session, tmp_path):
    experiment = Experiment(id=400, project_id=None, record_no="400")
    run = ExperimentRun(experiment_id=400, run_no=1, search_no=1, label="0")
    db_session.add_all([experiment, run])
    db_session.flush()

    omics_session.add(
        E2G(experiment_run_id=run.id, gene="999", geneidtype="GeneID", label="0", psms=99)
    )
    omics_session.commit()

    qual_path = tmp_path / "400_1_1_labelnone_e2g_QUAL.tsv"
    _write_tsv(
        qual_path,
        fieldnames=["EXPRecNo", "EXPRunNo", "EXPSearchNo", "LabelFLAG", "GeneID", "PSMs"],
        rows=[
            {
                "EXPRecNo": "400",
                "EXPRunNo": "1",
                "EXPSearchNo": "1",
                "LabelFLAG": "0",
                "GeneID": "123",
                "PSMs": "10",
            }
        ],
    )

    quant_path = tmp_path / "400_1_1_labelnone_e2g_QUANT.tsv"
    _write_tsv(
        quant_path,
        fieldnames=["EXPRecNo", "EXPRunNo", "EXPSearchNo", "LabelFLAG", "GeneID", "iBAQ_dstrAdj"],
        rows=[
            {
                "EXPRecNo": "400",
                "EXPRunNo": "1",
                "EXPSearchNo": "1",
                "LabelFLAG": "0",
                "GeneID": "123",
                "iBAQ_dstrAdj": "0.5",
            }
        ],
    )

    summary = import_e2g_files(
        core_session=db_session,
        omics_session=omics_session,
        qual_paths=[qual_path],
        quant_paths=[quant_path],
        force=True,
        store_metadata=True,
    )
    assert summary["errors"] == []

    # Old placeholder gene cleared.
    assert (
        omics_session.query(E2G)
        .filter_by(experiment_run_id=run.id, gene="999", geneidtype="GeneID")
        .first()
        is None
    )

    # New gene has both QUAL and QUANT fields populated (de-duped clear).
    row = (
        omics_session.query(E2G)
        .filter_by(experiment_run_id=run.id, gene="123", geneidtype="GeneID")
        .one()
    )
    assert row.psms == 10
    assert row.iBAQ_dstrAdj == 0.5

    meta = json.loads(row.metadata_json or "{}")
    assert meta.get("qual", {}).get("source") == "QUAL"
    assert meta.get("quant", {}).get("source") == "QUANT"

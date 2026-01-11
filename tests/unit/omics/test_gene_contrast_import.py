from __future__ import annotations

import csv
import json
from pathlib import Path

from ispec.db.models import Project
from ispec.omics.models import GeneContrast, GeneContrastStat
from ispec.omics.gene_contrast_import import import_gene_contrast_file


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_import_gene_contrast_file_infers_contrast_and_stores_rows(
    db_session, omics_session, tmp_path
):
    project = Project(id=1161, prj_AddedBy="test", prj_ProjectTitle="Project 1161")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "params_group_PCL_Bi_MSC_minus_PCLonly_dir_B.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=[
            "GeneID",
            "log2_FC",
            "CI.L",
            "CI.R",
            "AveExpr",
            "t",
            "pValue",
            "pAdj",
            "B",
            "20A",
            "GeneSymbol",
            "FunCats",
            "GeneDescription",
            "signedlogP",
        ],
        rows=[
            {
                "GeneID": "123",
                "log2_FC": "1.5",
                "CI.L": "1.0",
                "CI.R": "2.0",
                "AveExpr": "10.0",
                "t": "3.0",
                "pValue": "0.01",
                "pAdj": "0.02",
                "B": "5.0",
                "20A": "9.9",
                "GeneSymbol": "KRAS",
                "FunCats": "oncogene",
                "GeneDescription": "Kirsten rat sarcoma viral oncogene homolog",
                "signedlogP": "2.0",
            },
            {
                "GeneID": "456",
                "log2_FC": "-1.0",
                "CI.L": "-1.2",
                "CI.R": "-0.8",
                "AveExpr": "8.0",
                "t": "-2.0",
                "pValue": "0.05",
                "pAdj": "0.10",
                "B": "1.0",
                "20A": "7.7",
                "GeneSymbol": "EGFR",
                "FunCats": "",
                "GeneDescription": "Epidermal growth factor receptor",
                "signedlogP": "-1.3",
            },
        ],
    )

    result = import_gene_contrast_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1161,
    )
    assert result.skipped is False
    assert result.inserted == 2
    assert result.contrast == "PCL_Bi_MSC_minus_PCLonly"
    assert result.name == "params_group_PCL_Bi_MSC_minus_PCLonly_dir_B"

    contrast_row = (
        omics_session.query(GeneContrast).filter_by(id=result.gene_contrast_id).one()
    )
    assert contrast_row.project_id == 1161
    assert contrast_row.contrast == "PCL_Bi_MSC_minus_PCLonly"
    assert contrast_row.source_path.endswith(tsv_path.name)
    meta = json.loads(contrast_row.metadata_json or "{}")
    assert meta["inferred"]["direction"] == "B"
    assert "20A" in meta["source"]["sample_columns"]

    stats = (
        omics_session.query(GeneContrastStat)
        .filter_by(gene_contrast_id=contrast_row.id)
        .order_by(GeneContrastStat.gene_id.asc())
        .all()
    )
    assert [row.gene_id for row in stats] == [123, 456]
    assert stats[0].gene_symbol == "KRAS"
    assert stats[0].log2_fc == 1.5
    assert stats[0].p_adj == 0.02

    row_meta = json.loads(stats[0].metadata_json or "{}")
    assert row_meta["ci"]["low"] == 1.0
    assert row_meta["ci"]["high"] == 2.0
    assert row_meta["b"] == 5.0


def test_import_gene_contrast_file_skips_when_already_imported(
    db_session, omics_session, tmp_path
):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "x_group_A_vs_B_dir_A.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=["GeneID", "log2_FC", "pValue", "pAdj", "t", "signedlogP"],
        rows=[{"GeneID": "1", "log2_FC": "0.5", "pValue": "0.1", "pAdj": "0.2", "t": "1", "signedlogP": "1"}],
    )

    first = import_gene_contrast_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1,
    )
    assert first.skipped is False
    assert first.inserted == 1

    second = import_gene_contrast_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1,
    )
    assert second.skipped is True
    assert second.inserted == 0


def test_import_gene_contrast_file_force_overwrites_existing(
    db_session, omics_session, tmp_path
):
    project = Project(id=2, prj_AddedBy="test", prj_ProjectTitle="Project 2")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "x_group_A_vs_B_dir_A.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=["GeneID", "log2_FC", "pValue", "pAdj"],
        rows=[{"GeneID": "1", "log2_FC": "0.5", "pValue": "0.1", "pAdj": "0.2"}],
    )
    first = import_gene_contrast_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=2,
    )
    assert first.inserted == 1

    _write_tsv(
        tsv_path,
        fieldnames=["GeneID", "log2_FC", "pValue", "pAdj"],
        rows=[{"GeneID": "1", "log2_FC": "1.0", "pValue": "0.05", "pAdj": "0.1"}],
    )
    second = import_gene_contrast_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=2,
        force=True,
    )
    assert second.cleared_existing is True
    assert second.inserted == 1

    contrast_row = (
        omics_session.query(GeneContrast).filter_by(id=second.gene_contrast_id).one()
    )
    stat = (
        omics_session.query(GeneContrastStat)
        .filter_by(gene_contrast_id=contrast_row.id, gene_id=1)
        .one()
    )
    assert stat.log2_fc == 1.0
    assert stat.p_adj == 0.1

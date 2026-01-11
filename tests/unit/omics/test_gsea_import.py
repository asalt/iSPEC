from __future__ import annotations

import csv
import json
from pathlib import Path

from ispec.db.models import Project
from ispec.omics.models import GSEAAnalysis, GSEAResult
from ispec.omics.gsea_import import import_gsea_file


def _write_tsv(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_import_gsea_file_infers_collection_and_contrast(db_session, omics_session, tmp_path):
    project = Project(id=1161, prj_AddedBy="test", prj_ProjectTitle="Project 1161")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "H_treatA_minus_treatB.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=[
            "pathway",
            "pval",
            "padj",
            "log2err",
            "ES",
            "NES",
            "size",
            "leadingEdge_entrezid",
            "leadingEdge_genesymbol",
            "mainpathway",
        ],
        rows=[
            {
                "pathway": "HALLMARK_ADIPOGENESIS",
                "pval": "0.001",
                "padj": "0.01",
                "log2err": "0.5",
                "ES": "0.4",
                "NES": "1.2",
                "size": "100",
                "leadingEdge_entrezid": "1/2/3",
                "leadingEdge_genesymbol": "A/B/C",
                "mainpathway": "True",
            },
            {
                "pathway": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                "pval": "0.05",
                "padj": "0.10",
                "log2err": "0.1",
                "ES": "-0.2",
                "NES": "-1.1",
                "size": "50",
                "leadingEdge_entrezid": "",
                "leadingEdge_genesymbol": "",
                "mainpathway": "False",
            },
        ],
    )

    result = import_gsea_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1161,
        store_metadata=True,
    )
    assert result.skipped is False
    assert result.inserted == 2
    assert result.collection == "H"
    assert result.contrast == "treatA_minus_treatB"

    analysis = omics_session.query(GSEAAnalysis).filter_by(id=result.gsea_analysis_id).one()
    assert analysis.project_id == 1161
    assert analysis.collection == "H"
    assert analysis.contrast == "treatA_minus_treatB"

    meta = json.loads(analysis.metadata_json or "{}")
    assert meta["inferred"]["collection"] == "H"
    assert meta["inferred"]["contrast"] == "treatA_minus_treatB"

    rows = (
        omics_session.query(GSEAResult)
        .filter_by(gsea_analysis_id=analysis.id)
        .order_by(GSEAResult.pathway.asc())
        .all()
    )
    assert len(rows) == 2
    assert rows[0].pathway.startswith("HALLMARK_")
    assert rows[0].p_adj is not None


def test_import_gsea_file_skips_when_already_imported(db_session, omics_session, tmp_path):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "H_x.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=["pathway", "pval", "padj"],
        rows=[{"pathway": "P1", "pval": "0.1", "padj": "0.2"}],
    )

    first = import_gsea_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1,
    )
    assert first.skipped is False
    assert first.inserted == 1

    second = import_gsea_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=1,
    )
    assert second.skipped is True
    assert second.inserted == 0


def test_import_gsea_file_force_overwrites_existing(db_session, omics_session, tmp_path):
    project = Project(id=2, prj_AddedBy="test", prj_ProjectTitle="Project 2")
    db_session.add(project)
    db_session.commit()

    tsv_path = tmp_path / "H_x.tsv"
    _write_tsv(
        tsv_path,
        fieldnames=["pathway", "pval", "padj", "NES"],
        rows=[{"pathway": "P1", "pval": "0.1", "padj": "0.2", "NES": "1.0"}],
    )
    first = import_gsea_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=2,
    )
    assert first.inserted == 1

    _write_tsv(
        tsv_path,
        fieldnames=["pathway", "pval", "padj", "NES"],
        rows=[{"pathway": "P1", "pval": "0.05", "padj": "0.1", "NES": "2.0"}],
    )
    second = import_gsea_file(
        core_session=db_session,
        omics_session=omics_session,
        path=tsv_path,
        project_id=2,
        force=True,
    )
    assert second.cleared_existing is True
    assert second.inserted == 1

    analysis = omics_session.query(GSEAAnalysis).filter_by(id=second.gsea_analysis_id).one()
    row = (
        omics_session.query(GSEAResult)
        .filter_by(gsea_analysis_id=analysis.id, pathway="P1")
        .one()
    )
    assert row.nes == 2.0

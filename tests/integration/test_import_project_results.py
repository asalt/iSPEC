from pathlib import Path

from ispec.db import operations
from ispec.db.connect import get_session
from ispec.db.models import Project, ProjectFile
from ispec.omics.connect import get_omics_session
from ispec.omics.models import GeneContrast, GeneContrastStat


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_import_project_results_attaches_files_and_imports_volcano(tmp_path):
    results_dir = tmp_path / "Dec2025"
    _write_text(
        results_dir / "context" / "2more.tab",
        "\tMSPC1544\nname\tDec2025\n",
    )
    _write_text(
        results_dir / "metadata" / "2more.tab",
        "\trunno\nControl-1\t1\n",
    )
    _write_bytes(results_dir / "distribution" / "2more.png", b"png")
    _write_bytes(results_dir / "cluster2" / "cluster2.sqlite", b"sqlite-cache")
    _write_bytes(results_dir / "cluster2" / "cluster2_row_hclust.rds", b"rds-cache")
    _write_text(
        results_dir / "volcano" / "limma" / "contrast.tsv",
        "GeneID\tlog2_FC\tpValue\tpAdj\tGeneSymbol\tGeneDescription\n"
        "1\t0.1\t0.5\t0.5\tGeneA\tDescA\n"
        "2\t-0.2\t0.1\t0.2\tGeneB\tDescB\n",
    )

    core_db = tmp_path / "core.db"
    omics_db = tmp_path / "omics.db"

    with get_session(file_path=str(core_db)) as session:
        session.add(
            Project(
                id=1544,
                prj_AddedBy="tester",
                prj_ProjectTitle="Test Project",
                prj_ProjectBackground="bg",
            )
        )

    summary = operations.import_project_results(
        project_id=1544,
        results_dir=str(results_dir),
        db_file_path=str(core_db),
        omics_db_file_path=str(omics_db),
        prefix="Dec2025",
        added_by=None,
        skip_existing=True,
        force=False,
        dry_run=False,
        import_volcano=True,
    )

    assert summary["files_discovered"] == 6
    assert summary["files_total"] == 4
    assert summary["attachments_inserted"] == 4
    assert summary["volcano"] is not None
    assert summary["volcano"]["inserted_total"] == 2

    with get_session(file_path=str(core_db)) as session:
        rows = (
            session.query(ProjectFile.prjfile_FileName, ProjectFile.prjfile_AddedBy)
            .filter(ProjectFile.project_id == 1544)
            .order_by(ProjectFile.id.asc())
            .all()
        )
        names = [row[0] for row in rows]
        assert "Dec2025__context__2more.tab" in names
        assert "Dec2025__metadata__2more.tab" in names
        assert "Dec2025__volcano__limma__contrast.tsv" in names
        assert all(not name.endswith(".sqlite") for name in names)
        assert all(not name.endswith(".rds") for name in names)
        assert all(row[1] is None for row in rows)

    with get_omics_session(file_path=str(omics_db)) as session:
        contrast = session.query(GeneContrast).filter(GeneContrast.project_id == 1544).one()
        assert session.query(GeneContrastStat).filter(
            GeneContrastStat.gene_contrast_id == contrast.id
        ).count() == 2

    rerun = operations.import_project_results(
        project_id=1544,
        results_dir=str(results_dir),
        db_file_path=str(core_db),
        omics_db_file_path=str(omics_db),
        prefix="Dec2025",
        added_by="tester",
        skip_existing=True,
        force=False,
        dry_run=False,
        import_volcano=True,
    )
    assert rerun["attachments_inserted"] == 0
    assert rerun["attachments_skipped"] == 4
    assert rerun["attachments_metadata_updated"] == 4

    with get_session(file_path=str(core_db)) as session:
        assert (
            session.query(ProjectFile)
            .filter(ProjectFile.project_id == 1544)
            .filter(ProjectFile.prjfile_AddedBy == "tester")
            .count()
            == 4
        )

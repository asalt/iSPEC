import logging
import pytest

from ispec.db import operations
from ispec.omics import connect as omics_connect
from ispec.omics.connect import OmicsDatabaseUnavailableError


def test_check_status_logs_version(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    logger = operations.logger
    orig_prop = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO):
            operations.check_status()
    finally:
        logger.propagate = orig_prop
    assert "sqlite version" in caplog.text


def test_show_tables_logs_tables(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    logger = operations.logger
    orig_prop = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO):
            tables = operations.show_tables()
    finally:
        logger.propagate = orig_prop
    assert "person" in caplog.text
    assert "person" in tables
    person_columns = {column["name"]: column for column in tables["person"]}
    assert "ppl_Name_First" in person_columns
    assert not person_columns["ppl_Name_First"]["nullable"]
    assert "TEXT" in person_columns["ppl_Name_First"]["type"].upper()


def test_export_table_writes_csv(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))

    from ispec.db.connect import get_session
    from sqlalchemy import text
    import pandas as pd

    with get_session() as session:
        session.execute(
            text(
                "INSERT INTO person (ppl_Name_First, ppl_Name_Last, ppl_AddedBy, ppl_CreationTS, ppl_ModificationTS) "
                "VALUES ('Alice', 'Smith', 'tester', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    export_file = tmp_path / "out.csv"
    operations.export_table("person", str(export_file))
    df = pd.read_csv(export_file)
    assert df.loc[0, "ppl_Name_First"] == "Alice"


def test_import_e2g_rejects_missing_previously_known_omics_db(tmp_path, monkeypatch):
    core_db_path = tmp_path / "core.db"
    omics_db_path = tmp_path / "omics.db"
    qual_path = tmp_path / "sample_e2g_QUAL.tsv"
    qual_path.write_text("GeneID\tEXPRecNo\trunno\tsearchno\tlabel\n123\t1\t1\t1\t0\n")

    def _stub_import_e2g_files(**kwargs):
        return {"files": [], "inserted": 0, "updated": 0, "errors": []}

    monkeypatch.setattr("ispec.omics.e2g_import.import_e2g_files", _stub_import_e2g_files)

    summary = operations.import_e2g(
        qual_paths=[str(qual_path)],
        db_file_path=str(core_db_path),
        omics_db_file_path=str(omics_db_path),
    )
    assert summary["errors"] == []
    assert omics_db_path.exists()

    omics_connect._get_engine.cache_clear()
    omics_db_path.unlink()

    with pytest.raises(OmicsDatabaseUnavailableError, match="Refusing to auto-create"):
        operations.import_e2g(
            qual_paths=[str(qual_path)],
            db_file_path=str(core_db_path),
            omics_db_file_path=str(omics_db_path),
        )

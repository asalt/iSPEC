import logging
import pytest

from ispec.db import operations


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

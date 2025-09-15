import sys
import sqlite3
from pathlib import Path

import pandas as pd

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ispec.cli.main import main
from ispec.db import operations


def test_db_init_creates_tables(tmp_path, monkeypatch):
    """CLI db init should create a SQLite file with tables."""

    db_file = tmp_path / "test.db"
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "init", "--file", str(db_file)])
    main()

    assert db_file.exists()

    with sqlite3.connect(db_file) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }

    assert "person" in tables


def test_cli_import_inserts_data(tmp_path, monkeypatch):
    """Importing data via the CLI should insert rows into the database."""

    db_file = tmp_path / "test.db"

    # Initialize database
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "init", "--file", str(db_file)])
    main()
    assert db_file.exists()

    # Prepare sample CSV file
    csv_file = tmp_path / "people.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "ppl_AddedBy": "tester",
                "ppl_Name_First": "Alice",
                "ppl_Name_Last": "Smith",
            }
        ]
    ).to_csv(csv_file, index=False)

    # Ensure import uses the temp database
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_file))

    # Import data via CLI
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ispec",
            "db",
            "import",
            "--table-name",
            "person",
            "--file",
            str(csv_file),
        ],
    )
    main()

    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(
            "SELECT ppl_Name_First, ppl_Name_Last FROM person"
        ).fetchall()

    assert rows == [("Alice", "Smith")]


def test_db_status_prints_sqlite_version(tmp_path, monkeypatch, caplog):
    """Running `ispec db status` should output the SQLite version."""

    db_file = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_file))
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "status"])
    operations.logger.addHandler(caplog.handler)
    with caplog.at_level("INFO", logger=operations.logger.name):
        main()
    operations.logger.removeHandler(caplog.handler)

    assert sqlite3.sqlite_version in caplog.text


def test_db_show_lists_tables(tmp_path, monkeypatch, caplog):
    """After initialization, `ispec db show` should list tables."""

    db_file = tmp_path / "test.db"

    monkeypatch.setattr(sys, "argv", ["ispec", "db", "init", "--file", str(db_file)])
    operations.logger.addHandler(caplog.handler)
    with caplog.at_level("INFO", logger=operations.logger.name):
        main()
    operations.logger.removeHandler(caplog.handler)
    assert db_file.exists()
    caplog.clear()

    monkeypatch.setenv("ISPEC_DB_PATH", str(db_file))
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "show"])
    operations.logger.addHandler(caplog.handler)
    with caplog.at_level("INFO", logger=operations.logger.name):
        main()
    operations.logger.removeHandler(caplog.handler)

    assert "person" in caplog.text
    assert "project" in caplog.text


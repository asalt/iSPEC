import sys
import sqlite3
from pathlib import Path

import pandas as pd

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ispec.cli.main import main


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


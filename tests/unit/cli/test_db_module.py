import sys
import argparse
import types
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ispec.cli import db


def test_register_subcommands_parses_import_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args(["import", "--table-name", "person", "--file", "people.csv"])
    assert args.subcommand == "import"
    assert args.table_name == "person"
    assert args.file == "people.csv"


def test_dispatch_calls_correct_operations(monkeypatch):
    init_mock = MagicMock()
    status_mock = MagicMock()
    show_mock = MagicMock()
    import_mock = MagicMock()
    monkeypatch.setattr("ispec.cli.db.operations.initialize", init_mock)
    monkeypatch.setattr("ispec.cli.db.operations.check_status", status_mock)
    monkeypatch.setattr("ispec.cli.db.operations.show_tables", show_mock)
    monkeypatch.setattr("ispec.cli.db.operations.import_file", import_mock)

    db.dispatch(types.SimpleNamespace(subcommand="status"))
    status_mock.assert_called_once()

    db.dispatch(types.SimpleNamespace(subcommand="show"))
    show_mock.assert_called_once()

    db.dispatch(types.SimpleNamespace(subcommand="init", file="db.sqlite"))
    init_mock.assert_called_once_with(file_path="db.sqlite")

    db.dispatch(
        types.SimpleNamespace(subcommand="import", file="data.csv", table_name="person")
    )
    import_mock.assert_called_once_with("data.csv", "person")

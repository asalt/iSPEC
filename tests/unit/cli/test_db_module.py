import sys
import argparse
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ispec.cli import db
from rich.console import Console


@pytest.mark.parametrize("table_name", ["person", "project", "comment", "letter"])
def test_register_subcommands_parses_import_command(table_name):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args([
        "import",
        "--table-name",
        table_name,
        "--file",
        "people.csv",
    ])
    assert args.subcommand == "import"
    assert args.table_name == table_name
    assert args.file == "people.csv"


def test_register_subcommands_parses_export_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args(["export", "--table-name", "person", "--file", "out.csv"])
    assert args.subcommand == "export"
    assert args.table_name == "person"
    assert args.file == "out.csv"


def test_register_subcommands_parses_import_e2g_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args(
        [
            "import-e2g",
            "--dir",
            "/tmp/e2g",
            "--database",
            "db.sqlite",
            "--analysis-database",
            "analysis.sqlite",
            "--create-missing-runs",
            "--store-metadata",
        ]
    )
    assert args.subcommand == "import-e2g"
    assert args.data_dir == "/tmp/e2g"
    assert args.database == "db.sqlite"
    assert args.analysis_database == "analysis.sqlite"
    assert args.create_missing_runs is True
    assert args.create_missing_experiments is True
    assert args.skip_imported is True
    assert args.store_metadata is True


def test_register_subcommands_parses_import_psm_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args(
        [
            "import-psm",
            "--file",
            "/tmp/psm.tsv",
            "--database",
            "db.sqlite",
            "--psm-database",
            "psm.sqlite",
            "--experiment-run-id",
            "42",
            "--store-metadata",
        ]
    )
    assert args.subcommand == "import-psm"
    assert args.paths == ["/tmp/psm.tsv"]
    assert args.database == "db.sqlite"
    assert args.psm_database == "psm.sqlite"
    assert args.experiment_run_id == 42
    assert args.store_metadata is True


def test_register_subcommands_parses_audit_imports_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(subparsers)
    args = parser.parse_args(
        [
            "audit-imports",
            "--database",
            "db.sqlite",
            "--analysis-database",
            "analysis.sqlite",
            "--psm-database",
            "psm.sqlite",
            "--out-dir",
            "data",
        ]
    )
    assert args.subcommand == "audit-imports"
    assert args.database == "db.sqlite"
    assert args.analysis_database == "analysis.sqlite"
    assert args.psm_database == "psm.sqlite"
    assert args.out_dir == "data"


def test_dispatch_calls_correct_operations(monkeypatch):
    init_mock = MagicMock()
    status_mock = MagicMock()
    show_mock = MagicMock(return_value={})
    render_mock = MagicMock()
    import_mock = MagicMock()
    export_mock = MagicMock()

    monkeypatch.setattr("ispec.cli.db.operations.initialize", init_mock)
    monkeypatch.setattr("ispec.cli.db.operations.check_status", status_mock)
    monkeypatch.setattr("ispec.cli.db.operations.show_tables", show_mock)
    monkeypatch.setattr("ispec.cli.db._render_table_overview", render_mock)
    monkeypatch.setattr("ispec.cli.db.operations.import_file", import_mock)
    monkeypatch.setattr("ispec.cli.db.operations.export_table", export_mock)

    db.dispatch(types.SimpleNamespace(subcommand="status"))
    status_mock.assert_called_once()

    db.dispatch(types.SimpleNamespace(subcommand="show"))
    show_mock.assert_called_once()
    render_mock.assert_called_once_with({})

    db.dispatch(types.SimpleNamespace(subcommand="init", file="db.sqlite"))
    init_mock.assert_called_once_with(file_path="db.sqlite")


@pytest.mark.parametrize("table_name", ["person", "project", "comment", "letter"])
def test_dispatch_import_calls_operations(monkeypatch, table_name):
    import_mock = MagicMock()
    monkeypatch.setattr("ispec.cli.db.operations.import_file", import_mock)

    db.dispatch(
        types.SimpleNamespace(subcommand="import", file="data.csv", table_name=table_name)
    )
    import_mock.assert_called_once_with("data.csv", table_name)


def test_dispatch_import_e2g_calls_operations(monkeypatch):
    import_mock = MagicMock(return_value={"files": [], "inserted": 0, "updated": 0, "errors": []})
    monkeypatch.setattr("ispec.cli.db.operations.import_e2g", import_mock)

    db.dispatch(
        types.SimpleNamespace(
            subcommand="import-e2g",
            data_dir="/tmp/e2g",
            qual_paths=["a.tsv"],
            quant_paths=["b.tsv"],
            database="db.sqlite",
            analysis_database="analysis.sqlite",
            create_missing_runs=False,
            store_metadata=True,
        )
    )
    import_mock.assert_called_once()


def test_dispatch_import_psm_calls_operations(monkeypatch):
    import_mock = MagicMock(return_value={"files": [], "inserted": 0, "updated": 0, "errors": []})
    monkeypatch.setattr("ispec.cli.db.operations.import_psms", import_mock)

    db.dispatch(
        types.SimpleNamespace(
            subcommand="import-psm",
            paths=["/tmp/psm.tsv"],
            database="db.sqlite",
            psm_database="psm.sqlite",
            experiment_run_id=5,
            experiment_id=None,
            run_no=None,
            search_no=None,
            label=None,
            create_missing_runs=True,
            create_missing_experiments=True,
            store_metadata=False,
            skip_imported=True,
            force=False,
        )
    )
    import_mock.assert_called_once()


def test_dispatch_audit_imports_calls_operations(monkeypatch):
    audit_mock = MagicMock(return_value={"audit_json": "a.json", "gap_tsv": "b.tsv"})
    monkeypatch.setattr("ispec.cli.db.operations.audit_imports", audit_mock)

    db.dispatch(
        types.SimpleNamespace(
            subcommand="audit-imports",
            database="db.sqlite",
            analysis_database="analysis.sqlite",
            psm_database="psm.sqlite",
            legacy_schema=None,
            legacy_mapping=None,
            legacy_tables_file=None,
            scripts_dir=None,
            out_dir="data",
        )
    )
    audit_mock.assert_called_once_with(
        db_file_path="db.sqlite",
        analysis_db_file_path="analysis.sqlite",
        psm_db_file_path="psm.sqlite",
        legacy_schema_path=None,
        legacy_mapping_path=None,
        legacy_tables_file_path=None,
        scripts_dir=None,
        out_dir="data",
    )


def test_dispatch_export_calls_operations(monkeypatch):
    export_mock = MagicMock()
    monkeypatch.setattr("ispec.cli.db.operations.export_table", export_mock)

    db.dispatch(
        types.SimpleNamespace(subcommand="export", table_name="person", file="out.csv")
    )
    export_mock.assert_called_once_with("person", "out.csv")


def test_render_table_overview_outputs_columns():
    console = Console(record=True)
    table_definitions = {
        "person": [
            {"name": "id", "type": "INTEGER", "nullable": False, "default": None},
            {
                "name": "ppl_Name_First",
                "type": "TEXT",
                "nullable": False,
                "default": None,
            },
        ],
        "project": [],
    }

    db._render_table_overview(table_definitions, console=console)

    output = console.export_text()
    assert "person" in output
    assert "id" in output
    assert "INTEGER" in output

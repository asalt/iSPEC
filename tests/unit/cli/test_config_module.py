import sys
import argparse
import json
from pathlib import Path
import types
from unittest.mock import MagicMock

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ispec.cli import config


def test_register_subcommands_parses_paths_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    config.register_subcommands(subparsers)
    args = parser.parse_args(["paths", "--format", "json"])

    assert args.subcommand == "paths"
    assert args.format == "json"


def test_dispatch_paths_json_prints_resolved_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_DB_PATH", str(tmp_path / "core.db"))
    args = types.SimpleNamespace(subcommand="paths", format="json")
    print_json = MagicMock()
    monkeypatch.setattr("rich.console.Console.print_json", print_json)

    config.dispatch(args)

    assert print_json.call_count == 1
    payload = json.loads(print_json.call_args.args[0])
    assert payload["database"]["core"]["path"] == str(tmp_path / "core.db")
    assert payload["database"]["analysis"]["path"] == str(tmp_path / "ispec-analysis.db")
    assert payload["database"]["agent_state"]["path"] == str(tmp_path / "ispec-agent-state.db")

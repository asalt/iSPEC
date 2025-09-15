import sys
import argparse
import types
from pathlib import Path
import logging
from unittest.mock import MagicMock

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ispec.cli import logging as logging_cli


def test_register_subcommands_parses_set_level():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    logging_cli.register_subcommands(subparsers)
    args = parser.parse_args(["set-level", "DEBUG"])
    assert args.subcommand == "set-level"
    assert args.level == "DEBUG"


def test_register_subcommands_parses_show_level():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    logging_cli.register_subcommands(subparsers)
    args = parser.parse_args(["show-level"])
    assert args.subcommand == "show-level"


def test_dispatch_set_level_invokes_logging_helpers(monkeypatch):
    reset_mock = MagicMock()
    get_mock = MagicMock()
    monkeypatch.setattr(logging_cli, "reset_logger", reset_mock)
    monkeypatch.setattr(logging_cli, "get_logger", get_mock)
    args = types.SimpleNamespace(subcommand="set-level", level="INFO")
    logging_cli.dispatch(args)
    reset_mock.assert_called_once_with()
    get_mock.assert_called_once()
    assert get_mock.call_args.kwargs["level"] == logging.INFO


def test_dispatch_show_path_prints_resolved_path(monkeypatch, capsys):
    expected = Path("/tmp/ispec-test.log")
    monkeypatch.setattr(logging_cli, "_resolve_log_file", lambda: expected)
    args = types.SimpleNamespace(subcommand="show-path")
    logging_cli.dispatch(args)
    assert capsys.readouterr().out.strip() == str(expected.resolve())


def test_dispatch_show_level_prints_configured_level(monkeypatch, capsys):
    monkeypatch.setattr(logging_cli, "get_configured_level", lambda: "WARNING")
    args = types.SimpleNamespace(subcommand="show-level")
    logging_cli.dispatch(args)
    assert capsys.readouterr().out.strip() == "WARNING"

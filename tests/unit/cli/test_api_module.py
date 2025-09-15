import sys
import argparse
import types
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ispec.cli import api


def test_register_subcommands_parses_start_command():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    api.register_subcommands(subparsers)
    args = parser.parse_args(["start", "--host", "1.2.3.4", "--port", "1234"])
    assert args.subcommand == "start"
    assert args.host == "1.2.3.4"
    assert args.port == 1234


def test_dispatch_start_invokes_uvicorn(monkeypatch):
    captured = {}
    dummy_api_main = types.ModuleType("ispec.api.main")
    dummy_api_main.app = object()
    dummy_uvicorn = types.ModuleType("uvicorn")

    def fake_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    dummy_uvicorn.run = fake_run
    monkeypatch.setitem(sys.modules, "ispec.api.main", dummy_api_main)
    monkeypatch.setitem(sys.modules, "uvicorn", dummy_uvicorn)

    args = types.SimpleNamespace(subcommand="start", host="127.0.0.1", port=9000)
    api.dispatch(args)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000


def test_dispatch_status_does_not_start_server(monkeypatch):
    run_mock = MagicMock()
    dummy_api_main = types.ModuleType("ispec.api.main")
    dummy_api_main.app = object()
    dummy_uvicorn = types.ModuleType("uvicorn")
    dummy_uvicorn.run = run_mock
    monkeypatch.setitem(sys.modules, "ispec.api.main", dummy_api_main)
    monkeypatch.setitem(sys.modules, "uvicorn", dummy_uvicorn)

    args = types.SimpleNamespace(subcommand="status")
    api.dispatch(args)
    run_mock.assert_not_called()

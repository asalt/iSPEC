import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from ispec.cli import main, db, api


def invoke(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    main.main()


def test_db_subcommand_invokes_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(args):
        called["args"] = args

    monkeypatch.setattr(db, "dispatch", fake_dispatch)
    invoke(["ispec", "db", "status"], monkeypatch)
    assert called["args"].command == "db"
    assert called["args"].subcommand == "status"


def test_api_subcommand_invokes_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(args):
        called["args"] = args

    monkeypatch.setattr(api, "dispatch", fake_dispatch)
    invoke(["ispec", "api", "status"], monkeypatch)
    assert called["args"].command == "api"
    assert called["args"].subcommand == "status"


def test_dispatch_errors_propagate(monkeypatch):
    def boom(args):
        raise ValueError("boom")

    monkeypatch.setattr(db, "dispatch", boom)
    with pytest.raises(ValueError, match="boom"):
        invoke(["ispec", "db", "status"], monkeypatch)


def test_api_dispatch_unknown_subcommand():
    args = SimpleNamespace(subcommand="nope")
    with pytest.raises(ValueError, match="No handler for API subcommand"):
        api.dispatch(args)


def test_db_dispatch_unknown_subcommand(monkeypatch):
    dummy_ops = SimpleNamespace(
        check_status=lambda: None,
        show_tables=lambda: None,
        import_file=lambda f: None,
        initialize=lambda file_path=None: None,
    )
    dummy_pkg = ModuleType("db")
    dummy_pkg.operations = dummy_ops
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_pkg)
    monkeypatch.setitem(sys.modules, "ispec.db.operations", dummy_ops)
    args = SimpleNamespace(subcommand="nope")
    with pytest.raises(ValueError, match="No handler for DB subcommand"):
        db.dispatch(args)

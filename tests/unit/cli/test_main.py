import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

# Create lightweight stand-ins for database operations to avoid heavy imports
fake_operations = types.ModuleType("ispec.db.operations")

def _dummy(*args, **kwargs):
    pass

fake_operations.check_status = _dummy
fake_operations.show_tables = _dummy
fake_operations.import_file = _dummy
fake_operations.initialize = _dummy

fake_db = types.ModuleType("ispec.db")
fake_db.operations = fake_operations

sys.modules.setdefault("ispec.db", fake_db)
sys.modules.setdefault("ispec.db.operations", fake_operations)

from ispec.cli.main import main


def test_db_status(monkeypatch):
    mock_check_status = MagicMock()
    monkeypatch.setattr("ispec.cli.db.operations.check_status", mock_check_status)
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "status"])
    main()
    mock_check_status.assert_called_once()


def test_db_show(monkeypatch):
    mock_show_tables = MagicMock()
    monkeypatch.setattr("ispec.cli.db.operations.show_tables", mock_show_tables)
    monkeypatch.setattr(sys, "argv", ["ispec", "db", "show"])
    main()
    mock_show_tables.assert_called_once()


def test_api_status(monkeypatch):
    run_mock = MagicMock()
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = run_mock
    fake_api_main = types.ModuleType("ispec.api.main")
    fake_api_main.app = object()
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "ispec.api.main", fake_api_main)

    monkeypatch.setattr(sys, "argv", ["ispec", "api", "status"])
    main()
    run_mock.assert_not_called()


def test_api_start(monkeypatch):
    run_mock = MagicMock()
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = run_mock
    fake_api_main = types.ModuleType("ispec.api.main")
    fake_api_main.app = object()
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "ispec.api.main", fake_api_main)

    monkeypatch.setattr(
        sys,
        "argv",
        ["ispec", "api", "start", "--host", "0.0.0.0", "--port", "5000"],
    )
    main()
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["host"] == "0.0.0.0"
    assert run_mock.call_args.kwargs["port"] == 5000

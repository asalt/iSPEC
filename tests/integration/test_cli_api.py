import sys
from pathlib import Path
import types

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_api_start_invokes_uvicorn_run(monkeypatch):
    captured = {}

    # Stub out the database CLI module to avoid heavy dependencies
    dummy_db = types.ModuleType("db")
    dummy_db.register_subcommands = lambda subparsers: None
    dummy_db.dispatch = lambda args: None
    monkeypatch.setitem(sys.modules, "ispec.cli.db", dummy_db)

    # Stub out the FastAPI application module
    dummy_api_main = types.ModuleType("ispec.api.main")
    dummy_api_main.app = object()
    monkeypatch.setitem(sys.modules, "ispec.api.main", dummy_api_main)

    # Stub uvicorn to capture run arguments
    dummy_uvicorn = types.ModuleType("uvicorn")

    def fake_run(app, host, port, *args, **kwargs):
        captured["host"] = host
        captured["port"] = port

    dummy_uvicorn.run = fake_run
    monkeypatch.setitem(sys.modules, "uvicorn", dummy_uvicorn)

    monkeypatch.setattr(
        sys,
        "argv",
        ["ispec", "api", "start", "--host", "127.0.0.1", "--port", "9000"],
    )

    from ispec.cli.main import main

    main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000

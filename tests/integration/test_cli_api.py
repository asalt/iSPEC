import json
import os
import sys
from pathlib import Path
import types

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_api_start_invokes_uvicorn_run(monkeypatch, tmp_path):
    captured = {}

    state_file = tmp_path / "api_state.json"
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("ISPEC_API_STATE_FILE", str(state_file))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(log_dir))

    # Stub out the database CLI module to avoid heavy dependencies
    dummy_db = types.ModuleType("db")
    dummy_db.register_subcommands = lambda subparsers: None
    dummy_db.dispatch = lambda args: None
    monkeypatch.setitem(sys.modules, "ispec.cli.db", dummy_db)

    dummy_ispec_db = types.ModuleType("ispec.db")
    dummy_ispec_db.get_session = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_ispec_db)

    # Stub out the FastAPI application module
    dummy_api_main = types.ModuleType("ispec.api.main")
    dummy_api_main.app = object()
    monkeypatch.setitem(sys.modules, "ispec.api.main", dummy_api_main)

    # Stub uvicorn to capture run arguments
    dummy_uvicorn = types.ModuleType("uvicorn")

    def fake_run(app, host, port, *args, **kwargs):
        captured["host"] = host
        captured["port"] = port
        captured["state_exists"] = state_file.exists()
        if state_file.exists():
            captured["state_payload"] = json.loads(state_file.read_text())

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
    assert captured["state_exists"] is True
    payload = captured["state_payload"]
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 9000
    assert payload["pid"] == os.getpid()
    assert not state_file.exists()


def test_api_status_reports_not_running_without_state(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ISPEC_API_STATE_FILE", str(tmp_path / "missing.json"))

    dummy_ispec_db = types.ModuleType("ispec.db")
    dummy_ispec_db.get_session = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_ispec_db)

    from ispec.cli import api as api_cli

    called = {}

    def fail_get(*args, **kwargs):
        called["called"] = True
        raise AssertionError("status should not probe without state")

    monkeypatch.setattr(api_cli.requests, "get", fail_get)

    logger = api_cli.get_logger(api_cli.__file__)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO", logger=logger.name):
            api_cli.dispatch(types.SimpleNamespace(subcommand="status"))
    finally:
        logger.removeHandler(caplog.handler)

    assert "API server is not running." in caplog.text
    assert "called" not in called


def test_api_status_reports_running(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    state_file = tmp_path / "state.json"
    monkeypatch.setenv("ISPEC_API_STATE_FILE", str(state_file))
    state_file.write_text(json.dumps({"host": "localhost", "port": 9100}))

    dummy_ispec_db = types.ModuleType("ispec.db")
    dummy_ispec_db.get_session = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_ispec_db)

    from ispec.cli import api as api_cli

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True}

    captured = {}

    def fake_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(api_cli.requests, "get", fake_get)

    logger = api_cli.get_logger(api_cli.__file__)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO", logger=logger.name):
            api_cli.dispatch(types.SimpleNamespace(subcommand="status"))
    finally:
        logger.removeHandler(caplog.handler)

    assert "API server is running at localhost:9100" in caplog.text
    assert captured["url"] == "http://localhost:9100/status"
    assert captured["timeout"] == api_cli._REQUEST_TIMEOUT


def test_api_status_uses_loopback_for_wildcard_host(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    state_file = tmp_path / "wildcard.json"
    monkeypatch.setenv("ISPEC_API_STATE_FILE", str(state_file))
    state_file.write_text(json.dumps({"host": "0.0.0.0", "port": 9200}))

    dummy_ispec_db = types.ModuleType("ispec.db")
    dummy_ispec_db.get_session = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_ispec_db)

    from ispec.cli import api as api_cli

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True}

    captured = {}

    def fake_get(url, timeout):
        captured["url"] = url
        return DummyResponse()

    monkeypatch.setattr(api_cli.requests, "get", fake_get)

    logger = api_cli.get_logger(api_cli.__file__)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO", logger=logger.name):
            api_cli.dispatch(types.SimpleNamespace(subcommand="status"))
    finally:
        logger.removeHandler(caplog.handler)

    assert "API server is running at 0.0.0.0:9200" in caplog.text
    assert captured["url"] == "http://127.0.0.1:9200/status"

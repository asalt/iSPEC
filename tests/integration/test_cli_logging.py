import sys
import types
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_logging_show_level_reports_configured_level(tmp_path, monkeypatch, capsys):
    """`ispec logging show-level` should print the configured log level."""

    config_file = tmp_path / "config" / "logging.json"
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("ISPEC_LOG_CONFIG", str(config_file))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(log_dir))

    dummy_ispec_db = types.ModuleType("ispec.db")
    dummy_ispec_db.get_session = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ispec.db", dummy_ispec_db)

    from ispec.logging import reset_logger

    reset_logger()

    dummy_cli_db = types.ModuleType("ispec.cli.db")
    dummy_cli_db.register_subcommands = lambda subparsers: None
    dummy_cli_db.dispatch = lambda args: None
    monkeypatch.setitem(sys.modules, "ispec.cli.db", dummy_cli_db)

    dummy_cli_api = types.ModuleType("ispec.cli.api")
    dummy_cli_api.register_subcommands = lambda subparsers: None
    dummy_cli_api.dispatch = lambda args: None
    monkeypatch.setitem(sys.modules, "ispec.cli.api", dummy_cli_api)

    from ispec.cli.main import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["ispec", "logging", "set-level", "DEBUG"],
    )
    main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["ispec", "logging", "show-level"])
    main()
    captured = capsys.readouterr()

    assert captured.out.strip() == "DEBUG"

    reset_logger()

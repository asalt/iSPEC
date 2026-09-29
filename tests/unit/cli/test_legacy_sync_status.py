import argparse
import json

from ispec.cli import db


def test_status_cli_uses_explicit_agent_database_without_sync(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ISPEC_LEGACY_SYNC_ENABLED", "0")
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))

    def unexpected_sync(**kwargs):
        raise AssertionError("Status must not execute a sync")

    monkeypatch.setattr("ispec.db.legacy_sync_all.sync_legacy_all", unexpected_sync)
    parser = argparse.ArgumentParser()
    db.register_subcommands(parser.add_subparsers(dest="subcommand", required=True))
    args = parser.parse_args(
        [
            "legacy-sync-status",
            "--agent-database",
            str(tmp_path / "agent.db"),
        ]
    )
    db.dispatch(args)
    result = json.loads(capsys.readouterr().out)
    assert result["enabled"] is False
    assert result["active"] is None
    assert result["recent_runs"] == []
    assert result["log_directory"] == str(tmp_path / "logs" / "legacy-sync")

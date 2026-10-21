from datetime import UTC, datetime, timedelta
import json
import os
from types import SimpleNamespace

import psutil

from ispec.cli import supervisor as cli
from ispec.supervisor.runtime_status import (
    observed_run_status,
    observed_state,
    process_is_running,
)


def test_observed_state_does_not_rewrite_stale_history(monkeypatch):
    monkeypatch.setattr(
        "ispec.supervisor.runtime_status.process_is_running", lambda *a: False
    )
    raw = {"status": "running", "pid": 99999, "ended_at": None}
    observed = observed_state(raw)
    assert observed["status"] == "stale"
    assert observed["recorded_status"] == "running"
    assert observed["running"] is False
    assert observed["ended_at"] is None
    assert raw["status"] == "running"


def test_process_identity_rejects_reused_pid():
    created = psutil.Process().create_time()
    assert process_is_running(os.getpid(), created)
    assert not process_is_running(os.getpid(), created - 100)
    assert not process_is_running(None)


def test_cli_json_reports_observed_state(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_read_state_payload", lambda **_: {"status": "running", "pid": None}
    )
    cli.dispatch(SimpleNamespace(subcommand="status", json=True))
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "stale"
    assert value["recorded_status"] == "running"


def test_ops_status_expires_without_fabricating_stop_time():
    now = datetime.now(UTC)
    assert observed_run_status("running", now) == "running"
    assert observed_run_status("running", now - timedelta(minutes=10)) == "stale"
    assert observed_run_status("running", None) == "stale"
    assert observed_run_status("stopped", now) == "stopped"

from datetime import UTC, datetime, timedelta
import json

import pytest

from ispec.db import legacy_sync_all as sync
from ispec.db.legacy_sync_observability import (
    LegacySyncFailure,
    append_sync_event,
    safe_sync_error,
)
from ispec.supervisor.loop import _run_legacy_sync_all
from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand


@pytest.fixture
def phases(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path))
    calls = []
    for name in (
        "sync_legacy_projects",
        "sync_legacy_people",
        "sync_legacy_experiments",
        "sync_legacy_experiment_runs_incremental",
        "scan_recent_legacy_project_comment_projects",
        "sync_legacy_project_comments",
    ):

        def operation(_name=name, **kwargs):
            calls.append((_name, kwargs))
            return {
                "items": 1,
                "inserted": 1,
                "touched_ids": [123],
                "project_ids": [123],
            }

        monkeypatch.setattr(sync, name, operation)
    return calls


def events(tmp_path):
    return [
        json.loads(line)
        for path in (tmp_path / "legacy-sync").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_sync_logs_correlated_phases_and_summary_without_source_rows(phases, tmp_path):
    progress = []
    summary = sync.sync_legacy_all(
        command_id=42,
        progress=lambda value: progress.append(json.loads(json.dumps(value))),
    )
    assert summary["ok"] is True
    assert summary["sync_run"]["status"] == "succeeded"
    assert len(summary["sync_run"]["phases"]) == 6
    assert progress[0]["phases"] == []
    records = events(tmp_path)
    assert records[0]["event"] == "started"
    assert records[-1]["event"] == "finished"
    assert {record["command_id"] for record in records} == {42}
    assert len({record["sync_id"] for record in records}) == 1
    assert "touched_ids" not in json.dumps(records)


def test_failed_phase_retains_prior_work_and_redacts_request_url(
    phases, tmp_path, monkeypatch
):
    def fail(**kwargs):
        raise ConnectionError(
            "unavailable https://user:secret@legacy.invalid/path?token=secret"
        )

    monkeypatch.setattr(sync, "sync_legacy_experiments", fail)
    with pytest.raises(LegacySyncFailure) as caught:
        sync.sync_legacy_all(command_id=4)
    result = caught.value.summary
    assert result["ok"] is False
    assert result["projects"]["inserted"] == 1
    assert result["people"]["inserted"] == 1
    assert result["sync_run"]["failed_phase"] == "experiments"
    assert len(result["sync_run"]["phases"]) == 3
    assert len(phases) == 2
    assert "secret" not in json.dumps(result)
    assert "secret" not in json.dumps(events(tmp_path))
    assert events(tmp_path)[-1]["status"] == "failed"


def test_supervisor_failure_preserves_partial_summary(phases, monkeypatch):
    monkeypatch.setattr(
        sync,
        "sync_legacy_people",
        lambda **_: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    outcome = _run_legacy_sync_all({"max_project_comments": 0})
    assert outcome.ok is False
    assert outcome.result["projects"]["inserted"] == 1
    assert outcome.result["sync_run"]["failed_phase"] == "people"
    assert outcome.error == "TimeoutError: offline"


def test_zero_comments_stays_zero_through_supervisor(phases):
    result = _run_legacy_sync_all({"max_project_comments": 0})
    assert result.ok is True
    assert result.result["project_comments"]["requested"] == 0
    assert not any(name == "sync_legacy_project_comments" for name, _ in phases)
    scan = next(
        kwargs
        for name, kwargs in phases
        if name == "scan_recent_legacy_project_comment_projects"
    )
    assert scan["recent_days"] is None


def test_progress_committed_before_next_phase_and_failure(
    phases, tmp_path, monkeypatch
):
    path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(path))
    with get_agent_session(path) as db:
        row = AgentCommand(command_type=COMMAND_LEGACY_SYNC_ALL, status="running")
        db.add(row)
        db.flush()
        command_id = row.id

    def inspect_then_fail(**kwargs):
        with get_agent_session(path) as db:
            row = db.get(AgentCommand, command_id)
            progress = row.result_json["sync_run"]
            assert progress["phases"][0]["status"] == "succeeded"
            assert progress["phases"][1]["status"] == "running"
            assert progress["command_id"] == command_id
            # Observation persistence must not keep this DB locked during fetch.
            db.add(AgentCommand(command_type="unrelated", status="queued"))
        raise TimeoutError("bridge offline")

    monkeypatch.setattr(sync, "sync_legacy_people", inspect_then_fail)
    outcome = _run_legacy_sync_all({"max_project_comments": 0}, command_id=command_id)
    assert outcome.ok is False
    assert outcome.error == "TimeoutError: bridge offline"
    with get_agent_session(path) as db:
        progress = db.get(AgentCommand, command_id).result_json["sync_run"]
        assert progress["status"] == "failed"
        assert progress["failed_phase"] == "people"
        assert progress["ended_at"]


def test_unavailable_log_or_progress_does_not_hide_sync_result(phases, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic full disk")

    monkeypatch.setattr("ispec.db.legacy_sync_observability.append_sync_event", fail)
    summary = sync.sync_legacy_all(progress=fail)
    assert summary["ok"] is True
    assert "full disk" in summary["sync_run"]["log_error"]
    assert "full disk" in summary["sync_run"]["progress_error"]


def test_log_retention_removes_only_old_daily_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path))
    directory = tmp_path / "legacy-sync"
    directory.mkdir()
    old = directory / f"{datetime.now(UTC) - timedelta(days=40):%Y-%m-%d}.jsonl"
    unrelated = directory / "notes.jsonl"
    old.write_text("{}\n")
    unrelated.write_text("{}\n")
    append_sync_event({"event": "started"})
    assert not old.exists()
    assert unrelated.exists()


def test_connection_pool_error_keeps_cause_not_query():
    exc = ConnectionError(
        "HTTPConnectionPool(host='legacy.invalid', port=80): Max retries exceeded "
        "with url: /rows?token=secret&fields="
        + "long-field," * 200
        + " (Caused by NewConnectionError('[Errno 113] No route to host'))"
    )
    message = safe_sync_error(exc)
    assert "secret" not in message
    assert "long-field" not in message
    assert "No route to host" in message
    assert "legacy.invalid" in message

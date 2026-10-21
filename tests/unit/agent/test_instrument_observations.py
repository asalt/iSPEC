from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest
from pydantic import ValidationError

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, InstrumentObservationRow
from ispec.instruments import collector
from ispec.instruments.schema import InstrumentObservation
from ispec.instruments.store import (
    ObservationConflict,
    latest_status,
    record_observation,
)


def observation(**updates):
    payload = collector.blank_observation(
        collector.CollectorConfig(
            collector_id="windows-bruker",
            instrument_id="tims-ultra-0006",
            root=str(Path.cwd()),
        )
    )
    payload.update(updates)
    return InstrumentObservation.model_validate(payload)


def test_retries_are_atomic_and_do_not_refresh_heartbeat(tmp_path):
    path = tmp_path / "agent.db"
    now = datetime.now(UTC)
    payload = observation()
    with get_agent_session(path) as db:
        first = record_observation(db, payload, now=now)
        duplicate = record_observation(db, payload, now=now + timedelta(minutes=5))
        assert duplicate == {**first, "duplicate": True}
        assert db.query(InstrumentObservationRow).count() == 1
        assert db.query(AgentCommand).count() == 0
        conflicting = payload.model_copy(update={"source_root": "/different"})
        with pytest.raises(ObservationConflict):
            record_observation(db, conflicting, now=now)
        db.rollback()
        assert (
            db.get(InstrumentObservationRow, str(payload.event_id)).payload_json[
                "source_root"
            ]
            != "/different"
        )


def test_concurrent_retries_store_one_receipt(tmp_path):
    path = tmp_path / "agent.db"
    payload = observation()
    with get_agent_session(path):
        pass

    def submit(_):
        with get_agent_session(path) as db:
            return record_observation(db, payload, now=datetime.now(UTC))

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(submit, range(8)))
    assert sum(not receipt["duplicate"] for receipt in receipts) == 1


def test_source_order_freshness_and_unknown_instrument_state(tmp_path):
    now = datetime.now(UTC)
    with get_agent_session(tmp_path / "agent.db") as db:
        recent = observation(observed_at=now, source_updated_at=now - timedelta(days=4))
        record_observation(db, recent, now=now)
        record_observation(
            db,
            observation(observed_at=now - timedelta(hours=1)),
            now=now + timedelta(seconds=1),
        )
        status = latest_status(db, recent.instrument_id, now=now)
        assert status["observation"]["event_id"] == str(recent.event_id)
        assert status["collector_stale"] is False
        assert status["source_stale"] is True
        assert status["instrument_alive"] is None
        assert (
            latest_status(db, recent.instrument_id, now=now + timedelta(minutes=4))[
                "collector_stale"
            ]
            is True
        )
        assert latest_status(db, "missing", now=now) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"instrument_state": "complete"},
        {"pressure": 205},
        {"progress": 50},
        {"command": "stop"},
        {"observed_at": "2026-10-05T13:00:00"},
    ],
)
def test_metadata_cannot_claim_unsupported_vendor_fields(updates):
    with pytest.raises(ValidationError):
        observation(**updates)


def test_scan_only_stats_known_files_and_skips_links(tmp_path, monkeypatch):
    run = tmp_path / "project" / "example.d"
    run.mkdir(parents=True)
    (run / "analysis.tdf_bin").write_bytes(b"raw")
    (run / "unrelated.txt").write_text("not inspected")
    outside = tmp_path.parent / ("outside-" + uuid.uuid4().hex)
    outside.mkdir()
    (outside / "escape.d").mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    (run / "SampleInfo.xml").symlink_to(outside / "escape.d")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *a, **k: pytest.fail("Metadata collector must not read file contents"),
    )
    cfg = collector.CollectorConfig(
        collector_id="test", instrument_id="test", root=str(tmp_path)
    )
    result = collector.scan_directory(cfg)
    assert result["source_state"] == "readable"
    assert len(result["recent_runs"]) == 1
    assert result["recent_runs"][0]["relative_path"] == "project/example.d"
    assert result["recent_runs"][0]["observed_bytes"] == 3
    assert result["instrument_state"] == "unknown"
    InstrumentObservation.model_validate(result)


def test_scan_limits_and_missing_root(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.d").mkdir()
    cfg = collector.CollectorConfig(
        collector_id="test", instrument_id="test", root=str(tmp_path), max_entries=1
    )
    result = collector.scan_directory(cfg)
    assert result["truncated"] is True
    assert result["entries_examined"] == 1
    assert result["source_state"] == "partial"
    missing = collector.CollectorConfig(
        collector_id="test", instrument_id="test", root=str(tmp_path / "missing")
    )
    assert collector.scan_directory(missing)["source_state"] == "unavailable"


@pytest.mark.parametrize(
    "server",
    [
        "http://10.16.1.11:3001",
        "https://user:pass@host",
        "https://host/api",
        "file:///tmp",
        "https://host?x=1",
    ],
)
def test_unsafe_transport_rejected(server):
    with pytest.raises(ValueError):
        collector.endpoint_url(server)


def test_loopback_or_tls_origin_only():
    assert (
        collector.endpoint_url("http://127.0.0.1:8766")
        == "http://127.0.0.1:8766/api/instruments/observations"
    )
    assert (
        collector.endpoint_url("https://ispec.example")
        == "https://ispec.example/api/instruments/observations"
    )
    with pytest.raises(ValueError):
        collector.NoRedirect().redirect_request(
            None, None, 302, None, None, "https://elsewhere"
        )


def test_portable_cli_preview_without_ispec_imports_or_network(tmp_path):
    (tmp_path / "test.d").mkdir()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"collector_id": "demo", "instrument_id": "demo", "root": str(tmp_path)}
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", collector.__file__, "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["sent"] is False
    assert output["observation"]["recent_runs"][0]["relative_path"] == "test.d"
    assert output["observation"]["instrument_state"] == "unknown"


def test_send_requires_explicit_token_before_scan(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"collector_id": "demo", "instrument_id": "demo", "root": str(tmp_path)}
        )
    )
    monkeypatch.delenv("ISPEC_INSTRUMENT_COLLECTOR_TOKEN", raising=False)
    monkeypatch.setattr(
        collector, "collect", lambda *a: pytest.fail("should fail before scanning")
    )
    assert (
        collector.main(
            ["--config", str(config), "--send", "--server", "http://localhost:3001"]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_timeout_terminates_scan_worker(tmp_path, monkeypatch):
    actions = []
    reader = SimpleNamespace(poll=lambda timeout: False, close=lambda: None)
    writer = SimpleNamespace(close=lambda: None)
    worker = SimpleNamespace(
        start=lambda: actions.append("start"),
        join=lambda timeout: None,
        is_alive=lambda: True,
        terminate=lambda: actions.append("terminate"),
        kill=lambda: actions.append("kill"),
    )
    context = SimpleNamespace(
        Pipe=lambda **kw: (reader, writer), Process=lambda **kw: worker
    )
    monkeypatch.setattr(collector.multiprocessing, "get_context", lambda *args: context)
    cfg = collector.CollectorConfig(
        collector_id="test", instrument_id="test", root=str(tmp_path)
    )
    result = collector.collect(cfg)
    assert result["source_state"] == "timeout"
    assert result["instrument_error"] is None
    assert actions == ["start", "terminate", "kill"]


def test_unreadable_root_not_healthy(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError()

    monkeypatch.setattr(collector.os, "scandir", denied)
    cfg = collector.CollectorConfig(
        collector_id="test", instrument_id="test", root=str(tmp_path)
    )
    result = collector.scan_directory(cfg)
    assert result["source_state"] == "unavailable"
    assert result["read_errors"] == ["PermissionError"]


def test_foreground_watch_retries_identical_pending_event(
    tmp_path, monkeypatch, capsys
):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {"collector_id": "demo", "instrument_id": "demo", "root": str(tmp_path)}
        )
    )
    event = observation().model_dump(mode="json")
    scans = []
    deliveries = []
    sleeps = []
    monkeypatch.setenv(
        "ISPEC_INSTRUMENT_COLLECTOR_TOKEN", "test-only-credential-00000000000000"
    )

    def scan(*args):
        scans.append(1)
        return event

    def send(server, pending, token):
        deliveries.append(dict(pending))
        if len(deliveries) == 1:
            raise ValueError("unconfirmed delivery")
        return {"event_id": pending["event_id"], "stored": True, "duplicate": True}

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(collector, "collect", scan)
    monkeypatch.setattr(collector, "send_observation", send)
    monkeypatch.setattr(collector.time, "sleep", sleep)
    assert (
        collector.main(
            [
                "--config",
                str(cfg),
                "--send",
                "--watch",
                "--server",
                "http://localhost:3001",
            ]
        )
        == 0
    )
    assert len(scans) == 1
    assert deliveries[0] == deliveries[1]
    assert sleeps == [60, 60]
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert outputs[0]["sent"] is None
    assert outputs[1]["sent"] is True

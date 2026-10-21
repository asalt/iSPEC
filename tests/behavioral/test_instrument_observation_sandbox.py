"""Filesystem activity is not acquisition completion or authority to act."""

from datetime import UTC, datetime, timedelta
import os

import pytest

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.instruments.collector import CollectorConfig, scan_directory
from ispec.instruments.schema import InstrumentObservation
from ispec.instruments.store import latest_status, record_observation

pytestmark = pytest.mark.behavioral


def test_growing_then_quiet_run_never_becomes_completed(tmp_path):
    root = tmp_path / "instrument"
    run = root / "example.d"
    run.mkdir(parents=True)
    raw = run / "analysis.tdf_bin"
    cfg = CollectorConfig(
        collector_id="fixture", instrument_id="fixture", root=str(root)
    )
    now = datetime.now(UTC)
    with get_agent_session(tmp_path / "agent.db") as db:
        for index, content in enumerate((b"first", b"first more", b"first more")):
            raw.write_bytes(content)
            old = (now - timedelta(days=2)).timestamp()
            os.utime(raw, (old, old))
            os.utime(run, (old, old))
            snapshot = scan_directory(cfg)
            snapshot["observed_at"] = (now + timedelta(seconds=index)).isoformat()
            event = InstrumentObservation.model_validate(snapshot)
            record_observation(db, event, now=now + timedelta(seconds=index))
            status = latest_status(db, "fixture", now=now + timedelta(seconds=index))
            assert status["observation"]["instrument_state"] == "unknown"
            assert status["observation"]["progress"] is None
            assert status["source_stale"] is True
            assert status["collector_stale"] is False
            assert status["observation"]["recent_runs"][0]["observed_bytes"] == len(
                content
            )
        assert db.query(AgentCommand).count() == 0

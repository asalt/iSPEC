from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ispec.supervisor.sentinel import build_observation, build_sentinel_report, next_state_from_report


pytestmark = pytest.mark.behavioral


def _observation(text: str, *, target: str = "ispec:sentinel.1", active: bool = False):
    return build_observation(
        pane={
            "target": target,
            "pane_id": "%1",
            "session": "ispec",
            "window_name": "sentinel",
            "current_command": "bash",
            "pane_active": active,
        },
        snapshot={
            "target": target,
            "pane_id": "%1",
            "content": text,
            "last_nonempty_line": None,
            "pane_active": active,
        },
        captured_at="2026-04-27T12:00:00+00:00",
    )


def test_behavioral_sentinel_flags_risky_command_but_suppresses_repeated_candidate():
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    pane = _observation("Reviewing rsync -av --delete /source/ /media/backup/")

    first = build_sentinel_report(
        observations=[pane],
        previous_state={"panes": {}},
        now=now,
    )
    assert first["notifications"]["candidates"]
    assert first["notifications"]["candidates"][0]["risk"] is True
    assert first["notifications"]["candidates"][0]["attention"] == "human_review"
    assert first["notifications"]["slack"] == "simulated"

    state = next_state_from_report(
        report=first,
        observations=[pane],
        previous_state={"panes": {}},
        now=now,
    )
    repeated = build_sentinel_report(
        observations=[pane],
        previous_state=state,
        now=now + timedelta(seconds=30),
    )
    assert repeated["changed_panes"] == []
    assert repeated["notifications"]["candidates"] == []


def test_behavioral_sentinel_keeps_ordinary_progress_log_only():
    pane = _observation("Building wheel 42%", active=True)

    report = build_sentinel_report(
        observations=[pane],
        previous_state={"panes": {}},
        now=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
    )

    assert report["changed_panes"] == [pane.target]
    assert report["pane_states"][0]["state"] == "active"
    assert report["pane_states"][0]["notification_action"] == "log_only"
    assert report["notifications"]["candidates"] == []

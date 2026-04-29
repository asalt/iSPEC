from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ispec.agent_state.connect import get_agent_state_session
from ispec.agent_state.store import append_observation, list_heads, register_schema_version
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


def _pane_state(report: dict[str, Any], target: str = "ispec:sentinel.1") -> dict[str, Any]:
    for item in report.get("pane_states", []):
        if isinstance(item, dict) and item.get("target") == target:
            return item
    raise AssertionError(f"missing pane state for {target}")


def _review_packet(
    *,
    report: dict[str, Any],
    scenario: str,
    agent_state_head: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "sentinel_behavioral_review_packet",
        "scenario": scenario,
        "read_only": True,
        "sentinel_report": report,
        "agent_state_head": agent_state_head,
    }


def _seed_read_only_salience_head(db_path, *, agent_id: str = "sentinel-test") -> dict[str, Any]:
    with get_agent_state_session(db_path) as db:
        register_schema_version(
            db,
            schema_id=101,
            version=1,
            state_scope="mood",
            dims=[
                {"dim_index": 0, "name": "calm"},
                {"dim_index": 1, "name": "concerned"},
                {"dim_index": 2, "name": "risk_sensitive"},
                {"dim_index": 3, "name": "exploratory"},
            ],
            notes="behavioral sentinel read-only salience fixture",
        )
        append_observation(
            db,
            schema_id=101,
            schema_version=1,
            state_scope="mood",
            agent_id=agent_id,
            vector=[0.7, 0.35, 0.85, 0.45],
            source_kind="behavioral_fixture",
            source_ref="sentinel_salience_context",
        )
        heads = list_heads(db, agent_id=agent_id, state_scope="mood", limit=1)
    assert heads
    return heads[0]


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


def test_behavioral_sentinel_classifies_core_scenarios_without_live_side_effects():
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    scenarios = [
        ("ordinary-progress", "Building wheel 42%", True, "active", "log_only"),
        ("waiting-human", "Need approval before continuing", False, "waiting_for_human", "slack_candidate"),
        ("blocked", "blocked: missing credential", False, "blocked", "slack_candidate"),
        ("error", "Traceback\nRuntimeError: command failed", False, "error", "slack_candidate"),
        ("risky-rsync", "Reviewing rsync -av --delete /source/ /media/backup/", False, "idle", "slack_candidate"),
        ("complete", "30 passed in 12.3s", False, "complete", "log_only"),
        ("changed-low-importance", "I am reading logs for context", False, "idle", "log_only"),
    ]

    for label, text, active, expected_state, expected_action in scenarios:
        report = build_sentinel_report(
            observations=[_observation(text, active=active)],
            previous_state={"panes": {}},
            now=now,
        )
        state = _pane_state(report)
        assert state["state"] == expected_state, label
        assert state["notification_action"] == expected_action, label
        assert report["notifications"]["slack"] == "simulated"
        assert report["notifications"]["sent"] is False


def test_behavioral_sentinel_tracks_active_waiting_complete_transition():
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    active = _observation("Building wheel 42%", active=True)
    active_report = build_sentinel_report(
        observations=[active],
        previous_state={"panes": {}},
        now=now,
    )
    active_state = next_state_from_report(
        report=active_report,
        observations=[active],
        previous_state={"panes": {}},
        now=now,
    )
    assert active_report["changed_panes"] == [active.target]
    assert _pane_state(active_report)["state"] == "active"

    waiting = _observation("Need approval before continuing")
    waiting_report = build_sentinel_report(
        observations=[waiting],
        previous_state=active_state,
        now=now + timedelta(seconds=10),
    )
    waiting_state = next_state_from_report(
        report=waiting_report,
        observations=[waiting],
        previous_state=active_state,
        now=now + timedelta(seconds=10),
    )
    assert waiting_report["changed_panes"] == [waiting.target]
    assert _pane_state(waiting_report)["state"] == "waiting_for_human"
    assert waiting_report["notifications"]["candidates"]

    repeated_waiting = build_sentinel_report(
        observations=[waiting],
        previous_state=waiting_state,
        now=now + timedelta(seconds=20),
    )
    assert repeated_waiting["changed_panes"] == []
    assert repeated_waiting["notifications"]["candidates"] == []

    complete = _observation("30 passed in 12.3s")
    complete_report = build_sentinel_report(
        observations=[complete],
        previous_state=waiting_state,
        now=now + timedelta(seconds=30),
    )
    assert complete_report["changed_panes"] == [complete.target]
    assert _pane_state(complete_report)["state"] == "complete"
    assert _pane_state(complete_report)["notification_action"] == "log_only"


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


def test_behavioral_sentinel_review_packet_logs_salience_without_changing_decision(
    behavioral_datastore,
):
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    pane = _observation("Need approval before continuing")
    report_without_salience = build_sentinel_report(
        observations=[pane],
        previous_state={"panes": {}},
        now=now,
    )
    salience_head = _seed_read_only_salience_head(behavioral_datastore.agent_state_db_path)
    packet = _review_packet(
        report=report_without_salience,
        scenario="waiting-with-read-only-salience",
        agent_state_head=salience_head,
    )
    report_with_salience_nearby = build_sentinel_report(
        observations=[pane],
        previous_state={"panes": {}},
        now=now,
    )

    assert packet["read_only"] is True
    assert packet["agent_state_head"]["state_scope"] == "mood"
    assert packet["agent_state_head"]["dim_names"] == ["calm", "concerned", "risk_sensitive", "exploratory"]
    assert packet["agent_state_head"]["vector"] == pytest.approx([0.7, 0.35, 0.85, 0.45], rel=1e-6)
    assert packet["sentinel_report"] == report_without_salience
    assert report_with_salience_nearby == report_without_salience
    assert _pane_state(report_without_salience)["notification_action"] == "slack_candidate"

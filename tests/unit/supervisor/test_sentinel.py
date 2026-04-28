from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ispec.supervisor.sentinel import (
    build_observation,
    build_sentinel_report,
    classify_observation,
    content_hash,
    format_slack_candidate_text,
    next_state_from_report,
    normalize_pane_text,
    observe_tmux_panes,
)


def _observation(text: str, *, target: str = "ispec:worker.1", active: bool = False):
    pane = {
        "target": target,
        "pane_id": "%1",
        "session": "ispec",
        "window_name": "worker",
        "current_command": "bash",
        "pane_active": active,
    }
    return build_observation(
        pane=pane,
        snapshot={
            "target": target,
            "pane_id": "%1",
            "content": text,
            "last_nonempty_line": None,
            "pane_active": active,
        },
        captured_at="2026-04-27T12:00:00+00:00",
    )


def test_normalize_pane_text_strips_ansi_control_and_trailing_blank_lines():
    assert normalize_pane_text("\x1b[31mhello\x1b[0m  \r\nworld\x07\n\n") == "hello\nworld"
    assert content_hash("hello\n\n") == content_hash("hello")


def test_classify_waiting_blocked_error_complete_and_progress():
    waiting = classify_observation(_observation("Need approval before I run rsync"), changed=True)
    assert waiting.state == "waiting_for_human"
    assert waiting.attention == "human_review"

    blocked = classify_observation(_observation("blocked: missing credential"), changed=True)
    assert blocked.state == "blocked"

    error = classify_observation(_observation("Traceback\nRuntimeError: command failed"), changed=True)
    assert error.state == "error"
    assert error.notification_action == "slack_candidate"

    complete = classify_observation(_observation("30 passed in 12.3s"), changed=True)
    assert complete.state == "complete"
    assert complete.notification_action in {"log_only", "slack_candidate"}

    active = classify_observation(_observation("building wheel 42%", active=True), changed=True)
    assert active.state == "active"
    assert active.notification_action == "log_only"


def test_risky_rsync_is_attention_candidate_when_changed():
    state = classify_observation(
        _observation("reviewing rsync -av --delete /src/ /media/backup/"),
        changed=True,
    )

    assert state.risk is True
    assert state.attention == "human_review"
    assert state.notification_action == "slack_candidate"


def test_build_report_marks_only_changed_panes_and_suppresses_duplicates():
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    old = _observation("ordinary progress 10%", target="ispec:a.1")
    changed = _observation("Traceback\nValueError: bad", target="ispec:b.1")
    previous = {
        "panes": {
            old.target: {"content_hash": old.content_hash},
            changed.target: {"content_hash": "old-hash"},
        }
    }

    report = build_sentinel_report(
        observations=[old, changed],
        previous_state=previous,
        now=now,
    )

    assert report["changed_panes"] == [changed.target]
    states = {item["target"]: item for item in report["pane_states"]}
    assert states[old.target]["changed"] is False
    assert states[changed.target]["state"] == "error"
    assert report["notifications"]["candidates"][0]["target"] == changed.target

    next_state = next_state_from_report(
        report=report,
        observations=[old, changed],
        previous_state=previous,
        now=now,
    )
    repeated = build_sentinel_report(
        observations=[old, changed],
        previous_state=next_state,
        now=now + timedelta(seconds=10),
    )
    assert repeated["notifications"]["candidates"] == []


def test_format_slack_candidate_text_is_compact_and_structured():
    report = build_sentinel_report(
        observations=[_observation("blocked: waiting for approval")],
        previous_state={"panes": {}},
    )

    text = format_slack_candidate_text(report)
    assert text.startswith("orchestrator sentinel: attention candidate")
    assert "ispec:worker.1" in text
    assert report["notifications"]["message_preview"] == text


def test_observe_tmux_panes_uses_only_supplied_read_observer_functions():
    calls: list[tuple[str, object]] = []

    def list_panes():
        calls.append(("list", None))
        return [{"target": "ispec:worker.1", "pane_id": "%1"}]

    def capture_snapshot(**kwargs):
        calls.append(("capture", kwargs))
        return {"target": "ispec:worker.1", "pane_id": "%1", "content": "tests passed"}

    observations, errors = observe_tmux_panes(
        list_panes=list_panes,
        capture_snapshot=capture_snapshot,
        lines=40,
    )

    assert errors == []
    assert len(observations) == 1
    assert [call[0] for call in calls] == ["list", "capture"]
    assert calls[1][1]["include_history"] is False

from __future__ import annotations

import json
from datetime import UTC, datetime

from ispec.agent.commands import COMMAND_SLACK_POST_MESSAGE
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.supervisor.loop import _enqueue_command, _ensure_slack_scheduled_commands, _process_one_command


def test_supervisor_seeds_slack_schedules(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    fixed_now = datetime(2026, 1, 6, 9, 14, tzinfo=UTC)  # Tuesday
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)

    monkeypatch.setenv(
        "ISPEC_SLACK_SCHEDULE_JSON",
        json.dumps(
            [
                {
                    "name": "weekly_meeting_ready",
                    "weekday": "tue",
                    "time": "09:15",
                    "timezone": "UTC",
                    "channel": "C123",
                    "text": "Weekly meeting starting — are you ready?",
                }
            ]
        ),
    )

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=fixed_now,
                updated_at=fixed_now,
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        agent_db.commit()

    seeded = _ensure_slack_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["ok"] is True
    assert seeded["scheduled"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        rows = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_SLACK_POST_MESSAGE)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(rows) == 1
        cmd = rows[0]
        available_at = cmd.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=UTC)
        assert available_at == datetime(2026, 1, 6, 9, 15, tzinfo=UTC)
        assert cmd.payload_json["channel"] == "C123"
        assert cmd.payload_json["schedule"]["name"] == "weekly_meeting_ready"

    seeded_again = _ensure_slack_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded_again["ok"] is True
    assert seeded_again["scheduled"] == 0


def test_supervisor_processes_slack_post_message_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")

    fixed_now = datetime(2026, 1, 6, 9, 14, tzinfo=UTC)
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=fixed_now,
                updated_at=fixed_now,
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        agent_db.commit()

    calls: list[dict[str, object]] = []

    def fake_post(url, *, headers=None, json=None, timeout=None):  # type: ignore[no-untyped-def]
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})

        class FakeResponse:
            def raise_for_status(self):  # type: ignore[no-untyped-def]
                return None

            def json(self):  # type: ignore[no-untyped-def]
                return {"ok": True, "channel": (json or {}).get("channel"), "ts": "123.456"}

        return FakeResponse()

    monkeypatch.setattr(supervisor_loop.requests, "post", fake_post)

    cmd_id = _enqueue_command(
        command_type=COMMAND_SLACK_POST_MESSAGE,
        payload={
            "channel": "C123",
            "text": "Hello from the supervisor",
            "schedule": {"name": "weekly_meeting_ready", "key": "weekly_meeting_ready:2026-01-06T09:15:00+00:00"},
        },
    )
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True
    assert calls and str(calls[0]["url"]).endswith("/chat.postMessage")

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        scheduler = run.summary_json.get("scheduler") if isinstance(run.summary_json, dict) else None
        assert isinstance(scheduler, dict)
        slack = scheduler.get("slack")
        assert isinstance(slack, dict)
        schedule_state = slack.get("weekly_meeting_ready")
        assert isinstance(schedule_state, dict)
        assert schedule_state.get("last_sent_key") == "weekly_meeting_ready:2026-01-06T09:15:00+00:00"

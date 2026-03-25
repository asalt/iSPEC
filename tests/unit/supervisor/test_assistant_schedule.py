from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ispec.agent.commands import COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT, COMMAND_SLACK_POST_MESSAGE
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.assistant.service import AssistantReply
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import _enqueue_command, _ensure_assistant_scheduled_commands, _process_one_command


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def test_supervisor_seeds_assistant_schedules_in_central_time(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    fixed_now = datetime(2026, 1, 6, 14, 59, tzinfo=UTC)  # Tuesday 08:59 CST
    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "utcnow", lambda: fixed_now)
    monkeypatch.setenv(
        "ISPEC_ASSISTANT_SCHEDULE_JSON",
        json.dumps(
            [
                {
                    "name": "weekly_current_projects",
                    "weekday": "tue",
                    "time": "09:00",
                    "timezone": "America/Chicago",
                    "prompt": "Prepare the current projects update and post it to staff Slack.",
                    "allowed_tools": ["count_current_projects", "latest_projects", "assistant_enqueue_staff_slack_message"],
                    "required_tool": "assistant_enqueue_staff_slack_message",
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

    seeded = _ensure_assistant_scheduled_commands(agent_id="agent-1", run_id="run-1")
    assert seeded["ok"] is True
    assert seeded["scheduled"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        rows = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(rows) == 1
        cmd = rows[0]
        available_at = cmd.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=UTC)
        assert available_at == datetime(2026, 1, 6, 15, 0, tzinfo=UTC)
        assert cmd.payload_json["job"]["name"] == "weekly_current_projects"
        assert cmd.payload_json["job"]["required_tool"] == "assistant_enqueue_staff_slack_message"


def test_supervisor_processes_scheduled_assistant_prompt_and_queues_slack_post(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    core_db_path = tmp_path / "core.db"
    assistant_db_path = tmp_path / "assistant.db"
    schedule_db_path = tmp_path / "schedule.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_DB_PATH", str(core_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_SCHEDULE_DB_PATH", str(schedule_db_path))
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", "C123STAFF")

    fixed_now = datetime(2026, 1, 6, 15, 0, tzinfo=UTC)
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

    real_run_tool = supervisor_loop.run_tool

    def fake_run_tool(*, name, args, core_db, assistant_db, agent_db, schedule_db, omics_db, user, api_schema, user_message):  # type: ignore[no-untyped-def]
        if name == "count_current_projects":
            return {
                "ok": True,
                "tool": name,
                "result": {"count": 7, "scope": "current"},
            }
        return real_run_tool(
            name=name,
            args=args,
            core_db=core_db,
            assistant_db=assistant_db,
            agent_db=agent_db,
            schedule_db=schedule_db,
            omics_db=omics_db,
            user=user,
            api_schema=api_schema,
            user_message=user_message,
        )

    monkeypatch.setattr(supervisor_loop, "run_tool", fake_run_tool)

    calls: list[dict[str, object]] = []

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, **_):  # type: ignore[no-untyped-def]
        calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        index = len(calls)
        if index == 1:
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "count_current_projects", "arguments": "{}"},
                    }
                ],
            )
        if index == 2:
            return AssistantReply(
                content="FINAL:\nWe currently have 7 active projects.",
                provider="test",
                model="test-model",
                meta=None,
            )
        if index == 3:
            assert tool_choice == {
                "type": "function",
                "function": {"name": "assistant_enqueue_staff_slack_message"},
            }
            assert isinstance(tools, list) and len(tools) == 1
            return AssistantReply(
                content="",
                provider="test",
                model="test-model",
                meta=None,
                tool_calls=[
                    {
                        "id": "call_2",
                        "function": {
                            "name": "assistant_enqueue_staff_slack_message",
                            "arguments": json.dumps(
                                {
                                    "message": "We currently have 7 active projects.",
                                    "confirm": True,
                                    "reason": "weekly current projects update",
                                }
                            ),
                        },
                    }
                ],
            )
        return AssistantReply(
            content="FINAL:\nPosted the scheduled staff update.",
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(
        command_type=COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT,
        payload={
            "job": {
                "name": "weekly_current_projects",
                "prompt": "Prepare the weekly current projects update and post it to staff Slack.",
                "allowed_tools": ["count_current_projects", "assistant_enqueue_staff_slack_message"],
                "required_tool": "assistant_enqueue_staff_slack_message",
                "max_tool_calls": 4,
            },
            "schedule": {
                "name": "weekly_current_projects",
                "key": "weekly_current_projects:2026-01-06T15:00:00+00:00",
                "occurrence_utc": "2026-01-06T15:00:00+00:00",
                "timezone": "America/Chicago",
                "weekday": 1,
                "time": "09:00",
            },
        },
    )
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True
    assert len(calls) == 4

    with get_agent_session(agent_db_path) as agent_db:
        scheduled_cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert scheduled_cmd.status == "succeeded"
        assert scheduled_cmd.result_json["required_tool_called"] is True
        queued_ids = scheduled_cmd.result_json["queued_command_ids"]
        assert isinstance(queued_ids, list) and len(queued_ids) == 1

        slack_cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(queued_ids[0])).one()
        assert slack_cmd.command_type == COMMAND_SLACK_POST_MESSAGE
        assert slack_cmd.status == "queued"
        assert slack_cmd.payload_json["channel"] == "C123STAFF"
        assert "7 active projects" in slack_cmd.payload_json["text"]

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        scheduler = run.summary_json.get("scheduler") if isinstance(run.summary_json, dict) else None
        assert isinstance(scheduler, dict)
        assistant_jobs = scheduler.get("assistant_jobs")
        assert isinstance(assistant_jobs, dict)
        schedule_state = assistant_jobs.get("weekly_current_projects")
        assert isinstance(schedule_state, dict)
        assert schedule_state.get("last_completed_key") == "weekly_current_projects:2026-01-06T15:00:00+00:00"

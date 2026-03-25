from __future__ import annotations

import json

from ispec.assistant.tools import openai_tools_for_user, run_tool
from ispec.db.models import AuthUser, UserRole


def _internal_user() -> AuthUser:
    return AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )


def test_assistant_list_tools_shows_schedule_tools_unavailable_when_disabled(db_session, monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", raising=False)

    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True, "query": "scheduled"},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show scheduled job tools",
    )
    assert payload["ok"] is True
    unavailable = payload["result"].get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    names = {item.get("name") for item in unavailable if isinstance(item, dict)}
    assert "assistant_list_scheduled_jobs" in names
    assert "assistant_upsert_scheduled_job" in names
    assert "assistant_delete_scheduled_job" in names

    tool_names = {
        tool["function"]["name"]
        for tool in openai_tools_for_user(_internal_user())
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    assert "assistant_list_scheduled_jobs" not in tool_names


def test_assistant_list_tools_includes_schedule_tools_when_enabled(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_PATH", str(tmp_path / "assistant-schedules.json"))

    payload = run_tool(
        name="assistant_list_tools",
        args={"query": "scheduled", "limit": 50},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show scheduled job tools",
    )
    assert payload["ok"] is True
    tool_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "assistant_list_scheduled_jobs" in tool_names
    assert "assistant_upsert_scheduled_job" in tool_names
    assert "assistant_delete_scheduled_job" in tool_names

    openai_names = {
        tool["function"]["name"]
        for tool in openai_tools_for_user(_internal_user())
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    assert "assistant_list_scheduled_jobs" in openai_names
    assert "assistant_upsert_scheduled_job" in openai_names
    assert "assistant_delete_scheduled_job" in openai_names


def test_assistant_list_scheduled_jobs_reads_file(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    schedule_path = tmp_path / "assistant-schedules.json"
    schedule_path.write_text(
        json.dumps(
            [
                {
                    "name": "weekly_current_projects_update",
                    "weekday": "tue",
                    "time": "09:00",
                    "timezone": "America/Chicago",
                    "prompt": "Prepare the weekly projects staff update.",
                    "allowed_tools": ["latest_projects", "assistant_enqueue_staff_slack_message"],
                    "required_tool": "assistant_enqueue_staff_slack_message",
                    "enabled": True,
                },
                {
                    "name": "disabled_trial",
                    "weekday": "fri",
                    "time": "16:00",
                    "timezone": "America/Chicago",
                    "prompt": "Trial summary.",
                    "allowed_tools": ["project_counts_snapshot"],
                    "enabled": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_PATH", str(schedule_path))

    payload = run_tool(
        name="assistant_list_scheduled_jobs",
        args={"include_disabled": False},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="list scheduled jobs",
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["storage"]["path"] == str(schedule_path)
    assert result["storage"]["writable"] is True
    assert result["count"] == 1
    assert result["jobs"][0]["name"] == "weekly_current_projects_update"
    assert result["jobs"][0]["weekday"] == "tue"
    assert result["jobs"][0]["required_tool"] == "assistant_enqueue_staff_slack_message"


def test_assistant_schedule_write_tools_stay_hidden_without_path(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    monkeypatch.delenv("ISPEC_ASSISTANT_SCHEDULE_PATH", raising=False)
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_JSON", "[]")

    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True, "query": "scheduled"},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show scheduled job tools",
    )
    assert payload["ok"] is True
    available_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "assistant_list_scheduled_jobs" in available_names
    assert "assistant_upsert_scheduled_job" not in available_names

    unavailable = payload["result"].get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    reasons = {
        item["name"]: item.get("unavailable_reason")
        for item in unavailable
        if isinstance(item, dict) and item.get("name")
    }
    assert "ISPEC_ASSISTANT_SCHEDULE_PATH" in str(reasons.get("assistant_upsert_scheduled_job"))


def test_assistant_upsert_scheduled_job_creates_and_updates_file(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    schedule_path = tmp_path / "assistant-schedules.json"
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_PATH", str(schedule_path))

    created = run_tool(
        name="assistant_upsert_scheduled_job",
        args={
            "name": "weekly_current_projects_update",
            "weekday": "tue",
            "time": "09:00",
            "timezone": "America/Chicago",
            "prompt": "Prepare the weekly current projects update and post it to staff Slack.",
            "allowed_tools": ["count_current_projects", "assistant_enqueue_staff_slack_message"],
            "required_tool": "assistant_enqueue_staff_slack_message",
            "max_tool_calls": 5,
            "priority": 3,
            "grace_seconds": 900,
            "confirm": True,
        },
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="create the weekly current projects schedule",
    )
    assert created["ok"] is True
    assert created["result"]["action"] == "created"

    updated = run_tool(
        name="assistant_upsert_scheduled_job",
        args={
            "name": "weekly_current_projects_update",
            "weekday": "tue",
            "time": "09:30",
            "timezone": "America/Chicago",
            "prompt": "Prepare the weekly current projects update and post it to staff Slack.",
            "allowed_tools": ["count_current_projects"],
            "required_tool": "assistant_enqueue_staff_slack_message",
            "confirm": True,
            "enabled": False,
        },
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="update the weekly current projects schedule",
    )
    assert updated["ok"] is True
    assert updated["result"]["action"] == "updated"
    job = updated["result"]["job"]
    assert job["time"] == "09:30"
    assert job["enabled"] is False
    assert "assistant_enqueue_staff_slack_message" in job["allowed_tools"]

    parsed = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list) and len(parsed) == 1
    assert parsed[0]["time"] == "09:30"
    assert parsed[0]["enabled"] is False


def test_assistant_delete_scheduled_job_removes_entry(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    schedule_path = tmp_path / "assistant-schedules.json"
    schedule_path.write_text(
        json.dumps(
            [
                {
                    "name": "weekly_current_projects_update",
                    "weekday": "tue",
                    "time": "09:00",
                    "timezone": "America/Chicago",
                    "prompt": "Prepare the weekly projects staff update.",
                    "allowed_tools": ["latest_projects", "assistant_enqueue_staff_slack_message"],
                    "required_tool": "assistant_enqueue_staff_slack_message",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_PATH", str(schedule_path))

    payload = run_tool(
        name="assistant_delete_scheduled_job",
        args={"name": "weekly_current_projects_update", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="delete the weekly current projects schedule",
    )
    assert payload["ok"] is True
    assert payload["result"]["deleted"] is True
    assert json.loads(schedule_path.read_text(encoding="utf-8")) == []


def test_assistant_upsert_scheduled_job_requires_path_for_writes(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_TOOLS_ENABLED", "1")
    monkeypatch.delenv("ISPEC_ASSISTANT_SCHEDULE_PATH", raising=False)
    monkeypatch.setenv("ISPEC_ASSISTANT_SCHEDULE_JSON", "[]")

    payload = run_tool(
        name="assistant_upsert_scheduled_job",
        args={
            "name": "weekly_current_projects_update",
            "weekday": "tue",
            "time": "09:00",
            "prompt": "Prepare the weekly current projects update.",
            "allowed_tools": ["latest_projects"],
            "confirm": True,
        },
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="create the weekly current projects schedule",
    )
    assert payload["ok"] is False
    assert "ISPEC_ASSISTANT_SCHEDULE_PATH" in payload["error"]

from __future__ import annotations

import pytest

from ispec.agent.commands import COMMAND_SUPPORT_CHAT_TURN
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.api.routes import support as support_routes
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import _enqueue_command, _process_one_command, utcnow


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def test_supervisor_processes_support_chat_turn_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(tmp_path / "assistant.db"))
    monkeypatch.setenv("ISPEC_DB_PATH", str(tmp_path / "ispec.db"))
    monkeypatch.setenv("ISPEC_OMICS_DB_PATH", str(tmp_path / "omics.db"))
    monkeypatch.setenv("ISPEC_SCHEDULE_DB_PATH", str(tmp_path / "schedule.db"))
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "1")

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=utcnow(),
                updated_at=utcnow(),
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        agent_db.commit()

    captured: dict[str, str] = {}

    def fake_chat(
        payload,
        request=None,
        assistant_db=None,
        agent_db=None,
        core_db=None,
        omics_db=None,
        schedule_db=None,
        user=None,
    ):
        assert isinstance(payload, support_routes.ChatRequest)
        meta = payload.meta if isinstance(payload.meta, dict) else {}
        assert meta.get("_queue_force_inline") is True
        captured["session_id"] = payload.sessionId
        return support_routes.ChatResponse(
            sessionId=payload.sessionId,
            messageId=321,
            message="queued chat turn complete",
        )

    monkeypatch.setattr(support_routes, "chat", fake_chat)

    command_id = _enqueue_command(
        command_type=COMMAND_SUPPORT_CHAT_TURN,
        payload={
            "chat_request": {
                "sessionId": "queue-session-1",
                "message": "queued hello",
                "history": [],
                "ui": None,
            },
            "user_id": None,
        },
        priority=0,
    )
    assert isinstance(command_id, int)

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True
    assert captured.get("session_id") == "queue-session-1"

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).one()
        assert cmd.status == "succeeded"
        result = dict(cmd.result_json or {})
        chat_response = result.get("chat_response")
        assert isinstance(chat_response, dict)
        assert chat_response.get("sessionId") == "queue-session-1"
        assert chat_response.get("message") == "queued chat turn complete"


def test_supervisor_processes_support_chat_turn_with_api_key_service_user(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(tmp_path / "assistant.db"))
    monkeypatch.setenv("ISPEC_DB_PATH", str(tmp_path / "ispec.db"))
    monkeypatch.setenv("ISPEC_OMICS_DB_PATH", str(tmp_path / "omics.db"))
    monkeypatch.setenv("ISPEC_SCHEDULE_DB_PATH", str(tmp_path / "schedule.db"))
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "1")

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=utcnow(),
                updated_at=utcnow(),
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        agent_db.commit()

    captured: dict[str, object] = {}

    def fake_chat(
        payload,
        request=None,
        assistant_db=None,
        agent_db=None,
        core_db=None,
        omics_db=None,
        schedule_db=None,
        user=None,
    ):
        assert isinstance(payload, support_routes.ChatRequest)
        captured["user_id"] = getattr(user, "id", None)
        captured["username"] = getattr(user, "username", None)
        return support_routes.ChatResponse(
            sessionId=payload.sessionId,
            messageId=654,
            message="queued chat turn complete",
        )

    monkeypatch.setattr(support_routes, "chat", fake_chat)

    command_id = _enqueue_command(
        command_type=COMMAND_SUPPORT_CHAT_TURN,
        payload={
            "chat_request": {
                "sessionId": "queue-session-api-key",
                "message": "queued hello from slack",
                "history": [],
                "ui": None,
                "meta": {"source": "slack"},
            },
            "user_id": 0,
        },
        priority=0,
    )
    assert isinstance(command_id, int)

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True
    assert captured.get("user_id") == 0
    assert captured.get("username") == "api_key"

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).one()
        assert cmd.status == "succeeded"
        assert cmd.error in (None, "")

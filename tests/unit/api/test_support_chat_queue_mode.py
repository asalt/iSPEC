from __future__ import annotations

from ispec.agent.commands import COMMAND_ORCHESTRATOR_TICK, COMMAND_SUPPORT_CHAT_TURN
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, ChatResponse, chat, utcnow
from ispec.assistant.connect import get_assistant_session


def test_support_chat_queue_mode_enqueues_command_and_waits(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "1")

    assistant_path = tmp_path / "assistant.db"
    agent_path = tmp_path / "agent.db"

    with (
        get_assistant_session(assistant_path) as assistant_db,
        get_agent_session(agent_path) as agent_db,
    ):
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=utcnow(),
                updated_at=utcnow(),
                config_json={},
                state_json={},
                summary_json={},
            )
        )
        agent_db.commit()

        seen: dict[str, int] = {}

        def fake_wait_for_queue(*, agent_db, command_id, wait_seconds, poll_seconds):  # type: ignore[no-redef]
            row = agent_db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).one()
            assert row.command_type == COMMAND_SUPPORT_CHAT_TURN
            payload = dict(row.payload_json or {})
            request_payload = payload.get("chat_request")
            assert isinstance(request_payload, dict)
            assert request_payload.get("sessionId") == "session-queue"
            seen["command_id"] = int(command_id)
            assert wait_seconds >= 1
            assert poll_seconds >= 0.1
            return ChatResponse(sessionId="session-queue", messageId=42, message="queued response")

        monkeypatch.setattr(support_routes, "_wait_for_queued_chat_response", fake_wait_for_queue)

        response = chat(
            ChatRequest(
                sessionId="session-queue",
                message="hello from queue",
                history=[],
                ui=None,
                meta=None,
            ),
            assistant_db=assistant_db,
            agent_db=agent_db,
            core_db=db_session,
            user=None,
        )

        assert response.sessionId == "session-queue"
        assert response.messageId == 42
        assert response.message == "queued response"
        assert int(seen.get("command_id") or 0) > 0

        tick = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
            .order_by(AgentCommand.id.desc())
            .first()
        )
        assert tick is not None


def test_support_chat_queue_mode_falls_back_inline_without_supervisor_heartbeat(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "1")

    assistant_path = tmp_path / "assistant.db"
    agent_path = tmp_path / "agent.db"

    with (
        get_assistant_session(assistant_path) as assistant_db,
        get_agent_session(agent_path) as agent_db,
    ):
        def fail_wait(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("queue wait should not be called when supervisor heartbeat is missing")

        monkeypatch.setattr(support_routes, "_wait_for_queued_chat_response", fail_wait)

        response = chat(
            ChatRequest(
                sessionId="session-inline-fallback",
                message="hello inline fallback",
                history=[],
                ui=None,
                meta=None,
            ),
            assistant_db=assistant_db,
            agent_db=agent_db,
            core_db=db_session,
            user=None,
        )

        assert response.sessionId == "session-inline-fallback"
        assert response.messageId is not None
        assert isinstance(response.message, str) and response.message

        queued_chat = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_SUPPORT_CHAT_TURN)
            .order_by(AgentCommand.id.desc())
            .first()
        )
        assert queued_chat is None

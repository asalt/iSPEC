from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ispec.agent.commands import COMMAND_BUILD_SUPPORT_DIGEST, COMMAND_ORCHESTRATOR_TICK
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview
from ispec.assistant.tools import run_tool


def test_assistant_stats_reports_review_backlog(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_stats",
            args={},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["sessions_total"] == 1
        assert result["messages_total"] == 2
        assert result["sessions_needing_review"] == 1
        assert result["sessions_reviewed"] == 0


def test_assistant_recent_sessions_lists_state_fields(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_recent_sessions",
            args={"limit": 5},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        sessions = payload["result"]["sessions"]
        assert len(sessions) == 1
        item = sessions[0]
        assert item["session_id"] == "s1"
        assert item["message_count"] == 2
        assert item["last_user_message"] == "Hi"
        assert item["reviewed_up_to_id"] == 0


def test_assistant_stats_reports_supervisor_health_snapshot(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    agent_db_path = tmp_path / "agent.db"
    now = datetime.now(UTC)

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
            run_id="run-1",
            agent_id="agent-1",
            kind="supervisor",
            status="running",
            config_json={},
            state_json={},
            summary_json={
                "orchestrator": {
                    "next_tick_reason": "idle_backoff",
                    "next_tick_seconds": 60,
                }
            },
        )
        run.updated_at = now - timedelta(minutes=10)
        agent_db.add(run)
        agent_db.flush()

        queued_tick = AgentCommand(
            command_type=COMMAND_ORCHESTRATOR_TICK,
            status="queued",
            available_at=now - timedelta(minutes=3),
            payload_json={},
        )
        queued_digest = AgentCommand(
            command_type=COMMAND_BUILD_SUPPORT_DIGEST,
            status="queued",
            available_at=now - timedelta(minutes=2),
            payload_json={},
        )
        agent_db.add_all([queued_tick, queued_digest])
        agent_db.flush()

        failed_step = AgentStep(
            run_pk=int(run.id),
            step_index=0,
            kind=COMMAND_BUILD_SUPPORT_DIGEST,
            ok=False,
            error="invalid_digest_output",
            started_at=now - timedelta(minutes=4),
            ended_at=now - timedelta(minutes=4),
        )
        agent_db.add(failed_step)
        agent_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db, get_agent_session(agent_db_path) as agent_db:
        payload = run_tool(
            name="assistant_stats",
            args={},
            core_db=db_session,
            assistant_db=assistant_db,
            agent_db=agent_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        health = result["supervisor_health"]
        assert isinstance(health, dict)
        assert health["recent_failed_steps"]["total"] == 1
        assert health["recent_failed_steps"]["invalid_output_total"] == 1
        assert health["recent_failed_steps"]["items"][0]["kind"] == COMMAND_BUILD_SUPPORT_DIGEST
        assert health["commands"]["queued_by_type"][COMMAND_ORCHESTRATOR_TICK] == 1
        assert health["commands"]["queued_by_type"][COMMAND_BUILD_SUPPORT_DIGEST] == 1
        assert health["orchestrator"]["queued_tick"]["command_id"] is not None
        assert health["orchestrator"]["is_overdue"] is True
        assert health["orchestrator"]["overdue_seconds"] >= 60


def test_assistant_get_session_review_returns_review_when_present(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.add(
            SupportSessionReview(
                session_pk=session.id,
                target_message_id=2,
                schema_version=1,
                review_json={
                    "schema_version": 1,
                    "session_id": "s1",
                    "target_message_id": 2,
                    "summary": "ok",
                    "issues": [],
                    "repo_search_queries": [],
                    "followups": [],
                },
            )
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_get_session_review",
            args={"session_id": "s1"},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["session_id"] == "s1"
        assert result["reviewed_up_to_id"] == 2
        assert isinstance(result["review"], dict)

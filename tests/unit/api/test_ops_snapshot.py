from __future__ import annotations

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentRun
from ispec.api.routes.ops import snapshot
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession


def test_ops_snapshot_reports_counts_and_review_backlog(tmp_path):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi", provider="test"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello", provider="test"),
            ]
        )
        assistant_db.commit()

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="run-1",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                config_json={},
                state_json={"checks": {"backend": {"ok": True}}},
                summary_json={"orchestrator": {"ticks": 1}},
            )
        )
        agent_db.add(AgentCommand(command_type="orchestrator_tick_v1", status="queued", payload_json={"x": 1}))
        agent_db.commit()

    with (
        get_assistant_session(assistant_db_path) as assistant_db,
        get_agent_session(agent_db_path) as agent_db,
    ):
        result = snapshot(assistant_db=assistant_db, agent_db=agent_db, user=None)
        assert result.ok is True
        assert result.assistant.sessions_total == 1
        assert result.assistant.messages_total == 2
        assert result.assistant.sessions_needing_review == 1
        assert result.agent.commands_queued == 1
        assert result.agent.latest_supervisor_run is not None


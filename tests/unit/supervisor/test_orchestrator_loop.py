from __future__ import annotations

import json

from ispec.agent.commands import (
    COMMAND_ORCHESTRATOR_TICK,
    COMMAND_REVIEW_SUPPORT_SESSION,
)
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview
from ispec.assistant.service import AssistantReply
from ispec.supervisor.loop import _enqueue_command, _process_one_command, utcnow


def test_supervisor_processes_orchestrator_tick_and_schedules_followups(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

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

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
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
        agent_db.add(run)
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        context = json.loads(messages[-1]["content"])
        needs_review = context["assistant"]["sessions_needing_review"]
        assert needs_review and needs_review[0]["session_id"] == "s1"

        decision = {
            "schema_version": 1,
            "thoughts": "Review the newest session first.",
            "next_tick_seconds": 60,
            "commands": [
                {
                    "command_type": COMMAND_REVIEW_SUPPORT_SESSION,
                    "payload": {"session_id": "s1"},
                    "delay_seconds": 0,
                    "priority": 0,
                }
            ],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

        queued = agent_db.query(AgentCommand).filter(AgentCommand.status == "queued").all()
        queued_types = {row.command_type for row in queued}
        assert COMMAND_REVIEW_SUPPORT_SESSION in queued_types
        assert COMMAND_ORCHESTRATOR_TICK in queued_types

        step = agent_db.query(AgentStep).filter(AgentStep.kind == COMMAND_ORCHESTRATOR_TICK).one()
        assert step.ok is True
        assert isinstance(step.prompt_json, dict)
        assert isinstance(step.response_json, dict)


def test_orchestrator_tick_forces_session_review_when_model_schedules_none(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

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

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
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
        agent_db.add(run)
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        decision = {
            "schema_version": 1,
            "thoughts": "",
            "next_tick_seconds": 60,
            "commands": [],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"
        decision = cmd.result_json.get("decision")
        assert isinstance(decision, dict)
        commands = decision.get("commands")
        assert isinstance(commands, list) and commands
        assert commands[0]["command_type"] == COMMAND_REVIEW_SUPPORT_SESSION

        queued = agent_db.query(AgentCommand).filter(AgentCommand.status == "queued").all()
        queued_types = {row.command_type for row in queued}
        assert COMMAND_REVIEW_SUPPORT_SESSION in queued_types


def test_supervisor_processes_support_session_review_and_writes_review_table(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Question"),
                SupportMessage(session_pk=session.id, role="assistant", content="Answer"),
            ]
        )
        assistant_db.commit()
        assistant_db.refresh(session)
        last_id = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .first()
            .id
        )

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        run = AgentRun(
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
        agent_db.add(run)
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        context = json.loads(messages[-1]["content"])
        assert context["session"]["id"] == "s1"

        review = {
            "schema_version": 1,
            "session_id": "s1",
            "target_message_id": int(context["session"]["target_message_id"]),
            "summary": "Looks good overall.",
            "issues": [],
            "repo_search_queries": [],
            "followups": [],
        }
        return AssistantReply(content=json.dumps(review), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(
        command_type=COMMAND_REVIEW_SUPPORT_SESSION,
        payload={"session_id": "s1", "target_message_id": int(last_id)},
        priority=0,
    )
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

        step = agent_db.query(AgentStep).filter(AgentStep.kind == COMMAND_REVIEW_SUPPORT_SESSION).one()
        assert step.ok is True
        assert isinstance(step.response_json, dict)

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = assistant_db.query(SupportSession).filter(SupportSession.session_id == "s1").one()
        row = (
            assistant_db.query(SupportSessionReview)
            .filter(SupportSessionReview.session_pk == int(session.id))
            .order_by(SupportSessionReview.target_message_id.desc(), SupportSessionReview.id.desc())
            .first()
        )
        assert row is not None
        assert int(row.target_message_id) == int(last_id)
        assert isinstance(row.review_json, dict)


def test_orchestrator_tick_applies_idle_backoff(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_BASE_SECONDS", "60")

    # No sessions -> no work.
    with get_assistant_session(assistant_db_path):
        pass

    from ispec.agent.connect import get_agent_session

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

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "No work right now.",
            "next_tick_seconds": 30,
            "commands": [],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["idle_streak"] == 1
        assert orchestrator["next_tick_reason"] == "idle_backoff"
        assert orchestrator["next_tick_seconds"] == 60

        queued_tick = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
            .filter(AgentCommand.status == "queued")
            .order_by(AgentCommand.id.desc())
            .first()
        )
        assert queued_tick is not None
        queued_tick.available_at = utcnow()
        agent_db.commit()

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["idle_streak"] == 2
        assert orchestrator["next_tick_reason"] == "idle_backoff"
        assert orchestrator["next_tick_seconds"] == 120


def test_orchestrator_tick_schedules_fast_when_review_backlog(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_BASE_SECONDS", "60")

    with get_assistant_session(assistant_db_path) as assistant_db:
        for session_id in ("s1", "s2"):
            session = SupportSession(session_id=session_id, user_id=None)
            assistant_db.add(session)
            assistant_db.flush()
            assistant_db.add_all(
                [
                    SupportMessage(session_pk=session.id, role="user", content="Hi"),
                    SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
                ]
            )
        assistant_db.commit()

    from ispec.agent.connect import get_agent_session

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

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "Let the system force a review.",
            "next_tick_seconds": 600,
            "commands": [],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["idle_streak"] == 0
        assert orchestrator["next_tick_reason"] == "review_backlog"
        assert orchestrator["next_tick_seconds"] == 30


def test_orchestrator_tick_invalid_output_uses_error_backoff(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_BASE_SECONDS", "60")

    with get_assistant_session(assistant_db_path):
        pass

    from ispec.agent.connect import get_agent_session

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

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        return AssistantReply(content="not-json", provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "failed"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["error_streak"] == 1
        assert orchestrator["next_tick_reason"] == "invalid_output_backoff"
        assert orchestrator["next_tick_seconds"] == 120

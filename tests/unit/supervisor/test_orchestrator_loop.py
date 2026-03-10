from __future__ import annotations

import json
from datetime import timedelta

import pytest

from ispec.agent.commands import (
    COMMAND_BUILD_SUPPORT_DIGEST,
    COMMAND_ORCHESTRATOR_TICK,
    COMMAND_REVIEW_SUPPORT_SESSION,
)
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview
from ispec.assistant.service import AssistantReply
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import (
    _enqueue_command,
    _process_one_command,
    _recover_stale_running_commands,
    _seed_orchestrator_tick,
    utcnow,
)


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


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
        action_summary = cmd.result_json.get("action_summary")
        assert isinstance(action_summary, dict)
        totals = action_summary.get("totals")
        assert isinstance(totals, dict)
        assert int(totals.get("scheduled") or 0) >= 1

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
        assert isinstance(orchestrator.get("last_action_summary"), str)
        assert "No commands scheduled" in str(orchestrator.get("last_action_summary"))
        action_obj = orchestrator.get("last_action")
        assert isinstance(action_obj, dict)
        totals = action_obj.get("totals")
        assert isinstance(totals, dict)
        assert int(totals.get("scheduled") or 0) == 0

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
        assert cmd.status == "succeeded"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["error_streak"] == 1
        assert orchestrator["next_tick_reason"] == "invalid_output_backoff"
        assert orchestrator["next_tick_seconds"] == 120


def test_recover_stale_running_commands_requeues_orphaned_rows(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SUPERVISOR_RUNNING_STALE_SECONDS", "60")

    stale_at = utcnow() - timedelta(minutes=30)

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="old-run",
                agent_id="agent-1",
                kind="supervisor",
                status="stopped",
                created_at=stale_at,
                updated_at=stale_at,
                ended_at=stale_at,
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        cmd = AgentCommand(
            command_type=COMMAND_ORCHESTRATOR_TICK,
            status="running",
            priority=-5,
            created_at=stale_at,
            updated_at=stale_at,
            available_at=stale_at,
            claimed_at=stale_at,
            claimed_by_agent_id="agent-1",
            claimed_by_run_id="old-run",
            started_at=stale_at,
            attempts=1,
            max_attempts=3,
            payload_json={"source": "test"},
            result_json={},
        )
        agent_db.add(cmd)
        agent_db.commit()
        command_id = int(cmd.id)

    summary = _recover_stale_running_commands(agent_id="agent-1", run_id="new-run")
    assert summary["recovered"] == 1
    assert summary["running_total"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == command_id).one()
        assert cmd.status == "queued"
        assert cmd.claimed_at is None
        assert cmd.claimed_by_agent_id is None
        assert cmd.claimed_by_run_id is None
        assert cmd.started_at is None
        assert isinstance(cmd.result_json, dict)
        recovery = cmd.result_json.get("stale_recovery")
        assert isinstance(recovery, dict)
        assert recovery["stale_seconds"] == 60
        assert recovery["previous"]["claimed_by_run_id"] == "old-run"


def test_recover_stale_running_commands_skips_recent_active_run(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_SUPERVISOR_RUNNING_STALE_SECONDS", "60")

    stale_at = utcnow() - timedelta(minutes=30)

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentRun(
                run_id="active-run",
                agent_id="agent-1",
                kind="supervisor",
                status="running",
                created_at=stale_at,
                updated_at=utcnow(),
                config_json={},
                state_json={"checks": {}},
                summary_json={},
            )
        )
        cmd = AgentCommand(
            command_type=COMMAND_ORCHESTRATOR_TICK,
            status="running",
            priority=-5,
            created_at=stale_at,
            updated_at=stale_at,
            available_at=stale_at,
            claimed_at=stale_at,
            claimed_by_agent_id="agent-1",
            claimed_by_run_id="active-run",
            started_at=stale_at,
            attempts=1,
            max_attempts=3,
            payload_json={"source": "test"},
            result_json={},
        )
        agent_db.add(cmd)
        agent_db.commit()
        command_id = int(cmd.id)

    summary = _recover_stale_running_commands(agent_id="agent-1", run_id="new-run")
    assert summary["recovered"] == 0
    assert summary["skipped_recent_running_run"] == 1

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == command_id).one()
        assert cmd.status == "running"
        assert cmd.claimed_by_run_id == "active-run"


def test_seed_orchestrator_tick_recovers_stale_running_tick(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_ENABLED", "1")
    monkeypatch.setenv("ISPEC_SUPERVISOR_RUNNING_STALE_SECONDS", "60")

    stale_at = utcnow() - timedelta(minutes=30)

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        cmd = AgentCommand(
            command_type=COMMAND_ORCHESTRATOR_TICK,
            status="running",
            priority=-5,
            created_at=stale_at,
            updated_at=stale_at,
            available_at=stale_at,
            claimed_at=stale_at,
            claimed_by_agent_id="agent-1",
            claimed_by_run_id="old-run",
            started_at=stale_at,
            attempts=1,
            max_attempts=3,
            payload_json={"source": "test"},
            result_json={},
        )
        agent_db.add(cmd)
        agent_db.commit()
        command_id = int(cmd.id)

    seeded_id = _seed_orchestrator_tick(delay_seconds=10, agent_id="agent-1", run_id="new-run")
    assert seeded_id == command_id

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == command_id).one()
        assert cmd.status == "queued"
        assert cmd.claimed_by_agent_id is None
        assert cmd.claimed_by_run_id is None


def test_orchestrator_tick_skips_duplicate_inflight_review_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

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
        agent_db.add(
            AgentCommand(
                command_type=COMMAND_REVIEW_SUPPORT_SESSION,
                status="queued",
                priority=-5,
                available_at=utcnow(),
                payload_json={"session_id": "s1", "target_message_id": 101},
                result_json={},
            )
        )
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "Review session s1.",
            "next_tick_seconds": 30,
            "commands": [
                {
                    "command_type": COMMAND_REVIEW_SUPPORT_SESSION,
                    "payload": {"session_id": "s1", "target_message_id": 101},
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
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        queued_reviews = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_REVIEW_SUPPORT_SESSION)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(queued_reviews) == 1
        tick = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        skipped = tick.result_json.get("skipped")
        assert isinstance(skipped, list) and skipped
        assert skipped[0]["guard"]["reason"] == "duplicate_inflight"
        action_summary = tick.result_json.get("action_summary")
        assert isinstance(action_summary, dict)
        totals = action_summary.get("totals")
        assert isinstance(totals, dict)
        assert int(totals.get("skipped") or 0) >= 1


def test_orchestrator_tick_cools_down_after_recent_invalid_review_output(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_REVIEW_FAILURE_COOLDOWN_SECONDS", "600")

    from ispec.agent.connect import get_agent_session

    failed_at = utcnow() - timedelta(seconds=90)
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
        agent_db.add(
            AgentCommand(
                command_type=COMMAND_REVIEW_SUPPORT_SESSION,
                status="failed",
                priority=0,
                available_at=failed_at,
                created_at=failed_at,
                updated_at=failed_at,
                payload_json={"session_id": "s2", "target_message_id": 202},
                result_json={},
                error="invalid_review_output",
            )
        )
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "Retry review for s2.",
            "next_tick_seconds": 30,
            "commands": [
                {
                    "command_type": COMMAND_REVIEW_SUPPORT_SESSION,
                    "payload": {"session_id": "s2", "target_message_id": 202},
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
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        queued_reviews = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_REVIEW_SUPPORT_SESSION)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(queued_reviews) == 0
        tick = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        skipped = tick.result_json.get("skipped")
        assert isinstance(skipped, list) and skipped
        assert skipped[0]["guard"]["reason"] == "cooldown_after_invalid_review_output"
        assert int(skipped[0]["guard"]["retry_after_seconds"]) > 0


def test_orchestrator_tick_skips_duplicate_inflight_digest_command(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

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
        agent_db.add(
            AgentCommand(
                command_type=COMMAND_BUILD_SUPPORT_DIGEST,
                status="running",
                priority=-1,
                available_at=utcnow(),
                claimed_at=utcnow(),
                claimed_by_agent_id="agent-1",
                claimed_by_run_id="run-x",
                started_at=utcnow(),
                payload_json={"cursor_review_id": 30},
                result_json={},
            )
        )
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "Build digest.",
            "next_tick_seconds": 30,
            "commands": [
                {
                    "command_type": COMMAND_BUILD_SUPPORT_DIGEST,
                    "payload": {"cursor_review_id": 30},
                    "delay_seconds": 0,
                    "priority": -1,
                }
            ],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        queued_digests = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_BUILD_SUPPORT_DIGEST)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(queued_digests) == 0
        tick = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        skipped = tick.result_json.get("skipped")
        assert isinstance(skipped, list) and skipped
        assert skipped[0]["guard"]["reason"] == "duplicate_inflight"
        action_summary = tick.result_json.get("action_summary")
        assert isinstance(action_summary, dict)
        totals = action_summary.get("totals")
        assert isinstance(totals, dict)
        assert int(totals.get("scheduled") or 0) == 0
        assert int(totals.get("skipped") or 0) >= 1


def test_orchestrator_tick_cools_down_after_recent_invalid_digest_output(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_DIGEST_FAILURE_COOLDOWN_SECONDS", "600")

    from ispec.agent.connect import get_agent_session

    failed_at = utcnow() - timedelta(seconds=90)
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
        agent_db.add(
            AgentCommand(
                command_type=COMMAND_BUILD_SUPPORT_DIGEST,
                status="failed",
                priority=-1,
                available_at=failed_at,
                created_at=failed_at,
                updated_at=failed_at,
                payload_json={"cursor_review_id": 41},
                result_json={},
                error="invalid_digest_output",
            )
        )
        agent_db.commit()

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        assert tools is None
        decision = {
            "schema_version": 1,
            "thoughts": "Retry digest.",
            "next_tick_seconds": 30,
            "commands": [
                {
                    "command_type": COMMAND_BUILD_SUPPORT_DIGEST,
                    "payload": {"cursor_review_id": 41},
                    "delay_seconds": 0,
                    "priority": -1,
                }
            ],
        }
        return AssistantReply(content=json.dumps(decision), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_ORCHESTRATOR_TICK, payload={"source": "test"}, priority=0)
    assert isinstance(cmd_id, int)
    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        queued_digests = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_BUILD_SUPPORT_DIGEST)
            .filter(AgentCommand.status == "queued")
            .all()
        )
        assert len(queued_digests) == 0
        tick = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        skipped = tick.result_json.get("skipped")
        assert isinstance(skipped, list) and skipped
        assert skipped[0]["guard"]["reason"] == "cooldown_after_invalid_digest_output"
        assert int(skipped[0]["guard"]["retry_after_seconds"]) > 0

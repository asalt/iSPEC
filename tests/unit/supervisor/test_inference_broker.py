from __future__ import annotations

import time

import pytest

from ispec.agent.commands import COMMAND_RUN_TACKLE_PROMPT
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.service import AssistantReply
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.inference_broker import InferenceBroker
from ispec.supervisor.loop import _SupervisorCommandProcessor, _enqueue_command, utcnow


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def test_supervisor_inference_broker_processes_commands_while_inflight(tmp_path, monkeypatch) -> None:
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

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

    import ispec.supervisor.inference_broker as broker_mod

    def fake_generate_reply(*, messages=None, tools=None, tool_choice=None, stage=None, vllm_extra_body=None, **_) -> AssistantReply:
        # Slow enough that we can observe an in-flight inference.
        time.sleep(0.2)
        return AssistantReply(
            content="Looks good. Next steps: double-check labels and replicate counts.",
            provider="test",
            model="test-model",
            meta={"stage": stage, "has_tools": tools is not None},
            ok=True,
            error=None,
        )

    monkeypatch.setattr(broker_mod, "generate_reply", fake_generate_reply)

    broker = InferenceBroker()
    broker.start()
    processor = _SupervisorCommandProcessor(agent_id="agent-1", run_id="run-1", broker=broker)

    cmd_id = _enqueue_command(
        command_type=COMMAND_RUN_TACKLE_PROMPT,
        payload={"project_id": 123, "prompt": "Please interpret these PCA coords..."},
        priority=0,
    )
    assert isinstance(cmd_id, int)

    assert processor.tick() is True
    assert processor.inflight is not None

    # While inference is in-flight, we can still claim/finish non-LLM commands.
    other_id = _enqueue_command(command_type="noop_test_v1", payload={}, priority=10)
    assert isinstance(other_id, int)
    assert processor.tick() is True

    with get_agent_session(agent_db_path) as agent_db:
        other = agent_db.query(AgentCommand).filter(AgentCommand.id == int(other_id)).one()
        assert other.status == "failed"

    # Finish the in-flight inference.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        processor.tick()
        with get_agent_session(agent_db_path) as agent_db:
            cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(cmd_id)).one()
            if cmd.status == "succeeded":
                break
        time.sleep(0.05)

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(cmd_id)).one()
        assert cmd.status == "succeeded"
        assert isinstance(cmd.result_json, dict)
        assert cmd.result_json.get("ok") is True
        assert isinstance(cmd.result_json.get("output_text"), str)

        step = agent_db.query(AgentStep).filter(AgentStep.kind == COMMAND_RUN_TACKLE_PROMPT).one()
        assert step.ok is True

    processor.stop()


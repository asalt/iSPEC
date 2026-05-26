from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ispec.agent.commands import COMMAND_LOCAL_RELAY_REQUEST
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun
from ispec.agent.relay import EVENT_RELAY_RECEIPT
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import _enqueue_command, _process_one_command


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def test_supervisor_processes_relay_request_as_stage_only_receipt(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    root = tmp_path / "ispec-full"
    (root / "configs").mkdir(parents=True)
    (root / "Makefile").write_text("test:\n\t@true\n", encoding="utf-8")
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "user_id": "U123ALEX"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))

    import ispec.supervisor.loop as supervisor_loop

    def forbidden_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("relay stage mode must not post to Slack")

    monkeypatch.setattr(supervisor_loop.requests, "post", forbidden_post)

    fixed_now = datetime(2026, 1, 6, 9, 14, tzinfo=UTC)
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

    cmd_id = _enqueue_command(
        command_type=COMMAND_LOCAL_RELAY_REQUEST,
        payload={
            "relay_request": {
                "kind": "slack_message",
                "target": {"alias": "alex"},
                "body": "MSPC000936 LF rerun complete.",
                "metadata": {"project": "MSPC000936", "next_action": "gpGrouper comparison"},
            }
        },
    )
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"
        assert cmd.result_json["delivery_outcome"] == "staged"
        assert cmd.result_json["sent"] is False
        assert cmd.result_json["metadata"]["project"] == "MSPC000936"

        receipt = agent_db.query(AgentEvent).filter(AgentEvent.event_type == EVENT_RELAY_RECEIPT).one()
        payload = json.loads(receipt.payload_json)
        assert payload["receipt"]["delivery_outcome"] == "staged"
        assert payload["receipt"]["metadata"]["next_action"] == "gpGrouper comparison"

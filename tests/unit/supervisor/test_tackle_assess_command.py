from __future__ import annotations

import json

import pytest

from ispec.agent.commands import COMMAND_ASSESS_TACKLE_RESULTS
from ispec.agent.models import AgentCommand, AgentRun, AgentStep
from ispec.assistant.service import AssistantReply
from ispec.concurrency.thread_context import set_main_thread
from ispec.supervisor.loop import _enqueue_command, _process_one_command, utcnow


@pytest.fixture(autouse=True)
def _supervisor_main_thread() -> None:
    set_main_thread(owner="pytest")


def test_supervisor_processes_tackle_assess_command(tmp_path, monkeypatch) -> None:
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

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        assert isinstance(vllm_extra_body, dict)
        assert "guided_json" in vllm_extra_body

        return AssistantReply(
            content=json.dumps(
                {
                    "schema_version": 1,
                    "project_id": 123,
                    "summary": "PCA looks reasonable; limma has a few strong hits.",
                    "findings": [
                        {
                            "severity": "info",
                            "topic": "pca",
                            "description": "PC1/PC2 separation matches the intended groups.",
                            "evidence": None,
                        }
                    ],
                    "next_steps": ["Confirm sample labels and replicate structure."],
                    "questions": ["How many samples per condition were used?"],
                }
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(
        command_type=COMMAND_ASSESS_TACKLE_RESULTS,
        payload={
            "project_id": 123,
            "results": {
                "pca": {"explained_variance_ratio": [0.4, 0.2]},
                "limma": {"top_table_preview": [{"gene": "TP53", "logFC": 2.0, "adj.P.Val": 0.001}]},
            },
        },
        priority=0,
    )
    assert isinstance(cmd_id, int)

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == int(cmd_id)).one()
        assert cmd.status == "succeeded"
        assert isinstance(cmd.result_json, dict)
        assert cmd.result_json.get("ok") is True
        assessment = cmd.result_json.get("assessment")
        assert isinstance(assessment, dict)
        assert assessment.get("schema_version") == 1
        assert assessment.get("project_id") == 123

        step = agent_db.query(AgentStep).filter(AgentStep.kind == COMMAND_ASSESS_TACKLE_RESULTS).one()
        assert step.ok is True
        assert isinstance(step.response_json, dict)


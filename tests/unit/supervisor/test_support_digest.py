from __future__ import annotations

import json

from ispec.agent.commands import COMMAND_BUILD_SUPPORT_DIGEST
from ispec.agent.models import AgentCommand, AgentRun
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import (
    SupportMemory,
    SupportMemoryEvidence,
    SupportMessage,
    SupportSession,
    SupportSessionReview,
)
from ispec.assistant.service import AssistantReply
from ispec.supervisor.loop import _enqueue_command, _process_one_command, utcnow


def test_supervisor_processes_support_digest_and_writes_memory(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()

        message = SupportMessage(session_pk=int(session.id), role="assistant", content="Answer")
        assistant_db.add(message)
        assistant_db.flush()
        message_id = int(message.id)

        review = SupportSessionReview(
            session_pk=int(session.id),
            target_message_id=message_id,
            review_json={
                "schema_version": 1,
                "session_id": "s1",
                "target_message_id": message_id,
                "summary": "Looks good overall.",
                "issues": [],
                "repo_search_queries": [],
                "followups": [],
            },
        )
        assistant_db.add(review)
        assistant_db.commit()
        review_id = int(review.id)

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

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        assert isinstance(vllm_extra_body, dict)
        assert "guided_json" in vllm_extra_body
        digest = {
            "schema_version": 1,
            "from_review_id": 0,
            "to_review_id": 0,
            "summary": "Digest summary.",
            "highlights": ["One highlight."],
            "followups": ["One follow-up."],
            "sessions": [],
        }
        return AssistantReply(content=json.dumps(digest), provider="test", model="test-model", meta=None)

    import ispec.supervisor.loop as supervisor_loop

    monkeypatch.setattr(supervisor_loop, "generate_reply", fake_generate_reply)

    cmd_id = _enqueue_command(command_type=COMMAND_BUILD_SUPPORT_DIGEST, payload={"cursor_review_id": 0}, priority=0)
    assert isinstance(cmd_id, int)

    processed = _process_one_command(agent_id="agent-1", run_id="run-1")
    assert processed is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

        run = agent_db.query(AgentRun).filter(AgentRun.run_id == "run-1").one()
        orchestrator = run.summary_json.get("orchestrator") if isinstance(run.summary_json, dict) else None
        assert isinstance(orchestrator, dict)
        assert orchestrator["digest_last_review_id"] == review_id
        assert isinstance(orchestrator.get("digest_last_at"), str)

    with get_assistant_session(assistant_db_path) as assistant_db:
        memory = assistant_db.query(SupportMemory).filter(SupportMemory.kind == "digest").one()
        parsed = json.loads(memory.value_json)
        assert parsed["from_review_id"] == review_id
        assert parsed["to_review_id"] == review_id

        evidence_ids = [
            int(row.message_id)
            for row in assistant_db.query(SupportMemoryEvidence).filter(SupportMemoryEvidence.memory_id == int(memory.id)).all()
        ]
        assert message_id in evidence_ids


from __future__ import annotations

import json

from ispec.agent.commands import COMMAND_BUILD_SUPPORT_DIGEST, COMMAND_REVIEW_SUPPORT_SESSION
from ispec.agent.models import AgentCommand, AgentRun
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMemory, SupportMessage, SupportSession, SupportSessionReview
from ispec.assistant.service import AssistantReply
from ispec.supervisor.loop import _enqueue_command, _process_one_command, utcnow


def _seed_agent_run(*, agent_db_path):
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


def test_supervisor_repairs_session_review_json(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()

        assistant_msg = SupportMessage(session_pk=int(session.id), role="assistant", content="Answer")
        assistant_db.add(assistant_msg)
        assistant_db.commit()
        target_id = int(assistant_msg.id)

    _seed_agent_run(agent_db_path=agent_db_path)

    calls = {"n": 0}

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        assert isinstance(vllm_extra_body, dict)
        assert "guided_json" in vllm_extra_body
        calls["n"] += 1
        if calls["n"] == 1:
            # Malformed JSON that cannot be repaired by simply appending braces.
            return AssistantReply(
                content='{"schema_version": 1, "session_id": "s1", "summary": "bad}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert "repair" in str(messages[0].get("content") or "").lower()
        review = {
            "schema_version": 1,
            "session_id": "s1",
            "target_message_id": target_id,
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
        payload={"session_id": "s1", "target_message_id": target_id},
        priority=0,
    )
    assert isinstance(cmd_id, int)

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = assistant_db.query(SupportSession).filter(SupportSession.session_id == "s1").one()
        review = (
            assistant_db.query(SupportSessionReview)
            .filter(SupportSessionReview.session_pk == int(session.id))
            .filter(SupportSessionReview.target_message_id == target_id)
            .one()
        )
        assert isinstance(review.review_json, dict)

    assert calls["n"] == 2


def test_supervisor_repairs_support_digest_json(tmp_path, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")

    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()

        assistant_msg = SupportMessage(session_pk=int(session.id), role="assistant", content="Answer")
        assistant_db.add(assistant_msg)
        assistant_db.flush()
        target_id = int(assistant_msg.id)

        assistant_db.add(
            SupportSessionReview(
                session_pk=int(session.id),
                target_message_id=target_id,
                review_json={
                    "schema_version": 1,
                    "session_id": "s1",
                    "target_message_id": target_id,
                    "summary": "Looks good overall.",
                    "issues": [],
                    "repo_search_queries": [],
                    "followups": [],
                },
            )
        )
        assistant_db.commit()

    _seed_agent_run(agent_db_path=agent_db_path)

    calls = {"n": 0}

    def fake_generate_reply(*, messages=None, tools=None, vllm_extra_body=None, **_) -> AssistantReply:
        assert tools is None
        assert isinstance(messages, list)
        assert isinstance(vllm_extra_body, dict)
        assert "guided_json" in vllm_extra_body
        calls["n"] += 1
        if calls["n"] == 1:
            return AssistantReply(
                content='{"schema_version": 1, "summary": "Digest}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert "repair" in str(messages[0].get("content") or "").lower()
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

    assert _process_one_command(agent_id="agent-1", run_id="run-1") is True

    from ispec.agent.connect import get_agent_session

    with get_agent_session(agent_db_path) as agent_db:
        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == cmd_id).one()
        assert cmd.status == "succeeded"

    with get_assistant_session(assistant_db_path) as assistant_db:
        memory = assistant_db.query(SupportMemory).filter(SupportMemory.kind == "digest").one()
        assert isinstance(memory.value_json, str)

    assert calls["n"] == 2


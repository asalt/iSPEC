"""Cached conversation context is session-scoped; no live inference or jobs run."""
import json

import pytest

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession


def test_compaction_checkpoint_is_durable_and_session_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_KEEP_LAST", "6")
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_MIN_NEW_MESSAGES", "8")
    assistant_path = tmp_path / "assistant.db"
    agent_path = tmp_path / "agent.db"
    with get_assistant_session(assistant_path) as db, get_agent_session(agent_path) as queue:
        first = SupportSession(session_id="private-a", user_id=1, state_json="{}")
        other = SupportSession(session_id="channel-b", user_id=2, state_json="{}")
        db.add_all([first, other])
        db.flush()
        own_ids = []
        for index in range(14):
            message = SupportMessage(session_pk=first.id, role="user", content=f"Own message {index}")
            db.add_all([message, SupportMessage(session_pk=other.id, role="user", content="Other thread")])
            db.flush()
            own_ids.append(message.id)
        state, enqueued = support._maybe_enqueue_session_compaction(
            assistant_db=db, agent_db=queue, session=first, state={},
            triggered_by_message_id=own_ids[-1], reason="test",
        )
        assert enqueued
        first.state_json = json.dumps(state)
        queue.flush()
        job = queue.query(AgentCommand).one()
        assert job.payload_json["session_pk"] == first.id
        assert job.payload_json["target_message_id"] == own_ids[7]
        assert own_ids[8:]  # Last six messages stay outside this checkpoint.

    with get_assistant_session(assistant_path) as db, get_agent_session(agent_path) as queue:
        first = db.query(SupportSession).filter_by(session_id="private-a").one()
        state = json.loads(first.state_json)
        _, enqueued = support._maybe_enqueue_session_compaction(
            assistant_db=db, agent_db=queue, session=first, state=state,
            triggered_by_message_id=own_ids[-1], reason="retry",
        )
        assert not enqueued
        assert queue.query(AgentCommand).count() == 1


@pytest.mark.parametrize("provider,enabled", [("vllm", "0"), ("ollama", "1"), ("disabled", "1")])
def test_disabled_or_unsupported_compaction_never_touches_storage(monkeypatch, provider, enabled):
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", provider)
    monkeypatch.setenv("ISPEC_ASSISTANT_COMPACTION_ENABLED", enabled)
    state, queued = support._maybe_enqueue_session_compaction(
        assistant_db=None, agent_db=None, session=None, state={},
        triggered_by_message_id=None, reason="test",
    )
    assert not queued and state == {}


@pytest.mark.parametrize("summary_checkpoint,included", [(8, False), (9, True)])
def test_prompt_cache_prefers_covering_memory_without_internal_state(summary_checkpoint, included):
    state = support._prompt_state_from_session_state({
        "conversation_memory": {"schema_version": 1, "summary": "Prior context", "facts": [],
                                "open_tasks": [], "preferences": [], "entities": {}},
        "conversation_memory_up_to_id": 8,
        "conversation_summary": "Later context",
        "conversation_summary_up_to_id": summary_checkpoint,
        "conversation_memory_requested_at": "internal timestamp",
        "write_ticket": {"allow": True}, "controller": {"private": "state"},
    })
    assert state["conversation_memory"]["summary"] == "Prior context"
    assert ("conversation_summary" in state) == included
    assert "controller" not in state and "write_ticket" not in state
    assert "conversation_memory_requested_at" not in state

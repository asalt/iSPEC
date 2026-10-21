from fastapi import HTTPException
import pytest

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.reply_delivery import RETAINED_REPLY, failure_reply
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


def test_failed_model_acknowledges_only_committed_user_message(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOLS_ENABLED", "0")
    monkeypatch.setattr(
        support,
        "generate_reply",
        lambda **kw: AssistantReply(
            content="Assistant error: private backend detail",
            provider="test",
            model="test",
            ok=False,
            error="offline",
        ),
    )
    path = tmp_path / "assistant.db"
    with get_assistant_session(path) as db, get_agent_session(
        tmp_path / "agent.db"
    ) as agent, get_schedule_session(tmp_path / "schedule.db") as schedule:
        response = support.chat(
            support.ChatRequest(sessionId="test", message="retained request"),
            assistant_db=db,
            agent_db=agent,
            core_db=db_session,
            schedule_db=schedule,
            user=None,
        )
        assert response.responseStatus == "unavailable"
        assert response.message == RETAINED_REPLY
        assert response.retention.automaticDelivery is False
        # A separate connection must see the user row before the first context exits.
        with get_assistant_session(path) as other:
            assert (
                other.get(SupportMessage, response.retention.recordId).content
                == "retained request"
            )


def test_commit_failure_never_returns_retained_acknowledgement(
    tmp_path, db_session, monkeypatch
):
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOLS_ENABLED", "0")
    monkeypatch.setattr(
        support,
        "generate_reply",
        lambda **kw: AssistantReply(
            content="offline", provider="test", model="test", ok=False
        ),
    )
    with get_assistant_session(tmp_path / "assistant.db") as db, get_agent_session(
        tmp_path / "agent.db"
    ) as agent, get_schedule_session(tmp_path / "schedule.db") as schedule:
        original = db.commit

        def fail_commit():
            raise RuntimeError("disk failure")

        monkeypatch.setattr(db, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="disk failure"):
            support.chat(
                support.ChatRequest(sessionId="test", message="request"),
                assistant_db=db,
                agent_db=agent,
                core_db=db_session,
                schedule_db=schedule,
                user=None,
            )
        db.rollback()
        monkeypatch.setattr(db, "commit", original)


def test_queue_timeout_reports_existing_receipt_without_requeue_or_delete(
    tmp_path, monkeypatch
):
    ticks = iter([0, 2])
    monkeypatch.setattr(support.time, "monotonic", lambda: next(ticks))
    with get_agent_session(tmp_path / "agent.db") as db:
        row = AgentCommand(
            command_type="assistant_support_chat_turn_v1",
            status="queued",
            payload_json={"chat_request": {"message": "request"}},
        )
        db.add(row)
        db.commit()
        with pytest.raises(HTTPException) as caught:
            support._wait_for_queued_chat_response(
                agent_db=db, command_id=row.id, wait_seconds=1, poll_seconds=0.1
            )
        assert caught.value.status_code == 504
        assert failure_reply(caught.value.detail) == RETAINED_REPLY
        assert db.query(AgentCommand).count() == 1
        assert db.get(AgentCommand, row.id).status == "queued"

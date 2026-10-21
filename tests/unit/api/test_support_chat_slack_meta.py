from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session
from ispec.assistant.slack_outbound import ConversationScope, OutboundReceipt, store_outbound
from ispec.api.routes.agents import read_slack_outbound_context
from datetime import UTC, datetime, timedelta


def test_support_chat_persists_slack_meta(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    captured: dict[str, Any] = {}

    def fake_generate_reply(*, messages=None, tools=None, **_) -> AssistantReply:
        captured["messages"] = messages
        captured["tools"] = tools
        return AssistantReply(content="OK", provider="test", model="test-model", meta=None)

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    payload = ChatRequest.model_validate(
        {
            "sessionId": "slack-session",
            "message": "Hello",
            "history": [],
            "ui": None,
            "meta": {
                "source": "slack",
                "slack": {
                    "event": "app_mention",
                    "team_id": "T123",
                    "channel": "C123",
                    "thread_ts": "1700000000.123",
                    "message_ts": "1700000000.123",
                    "user_id": "U123",
                    "user_display_name": "Alex S",
                },
            },
        }
    )

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )

        assert response.sessionId == "slack-session"

        user_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.role == "user")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        assert user_message is not None
        assert user_message.provider == "slack"
        assert user_message.meta_json is not None
        meta = json.loads(user_message.meta_json)
        assert meta["client_meta"]["source"] == "slack"
        assert meta["client_meta"]["slack"]["user_display_name"] == "Alex S"


def test_confirmed_outbound_context_reaches_normal_reply_prompt(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "4000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    captured = []
    monkeypatch.setattr(support_routes, "generate_reply", lambda **kw: captured.append(kw) or AssistantReply(
        content="You're welcome; that report is ready for review.", provider="test", model="test", meta=None,
    ))
    now = datetime.now(UTC)
    parent = str((now - timedelta(minutes=1)).timestamp())
    with get_assistant_session(tmp_path / "assistant.db") as db:
        for channel, text in [("D1", "LeAP report delivered for review"), ("D2", "PRIVATE OTHER CONVERSATION")]:
            store_outbound(db, OutboundReceipt(
                team_id="T1", channel=channel, message_ts=parent, text=text,
                source="test", sent_at=now - timedelta(minutes=1), author_id="UBOT",
            ))
        context = read_slack_outbound_context(ConversationScope(
            team_id="T1", channel="D1", channel_type="im", thread_ts=parent, message_ts=str(now.timestamp()),
        ), db=db)
        request = ChatRequest(sessionId=f"slack:T1:D1:{parent}", message="Thank you!", history=[], meta={
            "source": "slack", "slack": {"team_id": "T1", "channel": "D1", "thread_ts": parent},
            "slack_outbound_context": context["messages"],
        })
        with get_schedule_session(tmp_path / "schedule.db") as schedule_db:
            response = chat(request, assistant_db=db, core_db=db_session, schedule_db=schedule_db, user=None)
        assert response.message
        prompt = json.dumps(captured[0]["messages"])
        assert "LeAP report delivered for review" in prompt
        assert "PRIVATE OTHER CONVERSATION" not in prompt
        assert captured[0]["tools"] is None
        user_turn = db.query(SupportMessage).filter_by(role="user").one()
        assert json.loads(user_turn.meta_json)["client_meta"]["slack_outbound_context"][0]["receipt_id"]


def test_thread_budget_preserves_recent_assistant_draft(tmp_path, db_session, monkeypatch):
    from ispec.assistant.models import SupportSession
    from ispec.assistant.slack_thread_context import SlackThreadContext, ThreadMessage

    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED", "0")
    captured = []
    monkeypatch.setattr(support_routes, "generate_reply", lambda **kw: captured.append(kw) or AssistantReply(
        content="The draft is still ready for your review.", provider="test", model="stub", meta=None,
    ))
    scope = ConversationScope(team_id="T1", channel="C1", thread_ts="1.0", message_ts="30.0")
    thread = SlackThreadContext(scope=scope, messages=tuple(
        ThreadMessage(inbox_id=i, user_id="U1", message_ts=f"{i}.0", text="background information " * 80)
        for i in range(1, 25)
    ))
    with get_assistant_session(tmp_path / "assistant.db") as db, get_schedule_session(tmp_path / "schedule.db") as schedule_db:
        session = SupportSession(session_id="thread-budget")
        db.add(session)
        db.flush()
        draft = "Draft: upload still needs to be completed. Please confirm before saving."
        db.add_all([
            SupportMessage(session_pk=session.id, role="user", content="Please draft the upload task."),
            SupportMessage(session_pk=session.id, role="assistant", content=draft),
        ])
        db.commit()
        chat(ChatRequest(sessionId=session.session_id, message="thanks", history=[], meta={
            "source": "slack", "slack": scope.model_dump(), "slack_thread_context": thread.model_dump(),
        }), assistant_db=db, core_db=db_session, schedule_db=schedule_db, user=None)
        assert {"role": "assistant", "content": draft} in captured[0]["messages"]
        result = db.query(SupportMessage).filter_by(role="assistant").order_by(SupportMessage.id.desc()).first()
        context_meta = json.loads(result.meta_json)["llm_trace"][0]["slack_thread_context"]
        assert context_meta["included"] and context_meta["truncated"]
        assert context_meta["message_count"] < 24

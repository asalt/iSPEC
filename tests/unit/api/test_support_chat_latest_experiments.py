from __future__ import annotations

import json
from typing import Any

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.assistant.service import AssistantReply
from ispec.db.models import Experiment
from ispec.schedule.connect import get_schedule_session


def test_support_chat_can_fetch_latest_experiments_via_tool(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")

    exp1 = Experiment(record_no="EXP-001", exp_Name="First")
    exp2 = Experiment(record_no="EXP-002", exp_Name="Second")
    db_session.add_all([exp1, exp2])
    db_session.commit()
    db_session.refresh(exp1)
    db_session.refresh(exp2)

    captured: list[dict[str, Any]] = []

    def fake_generate_reply(*, message: str, history=None, context=None) -> AssistantReply:
        captured.append({"message": message, "history": history, "context": context})

        if len(captured) == 1:
            return AssistantReply(
                content='TOOL_CALL {"name":"latest_experiments","arguments":{"limit":2}}',
                provider="test",
                model="test-model",
                meta=None,
            )

        assert isinstance(history, list)
        assert history[-1]["role"] == "system"
        assert history[-1]["content"].startswith("TOOL_RESULT latest_experiments")
        tool_payload = json.loads(history[-1]["content"].split("\n", 1)[1])
        assert tool_payload["ok"] is True
        ids = [row["id"] for row in tool_payload["result"]["experiments"]]
        assert ids == sorted(ids, reverse=True)

        return AssistantReply(
            content=(
                "PLAN:\n"
                "- List the most recent experiments\n"
                f"FINAL:\nLatest experiments: {ids[0]}, {ids[1]}"
            ),
            provider="test",
            model="test-model",
            meta=None,
        )

    monkeypatch.setattr(support_routes, "generate_reply", fake_generate_reply)

    db_path = tmp_path / "assistant.db"
    with get_assistant_session(db_path) as assistant_db:
        support_session = SupportSession(session_id="session-1", user_id=None)
        assistant_db.add(support_session)
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "Fetch the latest experiments",
                "history": [],
                "ui": None,
            }
        )

        schedule_path = tmp_path / "schedule.db"
        with get_schedule_session(schedule_path) as schedule_db:
            response = chat(
                payload,
                assistant_db=assistant_db,
                core_db=db_session,
                schedule_db=schedule_db,
                user=None,
            )
        assert response.sessionId == "session-1"
        assert response.message.startswith("Latest experiments:")
        assert "PLAN:" not in response.message

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.session_pk == support_session.id)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None
        meta = json.loads(assistant_row.meta_json)
        assert meta["tool_calls"][0]["name"] == "latest_experiments"
        assert meta["tool_calls"][0]["ok"] is True
        assert meta["plan"].startswith("- List")
        assert "FINAL:" in meta["raw_content"]

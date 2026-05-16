from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentEvent
from ispec.api.routes.agents import router as agents_router
from ispec.agent.connect import get_agent_session_dep
from ispec.assistant.slack_tmux_bridge import (
    BRIDGE_AGENT_ID,
    EVENT_SLACK_ARTIFACT_REPLY,
    EVENT_SLACK_ARTIFACT_SENT,
    EVENT_SLACK_TMUX_RELAY_SENT,
    build_artifact_sent_payload,
    recent_artifact_replies,
    stable_json,
)
from ispec.assistant.tools import run_tool
from ispec.db.models import AuthUser, UserRole


pytestmark = pytest.mark.testclient


def _admin_user() -> AuthUser:
    return AuthUser(
        username="admin",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.admin,
        is_active=True,
    )


def _receipt_payload() -> dict:
    return {
        "type": EVENT_SLACK_ARTIFACT_SENT,
        "artifact_id": "artifact-123",
        "file": {"path": "/tmp/report.pdf", "sha256": "abc"},
        "slack": {"channel": "D123", "thread_ts": "171.000", "file_id": "F123"},
        "origin_tmux": {
            "pane_id": "%1",
            "preferred_alias": "936-1:fish",
            "allowlist_match": "936-*",
        },
        "routing": {"submit_allowed": True},
    }


def test_artifact_reply_endpoint_records_only_known_threads(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_SLACK_ARTIFACT_REPLY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("CODEX_SLACK_DM_ALEX_USER_ID", raising=False)
    monkeypatch.delenv("SLACK_ALEX_USER_ID", raising=False)
    monkeypatch.delenv("ISPEC_SLACK_DM_ALEX_USER_ID", raising=False)

    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        agent_db.add(
            AgentEvent(
                agent_id=BRIDGE_AGENT_ID,
                event_type=EVENT_SLACK_ARTIFACT_SENT,
                ts=datetime.now(UTC),
                name="slack_artifact_sent",
                correlation_id="artifact-123",
                payload_json=stable_json(
                    {
                        "type": EVENT_SLACK_ARTIFACT_SENT,
                        "agent_id": BRIDGE_AGENT_ID,
                        "value": _receipt_payload(),
                    }
                ),
            )
        )
        agent_db.commit()

    app = FastAPI()
    app.include_router(agents_router, prefix="/api")

    def override_agent_session():
        with get_agent_session(agent_db_path) as session:
            yield session

    app.dependency_overrides[get_agent_session_dep] = override_agent_session

    with TestClient(app) as client:
        unmatched = client.post(
            "/api/agents/slack/artifact-replies",
            json={
                "channel": "D123",
                "thread_ts": "999.000",
                "message_ts": "999.001",
                "user_id": "U_ALEX",
                "text": "wrong thread",
            },
        )
        assert unmatched.status_code == 200
        assert unmatched.json()["matched"] is False

        matched = client.post(
            "/api/agents/slack/artifact-replies",
            json={
                "team_id": "T123",
                "channel": "D123",
                "channel_type": "im",
                "thread_ts": "171.000",
                "message_ts": "171.111",
                "user_id": "U_ALEX",
                "user_display_name": "Alex S",
                "text": "Looks good. Please update the caption wording.",
            },
        )
        assert matched.status_code == 200
        body = matched.json()
        assert body["matched"] is True
        assert body["artifact_id"] == "artifact-123"
        assert body["reply_event_id"] > 0

    with get_agent_session(agent_db_path) as agent_db:
        replies = recent_artifact_replies(agent_db)
        assert len(replies) == 1
        assert replies[0]["text"] == "Looks good. Please update the caption wording."
        assert replies[0]["origin_tmux"]["pane_id"] == "%1"


def test_assistant_relay_slack_reply_to_tmux_is_confirmed_and_logs_event(tmp_path, db_session, monkeypatch):
    agent_db_path = tmp_path / "agent.db"
    with get_agent_session(agent_db_path) as agent_db:
        reply = AgentEvent(
            agent_id=BRIDGE_AGENT_ID,
            event_type=EVENT_SLACK_ARTIFACT_REPLY,
            ts=datetime.now(UTC),
            name="slack_artifact_reply",
            correlation_id="artifact-123",
            payload_json=stable_json(
                {
                    "type": EVENT_SLACK_ARTIFACT_REPLY,
                    "artifact_id": "artifact-123",
                    "slack": {
                        "thread_ts": "171.000",
                        "message_ts": "171.111",
                        "user_display_name": "Alex S",
                    },
                    "text": "Please tighten the last paragraph.",
                    "origin_tmux": {"pane_id": "%1", "preferred_alias": "936-1:fish"},
                    "routing": {"submit_allowed": True},
                }
            ),
        )
        agent_db.add(reply)
        agent_db.commit()
        agent_db.refresh(reply)
        reply_id = int(reply.id)

        calls: list[dict[str, object]] = []
        monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
        monkeypatch.setattr(
            "ispec.assistant.tools._tmux_send_text",
            lambda *, target, text, press_enter: calls.append(
                {"target": target, "text": text, "press_enter": press_enter}
            )
            or {
                "target": "936-1:fish",
                "capture_target": "%1",
                "pane_id": "%1",
                "text_length": len(text),
                "press_enter": press_enter,
            },
        )

        denied = run_tool(
            name="assistant_relay_slack_reply_to_tmux",
            args={"reply_event_id": reply_id, "confirm": False},
            core_db=db_session,
            agent_db=agent_db,
            user=_admin_user(),
        )
        assert denied["ok"] is False
        assert calls == []

        payload = run_tool(
            name="assistant_relay_slack_reply_to_tmux",
            args={"reply_event_id": reply_id, "confirm": True, "press_enter": True},
            core_db=db_session,
            agent_db=agent_db,
            user=_admin_user(),
        )
        assert payload["ok"] is True
        assert calls[0]["target"] == "%1"
        assert calls[0]["press_enter"] is True
        assert "Slack review" in str(calls[0]["text"])

        relay_rows = agent_db.query(AgentEvent).filter(AgentEvent.event_type == EVENT_SLACK_TMUX_RELAY_SENT).all()
        assert len(relay_rows) == 1

        duplicate = run_tool(
            name="assistant_relay_slack_reply_to_tmux",
            args={"reply_event_id": reply_id, "confirm": True, "press_enter": True},
            core_db=db_session,
            agent_db=agent_db,
            user=_admin_user(),
        )
        assert duplicate["ok"] is False
        assert "already been relayed" in duplicate["error"]


def test_build_artifact_sent_payload_extracts_file_share_thread(tmp_path):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4\n")
    payload = build_artifact_sent_payload(
        artifact_id="artifact-abc",
        upload_result={
            "ok": True,
            "resolved": {"channel": "D123"},
            "file": {
                "path": str(report),
                "filename": "report.pdf",
                "size_bytes": report.stat().st_size,
                "mime_type": "application/pdf",
            },
            "file_id": "F123",
            "slack": {
                "files": [
                    {
                        "id": "F123",
                        "shares": {"private": {"D123": [{"ts": "171.222"}]}},
                    }
                ]
            },
        },
        origin_tmux={"pane_id": "%1"},
        submit_allowed=True,
    )

    assert payload["slack"]["thread_ts"] == "171.222"
    assert payload["file"]["sha256"]
    assert payload["origin_tmux"]["pane_id"] == "%1"

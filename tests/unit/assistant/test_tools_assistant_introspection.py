from __future__ import annotations

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview
from ispec.assistant.tools import run_tool


def test_assistant_stats_reports_review_backlog(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_stats",
            args={},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["sessions_total"] == 1
        assert result["messages_total"] == 2
        assert result["sessions_needing_review"] == 1
        assert result["sessions_reviewed"] == 0


def test_assistant_recent_sessions_lists_state_fields(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_recent_sessions",
            args={"limit": 5},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        sessions = payload["result"]["sessions"]
        assert len(sessions) == 1
        item = sessions[0]
        assert item["session_id"] == "s1"
        assert item["message_count"] == 2
        assert item["last_user_message"] == "Hi"
        assert item["reviewed_up_to_id"] == 0


def test_assistant_get_session_review_returns_review_when_present(tmp_path, db_session):
    assistant_db_path = tmp_path / "assistant.db"
    with get_assistant_session(assistant_db_path) as assistant_db:
        session = SupportSession(session_id="s1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.add(
            SupportSessionReview(
                session_pk=session.id,
                target_message_id=2,
                schema_version=1,
                review_json={
                    "schema_version": 1,
                    "session_id": "s1",
                    "target_message_id": 2,
                    "summary": "ok",
                    "issues": [],
                    "repo_search_queries": [],
                    "followups": [],
                },
            )
        )
        assistant_db.commit()

    with get_assistant_session(assistant_db_path) as assistant_db:
        payload = run_tool(
            name="assistant_get_session_review",
            args={"session_id": "s1"},
            core_db=db_session,
            assistant_db=assistant_db,
            schedule_db=None,
            omics_db=None,
            user=None,
            api_schema=None,
        )
        assert payload["ok"] is True
        result = payload["result"]
        assert result["session_id"] == "s1"
        assert result["reviewed_up_to_id"] == 2
        assert isinstance(result["review"], dict)

from __future__ import annotations

import json

from ispec.api.routes.support import FeedbackRequest, feedback
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession


def test_support_feedback_records_note_and_meta(tmp_path):
    db_path = tmp_path / "assistant.db"

    with get_assistant_session(db_path) as session:
        support_session = SupportSession(session_id="session-1", user_id=None)
        session.add(support_session)
        session.flush()

        message = SupportMessage(
            session_pk=support_session.id,
            role="assistant",
            content="Hello",
            provider="test",
            model="test-model",
        )
        session.add(message)
        session.flush()

        payload = FeedbackRequest.model_validate(
            {
                "sessionId": "session-1",
                "messageId": int(message.id),
                "rating": "down",
                "comment": "This was not helpful.",
                "ui": {
                    "name": "ProjectDetail",
                    "path": "/project/123",
                    "params": {"id": "123"},
                    "query": {},
                },
            }
        )

        result = feedback(payload, assistant_db=session, user=None)
        assert result == {"ok": True}

        session.refresh(message)
        assert message.feedback == -1
        assert message.feedback_note == "This was not helpful."
        assert message.feedback_at is not None
        assert message.feedback_meta_json is not None

        meta = json.loads(message.feedback_meta_json)
        assert meta["ui"]["name"] == "ProjectDetail"

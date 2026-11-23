from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ispec.api.routes.chat_threads import router
from ispec.api.security import get_current_user
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.models import AssistantBase, SupportMessage, SupportSession


@pytest.fixture
def browser():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    AssistantBase.metadata.create_all(engine)
    with Session(engine) as db:
        for name, owner in [
            ("older", 1),
            ("recent", 1),
            ("empty", 1),
            ("other", 2),
            ("anonymous", None),
            ("slack", 1),
        ]:
            thread = SupportSession(
                session_id=name, user_id=owner, updated_at=datetime(2025, 1, 1)
            )
            db.add(thread)
            if name == "empty":
                continue
            thread.messages.extend(
                [
                    SupportMessage(
                        role="user", content=f"{name}\n  question", provider="frontend"
                    ),
                    SupportMessage(role="system", content="Internal prompt"),
                    SupportMessage(
                        role="assistant",
                        content="First reply",
                        feedback=1,
                        meta_json="private",
                    ),
                    SupportMessage(
                        role="user",
                        content="Follow up",
                        provider="slack" if name == "slack" else "frontend",
                    ),
                ]
            )
        db.commit()

    def open_db():
        with Session(engine) as db:
            yield db

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_assistant_session_dep] = open_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)
    with TestClient(app) as client:
        yield client
    engine.dispose()


def test_personal_list_pages_use_stable_order_and_exclude_non_web_threads(browser):
    path = "/api/support/threads"
    recent = browser.get(path, params={"limit": 1})
    assert recent.status_code == 200
    assert recent.headers["Cache-Control"] == "private, no-store"
    page = recent.json()
    assert [(t["sessionId"], t["title"]) for t in page["threads"]] == [
        ("recent", "recent question")
    ]
    older = browser.get(path, params={"offset": page["nextOffset"]}).json()
    assert [t["sessionId"] for t in older["threads"]] == ["older"]
    assert older["nextOffset"] is None


def test_transcript_pages_contain_only_delivered_messages_and_feedback(browser):
    path = "/api/support/threads/recent/messages"
    page = browser.get(path, params={"limit": 2}).json()
    reply, followup = page["messages"]
    assert (reply["role"], reply["content"], reply["feedback"]) == (
        "assistant",
        "First reply",
        "up",
    )
    assert "meta_json" not in reply
    assert followup["content"] == "Follow up"
    assert page["beforeId"] == reply["id"]
    older = browser.get(path, params={"before": page["beforeId"]}).json()
    assert [m["content"] for m in older["messages"]] == ["recent\n  question"]
    assert older["beforeId"] is None
    assert browser.get(path, params={"limit": 2}).json() == page


@pytest.mark.parametrize(
    "session_id", ["empty", "other", "anonymous", "slack", "missing"]
)
def test_unavailable_threads_are_indistinguishable(browser, session_id):
    assert browser.get(f"/api/support/threads/{session_id}/messages").status_code == 404


@pytest.mark.parametrize("user, status", [(None, 401), (SimpleNamespace(id=0), 403)])
def test_personal_history_requires_browser_account(browser, user, status):
    browser.app.dependency_overrides[get_current_user] = lambda: user
    for path in ["/api/support/threads", "/api/support/threads/recent/messages"]:
        assert browser.get(path).status_code == status

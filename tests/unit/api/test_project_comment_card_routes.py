"""HTTP adapter tests with cookie identities and trusted receiver stubs."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ispec.api import security
from ispec.api.routes.comment_cards import router
from ispec.assistant.project_comment_cards import CardOrigin
from ispec.assistant.connect import get_assistant_session_dep
from ispec.db.connect import get_session_dep
from tests.unit.assistant.test_project_comment_cards import conversation  # noqa: F401


pytestmark = pytest.mark.testclient


@pytest.fixture
def client(conversation, monkeypatch):
    monkeypatch.setenv("ISPEC_REQUIRE_LOGIN", "1")
    monkeypatch.setenv("ISPEC_API_KEY", "synthetic-key")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_session_dep] = lambda: conversation.core
    app.dependency_overrides[get_assistant_session_dep] = lambda: conversation.assistant
    # Preserve actual origin checks and the distinction between cookies and keys.
    app.dependency_overrides[security.get_current_user] = lambda: conversation.user
    monkeypatch.setattr(
        security, "get_current_user", lambda request, db: conversation.user
    )
    with TestClient(app) as client:
        yield client


def payload_for(conversation, *, draft_id):
    from ispec.db.models import ProjectCommentDraft

    conversation.core.expire_all()
    return {
        "action": "save",
        "draft_id": draft_id,
        "content_hash": conversation.core.get(
            ProjectCommentDraft, draft_id
        ).content_hash,
    }


def test_browser_click_and_restored_card(client, conversation):
    card_id, draft_id = conversation.prepare()
    response = client.post(
        f"/api/support/project-comment-cards/{card_id}/actions",
        json=payload_for(conversation, draft_id=draft_id),
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200 and response.json()["card"]["status"] == "saved"
    cards = client.get(
        "/api/support/project-comment-cards", params={"session_id": "card-test"}
    )
    assert cards.status_code == 200
    assert cards.json()[0]["comment_id"] == response.json()["card"]["comment_id"]
    assert cards.json()[0]["can_undo"]
    undone = client.post(
        f"/api/support/project-comment-cards/{card_id}/actions",
        json={**payload_for(conversation, draft_id=draft_id), "action": "undo"},
        headers={"Origin": "http://testserver"},
    )
    assert undone.json()["ok"] and undone.json()["card"]["status"] == "undone"
    assert conversation.comments() == []


def test_cross_origin_click_is_rejected(client, conversation):
    card_id, draft_id = conversation.prepare()
    response = client.post(
        f"/api/support/project-comment-cards/{card_id}/actions",
        json=payload_for(conversation, draft_id=draft_id),
        headers={"Origin": "https://elsewhere.invalid"},
    )
    assert response.status_code == 403
    assert conversation.comments() == []


def test_browser_cannot_assert_slack_identity(client, conversation):
    card_id, draft_id = conversation.prepare()
    response = client.post(
        f"/api/support/project-comment-cards/{card_id}/actions",
        json={
            **payload_for(conversation, draft_id=draft_id),
            "slack": {
                "user_id": "U",
                "team_id": "T",
                "channel_id": "C",
                "message_ts": "2.0",
            },
        },
    )
    assert response.status_code == 401
    assert conversation.comments() == []


def test_delivery_scope_and_real_http_slack_action(client, conversation):
    conversation.user.id = 0
    conversation.user.username = "api_key"
    conversation.origin = CardOrigin(
        surface="slack", requester_id="U", team_id="T", channel_id="C", thread_ts="1.0"
    )
    card_id, draft_id = conversation.prepare()
    path = f"/api/support/project-comment-cards/{card_id}"
    delivery = {
        "team_id": "T",
        "channel_id": "C",
        "thread_ts": "1.0",
        "message_ts": "2.0",
    }
    headers = {"X-API-Key": "synthetic-key"}
    assert client.post(path + "/delivery", json=delivery).status_code == 401
    assert (
        client.post(
            path + "/delivery",
            json={**delivery, "channel_id": "OTHER"},
            headers=headers,
        ).status_code
        == 409
    )
    assert (
        client.post(path + "/delivery", json=delivery, headers=headers).status_code
        == 200
    )
    click = {
        **payload_for(conversation, draft_id=draft_id),
        "slack": {
            "user_id": "U",
            "team_id": "T",
            "channel_id": "C",
            "message_ts": "2.0",
        },
    }
    response = client.post(path + "/actions", json=click, headers=headers)
    assert response.status_code == 200 and response.json()["ok"]
    assert len(conversation.comments()) == 1


def test_unknown_button_fields_rejected(client, conversation):
    card_id, draft_id = conversation.prepare()
    response = client.post(
        f"/api/support/project-comment-cards/{card_id}/actions",
        json={
            **payload_for(conversation, draft_id=draft_id),
            "person_id": 99,
            "confirm": True,
        },
    )
    assert response.status_code == 422 and conversation.comments() == []

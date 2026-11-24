"""HTTP adapter tests with cookie identities and trusted receiver stubs."""

# ruff: noqa: F811 -- pytest injects the shared conversation fixture by name.

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


def test_review_delivery_requires_receiver_key_even_with_cookie(client):
    path = "/api/support/project-comment-cards/staff-review/claim"
    assert client.post(path, json={"team_id": "TSTAFF"}).status_code == 401
    response = client.post(
        path, headers={"X-API-Key": "synthetic-key"}, json={"team_id": "TSTAFF"}
    )
    assert response.status_code == 200 and response.json() == {"delivery": None}


def test_share_existing_card_and_list_visible_pending_only(
    client, conversation, monkeypatch
):
    from ispec.api.routes.comment_cards import CardDelivery, card_delivery
    from ispec.assistant.project_comment_cards import claim_staff_review
    from ispec.assistant.project_comment_workflow import comment_workflow_tools
    from ispec.db.models import AuthUser, UserRole

    monkeypatch.setenv("ISPEC_SLACK_AGENT_REVIEW_ENABLED", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL", "CREVIEW")
    monkeypatch.setenv("ISPEC_SLACK_ALLOWED_USER_IDS", "USTAFF")
    staff = conversation.user
    dot = AuthUser(
        id=19,
        username="dot",
        role=UserRole.viewer,
        is_comment_agent=True,
        is_active=True,
    )
    conversation.user = dot
    _, old_id = conversation.prepare(text="Old wording")
    card_id, draft_id = conversation.prepare(text="Exact pending draft")
    share = "request_staff_review"
    catalog = comment_workflow_tools(tools=[], prepare_only=True)
    assert share in {spec["function"]["name"] for spec in catalog}
    turn = conversation.turn(text="Send that draft to staff Slack.")

    def request_review(id=draft_id, **extra):
        return turn.run(name=share, args={"draft_id": id, **extra})

    def outstanding(user, **params):
        conversation.user = user
        response = client.get(
            "/api/support/project-comment-cards/outstanding", params=params
        )
        assert response.status_code == 200
        return [(r["comment"], r["delivery"]) for r in response.json()["drafts"]]

    assert outstanding(dot) == [("Exact pending draft", "private")]
    assert outstanding(staff) == []
    assert not request_review(old_id)["ok"]
    assert not request_review(channel="COTHER")["ok"]
    other = AuthUser(
        id=20,
        username="other-agent",
        role=UserRole.viewer,
        is_comment_agent=True,
        is_active=True,
    )
    conversation.user = other
    assert not conversation.turn().run(name=share, args={"draft_id": draft_id})["ok"]
    assert request_review()["result"]["delivery"] == "queued"
    assert request_review()["result"]["delivery"] == "queued"
    assert outstanding(staff, project_id=57) == [("Exact pending draft", "queued")]
    assert outstanding(staff, project_id=58) == []
    assert outstanding(other) == []

    def claim():
        return claim_staff_review(
            core_db=conversation.core,
            assistant_db=conversation.assistant,
            team_id="TSTAFF",
        )

    assert claim()["card"].card_id == card_id
    assert request_review()["result"]["delivery"] == "unconfirmed"
    assert claim() is None
    card_delivery(
        card_id,
        CardDelivery(team_id="TSTAFF", channel_id="CREVIEW", message_ts="1.0"),
        conversation.core,
    )
    assert request_review()["result"]["delivery"] == "posted"
    assert outstanding(staff) == [("Exact pending draft", "posted")]
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

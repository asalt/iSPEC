import json
from types import SimpleNamespace

import pytest

from ispec.assistant.project_comment_cards import ConfirmationCard
from ispec.cli.slack_comment_cards import (
    CANCEL_ACTION,
    SAVE_ACTION,
    UNDO_ACTION,
    SlackCommentCards,
    card_message,
)


@pytest.fixture
def card():
    return ConfirmationCard(
        card_id="a" * 32,
        draft_id="b" * 32,
        content_hash="c" * 64,
        project_id=57,
        session_id="slack:T:C:1.0",
        comment="Eight samples <@UOTHER> & exact\ntext.",
        status="pending",
        expires_at="2030-01-01T00:00:00Z",
        comment_id=None,
        message_ts=None,
    )


def test_card_exact_text_and_button_binding(card):
    message = card_message(card=card)
    assert message["blocks"][1]["text"] == {
        "type": "plain_text",
        "text": card.comment,
        "emoji": False,
    }
    buttons = message["blocks"][-1]["elements"]
    assert [button["action_id"] for button in buttons] == [SAVE_ACTION, CANCEL_ACTION]
    assert json.loads(buttons[0]["value"]) == {
        "card_id": card.card_id,
        "draft_id": card.draft_id,
        "content_hash": card.content_hash,
    }


def test_long_card_text_is_complete(card):
    long = card.model_copy(update={"comment": "x" * 20000})
    blocks = card_message(card=long)["blocks"]
    assert (
        "".join(block["text"]["text"] for block in blocks if block["type"] == "section")
        == long.comment
    )


def test_publishes_claimed_review_as_thread_root_and_binds_message(card, monkeypatch):
    calls = []
    posted = []

    def api(self, *, path, payload):
        calls.append((path, payload))
        if path == "staff-review/claim":
            return {"delivery": {"channel": "CREVIEW", "card": card.model_dump()}}
        return {"ok": True}

    monkeypatch.setattr(SlackCommentCards, "_post", api)
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: posted.append(kwargs) or {"ts": "2.0"}
    )
    adapter = SlackCommentCards(server="http://stub.invalid", api_key="synthetic")
    adapter.deliver_staff_review(client=client, team_id="T")
    assert posted == [{"channel": "CREVIEW", **card_message(card=card)}]
    assert calls[-1] == (
        f"{card.card_id}/delivery",
        {"channel_id": "CREVIEW", "team_id": "T", "thread_ts": "", "message_ts": "2.0"},
    )


def test_delivery_receipt_then_in_place_updates(card, monkeypatch):
    posts, updates, api_calls = [], [], []
    client = SimpleNamespace(
        chat_postMessage=lambda **kwargs: posts.append(kwargs) or {"ts": "2.0"},
        chat_update=lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        SlackCommentCards,
        "_post",
        lambda self, **kwargs: api_calls.append(kwargs) or {"ok": True},
    )
    adapter = SlackCommentCards(server="http://stub.invalid", api_key="synthetic")
    adapter.publish(
        payload={"commentCards": [card.model_dump()]},
        client=client,
        team_id="T",
        channel_id="C",
        thread_ts="1.0",
    )
    assert api_calls[0]["payload"]["message_ts"] == "2.0"
    assert posts[0]["thread_ts"] == "1.0"

    saved = card.model_copy(
        update={
            "message_ts": "2.0",
            "status": "saved",
            "comment_id": 123,
            "can_undo": True,
        }
    )
    adapter.publish(
        payload={"commentCards": [saved.model_dump()]},
        client=client,
        team_id="T",
        channel_id="C",
        thread_ts="1.0",
    )
    assert len(posts) == 1 and updates[0]["ts"] == "2.0"
    assert "123" in updates[0]["text"]
    assert updates[0]["blocks"][-1]["elements"][0]["action_id"] == UNDO_ACTION
    assert "Saved as comment #123" in updates[0]["text"]


@pytest.mark.parametrize("action_id", [SAVE_ACTION, UNDO_ACTION])
def test_click_ack_precedes_backend_and_updates_exact_message(
    card, monkeypatch, action_id
):
    order, updates, requests = [], [], []
    saved = card.model_copy(
        update={"status": "saved", "comment_id": 123, "message_ts": "2.0"}
    )

    def backend(self, **kwargs):
        order.append("backend")
        requests.append(kwargs)
        return {"ok": True, "card": saved.model_dump()}

    monkeypatch.setattr(SlackCommentCards, "_post", backend)
    client = SimpleNamespace(chat_update=lambda **kwargs: updates.append(kwargs))
    body = {
        "type": "block_actions",
        "team": {"id": "T"},
        "user": {"id": "UCLICKER"},
        "channel": {"id": "C"},
        "container": {"message_ts": "2.0"},
        "actions": [
            {
                "action_id": action_id,
                "value": card_message(card=card)["blocks"][-1]["elements"][0]["value"],
            }
        ],
    }
    SlackCommentCards(server="http://stub.invalid", api_key="synthetic").handle_action(
        ack=lambda: order.append("ack"), body=body, client=client
    )
    assert order == ["ack", "backend"]
    assert requests[0]["payload"]["slack"]["user_id"] == "UCLICKER"
    assert requests[0]["payload"]["action"] == (
        "save" if action_id == SAVE_ACTION else "undo"
    )
    assert updates[0]["ts"] == "2.0"


@pytest.mark.parametrize("save_confirmed", [True, False])
def test_save_click_reports_only_confirmed_receipt_when_request_fails(
    card, monkeypatch, save_confirmed
):
    private = []
    saved = card.model_copy(
        update={"status": "saved", "comment_id": 123, "message_ts": "2.0"}
    )

    def backend(self, **kwargs):
        if not save_confirmed:
            raise TimeoutError
        return {"ok": True, "card": saved.model_dump()}

    def refresh(**kwargs):
        assert save_confirmed
        raise RuntimeError("Slack update failed")

    monkeypatch.setattr(SlackCommentCards, "_post", backend)
    client = SimpleNamespace(
        chat_update=refresh, chat_postEphemeral=lambda **kwargs: private.append(kwargs)
    )
    body = {
        "type": "block_actions",
        "team": {"id": "T"},
        "user": {"id": "UCLICKER"},
        "channel": {"id": "C"},
        "container": {"message_ts": "2.0"},
        "actions": [
            {
                "action_id": SAVE_ACTION,
                "value": card_message(card=card)["blocks"][-1]["elements"][0]["value"],
            }
        ],
    }
    SlackCommentCards(server="http://stub.invalid", api_key="synthetic").handle_action(
        ack=lambda: None, body=body, client=client
    )
    expected = (
        "Saved as comment #123. Slack couldn't refresh the card. No need to click Save again."
        if save_confirmed
        else "Could not confirm the card action. Its saved state will be checked if you click again."
    )
    assert private == [{"channel": "C", "user": "UCLICKER", "text": expected}]


def test_denied_click_is_ephemeral_and_does_not_change_card(card, monkeypatch):
    private = []
    monkeypatch.setattr(
        SlackCommentCards,
        "_post",
        lambda self, **kwargs: {
            "ok": False,
            "error": "Not available to you",
            "card": None,
        },
    )
    client = SimpleNamespace(
        chat_update=lambda **kwargs: pytest.fail("Unauthorized mutation"),
        chat_postEphemeral=lambda **kwargs: private.append(kwargs),
    )
    body = {
        "type": "block_actions",
        "team": {"id": "T"},
        "user": {"id": "UBAD"},
        "channel": {"id": "C"},
        "container": {"message_ts": "2.0"},
        "actions": [
            {
                "action_id": SAVE_ACTION,
                "value": card_message(card=card)["blocks"][-1]["elements"][0]["value"],
            }
        ],
    }
    SlackCommentCards(server="http://stub.invalid", api_key="synthetic").handle_action(
        ack=lambda: None, body=body, client=client
    )
    assert private == [{"channel": "C", "user": "UBAD", "text": "Not available to you"}]


def test_bolt_action_registration():
    handlers = {}
    app = SimpleNamespace(action=lambda name: lambda fn: handlers.setdefault(name, fn))
    SlackCommentCards(server="http://stub.invalid", api_key="synthetic").register(
        app=app
    )
    assert set(handlers) == {SAVE_ACTION, CANCEL_ACTION, UNDO_ACTION}


def test_bolt_dispatches_save_with_injected_arguments(card, monkeypatch):
    """Exercise Bolt's argument injection, which a direct callback call skips."""
    bolt = pytest.importorskip("slack_bolt")
    from slack_bolt.request import BoltRequest
    from slack_bolt.response import BoltResponse
    from unittest.mock import Mock

    app = bolt.App(token="synthetic", token_verification_enabled=False)
    adapter = SlackCommentCards(server="http://stub.invalid", api_key="synthetic")
    adapter.register(app=app)
    body = {
        "type": "block_actions",
        "team": {"id": "T"},
        "user": {"id": "UCLICKER"},
        "channel": {"id": "C"},
        "container": {"message_ts": "2.0"},
        "actions": [card_message(card=card)["blocks"][-1]["elements"][0]],
    }
    request = BoltRequest(mode="socket_mode", body=body)
    client = Mock()
    request.context["client"] = client
    saved = card.model_copy(
        update={"status": "saved", "comment_id": 123, "can_undo": True}
    )

    def save(self, *, path, payload):
        assert request.context.ack.response.status == 200
        assert path == f"{card.card_id}/actions" and payload["action"] == "save"
        return {"ok": True, "card": saved.model_dump()}

    monkeypatch.setattr(SlackCommentCards, "_post", save)
    response = BoltResponse(status=200)
    listeners = [
        listener
        for listener in app._listeners
        if listener.matches(req=request, resp=response)
    ]
    assert len(listeners) == 1
    listeners[0].run_ack_function(request=request, response=response)
    client.chat_update.assert_called_once_with(
        channel="C", ts="2.0", **card_message(card=saved)
    )
    client.chat_postEphemeral.assert_not_called()

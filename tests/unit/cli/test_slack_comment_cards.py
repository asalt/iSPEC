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

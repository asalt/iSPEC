"""Use installed Bolt dispatch with fake authorization, never a Slack connection."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ispec.assistant.slack_inbox import SlackInbox
from ispec.assistant.reply_delivery import SlackEventClaims


def test_bolt_ack_is_after_commit_and_failure_does_not_ack(tmp_path, monkeypatch):
    bolt = pytest.importorskip("slack_bolt")
    from slack_bolt.authorization import AuthorizeResult
    from slack_bolt.request import BoltRequest
    from slack_sdk.web import WebClient

    monkeypatch.setattr(WebClient, "api_call", lambda *a, **kw: pytest.fail("No Slack API calls"))
    app = bolt.App(
        client=WebClient(token="xoxb-test"), token_verification_enabled=False,
        request_verification_enabled=False,
        authorize=lambda **kw: AuthorizeResult(
            enterprise_id=None, team_id="T1", bot_token="xoxb-test", bot_user_id="UBOT", bot_id="B1"),
    )
    inbox = SlackInbox(file=tmp_path / "assistant.db", channels=frozenset({("T1", "C1")}))
    app.middleware(inbox.middleware)
    handled = Event()
    seen = []

    @app.event("message")
    @inbox.handler
    def handler(context, event):
        assert inbox.claim(context, fallback=SlackEventClaims(), team_id="T1", event=event, route="reply")
        seen.extend(inbox.records())
        handled.set()

    body = {"type": "event_callback", "team_id": "T1", "event_id": "Ev1",
            "event": {"type": "message", "user": "U1", "channel": "C1",
                      "channel_type": "channel", "ts": "123.456", "text": "hello"}}
    response = app.dispatch(BoltRequest(body=body, mode="socket_mode"))
    assert response.status == 200
    assert handled.wait(3)
    assert seen[0]["text"] == "hello"
    handled.clear()

    def fail(*args):
        raise OSError("test failure")

    monkeypatch.setattr(inbox, "receive", fail)
    response = app.dispatch(BoltRequest(body=body, mode="socket_mode"))
    assert response.status == 503
    assert not handled.is_set()

    # Exercise the installed Socket Mode adapter's actual ACK decision too.
    from slack_bolt.adapter.socket_mode.internals import send_response
    client = SimpleNamespace(logger=Mock(level=20), send_socket_mode_response=Mock())
    request = SimpleNamespace(envelope_id="fixture-envelope")
    send_response(client, request, response, start_time=0)
    client.send_socket_mode_response.assert_not_called()

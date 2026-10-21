"""Received, handled, and reviewed never imply approval or automatic delivery."""

import pytest

from ispec.assistant.reply_delivery import RETAINED_REPLY, SlackEventClaims
from ispec.assistant.slack_inbox import SlackInbox

pytestmark = pytest.mark.behavioral


def test_receive_offline_review_restart_then_normal_question(tmp_path):
    path = tmp_path / "assistant.db"
    inbox = SlackInbox(file=path, channels=frozenset({("T1", "C1")}))
    calls = []

    def dispatch(store, *, stamp, channel="C1", content="Project 42: 12 IP samples received", offline=False):
        body = {"team_id": "T1", "event": {
            "type": "message", "channel": channel, "channel_type": "im" if channel == "D1" else "channel",
            "ts": stamp, "user": "U1", "text": content}}
        context = {}

        @store.handler
        def handle(*, context):
            if not store.claim(context, fallback=SlackEventClaims(), team_id="T1", event=body["event"], route="reply"):
                return
            if channel != "D1":
                return  # Passive channel capture does not invoke the model or writer.
            calls.append(content)
            if offline:
                return store.failed_reply(context, ConnectionError())
            return "Project 42: synthetic read-only response"

        return store.middleware(body, context, lambda: handle(context=context))

    assert dispatch(inbox, stamp="1.0") is None
    channel_row = inbox.records()[0]
    assert channel_row["handled_at"] is not None
    assert channel_row["reviewed_at"] is None
    assert calls == []
    assert dispatch(inbox, stamp="2.0", channel="D1", content="Tell me about project 42", offline=True) == RETAINED_REPLY
    failed = inbox.records()[0]
    assert failed["error_type"] == "ConnectionError"

    restarted = SlackInbox(file=path, channels=inbox.channels)
    assert dispatch(restarted, stamp="2.0", channel="D1", content="Tell me about project 42") is None
    assert len(calls) == 1  # Ambiguous prior attempts are not replayed after restart.
    restarted.review(channel_row["id"], actor="staff-reviewer")
    assert len(calls) == 1  # Review is not authorization or a command.
    assert restarted.records(row_id=channel_row["id"], unreviewed=False)[0]["reviewed_by"] == "staff-reviewer"
    assert dispatch(restarted, stamp="3.0", channel="D1", content="Tell me about project 42") == "Project 42: synthetic read-only response"
    assert len(calls) == 2
    assert len(restarted.records(unreviewed=False)) == 3

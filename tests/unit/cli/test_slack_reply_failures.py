from concurrent.futures import ThreadPoolExecutor
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import requests

from ispec.assistant.reply_delivery import (
    RETAINED_REPLY,
    UNRETAINED_REPLY,
    SlackEventClaims,
    chat_reply_text,
    failed_request_reply,
    failure_reply,
)
from ispec.cli import slack


def test_failure_words_require_structured_retention():
    assert (
        failed_request_reply(requests.ConnectionError("private backend address"))
        == UNRETAINED_REPLY
    )
    assert failure_reply({"message": "received", "retained": True}) == UNRETAINED_REPLY
    assert (
        failure_reply({"responseStatus": "unavailable", "retention": {"recordId": 5}})
        == UNRETAINED_REPLY
    )
    receipt = {
        "responseStatus": "unavailable",
        "retention": {"storage": "support_message", "recordId": 5},
    }
    assert failure_reply(receipt) == RETAINED_REPLY
    assert (
        chat_reply_text({**receipt, "message": "SECRET internal error"})
        == RETAINED_REPLY
    )
    response = requests.Response()
    response.status_code = 504
    response._content = json.dumps(
        {
            "detail": {
                **receipt,
                "retention": {"storage": "agent_command", "recordId": 3},
            }
        }
    ).encode()
    assert failed_request_reply(requests.HTTPError(response=response)) == RETAINED_REPLY


def test_claims_are_atomic_bounded_and_not_durable(monkeypatch):
    claims = SlackEventClaims(ttl_seconds=10, limit=2)
    event = {"channel": "DTEST", "ts": "1.1"}
    clock = [100.0]
    monkeypatch.setattr(
        "ispec.assistant.reply_delivery.time.monotonic", lambda: clock[0]
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda _: claims.claim(team_id="T", event=event, route="message"),
                range(10),
            )
        )
    assert sum(outcomes) == 1
    clock[0] += 11
    assert claims.claim(team_id="T", event=event, route="message")
    assert not claims.claim(team_id="T", event={}, route="message")
    for number in range(3):
        claims.claim(team_id="T", event={**event, "ts": str(number)}, route="message")
    assert len(claims._seen) == 2


@pytest.mark.parametrize("route", ["message", "app_mention"])
def test_receiver_allowlist_blocks_unapproved_senders_without_replies(receiver, monkeypatch, route):
    monkeypatch.setenv("ISPEC_SLACK_ALLOWED_USER_IDS", "UALLOWED")
    monkeypatch.setattr(slack, "_post_ispec_chat", lambda **kw: pytest.fail("Unapproved API call"))
    receiver.dispatch(name=route)
    assert not receiver.posts and not receiver.updates and not receiver.says


def test_receiver_allowlist_is_explicit_and_exact(monkeypatch):
    monkeypatch.delenv("ISPEC_SLACK_ALLOWED_USER_IDS", raising=False)
    assert slack._reply_user_allowed({"user": "UTEST"})
    monkeypatch.setenv("ISPEC_SLACK_ALLOWED_USER_IDS", "UALLOWED, USECOND")
    assert slack._reply_user_allowed({"user": "UALLOWED"})
    assert slack._reply_user_allowed({"user": "USECOND"})
    assert not slack._reply_user_allowed({"user": "UALLOWED-extra"})
    assert not slack._reply_user_allowed({})


@pytest.fixture
def receiver(monkeypatch, tmp_path):
    # Keep the real response type importable when replacing the Bolt app below.
    pytest.importorskip("slack_bolt.response")
    handlers = {}
    middleware = []
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(tmp_path / "assistant.db"))
    config = tmp_path / "inbox.json"
    config.write_text(json.dumps({"channels": [{"team_id": "T", "channel_id": "CTEST"}]}))
    monkeypatch.setenv("ISPEC_SLACK_INBOX_CONFIG", str(config))
    posts, updates, says = [], [], []
    client = SimpleNamespace(
        auth_test=lambda: {"team": "test", "team_id": "T", "user_id": "UBOT"},
        users_info=lambda **kw: {"user": {"name": "test"}},
        chat_postMessage=lambda **kw: posts.append(kw) or {"ok": True, "ts": "99.1"},
        chat_update=lambda **kw: updates.append(kw) or {"ok": True},
    )

    class FakeApp:
        def __init__(self, **kw):
            self.client = client

        def event(self, name):
            def decorate(fn):
                handlers[name] = fn
                return fn

            return decorate

        def error(self, fn):
            return fn

        def middleware(self, fn):
            middleware.append(fn)

    module = ModuleType("slack_bolt")
    module.App = FakeApp
    adapter = ModuleType("slack_bolt.adapter.socket_mode")
    adapter.SocketModeHandler = lambda *a: SimpleNamespace(start=lambda: None)
    monkeypatch.setitem(sys.modules, "slack_bolt", module)
    monkeypatch.setitem(sys.modules, "slack_bolt.adapter.socket_mode", adapter)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-bot")
    monkeypatch.setenv("SLACK_APP_TOKEN", "test-app")
    slack._run_socket_mode(
        SimpleNamespace(
            ispec_server="http://backend.test", api_key="test-key", timeout_seconds=1
        )
    )

    def dispatch(name="message", **changes):
        event = {
            "type": name,
            "channel": "DTEST",
            "channel_type": "im",
            "user": "UTEST",
            "ts": "2.1",
            "text": "hello",
            **changes,
        }
        context = {"team_id": "T", "bot_user_id": "UBOT"}
        return middleware[0](
            body={"team_id": "T", "event": event}, context=context,
            next=lambda: handlers[name](
                event=event, say=lambda **kw: says.append(kw), client=client,
                context=context, ack=lambda: None,
            ),
        )

    return SimpleNamespace(
        dispatch=dispatch, handlers=handlers, posts=posts, updates=updates, says=says,
        inbox=slack.SlackInbox(file=tmp_path / "assistant.db"),
    )


def test_backend_offline_then_recovery_no_duplicate_replies(receiver, monkeypatch):
    calls = []

    def post(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.ConnectionError("SECRET backend failure")
        return {"message": "Recovered answer", "responseStatus": "ok"}

    monkeypatch.setattr(slack, "_post_ispec_chat", post)
    receiver.dispatch()
    receiver.dispatch()  # Slack redelivery, not a new user request
    assert len(calls) == 1
    assert len(receiver.posts) == 1
    assert receiver.updates[-1]["text"] == RETAINED_REPLY
    record = receiver.inbox.records()[0]
    assert record["text"] == "hello" and record["error_type"] == "ConnectionError"
    assert record["handled_at"] and record["reviewed_at"] is None
    assert "SECRET" not in json.dumps(receiver.updates)
    receiver.dispatch(ts="3.1")
    assert len(calls) == 2
    assert receiver.updates[-1]["text"] == "Recovered answer"
    assert receiver.says == []


def test_ordinary_dm_thread_routes_to_chat_and_preserves_thread(receiver, monkeypatch):
    routes, chats = [], []
    monkeypatch.setattr(
        slack,
        "_post_ispec_slack_artifact_reply",
        lambda **kw: routes.append(kw) or {"matched": False},
    )
    monkeypatch.setattr(
        slack, "_post_ispec_chat", lambda **kw: chats.append(kw) or {"message": "reply"}
    )
    receiver.dispatch(thread_ts="1.1", text="Please clarify the report")
    assert len(routes) == len(chats) == 1
    assert chats[0]["meta"]["slack"]["thread_ts"] == "1.1"
    assert chats[0]["session_id"] == "slack:T:DTEST:1.1"
    assert receiver.posts[0]["thread_ts"] == "1.1"
    assert receiver.updates[-1]["text"] == "reply"


def test_dm_sessions_separate_threads_and_mentions_share_same_scope(receiver, monkeypatch):
    calls = []
    monkeypatch.setattr(slack, "_post_ispec_slack_artifact_reply", lambda **kw: {"matched": False})
    monkeypatch.setattr(slack, "_post_ispec_chat", lambda **kw: calls.append(kw) or {"message": "OK"})
    receiver.dispatch(thread_ts="1.1")
    receiver.dispatch(thread_ts="1.2", ts="2.2")
    receiver.dispatch(ts="2.3")
    receiver.dispatch(name="app_mention", ts="2.4", text="<@UBOT> thanks")
    assert calls[0]["session_id"] != calls[1]["session_id"]
    assert calls[2]["session_id"] == calls[3]["session_id"]
    assert calls[2]["session_id"] not in {calls[0]["session_id"], calls[1]["session_id"]}


def test_chat_transport_loads_context_then_posts_without_external_send(monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        result = {"messages": [{"text": "prior report", "receipt_id": 7}]} if url.endswith("outbound-context") else {"message": "OK"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)
    monkeypatch.setattr(slack.requests, "post", post)
    meta = {"source": "slack", "slack": {"team_id": "T1", "channel": "D1", "thread_ts": "1.1", "message_ts": "1.2"}}
    slack._post_ispec_chat(server="http://backend.test", api_key="test", session_id="test", message="thanks", meta=meta, timeout_seconds=60)
    assert len(calls) == 2 and all(url.startswith("http://backend.test/") for url, _ in calls)
    assert calls[0][1]["json"] == meta["slack"]
    assert calls[1][1]["json"]["meta"]["slack_outbound_context"][0]["receipt_id"] == 7
    assert "slack_outbound_context" not in meta


@pytest.mark.parametrize("channel_type,channel", [("channel", "CTEST"), ("im", "DTEST")])
def test_reply_receives_prior_thread_text_without_replacing_latest_request(receiver, monkeypatch, channel_type, channel):
    calls = []
    monkeypatch.setattr(slack, "_post_ispec_slack_artifact_reply", lambda **kw: {"matched": False})
    monkeypatch.setattr(slack, "_post_ispec_chat", lambda **kw: calls.append(kw) or {"message": "Draft ready"})
    receiver.dispatch(channel=channel, channel_type=channel_type, ts="1.0", text="Project 1131: uploaded")
    receiver.dispatch(channel=channel, channel_type=channel_type, thread_ts="1.0", ts="2.0", text="BioID is there too")
    receiver.dispatch(name="app_mention", channel=channel, channel_type=channel_type,
                      thread_ts="1.0", ts="3.0", text="<@UBOT> make a note these were reported")
    current = calls[-1]
    assert "BioID" not in current["message"]
    assert current["message"].endswith("make a note these were reported")
    assert [r["text"] for r in current["meta"]["slack_thread_context"]["messages"]] == [
        "Project 1131: uploaded", "BioID is there too",
    ]


def test_context_lookup_failure_does_not_prevent_normal_reply(monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("outbound-context"):
            raise requests.Timeout("SECRET")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"message": "OK"})
    monkeypatch.setattr(slack.requests, "post", post)
    result = slack._post_ispec_chat(server="http://backend.test", api_key="test", session_id="test", message="thanks",
                                   meta={"slack": {"channel": "D1"}}, timeout_seconds=60)
    assert result["message"] == "OK" and len(calls) == 2
    assert calls[-1][1]["json"]["meta"]["slack_outbound_context_status"] == "unavailable"


def test_provider_unavailable_with_receipt_and_channel_mention(receiver, monkeypatch):
    monkeypatch.setattr(
        slack,
        "_post_ispec_chat",
        lambda **kw: {
            "responseStatus": "unavailable",
            "message": "raw provider failure",
            "retention": {"storage": "support_message", "recordId": 10},
        },
    )
    receiver.dispatch(
        name="app_mention",
        channel="CTEST",
        channel_type="channel",
        text="<@UBOT> hello",
    )
    receiver.dispatch(
        name="app_mention",
        channel="CTEST",
        channel_type="channel",
        text="<@UBOT> hello",
    )
    assert len(receiver.posts) == 1
    assert receiver.updates[0]["text"] == RETAINED_REPLY


@pytest.mark.parametrize("first", ["message", "app_mention"])
def test_dm_mention_seen_by_both_handlers_replies_once(receiver, monkeypatch, first):
    calls = []
    monkeypatch.setattr(
        slack,
        "_post_ispec_chat",
        lambda **kw: calls.append(kw) or {"message": "one reply"},
    )
    second = "app_mention" if first == "message" else "message"
    receiver.dispatch(name=first, text="<@UBOT> hello")
    receiver.dispatch(name=second, text="<@UBOT> hello")
    assert len(calls) == len(receiver.posts) == len(receiver.updates) == 1


def test_bot_events_and_channel_echoes_never_reply(receiver, monkeypatch):
    monkeypatch.setattr(
        slack, "_post_ispec_chat", lambda **kw: pytest.fail("must not invoke backend")
    )
    receiver.dispatch(bot_id="BBOT")
    receiver.dispatch(subtype="message_changed")
    receiver.dispatch(
        channel="CTEST", channel_type="channel", text="<@UBOT> hello", thread_ts="1.1"
    )
    assert receiver.posts == receiver.updates == receiver.says == []
    acked = []
    for name in [
        "file_deleted",
        "reaction_removed",
        "file_created",
        "file_shared",
        "reaction_added",
    ]:
        receiver.handlers[name](
            event={"type": name},
            logger=SimpleNamespace(debug=lambda *args: None),
            ack=lambda: acked.append(name),
        )
    assert len(acked) == 5


def test_passive_channel_messages_are_retained_without_model_or_send(receiver, monkeypatch):
    monkeypatch.setattr(slack, "_post_ispec_chat", lambda **kw: pytest.fail("No model call"))
    receiver.dispatch(channel="CTEST", channel_type="channel", text="Project 42: 12 samples received")
    receiver.dispatch(channel="COTHER", channel_type="channel", ts="2.2", text="Not opted in")
    assert receiver.posts == receiver.says == receiver.updates == []
    rows = receiver.inbox.records()
    assert len(rows) == 1 and rows[0]["text"] == "Project 42: 12 samples received"
    assert rows[0]["reviewed_at"] is None


def test_question_still_goes_straight_to_support_with_receipt_reference(receiver, monkeypatch):
    calls = []
    monkeypatch.setattr(slack, "_post_ispec_chat", lambda **kw: calls.append(kw) or {"message": "Project details"})
    receiver.dispatch(text="Tell me about project 1131")
    row = receiver.inbox.records()[0]
    assert calls[0]["meta"]["slack_inbox_id"] == row["id"]
    assert receiver.updates[-1]["text"] == "Project details"
    assert row["reviewed_at"] is None

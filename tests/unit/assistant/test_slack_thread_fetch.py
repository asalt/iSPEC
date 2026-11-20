from types import SimpleNamespace

import pytest

from ispec.assistant.slack_inbox import SlackInbox
from ispec.assistant.slack_outbound import ConversationScope
from ispec.assistant.slack_thread_context import SlackThreadContext, ThreadMessage, render_thread_context
from ispec.assistant.slack_thread_fetch import fetch_thread_context


def context(**kwargs):
    return SlackThreadContext(scope=ConversationScope(team_id="T1", channel="C1", thread_ts="1.0", message_ts="50.0"), **kwargs)


def client(rows, **metadata):
    calls = []
    def replies(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "messages": rows, **metadata}
    return SimpleNamespace(conversations_replies=replies, calls=calls)


def row(ts, text="unmentioned message", **kw):
    return {"ts": ts, "thread_ts": "1.0", "user": "UOTHER", "text": text, **kw}


def test_missing_parent_fetched_with_other_humans_and_bot_as_evidence():
    api = client([row("1.0", "<@URECIPIENT> Project 42 ready for data analysis"),
                  row("2.0", "We should review the results first"),
                  row("3.0", "A prior bot draft", user="UBOT", bot_id="B1"),
                  row("50.0", "current request"), row("51.0", "future reply")])
    result = fetch_thread_context(context(), client=api)
    assert [r.message_ts for r in result.messages] == ["1.0", "2.0", "3.0"]
    assert api.calls == [{"channel": "C1", "ts": "1.0", "latest": "50.0", "inclusive": False, "limit": 100}]
    assert result.source == "slack_api" and result.coverage == "thread_snapshot"
    assert all(r.inbox_id is None for r in result.messages)
    prompt = render_thread_context(result)
    assert prompt.metadata["parent_present"] and prompt.metadata["fetch_status"] == "loaded"


def test_exact_receipt_required_and_network_runs_outside_db_session(tmp_path):
    inbox = SlackInbox(file=tmp_path / "assistant.db")
    current = inbox.receive({"team_id": "T1", "event": {
        "type": "app_mention", "channel": "C1", "ts": "50.0", "thread_ts": "1.0",
        "user": "USTAFF", "text": "make a note about this",
    }}, {"bot_user_id": "UBOT"})
    meta = {"source": "slack", "slack": context().scope.model_dump(), "slack_inbox_id": current}
    def replies(**kw):
        # A new SQLite writer succeeds; the context reader does not hold a transaction.
        inbox._update(current, reviewed_by="test")
        return {"ok": True, "messages": [row("1.0")]}
    result = inbox.thread_context(meta, client=SimpleNamespace(conversations_replies=replies))
    assert result["messages"][0]["text"] == "unmentioned message"
    assert len(inbox.records()) == 1  # No history backfill or new work queued.
    api = client([row("1.0")])
    meta["slack"]["channel"] = "PRIVATE"
    result = inbox.thread_context(meta, client=api)
    assert result["status"] == "unavailable" and not api.calls


def test_large_and_paginated_thread_is_bounded_and_honest():
    api = client([row(f"{n}.0", "x" * 2500) for n in range(1, 45)],
                 has_more=True, response_metadata={"next_cursor": "no-follow-up-request"})
    result = fetch_thread_context(context(), client=api)
    assert len(api.calls) == 1 and result.truncated
    assert result.coverage == "partial_thread_snapshot"
    assert result.messages[0].message_ts == "1.0"
    assert result.messages[-1].message_ts == "44.0"
    assert len(result.messages) <= 24 and len(result.model_dump_json()) <= 8000
    assert all(len(r.text) <= 2000 for r in result.messages)


def test_cross_thread_and_future_edits_are_excluded():
    api = client([row("1.0"), row("2.0", "foreign", thread_ts="0.0"),
                  row("3.0", "later edit", edited={"ts": "51.0"}),
                  row("4.0", "known edit", edited={"ts": "5.0"}),
                  row("NaN", "invalid"), {"bad": "row"}])
    result = fetch_thread_context(context(), client=api)
    assert [r.message_ts for r in result.messages] == ["1.0", "4.0"]
    assert result.truncated


def test_complete_snapshot_does_not_resurrect_absent_local_messages():
    old = ThreadMessage(inbox_id=3, user_id="U1", message_ts="3.0", text="removed locally cached text")
    result = fetch_thread_context(context(messages=(old,)), client=client([row("1.0")]))
    assert [r.message_ts for r in result.messages] == ["1.0"]


@pytest.mark.parametrize("failure", [TimeoutError("secret token"), RuntimeError("private path")])
def test_failure_retains_local_evidence_without_exposing_error_details(failure):
    def fail(**kw):
        raise failure
    local = ThreadMessage(inbox_id=1, user_id="U1", message_ts="1.0", text="prior locally retained text")
    result = fetch_thread_context(context(messages=(local,)), client=SimpleNamespace(conversations_replies=fail))
    assert result.messages == (local,)
    assert result.fetch_status == "failed" and result.fetch_error == "thread_fetch_failed"
    assert "secret" not in result.model_dump_json() and "private path" not in result.model_dump_json()


def test_rate_limit_is_reported_without_retry_or_send():
    class RateLimit(Exception):
        response = {"error": "ratelimited"}
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise RateLimit("token is hidden")
    result = fetch_thread_context(context(), client=SimpleNamespace(conversations_replies=fail))
    assert len(calls) == 1 and result.fetch_error == "ratelimited"


def test_new_root_and_unavailable_context_never_fetch():
    api = client([])
    fetch_thread_context(context(status="unavailable"), client=api)
    root = SlackThreadContext(scope=ConversationScope(team_id="T1", channel="C1", thread_ts="50.0", message_ts="50.0"))
    fetch_thread_context(root, client=api)
    assert api.calls == []

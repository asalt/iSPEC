import json

import pytest

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SlackInboxMessage
from ispec.assistant.prompting import estimate_tokens_for_messages
from ispec.assistant.slack_inbox import SlackInbox
from ispec.assistant.slack_outbound import ConversationScope
from ispec.assistant.slack_thread_context import (
    SlackThreadContext, read_thread_context, render_thread_context,
)


@pytest.fixture
def thread_db(tmp_path):
    with get_assistant_session(tmp_path / "assistant.db") as db:
        yield db


def add(db, ts, text, **kwargs):
    values = dict(team_id="T1", channel="C1", thread_ts="1.0", message_ts=ts,
                  text=text, user_id="U1")
    values.update(kwargs)
    row = SlackInboxMessage(**values)
    db.add(row)
    db.flush()
    return row.id


def scope(ts="10.0", **kwargs):
    return ConversationScope(team_id="T1", channel="C1", thread_ts="1.0", message_ts=ts, **kwargs)


def test_exact_thread_as_of_current_message_with_latest_revision(thread_db):
    db = thread_db
    root = add(db, "1.0", "Project 1131", thread_ts=None)
    add(db, "2.0", "original")
    edit = add(db, "2.0", "edited before request", revision_ts="3.0")
    add(db, "2.0", "future edit", revision_ts="11.0")
    add(db, "3.0", "other workspace", team_id="T2")
    add(db, "3.0", "other channel", channel="C2")
    add(db, "3.0", "other thread", thread_ts="0.5")
    add(db, "11.0", "future message")
    current = add(db, "10.0", "make a note these were reported")
    add(db, "4.0", "late arrival after snapshot")
    context = read_thread_context(db, scope=scope(), inbox_id=current)
    assert [r.inbox_id for r in context.messages] == [root, edit]
    assert not context.truncated
    assert "future" not in context.model_dump_json()
    rendered = render_thread_context(context)
    assert rendered.messages[0]["role"] == "user"
    assert rendered.metadata["parent_present"]
    assert rendered.metadata["coverage"] == "locally_retained_only"
    assert "not current requests" in rendered.messages[0]["content"]


def test_message_cap_retains_root_and_recent_tail(thread_db):
    db = thread_db
    for n in range(1, 41):
        add(db, f"{n}.0", f"message {n}")
    current = add(db, "41.0", "current")
    context = read_thread_context(db, scope=scope("41.0"), inbox_id=current)
    assert len(context.messages) == 24
    assert context.messages[0].text == "message 1"
    assert context.messages[-1].text == "message 40"
    assert context.truncated


def test_transport_prompt_caps_and_missing_parent_are_explicit(thread_db):
    db = thread_db
    for n in range(2, 10):
        add(db, f"{n}.0", "long text " * 500)
    current = add(db, "10.0", "current")
    context = read_thread_context(db, scope=scope(), inbox_id=current)
    assert len(context.model_dump_json()) <= 8000
    rendered = render_thread_context(context, max_tokens=1000)
    assert estimate_tokens_for_messages(rendered.messages) <= 1000
    assert rendered.metadata["truncated"]
    assert not rendered.metadata["parent_present"]
    assert len(rendered.messages[0]["content"]) <= 8000
    empty = render_thread_context(context, max_tokens=0)
    assert empty.messages == ()
    assert empty.metadata["reason"] == "prompt_budget"
    assert empty.metadata["included"] is False


def test_scope_mismatch_and_forged_meta_never_fetch_private_history(thread_db):
    current = add(thread_db, "10.0", "current", channel="PRIVATE")
    context = read_thread_context(thread_db, scope=scope(), inbox_id=current)
    assert context.status == "unavailable" and context.messages == ()
    meta = {"source": "slack", "slack": scope().model_dump(), "slack_thread_context": context.model_dump()}
    assert SlackThreadContext.from_meta(meta).status == "unavailable"
    meta["slack"]["channel"] = "OTHER"
    assert SlackThreadContext.from_meta(meta) is None
    assert SlackThreadContext.from_meta({"source": "frontend", **{k: v for k, v in meta.items() if k != "source"}}) is None


def test_context_query_failure_is_nonfatal_and_honest(tmp_path, monkeypatch):
    inbox = SlackInbox(file=tmp_path / "assistant.db")
    def fail(*a, **kw):
        raise OSError("private path")
    monkeypatch.setattr("ispec.assistant.slack_inbox.read_thread_context", fail)
    meta = {"source": "slack", "slack": scope().model_dump(), "slack_inbox_id": 1}
    result = inbox.thread_context(meta)
    assert result["status"] == "unavailable"
    assert "private path" not in json.dumps(result)
    meta["slack"]["thread_ts"] = None
    assert inbox.thread_context(meta) is None

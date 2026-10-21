from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.reply_delivery import RETAINED_REPLY
from ispec.assistant.reply_delivery import SlackEventClaims
from ispec.assistant.slack_inbox import SlackInbox, load_capture_channels


def event(**changes):
    return {"team_id": "T1", "event_id": "Ev1", "event": {
        "type": "message", "channel": "C1", "channel_type": "channel",
        "user": "U1", "ts": "123.456", "text": "Project 42: samples received", **changes}}


@pytest.fixture
def inbox(tmp_path):
    return SlackInbox(file=tmp_path / "assistant.db", channels=frozenset({("T1", "C1")}))


def test_capture_commits_before_handoff_and_deduplicates_across_restart(inbox):
    context = {"team_id": "T1"}
    def handoff():
        fresh = SlackInbox(file=inbox.file)
        row = fresh.records()[0]
        assert row["id"] == context["slack_inbox_id"]
        return "accepted"
    assert inbox.middleware(event(), context, handoff) == "accepted"
    fresh = SlackInbox(file=inbox.file, channels=inbox.channels)
    assert fresh.receive(event(), {}) == context["slack_inbox_id"]
    assert len(fresh.records()) == 1


def test_atomic_duplicate_delivery_and_claim(inbox):
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: inbox.receive(event(), {}), range(8)))
    assert len(set(ids)) == 1
    def claim(_):
        return inbox.claim({"slack_inbox_id": ids[0]}, fallback=SlackEventClaims(),
                           team_id="T1", event=event()["event"], route="reply")
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(claim, range(8))) == 1
    fresh = SlackInbox(file=inbox.file)
    assert not fresh.claim({"slack_inbox_id": ids[0]}, fallback=SlackEventClaims(),
                           team_id="T1", event=event()["event"], route="reply")
    assert fresh.records()[0]["handled_at"] is None  # Crash/unknown, not auto-retried.


def test_review_is_explicit_and_edits_are_separate_evidence(inbox):
    original = inbox.receive(event(), {})
    inbox.review(original, actor="alex")
    edited = event(subtype="message_changed", message={
        "user": "U1", "ts": "123.456", "text": "Correction: 10 samples",
        "edited": {"ts": "124.000"}})
    revision = inbox.receive(edited, {})
    assert revision != original
    assert len(inbox.records()) == 1
    rows = inbox.records(unreviewed=False)
    assert rows[1]["text"] == "Project 42: samples received"
    assert rows[1]["reviewed_by"] == "alex"
    assert rows[0]["reviewed_at"] is None
    assert rows[0]["processing_started_at"] is None


def test_failed_storage_neither_acks_nor_calls_handler(inbox, monkeypatch):
    def fail(*args):
        raise OSError("private detail")
    monkeypatch.setattr(inbox, "receive", fail)
    context = {}
    response = inbox.middleware(event(), context, lambda: pytest.fail("No handler on failed persistence"))
    assert response.status == 503
    assert "private detail" not in response.body
    assert "slack_inbox_id" not in context


def test_commit_failure_does_not_claim_retention(inbox, monkeypatch):
    def fail_commit(self):
        raise OSError("commit failed")
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", fail_commit)
        context = {}
        response = inbox.middleware(event(), context, lambda: pytest.fail("Commit must precede dispatch"))
        assert response.status == 503
        assert "slack_inbox_id" not in context
    assert inbox.records() == []


def test_capture_timeout_does_not_change_other_callers(inbox):
    with get_assistant_session(inbox.file) as db:
        timeout = db.scalar(text("PRAGMA busy_timeout"))
    with inbox.session() as db:
        assert db.scalar(text("PRAGMA busy_timeout")) == 1000
    with get_assistant_session(inbox.file) as db:
        assert db.scalar(text("PRAGMA busy_timeout")) == timeout


def test_failed_outcome_update_preserves_original_exception_and_retention(inbox, monkeypatch):
    row_id = inbox.receive(event(), {})
    def fail_update(*args, **kwargs):
        raise OSError("outcome storage unavailable")
    monkeypatch.setattr(inbox, "_update", fail_update)
    context = {"slack_inbox_id": row_id, "slack_inbox_claimed": True}
    assert inbox.failed_reply(context, TimeoutError()) == RETAINED_REPLY
    @inbox.handler
    def handle(*, context):
        raise ValueError("original error")
    with pytest.raises(ValueError, match="original error"):
        handle(context=context)
    assert inbox.records()[0]["handled_at"] is None


def test_scope_bot_filter_limits_and_pagination(inbox):
    assert inbox.receive(event(channel="OTHER"), {}) is None
    assert inbox.receive({**event(), "team_id": "OTHER"}, {}) is None
    assert inbox.receive(event(bot_id="B1"), {}) is None
    assert inbox.receive(event(), {"bot_user_id": "U1"}) is None
    first = inbox.receive(event(channel="DM1", channel_type="im"), {})
    second = inbox.receive(event(channel="OTHER", type="app_mention", text="x" * 40001), {})
    row = inbox.records(limit=1)[0]
    assert row["truncated"] and len(row["text"]) == 40000
    assert row["id"] == second
    assert inbox.records(before_id=second)[0]["id"] == first
    with pytest.raises(ValueError):
        inbox.records(limit=10000)


def test_inbound_configuration_is_exact_and_separate(tmp_path):
    path = tmp_path / "config.json"
    assert load_capture_channels(None) == frozenset()
    path.write_text(json.dumps({"channels": [{"team_id": "T1", "channel_id": "C1"}]}))
    assert load_capture_channels(str(path)) == frozenset({("T1", "C1")})
    path.write_text(json.dumps({"channels": [{"team_id": "T1", "channel_id": "*"}]}))
    with pytest.raises(ValueError):
        load_capture_channels(str(path))

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent, RelayDedupKey
from ispec.agent.relay_dedup import prune_relay_keys
from ispec.agent.relay_store import enqueue_relay_request, record_relay_receipt


@pytest.fixture
def relay_db(tmp_path, monkeypatch):
    path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(path))
    return path


def request(**overrides):
    return {
        "request_id": "first",
        "kind": "slack_message",
        "mode": "send",
        "confirm": True,
        "source": {"kind": "codex", "id": "test"},
        "target": {"alias": "alex"},
        "body": "Report ready",
        **overrides,
    }


def enqueue(path, **overrides):
    with get_agent_session(path) as db:
        row, result = enqueue_relay_request(db, request=request(**overrides))
        return row.id, result


def finish(path, command_id, payload, outcome="sent"):
    receipt = {
        "ok": outcome in {"sent", "staged"},
        "sent": outcome == "sent",
        "delivery_outcome": outcome,
    }
    record_relay_receipt(
        command_id=command_id, request=payload["request"], receipt=receipt
    )
    with get_agent_session(path) as db:
        db.get(AgentCommand, command_id).status = (
            "succeeded" if receipt["ok"] else "failed"
        )


def test_retries_and_duplicate_payload_share_one_command(relay_db):
    original, _ = enqueue(relay_db)
    repeated, by_id = enqueue(relay_db)
    duplicate, by_payload = enqueue(relay_db, request_id="second")
    repeated_alias, alias_result = enqueue(relay_db, request_id="second")
    assert original == repeated == duplicate == repeated_alias
    assert by_id["duplicate_match"] == alias_result["duplicate_match"] == "request_id"
    assert by_payload["duplicate_match"] == "payload"
    assert by_payload["submitted_request_id"] == "second"
    assert by_payload["request"]["request_id"] == "first"
    with get_agent_session(relay_db) as db:
        assert db.query(AgentCommand).count() == db.query(AgentEvent).count() == 1
        assert db.query(RelayDedupKey).count() == 3


def test_reusing_id_for_changed_payload_conflicts(relay_db):
    original, _ = enqueue(relay_db)
    with pytest.raises(ValueError, match="relay_request_id_conflict"):
        enqueue(relay_db, body="different")
    assert enqueue(relay_db)[0] == original
    with get_agent_session(relay_db) as db:
        assert db.query(RelayDedupKey).count() == 2


@pytest.mark.parametrize(
    "change",
    [
        {"target": {"alias": "another"}},
        {"thread_ts": "thread2"},
        {"mode": "stage"},
        {"press_enter": True},
        {"source": {"kind": "codex", "id": "other"}},
        {"provenance": {"pane": "%2"}},
        {"confirm": False},
    ],
)
def test_fingerprint_scopes_operation_destination_and_provenance(relay_db, change):
    original, _ = enqueue(relay_db)
    assert enqueue(relay_db, request_id="second", **change)[0] != original


def test_cwd_is_not_identity(relay_db):
    first, _ = enqueue(relay_db, source={"kind": "codex", "id": "test", "cwd": "/a"})
    assert (
        enqueue(relay_db, source={"kind": "codex", "id": "test", "cwd": "/b"})[0]
        == first
    )


def test_attachment_contents_participate_in_identity(relay_db, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"old report")
    original, _ = enqueue(relay_db, attachments=[str(pdf)])
    pdf.write_bytes(b"new report")
    with pytest.raises(ValueError, match="relay_request_id_conflict"):
        enqueue(relay_db, attachments=[str(pdf)])
    assert enqueue(relay_db, attachments=[str(pdf)], request_id="new")[0] != original


def test_failed_enqueue_rolls_back_keys_job_and_event(relay_db):
    with get_agent_session(relay_db) as db:

        def fail_before_commit(session):
            raise RuntimeError("injected commit failure")

        event.listen(db, "before_commit", fail_before_commit, once=True)
        with pytest.raises(RuntimeError, match="injected"):
            enqueue_relay_request(db, request=request())
    with get_agent_session(relay_db) as db:
        assert db.query(RelayDedupKey).count() == 0
        assert db.query(AgentCommand).count() == db.query(AgentEvent).count() == 0
    enqueue(relay_db)


def test_pruning_removes_only_expired_keys_not_audit_history(relay_db):
    original, payload = enqueue(relay_db)
    finish(relay_db, original, payload)
    future = datetime.now(UTC) + timedelta(minutes=11)
    with get_agent_session(relay_db) as db:
        assert prune_relay_keys(db, now=future)["eligible"] == 1
        assert db.query(RelayDedupKey).count() == 2
        assert prune_relay_keys(db, now=future, confirm=True)["deleted"] == 1
    # Stable ID remains protected, but a new request can now repeat the payload.
    assert enqueue(relay_db)[0] == original
    assert enqueue(relay_db, request_id="new")[0] != original
    with get_agent_session(relay_db) as db:
        assert (
            prune_relay_keys(db, now=future + timedelta(days=31), confirm=True)[
                "deleted"
            ]
            == 1
        )
        assert db.query(AgentCommand).count() == 2
        assert db.query(AgentEvent).count() == 3


def test_enqueue_expires_completed_payload_without_sweep(relay_db, monkeypatch):
    import ispec.agent.relay_store as store

    original, payload = enqueue(relay_db)
    finish(relay_db, original, payload)
    future = datetime.now(UTC) + timedelta(minutes=11)
    monkeypatch.setattr(store, "utcnow", lambda: future)
    assert enqueue(relay_db, request_id="new")[0] != original
    assert enqueue(relay_db)[0] == original


def test_prune_is_bounded(relay_db):
    original, payload = enqueue(relay_db)
    finish(relay_db, original, payload)
    with get_agent_session(relay_db) as db:
        assert (
            prune_relay_keys(
                db, now=datetime.now(UTC) + timedelta(days=31), confirm=True, limit=1
            )["deleted"]
            == 1
        )
        assert db.query(RelayDedupKey).count() == 1


@pytest.mark.parametrize(
    "state", ["queued", "running", "unknown", "partial", "missing_receipt"]
)
def test_pruning_never_unlocks_pending_or_uncertain_delivery(relay_db, state):
    original, payload = enqueue(relay_db)
    if state in {"unknown", "partial"}:
        finish(relay_db, original, payload, outcome=state)
    with get_agent_session(relay_db) as db:
        row = db.get(AgentCommand, original)
        if state in {"queued", "running"}:
            row.status = state
            row.result_json = {"delivery_outcome": "sent"}
        elif state == "missing_receipt":
            row.status = "failed"
        for key in db.query(RelayDedupKey):
            key.expires_at = datetime.now(UTC) - timedelta(days=1)
    with get_agent_session(relay_db) as db:
        assert prune_relay_keys(db, now=datetime.now(UTC), confirm=True)["deleted"] == 0
    assert enqueue(relay_db, request_id="again")[0] == original


@pytest.fixture
def fake_dispatch(monkeypatch):
    import ispec.agent.relay_dispatcher as dispatcher

    monkeypatch.setattr(dispatcher, "load_canonical_env", lambda: {})
    monkeypatch.setattr(dispatcher, "relay_config_probe", lambda **kw: {})
    monkeypatch.setattr(dispatcher, "relay_live_enabled", lambda env: True)
    monkeypatch.setattr(dispatcher, "source_policy", lambda req, **kw: (True, {}))
    monkeypatch.setattr(
        dispatcher,
        "resolve_slack_destination",
        lambda *a, **kw: ({"channel": "fake"}, None),
    )
    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return True, {"sent": True, "slack": {"ts": "123.4"}}, None

    monkeypatch.setattr(dispatcher, "execute_slack_send", send)
    monkeypatch.setattr(dispatcher, "execute_slack_uploads", send)
    return dispatcher, calls


def test_delivery_receipt_prevents_second_execution(relay_db, fake_dispatch):
    dispatcher, calls = fake_dispatch
    command, payload = enqueue(relay_db)
    first = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    again = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    assert len(calls) == 1
    assert first["slack"] == again["slack"] == {"ts": "123.4"}


def test_timeout_keeps_claim_and_never_resends(relay_db, fake_dispatch, monkeypatch):
    dispatcher, _ = fake_dispatch
    calls = []

    def timeout(**kwargs):
        calls.append(1)
        raise TimeoutError("accepted remotely but acknowledgement lost")

    monkeypatch.setattr(dispatcher, "execute_slack_send", timeout)
    command, payload = enqueue(relay_db)
    for _ in range(2):
        result = dispatcher.dispatch_relay_request(
            payload["request"], command_id=command
        )
        assert result["delivery_outcome"] == "unknown"
        assert result["sent"] is None
    assert calls == [1]
    assert enqueue(relay_db, request_id="retry")[0] == command
    with get_agent_session(relay_db) as db:
        assert all(key.expires_at is None for key in db.query(RelayDedupKey))


def test_changed_attachment_is_not_sent(relay_db, fake_dispatch, tmp_path):
    dispatcher, calls = fake_dispatch
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"original")
    command, payload = enqueue(relay_db, attachments=[str(pdf)])
    pdf.write_bytes(b"changed")
    result = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    assert result["error_type"] == "relay_payload_changed"
    assert not calls


def test_different_request_id_cannot_bypass_stored_payload_check(
    relay_db, fake_dispatch
):
    dispatcher, calls = fake_dispatch
    command, payload = enqueue(relay_db)
    altered = {**payload["request"], "request_id": "different", "body": "not approved"}
    result = dispatcher.dispatch_relay_request(altered, command_id=command)
    assert result["error_type"] == "relay_payload_changed"
    assert not calls


def test_crash_after_execution_marker_never_sends_again(relay_db, fake_dispatch):
    from ispec.agent.relay_store import begin_relay_delivery

    dispatcher, calls = fake_dispatch
    command, payload = enqueue(relay_db)
    assert begin_relay_delivery(command, payload["request"]) is None
    receipt = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    assert receipt["delivery_outcome"] == "unknown"
    assert not calls


def test_partial_upload_stays_protected(relay_db, fake_dispatch, monkeypatch):
    dispatcher, _ = fake_dispatch
    monkeypatch.setattr(
        dispatcher,
        "execute_slack_send",
        lambda **kw: (
            False,
            {"sent": True, "uploaded": [{"file_id": "F123"}]},
            "second_upload_failed",
        ),
    )
    command, payload = enqueue(relay_db)
    receipt = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    assert receipt["delivery_outcome"] == "partial"
    assert receipt["uploaded"] == [{"file_id": "F123"}]
    with get_agent_session(relay_db) as db:
        assert all(key.expires_at is None for key in db.query(RelayDedupKey))


def test_config_change_does_not_rewrite_success(relay_db, fake_dispatch, monkeypatch):
    dispatcher, calls = fake_dispatch
    command, payload = enqueue(relay_db)
    dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    monkeypatch.setattr(dispatcher, "relay_live_enabled", lambda env: False)
    receipt = dispatcher.dispatch_relay_request(payload["request"], command_id=command)
    assert receipt["delivery_outcome"] == "sent"
    assert len(calls) == 1

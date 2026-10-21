import json

import pytest
import requests

from ispec.cli import slack_history as cli


@pytest.fixture
def provider(monkeypatch, tmp_path):
    calls = []
    responses = []
    monkeypatch.setattr(cli.slack, "_cli_slack_bot_token", lambda: "xoxb-secret-test")
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(tmp_path / "assistant.db"))

    def call(**kwargs):
        calls.append((kwargs["endpoint"], kwargs["payload"]))
        if kwargs["endpoint"] == "auth.test":
            return {"ok": True, "bot_id": "B123", "team_id": "T123", "user_id": "UBOT"}
        if kwargs["endpoint"] == "conversations.info":
            return {"ok": True, "channel": {"id": "GC420B63V", "name": "proteomics_core"}}
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(cli.slack, "_slack_direct_api_call", call)
    return calls, responses


def run(capsys, *args):
    code = cli.main(["--channel", "GC420B63V", *args])
    return code, json.loads(capsys.readouterr().out)


def test_default_read_and_redaction(provider, capsys):
    calls, responses = provider
    responses.append({"ok": True, "messages": [{"ts": "1.1", "text": "xoxb-secret-test"}]})
    code, result = run(capsys)
    assert code == 0 and result["source"] == "live_slack"
    assert result["messages"][0]["text"] == "[REDACTED]"
    assert [x[0] for x in calls] == ["auth.test", "conversations.info", "conversations.history"]


def test_thread_pagination_and_bound(provider, capsys):
    calls, responses = provider
    responses.extend([
        {"ok": True, "messages": [{"ts": "1.1"}], "response_metadata": {"next_cursor": "c1"}},
        {"ok": True, "messages": [{"ts": "1.2"}], "response_metadata": {"next_cursor": "c2"}},
    ])
    code, result = run(capsys, "--thread-ts", "1.1", "--max-pages", "2", "--limit", "3")
    assert code == 0 and result["next_cursor"] == "c2" and result["has_more"]
    assert len(result["messages"]) == 2
    assert calls[-1] == ("conversations.replies", {"channel": "GC420B63V", "limit": 3, "ts": "1.1", "cursor": "c1"})


def test_resume_cursor_and_missing_cursor_warning(provider, capsys):
    calls, responses = provider
    responses.append({"ok": True, "messages": [], "has_more": True})
    code, result = run(capsys, "--cursor", "old")
    assert code == 0 and result["warning"]
    assert calls[-1][1]["cursor"] == "old"


def test_repeated_cursor_stops(provider, capsys):
    calls, responses = provider
    responses.append({"ok": True, "messages": [], "response_metadata": {"next_cursor": "old"}})
    code, result = run(capsys, "--cursor", "old", "--max-pages", "3")
    assert code == 1 and "repeated" in result["error"]
    assert len(calls) == 3


@pytest.mark.parametrize("args", [
    ["--text", "hello"], ["--send"], ["--send", "--text", "hello"],
    ["--thread-ts", "not-a-timestamp"], ["--limit", "0"], ["--max-pages", "101"],
    ["--timeout-seconds", "nan"],
])
def test_invalid_or_implicit_send_never_calls_provider(provider, capsys, args):
    calls, _ = provider
    code, result = run(capsys, *args)
    assert code == 1 and not result["ok"] and calls == []


def test_dry_run_no_credentials_or_network(provider, monkeypatch, capsys):
    calls, _ = provider
    monkeypatch.setattr(cli.slack, "_cli_slack_bot_token", lambda: "")
    code, result = run(capsys, "--thread-ts", "1.1", "--text", "hello", "--dry-run")
    assert code == 0 and result["dry_run"] and calls == []
    assert result["payload"]["text"] == "hello"


def test_explicit_send_exactly_once_mocked(provider, capsys):
    calls, responses = provider
    responses.append({"ok": True, "ts": "1.2"})
    code, result = run(capsys, "--thread-ts", "1.1", "--text", "hello", "--send")
    assert code == 0 and result["sent"]
    assert result["retention"]["retained"]
    assert calls[-1] == ("chat.postMessage", {"channel": "GC420B63V", "thread_ts": "1.1", "text": "hello", "unfurl_links": "false", "unfurl_media": "false"})
    assert len(calls) == 3


def test_duplicate_send_receipt_is_one_row(provider, capsys):
    calls, responses = provider
    responses.extend([{"ok": True, "ts": "1.2"}] * 2)
    _, first = run(capsys, "--thread-ts", "1.1", "--text", "hello", "--send")
    _, second = run(capsys, "--thread-ts", "1.1", "--text", "hello", "--send")
    assert first["retention"]["record_id"] == second["retention"]["record_id"]
    assert second["retention"]["duplicate"]


@pytest.mark.parametrize("error", ["missing_scope", "not_allowed_token_type", "not_in_channel"])
def test_permission_failure_no_join_or_retry(provider, capsys, error):
    calls, responses = provider
    responses.append({"ok": False, "error": error})
    code, result = run(capsys, "--thread-ts", "1.1")
    assert code == 1 and error in result["error"] and len(calls) == 3


def test_send_transport_error_no_retry_or_secret_output(provider, capsys):
    calls, responses = provider
    responses.append(requests.Timeout("credential xoxb-secret-test"))
    code, result = run(capsys, "--thread-ts", "1.1", "--text", "hello", "--send")
    assert code == 1 and "uncertain" in result["error"]
    assert "secret" not in result["error"] and len(calls) == 3


def test_rate_limit(provider, capsys):
    _, responses = provider
    response = requests.Response()
    response.status_code = 429
    responses.append(requests.HTTPError(response=response))
    code, result = run(capsys)
    assert code == 1 and "Rate limited" in result["error"]


def test_exact_import_never_sends_and_keeps_file_reference(provider, capsys):
    calls, responses = provider
    responses.append({"ok": True, "messages": [{"ts": "1.1", "thread_ts": "1.1", "user": "UBOT",
        "text": "approved report", "files": [{"id": "F1", "name": "short.pdf", "mimetype": "application/pdf", "size": 20}]}]})
    code, result = run(capsys, "--thread-ts", "1.1", "--record-outbound")
    assert code == 0 and result["imported"] and result["sent"] is False
    assert result["files"][0]["file_id"] == "F1"
    assert calls[-1] == ("conversations.replies", {"channel": "GC420B63V", "ts": "1.1", "limit": 1})
    assert all(endpoint != "chat.postMessage" for endpoint, _ in calls)


@pytest.mark.parametrize("row", [{"ts": "1.1", "user": "UHUMAN"}, {"ts": "1.2", "user": "UBOT"}])
def test_exact_import_rejects_other_authors_or_missing_message(provider, capsys, row):
    calls, responses = provider
    responses.append({"ok": True, "messages": [row]})
    code, result = run(capsys, "--thread-ts", "1.1", "--record-outbound")
    assert code == 1 and "nothing imported" in result["error"]
    assert len(calls) == 3


def test_import_cannot_be_combined_with_sending(provider, capsys):
    calls, _ = provider
    code, _ = run(capsys, "--thread-ts", "1.1", "--record-outbound", "--send", "--text", "no")
    assert code == 1 and calls == []


def test_exact_dm_import_verifies_actual_peer_before_recording(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(tmp_path / "assistant.db"))
    monkeypatch.setattr(cli.slack, "_cli_slack_bot_token", lambda: "xoxb-test")
    calls = []
    results = {
        "auth.test": {"ok": True, "team_id": "T1", "user_id": "UBOT"},
        "conversations.info": {"ok": True, "channel": {"id": "D1", "is_im": True, "user": "UACTUAL"}},
        "users.info": {"ok": True, "user": {"id": "UACTUAL", "name": "actual_peer"}},
        "conversations.replies": {"ok": True, "messages": [{"ts": "1.1", "user": "UBOT", "text": "draft intended for someone else"}]},
    }
    def call(**kw):
        calls.append(kw["endpoint"])
        return results[kw["endpoint"]]
    monkeypatch.setattr(cli.slack, "_slack_direct_api_call", call)
    assert cli.main(["--channel", "D1", "--thread-ts", "1.1", "--record-outbound"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recipient"]["id"] == "UACTUAL"
    assert calls == ["auth.test", "conversations.info", "users.info", "conversations.replies"]

import json

import pytest
import requests

from ispec.cli import slack_history as cli


@pytest.fixture(autouse=True)
def local_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config/slack-history.json"


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


def test_date_scan_bounds_and_read_only(provider, capsys):
    calls, responses = provider
    start, end = cli.date_window("2026-10-21", "America/Chicago")
    responses.append({"ok": True, "messages": [
        {"ts": str(start - 1)}, {"ts": str(start)},
        {"ts": str(end - 1)}, {"ts": str(end)}, {"ts": "NaN"},
    ]})
    code, result = run(capsys, "--date", "2026-10-21")
    assert code == 0 and len(result["messages"]) == 2
    assert "older threads" in result["coverage"]
    assert calls[-1][1]["oldest"] == str(start)
    assert calls[-1][1]["latest"] == str(end)
    assert all(endpoint != "chat.postMessage" for endpoint, _ in calls)


@pytest.mark.parametrize("day,hours", [("2026-03-08", 23), ("2026-11-01", 25)])
def test_date_scan_handles_daylight_saving(day, hours):
    start, end = cli.date_window(day, "America/Chicago")
    assert end - start == hours * 3600


def test_timezone_remembered_across_calls_and_working_directories(provider, capsys, local_settings, monkeypatch, tmp_path):
    calls, responses = provider
    responses.extend([{"ok": True, "messages": []}] * 3)
    assert run(capsys, "--date", "2026-10-21", "--timezone", "Asia/Tokyo")[0] == 0
    assert json.loads(local_settings.read_text()) == {"timezone": "Asia/Tokyo"}
    assert local_settings.stat().st_mode & 0o077 == 0
    monkeypatch.chdir(tmp_path)
    code, result = run(capsys, "--date", "2026-10-22")
    assert code == 0 and result["date_window"]["timezone"] == "Asia/Tokyo"
    assert calls[-1][1]["oldest"] == str(cli.date_window("2026-10-22", "Asia/Tokyo")[0])
    assert run(capsys, "--date", "2026-10-22", "--timezone", "UTC")[0] == 0
    assert cli.saved_timezone() == "UTC"


@pytest.mark.parametrize("args", [
    ["--date", "2026-10-21", "--timezone", "invalid/timezone"],
    ["--date", "2026-02-30", "--timezone", "UTC"],
    ["--date", "2026-10-21", "--timezone", "UTC", "--send", "--thread-ts", "1.1", "--text", "no"],
])
def test_invalid_command_preserves_saved_timezone(provider, capsys, args):
    cli.save_timezone("America/Chicago")
    assert run(capsys, *args)[0] == 1
    assert cli.saved_timezone() == "America/Chicago"
    assert provider[0] == []


def test_date_default_does_not_create_settings(provider, capsys, local_settings):
    provider[1].append({"ok": True, "messages": []})
    _, result = run(capsys, "--date", "2026-10-21")
    assert result["date_window"]["timezone"] == "America/Chicago"
    assert not local_settings.exists()


@pytest.mark.parametrize("contents", ["invalid json", "[]", '{"timezone": "bad/zone"}'])
def test_corrupt_preferences_require_explicit_timezone(provider, capsys, local_settings, contents):
    local_settings.parent.mkdir()
    local_settings.write_text(contents)
    code, result = run(capsys, "--date", "2026-10-21")
    assert code == 1 and "specify --timezone" in result["error"]
    assert provider[0] == []
    provider[1].append({"ok": True, "messages": []})
    assert run(capsys, "--date", "2026-10-21", "--timezone", "UTC")[0] == 0
    assert cli.saved_timezone() == "UTC"


def test_settings_write_failure_happens_before_provider(provider, capsys, monkeypatch):
    monkeypatch.setattr(cli.tempfile, "NamedTemporaryFile", lambda **kw: (_ for _ in ()).throw(PermissionError()))
    code, result = run(capsys, "--date", "2026-10-21", "--timezone", "UTC")
    assert code == 1 and "no Slack request" in result["error"]
    assert provider[0] == []


def test_timezone_selection_survives_provider_failure(provider, capsys):
    provider[1].append(requests.Timeout("synthetic outage"))
    code, _ = run(capsys, "--date", "2026-10-21", "--timezone", "UTC")
    assert code == 1
    assert cli.saved_timezone() == "UTC"


def test_failed_atomic_replace_preserves_previous_setting(provider, capsys, monkeypatch, local_settings):
    cli.save_timezone("America/Chicago")
    monkeypatch.setattr(cli.Path, "replace", lambda *a: (_ for _ in ()).throw(PermissionError()))
    code, _ = run(capsys, "--date", "2026-10-21", "--timezone", "UTC")
    assert code == 1 and cli.saved_timezone() == "America/Chicago"
    assert not list(local_settings.parent.glob(".slack-history-*"))
    assert provider[0] == []


def test_send_dry_run_does_not_save_timezone(provider, capsys, local_settings):
    assert run(capsys, "--dry-run", "--thread-ts", "1.1", "--text", "hello", "--timezone", "UTC")[0] == 0
    assert not local_settings.exists()


@pytest.mark.parametrize("args", [
    ["--date", "2026-02-30"], ["--date", "2026-10-21", "--timezone", "not/a/timezone"],
    ["--date", "2026-10-21", "--thread-ts", "1.1", "--send", "--text", "no"],
    ["--date", "2026-10-21", "--thread-ts", "1.1", "--record-outbound"],
])
def test_invalid_date_or_date_send_calls_no_provider(provider, capsys, args):
    calls, _ = provider
    code, _ = run(capsys, *args)
    assert code == 1 and calls == []


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

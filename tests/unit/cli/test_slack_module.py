from __future__ import annotations

from datetime import UTC, datetime
import json

from ispec.cli import slack


class _FakeSlackResponse(dict):
    pass


class _FakeSlackApiError(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.response = {"error": error}


class _FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def users_info(self, *, user: str):  # type: ignore[no-untyped-def]
        self.calls.append(user)
        return {
            "user": {
                "name": "alex",
                "real_name": "Alex Smith",
                "profile": {
                    "display_name": "Alex S",
                    "real_name": "Alex Smith",
                },
            }
        }


class _FakePostingSlackClient:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, str]] = []
        self.update_calls: list[dict[str, str]] = []
        self.join_calls: list[str] = []
        self.fail_first_post = False

    def chat_postMessage(self, **kwargs):  # type: ignore[no-untyped-def]
        self.post_calls.append(dict(kwargs))
        if self.fail_first_post and len(self.post_calls) == 1:
            raise _FakeSlackApiError("not_in_channel")
        return _FakeSlackResponse({"ok": True, "ts": "171234.5678"})

    def chat_update(self, **kwargs):  # type: ignore[no-untyped-def]
        self.update_calls.append(dict(kwargs))
        return _FakeSlackResponse({"ok": True, "ts": kwargs.get("ts")})

    def conversations_join(self, *, channel: str):  # type: ignore[no-untyped-def]
        self.join_calls.append(channel)
        return _FakeSlackResponse({"ok": True})


def test_slack_user_summary_prefers_readable_fields_and_caches() -> None:
    client = _FakeSlackClient()
    cache: dict[str, dict[str, str]] = {}

    first = slack._slack_user_summary(client=client, user_cache=cache, user_id="U123")
    second = slack._slack_user_summary(client=client, user_cache=cache, user_id="U123")

    assert first["user_id"] == "U123"
    assert first["user_name"] == "alex"
    assert first["user_display_name"] == "Alex S"
    assert first["user_real_name"] == "Alex Smith"
    assert second == first
    assert client.calls == ["U123"]


def test_format_message_for_ispec_omits_raw_slack_id_when_name_missing() -> None:
    message = slack._format_message_for_ispec(
        text="Hello can you hear me?",
        slack_user={"user_id": "UC54CNNUW"},
    )

    assert message == "Hello can you hear me?"


def test_format_message_for_ispec_uses_display_name_when_available() -> None:
    message = slack._format_message_for_ispec(
        text="Hello can you hear me?",
        slack_user={
            "user_id": "UC54CNNUW",
            "user_display_name": "Alex S",
            "user_real_name": "Alex Smith",
        },
    )

    assert message == "[Alex S] Hello can you hear me?"


def test_safe_slack_post_message_retries_after_not_in_channel() -> None:
    client = _FakePostingSlackClient()
    client.fail_first_post = True

    ts = slack._safe_slack_post_message(
        client=client,
        channel="C123",
        thread_ts="171000.0001",
        text="Working on it...",
    )

    assert ts == "171234.5678"
    assert client.join_calls == ["C123"]
    assert len(client.post_calls) == 2
    assert client.post_calls[0]["thread_ts"] == "171000.0001"


def test_safe_slack_update_message_uses_existing_placeholder() -> None:
    client = _FakePostingSlackClient()

    ok = slack._safe_slack_update_message(
        client=client,
        channel="C123",
        message_ts="171234.5678",
        text="Finished.",
    )

    assert ok is True
    assert client.update_calls == [
        {"channel": "C123", "ts": "171234.5678", "text": "Finished."}
    ]


def test_session_id_for_dm_rotates_by_utc_day_bucket() -> None:
    session_id = slack._session_id_for_dm(
        team_id="T123",
        channel="D456",
        now=datetime(2026, 4, 2, 23, 59, tzinfo=UTC),
    )

    assert session_id == "slack:T123:D456:dm24:20260402"


def test_thread_session_id_stays_thread_scoped() -> None:
    session_id = slack._session_id(team_id="T123", channel="C456", thread_ts="171000.0001")

    assert session_id == "slack:T123:C456:171000.0001"


def test_send_slack_text_dry_run_resolves_alias_channel(monkeypatch) -> None:
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_SLACK_RECIPIENTS_JSON", '{"alex": "D123DM"}')

    result = slack.send_slack_text(
        recipient="alex",
        text="hello from test",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["resolved"]["channel"] == "D123DM"
    assert result["payload"]["text"] == "hello from test"


def test_send_slack_text_dry_run_resolves_assistant_destination_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("ISPEC_SLACK_RECIPIENTS_JSON", raising=False)
    destinations_path = tmp_path / "assistant-slack-destinations.local.json"
    destinations_path.write_text(
        json.dumps(
            {
                "destinations": {
                    "proteomics_core": {
                        "kind": "channel",
                        "channel": "GC420B63V",
                        "audience": "staff",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISPEC_ASSISTANT_SLACK_DESTINATIONS_PATH", str(destinations_path))

    result = slack.send_slack_text(
        recipient="proteomics_core",
        text="hello channel",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["resolved"]["channel"] == "GC420B63V"
    assert result["payload"]["text"] == "hello channel"


def test_send_slack_text_resolves_email_to_dm_and_posts(monkeypatch) -> None:
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    calls: list[dict[str, object]] = []

    def fake_post(url, *, headers=None, json=None, timeout=None):  # type: ignore[no-untyped-def]
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})

        class FakeResponse:
            def raise_for_status(self):  # type: ignore[no-untyped-def]
                return None

            def json(self):  # type: ignore[no-untyped-def]
                if str(url).endswith("/users.lookupByEmail"):
                    return {"ok": True, "user": {"id": "U123"}}
                if str(url).endswith("/conversations.open"):
                    return {"ok": True, "channel": {"id": "D123"}}
                if str(url).endswith("/chat.postMessage"):
                    return {"ok": True, "channel": "D123", "ts": "123.456"}
                return {"ok": False, "error": "unexpected_endpoint"}

        return FakeResponse()

    monkeypatch.setattr(slack.requests, "post", fake_post)

    result = slack.send_slack_text(
        email="alex@example.test",
        text="hello by email",
    )

    assert result["ok"] is True
    assert [str(call["url"]).rsplit("/", 1)[-1] for call in calls] == [
        "users.lookupByEmail",
        "conversations.open",
        "chat.postMessage",
    ]
    assert calls[-1]["json"] == {"channel": "D123", "text": "hello by email"}


def test_upload_slack_file_dry_run_resolves_file_and_alias(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ISPEC_SLACK_RECIPIENTS_JSON", '{"alex": "D123DM"}')
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4\n")

    result = slack.upload_slack_file(
        recipient="alex",
        file_path=report,
        text="report ready",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["resolved"]["channel"] == "D123DM"
    assert result["file"]["filename"] == "report.pdf"
    assert result["upload_request"]["length"] == len(b"%PDF-1.4\n")
    assert result["complete_payload"]["channel_id"] == "D123DM"
    assert result["complete_payload"]["initial_comment"] == "report ready"


def test_upload_slack_file_uses_external_upload_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ISPEC_SLACK_BOT_TOKEN", "xoxb-test")
    report = tmp_path / "report.txt"
    report.write_text("hello report\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_post(url, *, headers=None, json=None, data=None, files=None, timeout=None):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "data": data,
                "files": files,
                "timeout": timeout,
            }
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self):  # type: ignore[no-untyped-def]
                return None

            def json(self):  # type: ignore[no-untyped-def]
                if str(url).endswith("/conversations.open"):
                    return {"ok": True, "channel": {"id": "D123"}}
                if str(url).endswith("/files.getUploadURLExternal"):
                    return {
                        "ok": True,
                        "upload_url": "https://upload.example.test/abc",
                        "file_id": "F123",
                    }
                if str(url) == "https://upload.example.test/abc":
                    return {"ok": True}
                if str(url).endswith("/files.completeUploadExternal"):
                    return {"ok": True, "files": [{"id": "F123", "title": "Report"}]}
                return {"ok": False, "error": "unexpected_endpoint"}

        return FakeResponse()

    monkeypatch.setattr(slack.requests, "post", fake_post)

    result = slack.upload_slack_file(
        user_id="U123",
        file_path=report,
        text="report ready",
        title="Report",
    )

    assert result["ok"] is True
    assert [str(call["url"]).rsplit("/", 1)[-1] for call in calls] == [
        "conversations.open",
        "files.getUploadURLExternal",
        "abc",
        "files.completeUploadExternal",
    ]
    assert calls[1]["json"] is None
    assert calls[1]["data"]["filename"] == "report.txt"
    assert calls[1]["data"]["length"] == len("hello report\n")
    assert calls[2]["files"] is not None
    assert calls[3]["json"]["files"] == [{"id": "F123", "title": "Report"}]
    assert calls[3]["json"]["channel_id"] == "D123"
    assert calls[3]["json"]["initial_comment"] == "report ready"

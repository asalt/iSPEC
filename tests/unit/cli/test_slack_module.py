from __future__ import annotations

from datetime import UTC, datetime

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

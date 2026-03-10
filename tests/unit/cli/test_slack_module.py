from __future__ import annotations

from ispec.cli import slack


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

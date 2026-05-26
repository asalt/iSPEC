from __future__ import annotations

import json

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentEvent
from ispec.agent.relay import (
    EVENT_RELAY_RECEIPT,
    dispatch_relay_request,
    relay_config_probe,
)


def _clear_relay_env(monkeypatch) -> None:
    for key in (
        "ISPEC_SLACK_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "ISPEC_SLACK_APP_TOKEN",
        "SLACK_APP_TOKEN",
        "ISPEC_SLACK_CONFIG_TOML",
        "ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON",
        "ISPEC_ASSISTANT_SLACK_ALLOWED_DESTINATIONS_JSON",
        "ISPEC_ASSISTANT_SLACK_DESTINATIONS_PATH",
        "ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL",
        "ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST",
        "ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH",
        "ISPEC_ASSISTANT_TMUX_TARGET_BLACKLIST",
        "ISPEC_ASSISTANT_TMUX_TARGET_BLACKLIST_PATH",
        "ISPEC_RELAY_LIVE_SEND_ENABLED",
        "ISPEC_RELAY_ALLOWED_SOURCES",
        "ISPEC_SLACK_UPLOAD_MAX_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)


def _relay_root(tmp_path):
    root = tmp_path / "ispec-full"
    (root / "configs").mkdir(parents=True)
    (root / "Makefile").write_text("test:\n\t@true\n", encoding="utf-8")
    return root


def test_relay_config_probe_loads_canonical_slack_env_independent_of_cwd(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    (root / ".env.local").write_text("ISPEC_RELAY_LIVE_SEND_ENABLED=1\n", encoding="utf-8")
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "user_id": "U123ALEX"}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path / "ispec-full" / "configs")

    probe = relay_config_probe(target_alias="alex")

    assert probe["root"] == str(root)
    token = probe["slack"]["bot_token"]
    assert token["present"] is True
    assert "xoxb" not in json.dumps(probe)
    assert ".env.slack" in str(token["source"])
    assert probe["relay"]["live_send_enabled"] is True
    assert probe["relay"]["pdf_attachment_upload"]["supported"] is True
    assert probe["slack"]["target"]["alias"] == "alex"
    assert probe["slack"]["target"]["user_id"] == "U123ALEX"


def test_relay_stages_slack_message_without_calling_external_slack(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "user_id": "U123ALEX"}}}),
        encoding="utf-8",
    )

    def forbidden_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("stage-only relay must not call Slack")

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "target": {"alias": "alex"},
            "body": "MSPC000936 LF rerun complete; gpGrouper comparison next.",
            "metadata": {
                "work_state": {
                    "project": "MSPC000936",
                    "status": "complete_idle",
                    "next_action": "gpGrouper comparison",
                }
            },
        },
        command_id=42,
        slack_post=forbidden_post,
    )

    assert receipt["ok"] is True
    assert receipt["delivery_outcome"] == "staged"
    assert receipt["sent"] is False
    assert receipt["metadata"]["work_state"]["status"] == "complete_idle"

    with get_agent_session(agent_db_path) as db:
        rows = db.query(AgentEvent).filter(AgentEvent.event_type == EVENT_RELAY_RECEIPT).all()
        assert len(rows) == 1
        payload = json.loads(rows[0].payload_json)
        assert payload["receipt"]["delivery_outcome"] == "staged"
        assert payload["receipt"]["metadata"]["work_state"]["project"] == "MSPC000936"


def test_relay_send_mode_reports_missing_token_precisely(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("ISPEC_RELAY_LIVE_SEND_ENABLED", "1")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "user_id": "U123ALEX"}}}),
        encoding="utf-8",
    )

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "mode": "send",
            "confirm": True,
            "target": {"alias": "alex"},
            "body": "This should not send without a token.",
        },
        command_id=43,
    )

    assert receipt["ok"] is False
    assert receipt["delivery_outcome"] == "failed"
    assert receipt["error_type"] == "missing_token"
    assert receipt["sent"] is False


def test_relay_reports_unallowlisted_slack_target(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "target": {"alias": "not_allowlisted"},
            "body": "Nope.",
        },
        command_id=44,
    )

    assert receipt["ok"] is False
    assert receipt["error_type"] == "target_not_allowed"
    assert receipt["sent"] is False


def test_relay_can_fail_closed_for_untrusted_source(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("ISPEC_RELAY_ALLOWED_SOURCES", "codex:trusted")
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "user_id": "U123ALEX"}}}),
        encoding="utf-8",
    )

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "source": {"kind": "codex", "id": "untrusted"},
            "target": {"alias": "alex"},
            "body": "Blocked by source policy.",
        },
        command_id=46,
    )

    assert receipt["ok"] is False
    assert receipt["error_type"] == "source_not_allowed"
    assert receipt["sent"] is False
    assert receipt["policy"]["source"]["allowed_sources_configured"] is True


def test_relay_send_mode_uploads_pdf_attachment_from_canonical_env(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    (root / ".env.local").write_text(
        "\n".join(
            [
                "ISPEC_RELAY_LIVE_SEND_ENABLED=1",
                "ISPEC_SLACK_BOT_TOKEN=xoxb-from-file",
            ]
        ),
        encoding="utf-8",
    )
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "channel": "D123ALEX"}}}),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class FakeResponse:
        def __init__(self, payload=None, *, status_code=200, text="OK"):
            self._payload = payload if isinstance(payload, dict) else {}
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return dict(self._payload)

    calls: list[str] = []

    def fake_post(url, **kwargs):  # type: ignore[no-untyped-def]
        url_text = str(url)
        calls.append(url_text)
        if url_text.endswith("/files.getUploadURLExternal"):
            assert kwargs["data"]["filename"] == "report.pdf"
            return FakeResponse({"ok": True, "upload_url": "https://upload.example.test/file", "file_id": "F123"})
        if url_text == "https://upload.example.test/file":
            assert "files" in kwargs
            return FakeResponse({"ok": True})
        if url_text.endswith("/files.completeUploadExternal"):
            payload = kwargs["json"]
            assert payload["channel_id"] == "D123ALEX"
            assert payload["initial_comment"] == "A report is attached."
            assert payload["files"][0]["id"] == "F123"
            return FakeResponse({"ok": True, "files": [{"id": "F123"}]})
        raise AssertionError(f"unexpected Slack call: {url_text}")

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "mode": "send",
            "confirm": True,
            "target": {"alias": "alex"},
            "body": "A report is attached.",
            "attachments": [{"path": str(pdf_path)}],
        },
        command_id=47,
        slack_post=fake_post,
    )

    assert receipt["ok"] is True
    assert receipt["delivery_outcome"] == "sent"
    assert receipt["sent"] is True
    assert receipt["attachments_uploaded"][0]["file_id"] == "F123"
    assert calls == [
        "https://slack.com/api/files.getUploadURLExternal",
        "https://upload.example.test/file",
        "https://slack.com/api/files.completeUploadExternal",
    ]


def test_relay_send_mode_rejects_non_pdf_attachment(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("ISPEC_RELAY_LIVE_SEND_ENABLED", "1")
    (root / ".env.slack").write_text("ISPEC_SLACK_BOT_TOKEN=xoxb-from-file\n", encoding="utf-8")
    (root / "configs" / "assistant-slack-destinations.local.json").write_text(
        json.dumps({"destinations": {"alex": {"kind": "dm", "channel": "D123ALEX"}}}),
        encoding="utf-8",
    )
    text_path = tmp_path / "report.txt"
    text_path.write_text("not a pdf", encoding="utf-8")

    def forbidden_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("non-PDF attachments must fail before Slack calls")

    receipt = dispatch_relay_request(
        {
            "kind": "slack_message",
            "mode": "send",
            "confirm": True,
            "target": {"alias": "alex"},
            "body": "A report is attached.",
            "attachments": [{"path": str(text_path)}],
        },
        command_id=48,
        slack_post=forbidden_post,
    )

    assert receipt["ok"] is False
    assert receipt["error_type"] == "attachment_upload_unsupported"
    assert receipt["sent"] is False


def test_relay_tmux_stage_honors_wildcard_allowlist_without_sending_keys(tmp_path, monkeypatch):
    _clear_relay_env(monkeypatch)
    root = _relay_root(tmp_path)
    monkeypatch.setenv("ISPEC_RELAY_CONFIG_ROOT", str(root))
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST", "936-* codex2-*")

    receipt = dispatch_relay_request(
        {
            "kind": "tmux_send",
            "target": {"target": "936-accumulating"},
            "body": "Review from Slack: please inspect the Figure 5 report.",
            "press_enter": True,
        },
        command_id=45,
    )

    assert receipt["ok"] is True
    assert receipt["delivery_outcome"] == "staged"
    assert receipt["sent"] is False
    assert receipt["policy"]["tmux"]["allowlist_match"] == "936-*"

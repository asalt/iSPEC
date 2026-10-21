import json

from ispec.assistant.slack_inbox import SlackInbox
from ispec.cli.main import main


def test_local_inbox_review_cli_without_slack_auth_or_network(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    database = tmp_path / "assistant.db"
    inbox = SlackInbox(file=database)
    row_id = inbox.receive({"team_id": "T1", "event": {
        "type": "message", "channel_type": "im", "channel": "D1", "user": "U1",
        "ts": "123.456", "text": "Review this later"}}, {})
    def run(*args):
        monkeypatch.setattr("sys.argv", ["ispec", "slack", "inbox", "--database", str(database), *args])
        main()
        return json.loads(capsys.readouterr().out)
    assert run("list")[0]["id"] == row_id
    result = run("review", str(row_id), "--by", "alex")
    assert result["reviewed"] and not result["write_authorized"]
    assert run("list") == []
    assert run("show", str(row_id))[0]["reviewed_by"] == "alex"

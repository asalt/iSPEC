"""Replay local capture -> routing -> response, with no network or real writers."""

import json
from pathlib import Path

import pytest
import requests

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.service import AssistantReply
from ispec.assistant.slack_inbox import SlackInbox
from ispec.db.connect import get_session
from ispec.schedule.connect import get_schedule_session

pytestmark = pytest.mark.behavioral


@pytest.mark.parametrize("request_text", [None, "thanks"])
@pytest.mark.parametrize("provider", ["vllm", "ollama"])
def test_thread_reaches_decision_router_and_planner_without_becoming_approval(tmp_path, behavioral_datastore, monkeypatch, request_text, provider):
    for key, value in behavioral_datastore.env_overrides().items():
        monkeypatch.setenv(key, value)
    def forbidden(*a, **kw):
        pytest.fail("No network, Slack send, or tool execution in this replay")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(support, "run_tool", forbidden)
    for name, value in {
        "PROVIDER": provider, "CLASSIFIER_PROVIDER": "inherit", "TOOL_PROTOCOL": "openai",
        "TURN_DECISION_MODE": "shadow", "MAX_TOOL_CALLS": "2", "MAX_PROMPT_TOKENS": "4000",
        "SUMMARY_MAX_CHARS": "0", "SELF_REVIEW": "0", "SELF_REVIEW_DECIDER": "0",
        "COMPACTION_ENABLED": "0", "CHAT_QUEUE_ENABLED": "0", "PROJECT_COMMENT_INTENT_HINTS": "0",
        "ENABLE_RESPONSE_CONTRACTS": "off", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "shadow",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + name, value)

    fixture = json.loads((Path(__file__).parent / "data/slack_reported_thread.json").read_text())
    latest = request_text or fixture["request"]
    inbox = SlackInbox(file=tmp_path / "assistant.db", channels=frozenset({("T1", "C1")}))
    for index, row in enumerate(fixture["messages"], start=1):
        inbox.receive({"team_id": "T1", "event": {
            "type": "message", "channel": "C1", "user": row["user"], "text": row["text"],
            "ts": f"{index}.0", "thread_ts": "1.0" if index > 1 else None,
        }}, {"bot_user_id": "UBOT"})
    if request_text:
        # Other people's old save words are reference, not the latest user's approval.
        inbox.receive({"team_id": "T1", "event": {
            "type": "message", "channel": "C1", "user": "UOTHER", "text": "yes confirm save it",
            "ts": "8.0", "thread_ts": "1.0",
        }}, {"bot_user_id": "UBOT"})
    current = inbox.receive({"team_id": "T1", "event": {
        "type": "app_mention", "channel": "C1", "user": "USTAFF", "text": latest,
        "ts": "10.0", "thread_ts": "1.0",
    }}, {"bot_user_id": "UBOT"})
    meta = {"source": "slack", "slack_inbox_id": current, "slack": {
        "team_id": "T1", "channel": "C1", "thread_ts": "1.0", "message_ts": "10.0", "user_id": "USTAFF",
    }}
    meta["slack_thread_context"] = inbox.thread_context(meta)
    calls = []
    def generate(**kw):
        calls.append(kw)
        stage = kw.get("observability_context", {}).get("stage")
        if stage == "turn_decision":
            inputs = json.loads(kw["messages"][-1]["content"])
            assert inputs["user_message"] == latest
            assert inputs["last_assistant_message"] == ""
            assert inputs["context"]["confirmation_reply"] is False
            assert "BioID" in json.dumps(inputs["context"]["quoted_slack_thread"])
            return AssistantReply(content="{}", provider="test", model="stub", meta=None)
        if stage == "tool_router":
            assert "1131" in json.dumps(kw["messages"])
            assert "BioID" in json.dumps(kw["messages"])
            assert kw["messages"][-1]["content"] == f'User request: "{latest}"'
            return AssistantReply(content=json.dumps({"primary": "projects", "secondary": [],
                                  "confidence": 1, "clarify": False}), provider="test", model="stub", meta=None)
        assert stage == "planner"
        assert kw["messages"][-1] == {"role": "user", "content": latest}
        assert "BioID is there too" in json.dumps(kw["messages"])
        assert "get_project" in {t["function"]["name"] for t in kw["tools"]}
        assert kw["tool_choice"] is None
        return AssistantReply(content="Draft: Project 1131 prof and BioID results were shared. Save this note?",
                              provider="test", model="stub", meta=None)
    monkeypatch.setattr(support, "generate_reply", generate)
    with (
        get_assistant_session(tmp_path / "assistant.db") as db,
        get_schedule_session(behavioral_datastore.schedule_db_path) as schedule_db,
        get_session(behavioral_datastore.core_db_path) as core_db,
    ):
        response = support.chat(support.ChatRequest(
            sessionId="slack:T1:C1:1.0", message=latest, history=[], meta=meta,
        ), assistant_db=db, core_db=core_db, schedule_db=schedule_db, user=None)
        assert "Draft:" in response.message
        saved_user = db.query(SupportMessage).filter_by(role="user").one()
        assert saved_user.content == latest
        reply = db.query(SupportMessage).filter_by(role="assistant").one()
        metadata = json.loads(reply.meta_json)
        assert metadata["tooling"]["used_tool_calls"] == 0
        assert metadata["tooling"]["forced_tool_choice"] is None
        if request_text:
            assert "project_comment_approval_eval" not in metadata
        else:
            assert metadata["project_comment_approval_eval"]["write_outcome"]["status"] == "not_attempted"
        assert metadata["context"]["slack_thread_context"]["message_count"] == (8 if request_text else 7)
        planner_trace = next(t for t in metadata["llm_trace"] if t["prompt"] == "planner")
        assert planner_trace["slack_thread_context"]["included"]
        assert not planner_trace["slack_thread_context"]["truncated"]
    assert len(calls) == 3

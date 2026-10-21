"""Opt-in real inference through the queue worker; all tools are intercepted."""

import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import pytest
import requests

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.supervisor.loop import _run_support_chat_turn


pytestmark = [
    pytest.mark.behavioral,
    pytest.mark.skipif(os.getenv("ISPEC_RUN_VLLM_TESTS") != "1", reason="Opt-in local inference"),
]


@pytest.mark.parametrize("request_text", [
    "make a note these were reported",
    "hey save a project comment I provided full thread context so you can read the discussion",
])
def test_queue_worker_with_thread_context(behavioral_datastore, monkeypatch, tmp_path, request_text):
    for key, value in behavioral_datastore.env_overrides().items():
        monkeypatch.setenv(key, value)
    for kind in ("ANALYSIS", "PSM", "PROMPTS"):
        monkeypatch.setenv(f"ISPEC_{kind}_DB_PATH", str(tmp_path / f"{kind.lower()}.db"))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    for name, value in {
        "PROVIDER": "vllm", "CLASSIFIER_PROVIDER": "inherit", "TOOL_PROTOCOL": "openai",
        "TURN_DECISION_MODE": "shadow", "MAX_TOOL_CALLS": "2", "MAX_PROMPT_TOKENS": "4000",
        "SUMMARY_MAX_CHARS": "0", "SELF_REVIEW": "0", "SELF_REVIEW_DECIDER": "0",
        "COMPACTION_ENABLED": "0", "CHAT_QUEUE_ENABLED": "0", "PROJECT_COMMENT_INTENT_HINTS": "0",
        "ENABLE_RESPONSE_CONTRACTS": "off", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "shadow",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + name, value)
    monkeypatch.setenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("ISPEC_VLLM_MODEL", "allenai/Llama-3.1-Tulu-3-8B")

    original_request = requests.sessions.Session.request

    def local_inference_only(self, method, url, **kwargs):
        target = urlsplit(url)
        assert target.scheme == "http" and target.hostname == "127.0.0.1" and target.port == 8000
        assert target.path in {"/v1/models", "/v1/chat/completions"}
        kwargs["allow_redirects"] = False
        return original_request(self, method, url, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_inference_only)
    proposed_tools = []

    def intercept_tool(*, name, args, **kwargs):
        proposed_tools.append({"name": name, "arguments": args})
        if name == "get_project" and args.get("project_id") == 1131:
            return {"ok": True, "result": {"id": 1131, "prj_ProjectTitle": "Oocyte proteomics (synthetic fixture)"}}
        return {"ok": False, "error": "offline_review_only", "result": {
            "message": "No action executed. Present the proposed note for human review; do not claim it was saved.",
        }}

    monkeypatch.setattr(support, "run_tool", intercept_tool)
    fixture = json.loads((Path(__file__).parent / "data/slack_reported_thread.json").read_text())
    slack = {"team_id": "TDEMO", "channel": "CDEMO", "thread_ts": "1.0",
             "message_ts": "10.0", "user_id": "USTAFF"}
    context = {"scope": slack, "status": "loaded", "truncated": False, "messages": [
        {"inbox_id": i, "user_id": row["user"], "message_ts": f"{i}.0", "text": row["text"]}
        for i, row in enumerate(fixture["messages"], 1)
    ]}
    start = time.monotonic()
    execution = _run_support_chat_turn(payload={"user_id": 0, "chat_request": {
        "sessionId": "offline-thread-replay", "message": request_text, "history": [],
        "meta": {"source": "slack", "slack": slack, "slack_thread_context": context},
    }}, agent_id="offline-test", run_id="offline-test")
    assert execution.ok, execution.error
    with get_assistant_session(behavioral_datastore.assistant_db_path) as db:
        reply = db.query(SupportMessage).filter_by(role="assistant").order_by(SupportMessage.id.desc()).first()
        metadata = json.loads(reply.meta_json)
        report = {"request": request_text, "reply": reply.content, "proposed_tools": proposed_tools,
                  "elapsed_ms": round(1000 * (time.monotonic() - start)),
                  "thread_context": metadata["context"]["slack_thread_context"],
                  "trace": metadata["llm_trace"]}
        print("\nOFFLINE_QUEUE_REPLAY " + json.dumps(report, ensure_ascii=False))
        assert report["thread_context"]["message_count"] == 7
        assert all(t["slack_thread_context"]["included"] for t in metadata["llm_trace"])
        combined = reply.content + json.dumps(proposed_tools)
        assert "1131" in combined and "bioid" in combined.lower()
        assert "1234" not in combined and "5000" not in combined

"""Opt-in vLLM/Ollama approval replay: no tools or live database writes."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

from ispec.assistant.classifier_service import generate_classifier_reply
from ispec.assistant.project_comment_approval import (
    ProjectCommentApprovalSettings, ProjectCommentGate, decide_project_comment_policy,
    detect_project_comment_trigger, run_project_comment_approval_classifier, stable_hash,
)
from ispec.assistant.service import generate_reply

pytestmark = [pytest.mark.behavioral, pytest.mark.skipif(
    os.getenv("ISPEC_RUN_VLLM_TESTS") != "1" and os.getenv("ISPEC_RUN_LLM_TESTS") != "1",
    reason="Opt-in local inference",
)]
CORPUS = json.loads(Path(os.getenv("ISPEC_APPROVAL_EVAL_CASES_PATH",
    str(Path(__file__).parent / "data/project_comment_approval_cases.json"))).read_text())


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_local_classifier_policy(case, monkeypatch, tmp_path):
    provider = os.getenv("ISPEC_APPROVAL_EVAL_PROVIDER", "vllm")
    assert provider in {"vllm", "ollama"}
    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", provider)
    monkeypatch.setenv("ISPEC_ASSISTANT_CLASSIFIER_PROVIDER", "inherit")
    monkeypatch.setenv("ISPEC_VLLM_URL", os.getenv("ISPEC_VLLM_URL", "http://127.0.0.1:8000/v1"))
    monkeypatch.setenv("ISPEC_VLLM_MODEL", os.getenv("ISPEC_VLLM_MODEL", "allenai/Llama-3.1-Tulu-3-8B"))
    monkeypatch.setenv("ISPEC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    original = requests.sessions.Session.request
    base_url = os.getenv("ISPEC_OLLAMA_URL", "http://127.0.0.1:11434") if provider == "ollama" else os.environ["ISPEC_VLLM_URL"]
    base = urlsplit(base_url)
    assert base.scheme == "http" and base.hostname == "127.0.0.1"

    def local_only(self, method, url, **kwargs):
        target = urlsplit(url)
        assert target.scheme == base.scheme and target.hostname == base.hostname and target.port == base.port
        assert target.path in {"/v1/models", "/v1/chat/completions", "/api/chat"}
        kwargs["allow_redirects"] = False
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_only)
    settings = ProjectCommentApprovalSettings(mode="own")
    gate = ProjectCommentGate(eligible=case["pending"],
        kind="pending_save_confirmation" if case["pending"] else "none", reason="synthetic_fixture",
        project_id=42, pending_tool="create_project_comment", session_id="eval", thread_key="eval",
        current_turn_id="1", prior_assistant_message_id=1,
        pending_draft_hash=stable_hash(CORPUS["prior_assistant_message"]), source_message_hash=stable_hash(case["reply"]))
    trigger = detect_project_comment_trigger(message=case["reply"], legacy_confirmation_kind="none", legacy_save_requested=False)
    replies = []
    def capture_reply(**kwargs):
        reply = generate_reply(**kwargs)
        replies.append({"content": reply.content, "ok": reply.ok, "error": reply.error, "meta": reply.meta})
        return reply
    classifier = run_project_comment_approval_classifier(settings=settings, gate=gate, trigger=trigger,
        user_message=case["reply"], prior_assistant_message=CORPUS["prior_assistant_message"], focused_project_id=42,
        generate_reply_fn=capture_reply, generate_classifier_reply_fn=generate_classifier_reply)
    policy, ticket = decide_project_comment_policy(settings=settings, gate=gate, trigger=trigger, classifier=classifier)
    print("\nAPPROVAL_REPLAY " + json.dumps({"case": case["id"], "expected_label": case["expected_label"],
        "classifier": classifier.to_dict(), "policy": policy.to_dict(), "would_write": ticket.issued,
        "raw_replies": replies, "tools_executed": False, "saved": False}))
    assert ticket.issued == (case["expected_label"] == "approve_save")
    if case["pending"]:
        assert classifier.ok, classifier.error
    else:
        assert not classifier.ran

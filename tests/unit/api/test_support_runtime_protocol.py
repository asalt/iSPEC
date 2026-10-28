import json

import pytest

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage
from ispec.assistant.service import AssistantReply
from ispec.schedule.connect import get_schedule_session


@pytest.mark.parametrize("ok", [True, False])
def test_strict_runtime_never_uses_controller_prose_tool_fallback(tmp_path, db_session, monkeypatch, ok):
    # Stub provider avoids routing inference; the generated reply exercises the
    # vLLM result boundary in the real controller with temporary databases.
    for key, value in {
        "PROVIDER": "stub", "TOOL_PROTOCOL": "openai", "MAX_TOOL_CALLS": "2",
        "TURN_DECISION_MODE": "off", "PROJECT_COMMENT_APPROVAL_EVAL_MODE": "off",
        "ENABLE_SELF_REVIEW": "0", "ENABLE_RESPONSE_CONTRACTS": "off",
        "SUMMARY_MAX_CHARS": "0", "CHAT_QUEUE_ENABLED": "0",
    }.items():
        monkeypatch.setenv("ISPEC_ASSISTANT_" + key, value)
    monkeypatch.setenv("ISPEC_VLLM_STRICT_PROTOCOL", "1")
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Strict runtime used a prose tool fallback or executed a tool")

    monkeypatch.setattr(support, "parse_tool_call", forbidden)
    monkeypatch.setattr(support, "_suggested_tool_from_text", forbidden)
    monkeypatch.setattr(support, "run_tool", forbidden)
    monkeypatch.setattr(support, "generate_reply", lambda **kwargs: AssistantReply(
        content="FINAL:\nI could use count_all_projects." if ok else "Model response was incomplete.",
        provider="vllm", model="candidate", ok=ok,
        error=None if ok else "invalid_response",
        meta={"runtime": {"profile": "candidate", "strict_protocol": True},
              "tool_preamble": {"prompt_family": "assistant.tools.olmo3_preamble", "prompt_sha256": "fixture-hash"}},
    ))
    with get_assistant_session(tmp_path / "assistant.db") as adb, get_schedule_session(tmp_path / "schedule.db") as sdb:
        response = support.chat(support.ChatRequest(sessionId="synthetic-runtime", message="Hello", history=[]),
                                assistant_db=adb, core_db=db_session, schedule_db=sdb,
                                agent_db=None, omics_db=None, user=None)
        row = adb.get(SupportMessage, response.messageId)
        meta = json.loads(row.meta_json)
        trace = meta["llm_trace"]
        assert len(trace) == 1
        assert trace[0]["provider_meta"]["tool_preamble"] == {
            "prompt_family": "assistant.tools.olmo3_preamble", "prompt_sha256": "fixture-hash"}
        assert trace[0]["action"] == ("final_no_tool" if ok else "model_protocol_error")
        assert response.responseStatus == ("ok" if ok else "unavailable")

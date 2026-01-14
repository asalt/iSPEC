from __future__ import annotations

import json
import os

import pytest
import requests

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.tools import TOOL_CALL_PREFIX
from ispec.db.connect import get_session as get_core_session
from ispec.omics.connect import get_omics_session
from ispec.schedule.connect import get_schedule_session


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in _TRUTHY


if not _truthy_env("ISPEC_RUN_VLLM_TESTS"):
    pytest.skip(
        "Set ISPEC_RUN_VLLM_TESTS=1 to enable live vLLM integration tests.",
        allow_module_level=True,
    )


def _vllm_base_url() -> str:
    raw = (os.getenv("ISPEC_VLLM_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/models", "/v1"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].rstrip("/")
            break
    return raw


def _skip_if_vllm_unreachable(url: str) -> None:
    try:
        response = requests.get(f"{url}/v1/models", timeout=2)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"vLLM is not reachable at {url}: {type(exc).__name__}: {exc}")


def test_support_chat_can_use_assistant_stats_with_live_vllm(tmp_path, monkeypatch):
    url = _vllm_base_url()
    _skip_if_vllm_unreachable(url)

    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_ASSISTANT_TOOL_PROTOCOL", "openai")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("ISPEC_ASSISTANT_HISTORY_LIMIT", "10")
    monkeypatch.setenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS", "2000")
    monkeypatch.setenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_SELF_REVIEW_DECIDER", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_TEMPERATURE", "0")
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", "0")
    monkeypatch.setenv("ISPEC_VLLM_URL", url)
    monkeypatch.setenv("ISPEC_VLLM_TIMEOUT_SECONDS", "120")

    core_db_path = tmp_path / "core.db"
    assistant_db_path = tmp_path / "assistant.db"
    schedule_db_path = tmp_path / "schedule.db"
    omics_db_path = tmp_path / "omics.db"

    with (
        get_core_session(file_path=core_db_path) as core_db,
        get_assistant_session(assistant_db_path) as assistant_db,
        get_schedule_session(schedule_db_path) as schedule_db,
        get_omics_session(omics_db_path) as omics_db,
    ):
        session = SupportSession(session_id="session-1", user_id=None)
        assistant_db.add(session)
        assistant_db.flush()
        assistant_db.add_all(
            [
                SupportMessage(session_pk=session.id, role="user", content="Hi"),
                SupportMessage(session_pk=session.id, role="assistant", content="Hello"),
            ]
        )
        assistant_db.commit()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "Use the assistant_stats tool and tell me how many support sessions and messages exist.",
                "history": [],
                "ui": None,
            }
        )

        response = chat(
            payload,
            assistant_db=assistant_db,
            core_db=core_db,
            schedule_db=schedule_db,
            omics_db=omics_db,
            user=None,
        )
        assert response.message

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None

        meta = json.loads(assistant_row.meta_json or "{}")
        tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
        assert isinstance(tool_calls, list)

        stats_call = next((call for call in tool_calls if isinstance(call, dict) and call.get("name") == "assistant_stats"), None)
        assert isinstance(stats_call, dict), tool_calls
        assert stats_call.get("protocol") in {"openai", "line", "suggested"}
        assert stats_call.get("ok") is True

        preview = stats_call.get("result_preview")
        assert isinstance(preview, str) and preview
        result = json.loads(preview)
        assert result["sessions_total"] == 1
        assert result["messages_total"] == 3  # includes the new user message added by chat()

        raw_content = meta.get("raw_content") if isinstance(meta, dict) else None
        if isinstance(raw_content, str):
            assert TOOL_CALL_PREFIX not in raw_content


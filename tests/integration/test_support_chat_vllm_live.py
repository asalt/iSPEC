from __future__ import annotations

import json
import os
import re

import pytest
import requests

from ispec.api.routes import support as support_routes
from ispec.api.routes.support import ChatRequest, chat
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSession
from ispec.db.connect import get_session as get_core_session
from ispec.db.models import Project
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


def _find_count_tool_call(tool_calls: list[dict[str, object]]) -> dict[str, object] | None:
    for call in tool_calls:
        if isinstance(call, dict) and call.get("name") == "count_all_projects":
            return call
    return None


def test_support_chat_counts_total_projects_with_live_vllm(tmp_path, monkeypatch):
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
        core_db.add_all(
            [
                Project(prj_AddedBy="test", prj_ProjectTitle="One"),
                Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
                Project(prj_AddedBy="test", prj_ProjectTitle="Three"),
            ]
        )
        core_db.commit()

        assistant_db.add(SupportSession(session_id="session-1", user_id=None))
        assistant_db.flush()

        payload = ChatRequest.model_validate(
            {
                "sessionId": "session-1",
                "message": "Use a tool to answer: how many total projects do we have?",
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
        assert re.search(r"\bprojects?\b", response.message, re.IGNORECASE)

        assistant_row = (
            assistant_db.query(support_routes.SupportMessage)
            .filter(support_routes.SupportMessage.role == "assistant")
            .order_by(support_routes.SupportMessage.id.desc())
            .first()
        )
        assert assistant_row is not None

        meta = json.loads(assistant_row.meta_json or "{}")

        tool_router = meta.get("tool_router") if isinstance(meta, dict) else None
        assert isinstance(tool_router, dict)
        assert isinstance(tool_router.get("decision"), dict)
        assert tool_router["decision"].get("primary") == "projects"
        assert "count_all_projects" in (tool_router.get("selected_tool_names") or [])

        tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
        assert isinstance(tool_calls, list)
        count_call = _find_count_tool_call(tool_calls)
        assert isinstance(count_call, dict)

        assert count_call.get("protocol") in {"openai", "line", "suggested"}
        assert count_call.get("ok") is True

        preview = count_call.get("result_preview")
        assert isinstance(preview, str) and preview
        result = json.loads(preview)
        assert result["count"] == 3
        assert result["scope"] == "all"


from __future__ import annotations

import os

import pytest
import requests

from ispec.agent.commands import COMMAND_REVIEW_SUPPORT_SESSION
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportSessionReview
from ispec.supervisor.loop import SupervisorConfig, run_supervisor
from ispec.supervisor.smoke import enqueue_orchestrator_tick, seed_support_session_for_review


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


def test_supervisor_reviews_support_session_with_live_vllm(tmp_path, monkeypatch):
    url = _vllm_base_url()
    _skip_if_vllm_unreachable(url)

    agent_db_path = tmp_path / "agent.db"
    assistant_db_path = tmp_path / "assistant.db"

    monkeypatch.setenv("ISPEC_ASSISTANT_PROVIDER", "vllm")
    monkeypatch.setenv("ISPEC_VLLM_URL", url)
    monkeypatch.setenv("ISPEC_VLLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_MAX_COMMANDS_PER_TICK", "1")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_MIN_SECONDS", "10")
    monkeypatch.setenv("ISPEC_ORCHESTRATOR_MAX_SECONDS", "60")
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_ASSISTANT_DB_PATH", str(assistant_db_path))

    seeded = seed_support_session_for_review(
        session_id="smoke-session-1",
        user_message="Smoke test: user",
        assistant_message="Smoke test: assistant",
        assistant_db_path=assistant_db_path,
    )
    tick_id = enqueue_orchestrator_tick(
        payload={"source": "pytest_smoke", "session_id": seeded.session_id},
        priority=10,
        allow_existing=False,
        agent_db_path=agent_db_path,
    )
    assert tick_id is not None

    config = SupervisorConfig(
        agent_id="pytest-supervisor",
        backend_base_url="http://127.0.0.1:3001",
        frontend_url="http://127.0.0.1:3000/",
        interval_seconds=1,
        timeout_seconds=1.0,
    )

    # First run: consume orchestrator tick and enqueue a review command.
    run_supervisor(config, once=True)

    with get_agent_session(agent_db_path) as db:
        review_cmd = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_REVIEW_SUPPORT_SESSION)
            .order_by(AgentCommand.id.desc())
            .first()
        )
        assert review_cmd is not None
        assert str(review_cmd.status) == "queued"

    # Second run: consume review command and persist review record in assistant DB.
    run_supervisor(config, once=True)

    with get_agent_session(agent_db_path) as db:
        review_cmd = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_REVIEW_SUPPORT_SESSION)
            .order_by(AgentCommand.id.desc())
            .first()
        )
        assert review_cmd is not None
        assert str(review_cmd.status) == "succeeded"

    with get_assistant_session(assistant_db_path) as db:
        review_row = (
            db.query(SupportSessionReview)
            .filter(SupportSessionReview.session_pk == int(seeded.session_pk))
            .filter(SupportSessionReview.target_message_id == int(seeded.assistant_message_id))
            .first()
        )
        assert review_row is not None

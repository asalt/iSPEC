from __future__ import annotations

import pytest

from ispec.agent.commands import COMMAND_ASSESS_TACKLE_RESULTS
from ispec.agent.connect import get_agent_session
from ispec.api.routes.agents import EnqueueCommandRequest, enqueue_command, get_command
from fastapi import HTTPException


def test_agents_can_enqueue_and_fetch_command(tmp_path, monkeypatch) -> None:
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    request = EnqueueCommandRequest(
        command_type=COMMAND_ASSESS_TACKLE_RESULTS,
        payload={
            "project_id": 123,
            "results": {"pca": {"explained_variance_ratio": [0.4, 0.2]}},
        },
        priority=10,
        delay_seconds=0,
        max_attempts=2,
    )

    with get_agent_session(agent_db_path) as agent_db:
        resp = enqueue_command(request, db=agent_db)
        assert resp.command_id > 0

        fetched = get_command(resp.command_id, db=agent_db)
        assert fetched.id == resp.command_id
        assert fetched.command_type == COMMAND_ASSESS_TACKLE_RESULTS
        assert fetched.status == "queued"
        assert fetched.priority == 10
        assert fetched.max_attempts == 2
        assert isinstance(fetched.payload, dict)
        assert fetched.payload.get("project_id") == 123


def test_agents_enqueue_rejects_unknown_command_type(tmp_path, monkeypatch) -> None:
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    request = EnqueueCommandRequest(command_type="totally_not_a_command", payload={})
    with get_agent_session(agent_db_path) as agent_db:
        with pytest.raises(HTTPException) as excinfo:
            enqueue_command(request, db=agent_db)
        assert excinfo.value.status_code == 400

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from ispec.agent.commands import COMMAND_LOCAL_RELAY_REQUEST
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent
from ispec.agent.relay import EVENT_RELAY_REQUEST_ENQUEUED
from ispec.api.routes.agents import RelayEnqueueRequest, enqueue_relay_request_endpoint


def test_agents_can_enqueue_local_relay_request(tmp_path, monkeypatch) -> None:
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    request = RelayEnqueueRequest(
        kind="slack_message",
        source={"kind": "codex", "id": "pytest"},
        target={"alias": "alex"},
        body="Stage this status update.",
        metadata={"project": "MSPC000936"},
        priority=5,
    )

    with get_agent_session(agent_db_path) as agent_db:
        resp = enqueue_relay_request_endpoint(request, db=agent_db)
        assert resp.command_id > 0
        assert resp.request_id
        assert resp.relay_request["kind"] == "slack_message"
        assert resp.relay_request["mode"] == "stage"

        cmd = agent_db.query(AgentCommand).filter(AgentCommand.id == resp.command_id).one()
        assert cmd.command_type == COMMAND_LOCAL_RELAY_REQUEST
        assert cmd.status == "queued"
        assert cmd.priority == 5
        assert cmd.payload_json["relay_request"]["target"]["alias"] == "alex"

        event = agent_db.query(AgentEvent).filter(AgentEvent.event_type == EVENT_RELAY_REQUEST_ENQUEUED).one()
        payload = json.loads(event.payload_json)
        assert payload["command_id"] == resp.command_id
        assert payload["request"]["metadata"]["project"] == "MSPC000936"


def test_agents_relay_enqueue_rejects_unknown_kind(tmp_path, monkeypatch) -> None:
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    request = RelayEnqueueRequest(kind="do_whatever", target="alex", body="bad")
    with get_agent_session(agent_db_path) as agent_db:
        with pytest.raises(HTTPException) as excinfo:
            enqueue_relay_request_endpoint(request, db=agent_db)
        assert excinfo.value.status_code == 400

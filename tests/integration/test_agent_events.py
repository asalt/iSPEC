import json
import logging

import pytest
from fastapi.testclient import TestClient

from ispec.api.main import app
from ispec.agent.connect import get_agent_session as agent_get_session, get_agent_session_dep
from ispec.agent.models import AgentEvent
from ispec.db.connect import get_session as db_get_session, get_session_dep
from ispec.db.models import logger as db_logger

pytestmark = pytest.mark.testclient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))

    db_logger.setLevel(logging.ERROR)

    def override_get_session():
        with db_get_session() as session:
            yield session

    def override_get_agent_session():
        with agent_get_session(agent_db_path) as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session
    app.dependency_overrides[get_agent_session_dep] = override_get_agent_session
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_agent_event_ingest_persists_payload(client):
    payload = [
        {
            "type": "metric",
            "agent_id": "ms01",
            "ts": "2026-01-01T00:00:00Z",
            "name": "disk_free_bytes",
            "dimensions": {"path": "D:\\"},
            "value": 123,
        }
    ]
    resp = client.post("/api/agents/events", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ingested": 1}

    with agent_get_session() as session:
        row = session.query(AgentEvent).one()
        assert row.agent_id == "ms01"
        assert row.event_type == "metric"
        assert row.name == "disk_free_bytes"
        assert json.loads(row.payload_json)["value"] == 123


def test_agent_command_poll_returns_empty(client):
    resp = client.get("/api/agents/commands/poll", params={"agent_id": "ms01"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"commands": []}


def test_agent_ingest_requires_api_key_when_configured(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_API_KEY", "secret")

    db_logger.setLevel(logging.ERROR)

    def override_get_session():
        with db_get_session() as session:
            yield session

    def override_get_agent_session():
        with agent_get_session(agent_db_path) as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session
    app.dependency_overrides[get_agent_session_dep] = override_get_agent_session
    try:
        with TestClient(app) as client:

            payload = [
                {
                    "type": "metric",
                    "agent_id": "ms01",
                    "ts": "2026-01-01T00:00:00Z",
                    "name": "disk_free_bytes",
                    "dimensions": {"path": "D:\\"},
                    "value": 123,
                }
            ]

            resp = client.post("/api/agents/events", json=payload)
            assert resp.status_code == 401, resp.text

            resp = client.post(
                "/api/agents/events",
                json=payload,
                headers={"X-API-Key": "secret"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"ingested": 1}
    finally:
        app.dependency_overrides.clear()

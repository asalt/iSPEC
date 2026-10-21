from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from ispec.agent.connect import get_agent_session, get_agent_session_dep
from ispec.agent.models import AgentCommand, InstrumentObservationRow
from ispec.api.routes.instruments import router
from ispec.api.security import get_current_user, require_api_key
from ispec.db.models import UserRole
from ispec.instruments.collector import CollectorConfig, blank_observation


pytestmark = pytest.mark.testclient
TOKEN = "test-only-instrument-credential-0000000000"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = tmp_path / "collectors.json"
    cfg.write_text(
        json.dumps(
            {
                "collectors": {
                    "bruker": {
                        "token_env": "ISPEC_INSTRUMENT_TOKEN_BRUKER",
                        "instrument_ids": ["tims"],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("ISPEC_INSTRUMENT_COLLECTORS_FILE", str(cfg))
    monkeypatch.setenv("ISPEC_INSTRUMENT_TOKEN_BRUKER", TOKEN)
    monkeypatch.setenv("ISPEC_API_KEY", "general-key-not-a-collector-key")
    path = tmp_path / "agent.db"
    app = FastAPI()
    app.include_router(router, prefix="/api")

    @app.post("/api/agents/commands", dependencies=[Depends(require_api_key)])
    def unrelated_command_surface():
        pytest.fail("Collector credentials must not reach command endpoints")

    def session():
        with get_agent_session(path) as db:
            yield db

    app.dependency_overrides[get_agent_session_dep] = session
    app.dependency_overrides[get_current_user] = lambda: None
    with TestClient(app) as client:
        yield client, app, path


def payload():
    return blank_observation(
        CollectorConfig(
            collector_id="bruker", instrument_id="tims", root="/approved/root"
        )
    )


def submit(client, value, token=TOKEN):
    return client.post(
        "/api/instruments/observations",
        json=value,
        headers={"Authorization": "Bearer " + token},
    )


def test_ingest_replay_status_and_no_commands(setup):
    client, app, path = setup
    value = payload()
    result = submit(client, value)
    assert result.status_code == 200, result.text
    assert result.json()["duplicate"] is False
    assert submit(client, value).json()["duplicate"] is True
    assert submit(client, {**value, "source_root": "/different"}).status_code == 409
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        role=UserRole.editor, must_change_password=False
    )
    status = client.get("/api/instruments/tims/status")
    assert status.status_code == 200
    assert status.json()["instrument_alive"] is None
    assert status.json()["observation"]["instrument_state"] == "unknown"
    with get_agent_session(path) as db:
        assert db.query(InstrumentObservationRow).count() == 1
        assert db.query(AgentCommand).count() == 0


def test_unauthorized_and_identity_mismatch(setup):
    client, _, _ = setup
    assert (
        submit(client, payload(), token="general-key-not-a-collector-key").status_code
        == 401
    )
    assert submit(client, {**payload(), "collector_id": "other"}).status_code == 403
    assert submit(client, {**payload(), "instrument_id": "other"}).status_code == 403
    assert (
        client.post(
            "/api/agents/commands", headers={"Authorization": "Bearer " + TOKEN}
        ).status_code
        == 401
    )


def test_staff_only_read_not_collector_or_client(setup):
    client, app, _ = setup
    submit(client, payload())
    assert (
        client.get(
            "/api/instruments/tims/status", headers={"Authorization": "Bearer " + TOKEN}
        ).status_code
        == 401
    )
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        role=UserRole.viewer, must_change_password=False
    )
    assert client.get("/api/instruments/tims/status").status_code == 403


def test_fail_closed_disabled_or_broken_config(setup, monkeypatch):
    client, _, _ = setup
    monkeypatch.delenv("ISPEC_INSTRUMENT_COLLECTORS_FILE")
    assert submit(client, payload()).status_code == 503
    monkeypatch.setenv("ISPEC_INSTRUMENT_COLLECTORS_FILE", "relative.json")
    assert submit(client, payload()).status_code == 503


def test_body_limits_schema_and_future_clock(setup):
    client, _, _ = setup
    assert submit(client, {**payload(), "unrecognized": "field"}).status_code == 422
    assert submit(client, {**payload(), "pressure": 205}).status_code == 422
    future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    assert submit(client, {**payload(), "observed_at": future}).status_code == 422
    result = client.post(
        "/api/instruments/observations",
        content=b"x" * 65537,
        headers={"Authorization": "Bearer " + TOKEN},
    )
    assert result.status_code == 413


def test_no_instrument_control_routes(setup):
    client, _, _ = setup
    assert (
        client.post(
            "/api/instruments/tims/commands",
            json={"command": "stop"},
            headers={"Authorization": "Bearer " + TOKEN},
        ).status_code
        == 404
    )

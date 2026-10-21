from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ispec.api.routes.agents import router
from ispec.assistant.connect import get_assistant_session, get_assistant_session_dep
from ispec.api.security import require_configured_api_key
from fastapi import HTTPException


def test_receipt_context_requires_existing_key_even_when_optional_auth_disabled(monkeypatch):
    monkeypatch.delenv("ISPEC_API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc:
        require_configured_api_key(None)
    assert exc.value.status_code == 503
    monkeypatch.setenv("ISPEC_API_KEY", "test-secret")
    for invalid in [None, "wrong"]:
        with pytest.raises(HTTPException) as exc:
            require_configured_api_key(invalid)
        assert exc.value.status_code == 401
    require_configured_api_key("test-secret")


@pytest.mark.testclient
def test_receipt_api_validation_auth_and_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPEC_API_KEY", "test-secret")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    def db():
        with get_assistant_session(tmp_path / "assistant.db") as session:
            yield session
    app.dependency_overrides[get_assistant_session_dep] = db
    with TestClient(app) as client:
        payload = {"team_id": "T1", "channel": "D1", "message_ts": "1791547200.1", "text": "report",
                   "source": "test", "sent_at": "2026-10-09T12:00:00Z"}
        assert client.post("/api/agents/slack/outbound", json=payload).status_code == 401
        headers = {"X-API-Key": "test-secret"}
        first = client.post("/api/agents/slack/outbound", json=payload, headers=headers).json()
        second = client.post("/api/agents/slack/outbound", json=payload, headers=headers).json()
        assert first["retained"] and second["duplicate"] and first["record_id"] == second["record_id"]
        assert client.post("/api/agents/slack/outbound", json={**payload, "token": "SECRET"}, headers=headers).status_code == 422
        assert client.post("/api/agents/slack/outbound-context", json={
            "team_id": "T1", "channel": "D1", "message_ts": "1791547201.1",
        }).status_code == 401

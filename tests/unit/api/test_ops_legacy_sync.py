from __future__ import annotations

import pytest
from fastapi import HTTPException

from ispec.agent.commands import COMMAND_LEGACY_SYNC_ALL
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.api.routes.ops import LegacySyncRunRequest, run_legacy_sync
from ispec.db.models import AuthUser, UserRole


def _user(
    username: str,
    *,
    role: UserRole = UserRole.editor,
    user_id: int = 100,
) -> AuthUser:
    return AuthUser(
        id=user_id,
        username=username,
        password_hash="test",
        password_salt="test",
        role=role,
        is_active=True,
    )


def test_ops_legacy_sync_alex_can_enqueue(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS", raising=False)

    with get_agent_session(tmp_path / "agent.db") as agent_db:
        response = run_legacy_sync(
            request=LegacySyncRunRequest(
                dry_run=True,
                limit=10,
                max_project_comments=0,
            ),
            agent_db=agent_db,
            user=_user("alex", user_id=7),
        )

        assert response.ok is True
        assert response.queued is True
        assert response.status == "queued"

        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.id == response.command_id)
            .one()
        )
        assert cmd.command_type == COMMAND_LEGACY_SYNC_ALL
        assert cmd.status == "queued"
        assert cmd.max_attempts == 1
        assert cmd.payload_json["dry_run"] is True
        assert cmd.payload_json["limit"] == 10
        assert cmd.payload_json["max_project_comments"] == 0
        assert "reset_cursor" not in cmd.payload_json
        assert cmd.payload_json["meta"]["enqueued_by"] == "api_ops_legacy_sync"
        assert cmd.payload_json["meta"]["username"] == "alex"
        assert cmd.payload_json["meta"]["user_id"] == 7
        assert cmd.payload_json["meta"]["requested_at"]


def test_ops_legacy_sync_rejects_non_alex_staff_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS", raising=False)

    with get_agent_session(tmp_path / "agent.db") as agent_db:
        with pytest.raises(HTTPException) as excinfo:
            run_legacy_sync(
                request=LegacySyncRunRequest(),
                agent_db=agent_db,
                user=_user("casey", role=UserRole.admin),
            )

        assert excinfo.value.status_code == 403
        assert agent_db.query(AgentCommand).count() == 0


@pytest.mark.parametrize("role", [UserRole.viewer, UserRole.client])
def test_ops_legacy_sync_rejects_non_staff_even_when_allowlisted(
    tmp_path,
    monkeypatch,
    role,
):
    monkeypatch.delenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS", raising=False)

    with get_agent_session(tmp_path / "agent.db") as agent_db:
        with pytest.raises(HTTPException) as excinfo:
            run_legacy_sync(
                request=LegacySyncRunRequest(),
                agent_db=agent_db,
                user=_user("alex", role=role),
            )

        assert excinfo.value.status_code == 403
        assert agent_db.query(AgentCommand).count() == 0


def test_ops_legacy_sync_reuses_existing_queued_command(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_OPS_LEGACY_SYNC_ALLOWED_USERS", raising=False)

    with get_agent_session(tmp_path / "agent.db") as agent_db:
        existing = AgentCommand(
            command_type=COMMAND_LEGACY_SYNC_ALL,
            status="queued",
            payload_json={"meta": {"enqueued_by": "test"}},
            result_json={},
        )
        agent_db.add(existing)
        agent_db.commit()
        agent_db.refresh(existing)

        response = run_legacy_sync(
            request=LegacySyncRunRequest(reset_cursor=True),
            agent_db=agent_db,
            user=_user("alex"),
        )

        assert response.ok is True
        assert response.queued is False
        assert response.status == "existing"
        assert response.command_id == existing.id
        assert response.payload == {"meta": {"enqueued_by": "test"}}
        assert agent_db.query(AgentCommand).count() == 1

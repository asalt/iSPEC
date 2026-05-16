from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.cli import agent


def _args(**kwargs):
    defaults = {
        "database": None,
        "status": [],
        "command_type": [],
        "older_than_hours": 24.0,
        "limit": 50,
        "reason": "test cleanup",
        "confirm": False,
        "json": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_agent_queue_cancel_stale_dry_run_does_not_mutate(tmp_path, capsys):
    agent_db_path = tmp_path / "agent.db"
    old = datetime.now(UTC) - timedelta(days=3)

    with get_agent_session(agent_db_path) as db:
        db.add(
            AgentCommand(
                command_type="assistant_post_send_prepare_v1",
                status="queued",
                created_at=old,
                updated_at=old,
                available_at=old,
                payload_json={},
                result_json={},
            )
        )
        db.commit()

    agent._run_queue_cancel_stale(
        _args(database=str(agent_db_path), older_than_hours=24.0, confirm=False)
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert payload["selected"] == 1
    assert payload["cancelled"] == 0

    with get_agent_session(agent_db_path) as db:
        row = db.query(AgentCommand).one()
        assert row.status == "queued"
        assert row.error is None


def test_agent_queue_cancel_stale_marks_matching_rows_failed(tmp_path, capsys):
    agent_db_path = tmp_path / "agent.db"
    old = datetime.now(UTC) - timedelta(days=3)
    recent = datetime.now(UTC) - timedelta(minutes=5)

    with get_agent_session(agent_db_path) as db:
        stale = AgentCommand(
            command_type="assistant_post_send_prepare_v1",
            status="queued",
            created_at=old,
            updated_at=old,
            available_at=old,
            payload_json={},
            result_json={},
        )
        fresh = AgentCommand(
            command_type="assistant_post_send_prepare_v1",
            status="queued",
            created_at=recent,
            updated_at=recent,
            available_at=recent,
            payload_json={},
            result_json={},
        )
        db.add_all([stale, fresh])
        db.commit()
        stale_id = int(stale.id)
        fresh_id = int(fresh.id)

    agent._run_queue_cancel_stale(
        _args(database=str(agent_db_path), older_than_hours=24.0, confirm=True)
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is False
    assert payload["selected"] == 1
    assert payload["cancelled"] == 1

    with get_agent_session(agent_db_path) as db:
        stale = db.get(AgentCommand, stale_id)
        fresh = db.get(AgentCommand, fresh_id)
        assert stale is not None
        assert stale.status == "failed"
        assert stale.error == "cancelled_stale_queue: test cleanup"
        assert stale.result_json["cancelled"] is True
        assert stale.result_json["previous_status"] == "queued"
        assert fresh is not None
        assert fresh.status == "queued"


def test_agent_queue_list_filters_status_and_type(tmp_path, capsys):
    agent_db_path = tmp_path / "agent.db"
    now = datetime.now(UTC)

    with get_agent_session(agent_db_path) as db:
        db.add_all(
            [
                AgentCommand(
                    command_type="orchestrator_tick_v1",
                    status="queued",
                    created_at=now,
                    updated_at=now,
                    available_at=now,
                    payload_json={},
                    result_json={},
                ),
                AgentCommand(
                    command_type="slack_post_message_v1",
                    status="failed",
                    created_at=now,
                    updated_at=now,
                    available_at=now,
                    payload_json={},
                    result_json={},
                ),
            ]
        )
        db.commit()

    agent._run_queue_list(
        _args(
            database=str(agent_db_path),
            status=["queued"],
            command_type=["orchestrator_tick_v1"],
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["selected"] == 1
    assert payload["commands"][0]["command_type"] == "orchestrator_tick_v1"

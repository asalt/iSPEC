"""Separate processes share one atomic SQLite reservation, with no external IO."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import pytest

from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand, AgentEvent, RelayDedupKey
from ispec.agent.relay_store import enqueue_relay_request


def _submit(args):
    path, request_id = args
    with get_agent_session(path) as db:
        row, result = enqueue_relay_request(
            db,
            request={
                "kind": "slack_message",
                "target": "alex",
                "body": "A staged report",
                "request_id": request_id,
            },
        )
        return row.id, result["deduplicated"]


def _reserve_execution(args):
    from ispec.agent.relay_store import begin_relay_delivery

    path, command_id, request = args
    os.environ["ISPEC_AGENT_DB_PATH"] = path
    return begin_relay_delivery(command_id, request) is None


@pytest.mark.parametrize("same_id", [True, False])
def test_concurrent_processes_enqueue_once(tmp_path, same_id):
    path = str(tmp_path / "agent.db")
    # Deliberately do not pre-create the schema: first-start initialization must
    # also be safe when several independent clients arrive together.
    args = [(path, "same" if same_id else f"request-{i}") for i in range(12)]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_submit, args))
    assert len({command for command, duplicate in results}) == 1
    assert sum(not duplicate for command, duplicate in results) == 1
    with get_agent_session(path) as db:
        assert db.query(AgentCommand).count() == db.query(AgentEvent).count() == 1
        assert db.query(RelayDedupKey).count() == (2 if same_id else 13)
        assert all(key.command_id is not None for key in db.query(RelayDedupKey))


def test_concurrent_execution_reservations_have_one_winner(tmp_path):
    path = str(tmp_path / "agent.db")
    with get_agent_session(path) as db:
        row, payload = enqueue_relay_request(
            db,
            request={
                "kind": "slack_message",
                "mode": "send",
                "confirm": True,
                "target": "alex",
                "body": "A report",
                "request_id": "one",
            },
        )
        command_id = row.id
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        winners = list(
            pool.map(_reserve_execution, [(path, command_id, payload["request"])] * 8)
        )
    assert sum(winners) == 1

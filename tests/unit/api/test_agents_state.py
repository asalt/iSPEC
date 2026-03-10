from __future__ import annotations

import pytest

from ispec.agent_state.connect import get_agent_state_session
from ispec.api.routes.agents import (
    AgentStateObservationRequest,
    AgentStateSchemaUpsertRequest,
    get_state_heads,
    get_state_schema,
    observe_state,
    upsert_state_schema,
)


def test_agents_state_api_round_trip(tmp_path, monkeypatch) -> None:
    agent_state_db_path = tmp_path / "agent-state.db"
    monkeypatch.setenv("ISPEC_AGENT_STATE_DB_PATH", str(agent_state_db_path))

    with get_agent_state_session(agent_state_db_path) as agent_state_db:
        schema = upsert_state_schema(
            AgentStateSchemaUpsertRequest(
                schema_id=7,
                version=1,
                state_scope="mood",
                dims=[
                    {"dim_index": 0, "name": "caution"},
                    {"dim_index": 1, "name": "curiosity"},
                    {"dim_index": 2, "name": "patience"},
                ],
            ),
            db=agent_state_db,
        )
        assert schema.schema_id == 7
        assert schema.dim_count == 3

        fetched = get_state_schema(7, 1, db=agent_state_db)
        assert fetched.state_scope == "mood"

        observed = observe_state(
            AgentStateObservationRequest(
                schema_id=7,
                schema_version=1,
                state_scope="mood",
                agent_id="agent-1",
                job_id="job-42",
                step_index=3,
                vector=[0.2, 0.6, 0.9],
                source_kind="job_end",
                source_ref="job-42",
            ),
            db=agent_state_db,
        )
        assert observed.observation_id > 0
        assert observed.head is not None
        assert observed.head.vector[1] == pytest.approx(0.6, rel=1e-6)
        assert observed.schema_info.state_scope == "mood"

        heads = get_state_heads(agent_id="agent-1", state_scope="mood", limit=5, db=agent_state_db)
        assert len(heads.heads) == 1
        assert heads.heads[0].dim_names == ["caution", "curiosity", "patience"]

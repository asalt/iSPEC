from __future__ import annotations

import pytest

from ispec.agent_state.connect import get_agent_state_session
from ispec.agent_state.store import append_observation, get_schema, list_heads, register_schema_version


def test_agent_state_store_round_trip_schema_and_head(tmp_path):
    db_path = tmp_path / "agent-state.db"

    with get_agent_state_session(db_path) as db:
        schema = register_schema_version(
            db,
            schema_id=1,
            version=1,
            state_scope="mood",
            dims=[
                {"dim_index": 0, "name": "caution"},
                {"dim_index": 1, "name": "curiosity"},
                {"dim_index": 2, "name": "patience"},
            ],
            notes="initial mood schema",
        )
        assert schema["dim_count"] == 3

        observation = append_observation(
            db,
            schema_id=1,
            schema_version=1,
            state_scope="mood",
            agent_id="agent-1",
            vector=[0.25, 0.5, 0.75],
            reward=1.0,
            source_kind="job_end",
            source_ref="job-123",
        )
        assert observation["head_updated"] is True
        assert observation["observation_id"] > 0

        fetched = get_schema(db, schema_id=1, version=1)
        assert fetched is not None
        assert fetched["dims"][1]["name"] == "curiosity"

        heads = list_heads(db, agent_id="agent-1", state_scope="mood", limit=5)
        assert len(heads) == 1
        assert heads[0]["dim_names"] == ["caution", "curiosity", "patience"]
        assert heads[0]["vector"] == pytest.approx([0.25, 0.5, 0.75], rel=1e-6)


def test_agent_state_store_rejects_vector_length_mismatch(tmp_path):
    db_path = tmp_path / "agent-state.db"

    with get_agent_state_session(db_path) as db:
        register_schema_version(
            db,
            schema_id=2,
            version=1,
            state_scope="mood",
            dims=[
                {"dim_index": 0, "name": "caution"},
                {"dim_index": 1, "name": "curiosity"},
            ],
        )

        with pytest.raises(ValueError, match="Vector length"):
            append_observation(
                db,
                schema_id=2,
                schema_version=1,
                state_scope="mood",
                agent_id="agent-1",
                vector=[0.1, 0.2, 0.3],
            )

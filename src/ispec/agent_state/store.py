from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
import math
import struct
from typing import Any

from sqlalchemy.orm import Session

from .models import AgentStateHead, AgentStateObservation, AgentStateSchemaDim, AgentStateSchemaVersion


_SUPPORTED_CODECS = {"f32le"}


def utcnow_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_scope(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("state_scope is required.")
    return text


def _normalize_codec(value: str) -> str:
    codec = str(value or "").strip().lower() or "f32le"
    if codec not in _SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec: {codec}. Supported: {sorted(_SUPPORTED_CODECS)}")
    return codec


def _normalize_vector(values: Sequence[float | int]) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("vector must be a sequence of numbers.")
    out: list[float] = []
    for idx, raw in enumerate(values):
        try:
            value = float(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"vector[{idx}] is not numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"vector[{idx}] must be finite.")
        out.append(value)
    if not out:
        raise ValueError("vector must not be empty.")
    return out


def encode_vector(values: Sequence[float | int], *, codec: str = "f32le") -> bytes:
    codec = _normalize_codec(codec)
    vector = _normalize_vector(values)
    if codec == "f32le":
        return struct.pack("<" + ("f" * len(vector)), *vector)
    raise ValueError(f"Unsupported codec: {codec}")


def decode_vector(blob: bytes, *, codec: str = "f32le") -> list[float]:
    codec = _normalize_codec(codec)
    if codec != "f32le":
        raise ValueError(f"Unsupported codec: {codec}")
    if len(blob) % 4 != 0:
        raise ValueError("Invalid f32le blob length.")
    if not blob:
        return []
    count = len(blob) // 4
    return [float(x) for x in struct.unpack("<" + ("f" * count), blob)]


def _schema_header(db: Session, *, schema_id: int, version: int) -> AgentStateSchemaVersion | None:
    return (
        db.query(AgentStateSchemaVersion)
        .filter(
            AgentStateSchemaVersion.schema_id == int(schema_id),
            AgentStateSchemaVersion.version == int(version),
        )
        .first()
    )


def _schema_dims(db: Session, *, schema_id: int, version: int) -> list[AgentStateSchemaDim]:
    return (
        db.query(AgentStateSchemaDim)
        .filter(
            AgentStateSchemaDim.schema_id == int(schema_id),
            AgentStateSchemaDim.version == int(version),
        )
        .order_by(AgentStateSchemaDim.dim_index.asc())
        .all()
    )


def _schema_payload(
    header: AgentStateSchemaVersion,
    dims: Iterable[AgentStateSchemaDim],
) -> dict[str, Any]:
    dim_rows = list(dims)
    return {
        "schema_id": int(header.schema_id),
        "version": int(header.version),
        "state_scope": str(header.state_scope),
        "dim_count": int(header.dim_count),
        "codec": str(header.codec),
        "created_at": header.created_at.isoformat() if getattr(header, "created_at", None) else None,
        "notes": header.notes,
        "dims": [
            {
                "dim_index": int(row.dim_index),
                "name": str(row.name),
                "description": row.description,
            }
            for row in dim_rows
        ],
    }


def get_schema(db: Session, *, schema_id: int, version: int) -> dict[str, Any] | None:
    header = _schema_header(db, schema_id=int(schema_id), version=int(version))
    if header is None:
        return None
    dims = _schema_dims(db, schema_id=int(schema_id), version=int(version))
    return _schema_payload(header, dims)


def register_schema_version(
    db: Session,
    *,
    schema_id: int,
    version: int,
    state_scope: str,
    dims: Sequence[dict[str, Any]],
    codec: str = "f32le",
    notes: str | None = None,
) -> dict[str, Any]:
    if int(schema_id) <= 0:
        raise ValueError("schema_id must be > 0.")
    if int(version) <= 0:
        raise ValueError("version must be > 0.")

    normalized_scope = _normalize_scope(state_scope)
    normalized_codec = _normalize_codec(codec)
    normalized_notes = _clean_optional_text(notes)

    dim_rows: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for raw in dims:
        if not isinstance(raw, dict):
            raise ValueError("dims must be a sequence of dicts.")
        idx = int(raw.get("dim_index"))
        if idx < 0:
            raise ValueError("dim_index must be >= 0.")
        if idx in seen_indexes:
            raise ValueError(f"Duplicate dim_index: {idx}")
        seen_indexes.add(idx)
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"Dimension {idx} is missing a name.")
        dim_rows.append(
            {
                "dim_index": idx,
                "name": name,
                "description": _clean_optional_text(raw.get("description")),
            }
        )
    dim_rows.sort(key=lambda row: int(row["dim_index"]))
    expected_indexes = list(range(len(dim_rows)))
    actual_indexes = [int(row["dim_index"]) for row in dim_rows]
    if actual_indexes != expected_indexes:
        raise ValueError(f"dim_index values must be contiguous starting at 0 (got {actual_indexes}).")

    header = _schema_header(db, schema_id=int(schema_id), version=int(version))
    existing_dims = _schema_dims(db, schema_id=int(schema_id), version=int(version)) if header is not None else []
    if header is not None:
        payload = _schema_payload(header, existing_dims)
        if payload["state_scope"] != normalized_scope:
            raise ValueError("Existing schema has a different state_scope.")
        if payload["codec"] != normalized_codec:
            raise ValueError("Existing schema has a different codec.")
        if int(payload["dim_count"]) != len(dim_rows):
            raise ValueError("Existing schema has a different dim_count.")
        existing_dim_rows = payload["dims"]
        if existing_dim_rows != dim_rows:
            raise ValueError("Existing schema dimensions do not match.")
        if normalized_notes is not None and normalized_notes != payload["notes"]:
            raise ValueError("Existing schema notes do not match.")
        return payload

    header = AgentStateSchemaVersion(
        schema_id=int(schema_id),
        version=int(version),
        state_scope=normalized_scope,
        dim_count=len(dim_rows),
        codec=normalized_codec,
        notes=normalized_notes,
    )
    db.add(header)
    db.flush()
    for row in dim_rows:
        db.add(
            AgentStateSchemaDim(
                schema_id=int(schema_id),
                version=int(version),
                dim_index=int(row["dim_index"]),
                name=str(row["name"]),
                description=row["description"],
            )
        )
    db.flush()
    return _schema_payload(header, _schema_dims(db, schema_id=int(schema_id), version=int(version)))


def append_observation(
    db: Session,
    *,
    schema_id: int,
    schema_version: int,
    state_scope: str,
    vector: Sequence[float | int],
    ts_ms: int | None = None,
    agent_id: str | None = None,
    job_id: str | None = None,
    task_id: str | None = None,
    step_index: int | None = None,
    reward: float | None = None,
    source_kind: str | None = None,
    source_ref: str | None = None,
    update_head: bool = True,
) -> dict[str, Any]:
    schema = get_schema(db, schema_id=int(schema_id), version=int(schema_version))
    if schema is None:
        raise ValueError(f"Unknown schema ({schema_id}, version={schema_version}).")

    normalized_scope = _normalize_scope(state_scope)
    if normalized_scope != str(schema["state_scope"]):
        raise ValueError("Observation state_scope does not match schema state_scope.")

    normalized_vector = _normalize_vector(vector)
    if len(normalized_vector) != int(schema["dim_count"]):
        raise ValueError(
            f"Vector length {len(normalized_vector)} does not match schema dim_count {schema['dim_count']}."
        )

    normalized_reward = None
    if reward is not None:
        normalized_reward = float(reward)
        if not math.isfinite(normalized_reward):
            raise ValueError("reward must be finite.")

    normalized_agent_id = _clean_optional_text(agent_id)
    normalized_job_id = _clean_optional_text(job_id)
    normalized_task_id = _clean_optional_text(task_id)
    normalized_source_kind = _clean_optional_text(source_kind)
    normalized_source_ref = _clean_optional_text(source_ref)
    normalized_step_index = int(step_index) if step_index is not None else None
    if normalized_step_index is not None and normalized_step_index < 0:
        raise ValueError("step_index must be >= 0.")

    payload_blob = encode_vector(normalized_vector, codec=str(schema["codec"]))
    row = AgentStateObservation(
        ts_ms=int(ts_ms) if ts_ms is not None else utcnow_ms(),
        agent_id=normalized_agent_id,
        job_id=normalized_job_id,
        task_id=normalized_task_id,
        step_index=normalized_step_index,
        schema_id=int(schema_id),
        schema_version=int(schema_version),
        state_scope=normalized_scope,
        vector_blob=payload_blob,
        reward=normalized_reward,
        source_kind=normalized_source_kind,
        source_ref=normalized_source_ref,
    )
    db.add(row)
    db.flush()

    head_updated = False
    if update_head and normalized_agent_id:
        head = (
            db.query(AgentStateHead)
            .filter(
                AgentStateHead.agent_id == normalized_agent_id,
                AgentStateHead.state_scope == normalized_scope,
            )
            .first()
        )
        if head is None:
            head = AgentStateHead(
                agent_id=normalized_agent_id,
                state_scope=normalized_scope,
                schema_id=int(schema_id),
                schema_version=int(schema_version),
                ts_ms=int(row.ts_ms),
                vector_blob=payload_blob,
                observation_id=int(row.id),
            )
            db.add(head)
        else:
            head.schema_id = int(schema_id)
            head.schema_version = int(schema_version)
            head.ts_ms = int(row.ts_ms)
            head.vector_blob = payload_blob
            head.observation_id = int(row.id)
        db.flush()
        head_updated = True

    return {
        "observation_id": int(row.id),
        "head_updated": head_updated,
        "schema": schema,
        "vector": normalized_vector,
    }


def list_heads(
    db: Session,
    *,
    agent_id: str | None = None,
    state_scope: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    query = db.query(AgentStateHead).order_by(AgentStateHead.ts_ms.desc())
    normalized_agent_id = _clean_optional_text(agent_id)
    if normalized_agent_id is not None:
        query = query.filter(AgentStateHead.agent_id == normalized_agent_id)
    normalized_scope = _clean_optional_text(state_scope)
    if normalized_scope is not None:
        query = query.filter(AgentStateHead.state_scope == _normalize_scope(normalized_scope))

    rows = query.limit(max(1, int(limit))).all()
    out: list[dict[str, Any]] = []
    for row in rows:
        schema = get_schema(db, schema_id=int(row.schema_id), version=int(row.schema_version))
        codec = str(schema["codec"]) if isinstance(schema, dict) else "f32le"
        dim_names = [str(item.get("name") or "") for item in (schema or {}).get("dims", [])]
        out.append(
            {
                "agent_id": str(row.agent_id),
                "state_scope": str(row.state_scope),
                "schema_id": int(row.schema_id),
                "schema_version": int(row.schema_version),
                "ts_ms": int(row.ts_ms),
                "observation_id": int(row.observation_id) if row.observation_id is not None else None,
                "vector": decode_vector(bytes(row.vector_blob), codec=codec),
                "dim_names": dim_names,
            }
        )
    return out

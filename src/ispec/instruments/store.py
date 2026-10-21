from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ispec.agent.models import InstrumentObservationRow
from .schema import InstrumentObservation, as_utc


class ObservationConflict(ValueError):
    pass


def record_observation(
    db: Session, observation: InstrumentObservation, *, now: datetime
) -> dict:
    payload = observation.model_dump(mode="json")
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    event_id = str(observation.event_id)
    result = db.execute(
        insert(InstrumentObservationRow)
        .values(
            event_id=event_id,
            collector_id=observation.collector_id,
            instrument_id=observation.instrument_id,
            observed_at=as_utc(observation.observed_at),
            received_at=now,
            payload_hash=fingerprint,
            payload_json=payload,
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
    )
    row = db.get(InstrumentObservationRow, event_id)
    if row.payload_hash != fingerprint:
        raise ObservationConflict("Event ID already exists with different evidence.")
    db.commit()
    return {
        "event_id": event_id,
        "stored": True,
        "duplicate": result.rowcount == 0,
        "received_at": as_utc(row.received_at).isoformat(),
    }


def latest_status(
    db: Session, instrument_id: str, *, now: datetime, stale_after_seconds: int = 180
) -> dict | None:
    # Source observation order, not HTTP arrival order, selects the latest state.
    row = db.scalars(
        select(InstrumentObservationRow)
        .where(
            InstrumentObservationRow.instrument_id == instrument_id,
        )
        .order_by(
            InstrumentObservationRow.observed_at.desc(),
            InstrumentObservationRow.received_at.desc(),
        )
        .limit(1)
    ).first()
    if row is None:
        return None
    now = as_utc(now)
    observation_age = (now - as_utc(row.observed_at)).total_seconds()
    receipt_age = (now - as_utc(row.received_at)).total_seconds()
    payload = dict(row.payload_json)
    updated = payload.get("source_updated_at")
    source_age = (
        (now - datetime.fromisoformat(updated).astimezone(UTC)).total_seconds()
        if updated
        else None
    )
    return {
        "instrument_id": instrument_id,
        "observation": payload,
        "received_at": as_utc(row.received_at).isoformat(),
        "observation_age_seconds": observation_age,
        "receipt_age_seconds": receipt_age,
        "source_age_seconds": source_age,
        "collector_stale": observation_age > stale_after_seconds
        or receipt_age > stale_after_seconds,
        "source_stale": None
        if source_age is None
        else source_age > stale_after_seconds,
        "clock_in_future": observation_age < 0
        or (source_age is not None and source_age < 0),
        "instrument_alive": None,
        "read_only": True,
        "note": "File metadata is not acquisition state. A fresh collector does not prove a fresh source or healthy instrument.",
    }

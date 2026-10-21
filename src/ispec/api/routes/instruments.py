from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ispec.agent.connect import get_agent_session_dep
from ispec.api.security import require_staff
from ispec.instruments.auth import CollectorIdentity, require_collector
from ispec.instruments.schema import InstrumentObservation
from ispec.instruments.store import (
    ObservationConflict,
    latest_status,
    record_observation,
)


router = APIRouter(prefix="/instruments", tags=["Instrument observations"])


async def read_observation(
    request: Request, identity: CollectorIdentity = Depends(require_collector)
) -> InstrumentObservation:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
            raise HTTPException(413, "Observation exceeds 64 KiB.")
    try:
        observation = InstrumentObservation.model_validate(json.loads(body))
    except (ValueError, ValidationError):
        raise HTTPException(422, "Invalid instrument observation schema.") from None
    if (
        observation.collector_id != identity.collector_id
        or observation.instrument_id not in identity.instrument_ids
    ):
        raise HTTPException(
            403, "Collector is not authorized for this identity/instrument."
        )
    if observation.observed_at > datetime.now(UTC) + timedelta(seconds=60):
        raise HTTPException(
            422, "Collector clock is more than 60 seconds in the future."
        )
    return observation


@router.post("/observations")
def ingest_observation(
    observation: InstrumentObservation = Depends(read_observation),
    db: Session = Depends(get_agent_session_dep),
) -> dict:
    try:
        return record_observation(db, observation, now=datetime.now(UTC))
    except ObservationConflict as exc:
        db.rollback()
        raise HTTPException(409, str(exc)) from None


@router.get("/{instrument_id}/status", dependencies=[Depends(require_staff)])
def instrument_status(
    instrument_id: str,
    stale_after_seconds: int = Query(default=180, ge=30, le=86400),
    db: Session = Depends(get_agent_session_dep),
) -> dict:
    result = latest_status(
        db,
        instrument_id,
        now=datetime.now(UTC),
        stale_after_seconds=stale_after_seconds,
    )
    if result is None:
        raise HTTPException(404, "No observations for this instrument.")
    return result

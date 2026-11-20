from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import StaleDataError

from ispec.agent.connect import get_agent_session_dep
from ispec.api.security import require_staff
from ispec.db.connect import get_session_dep
from ispec.db.models import InstrumentRegistry
from ispec.instruments import dashboard, registry
from ispec.timeutil import format_utc, utcnow
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
    core: Session = Depends(get_session_dep),
) -> dict:
    result = latest_status(
        db,
        instrument_id,
        now=datetime.now(UTC),
        stale_after_seconds=stale_after_seconds,
    )
    row = core.get(InstrumentRegistry, instrument_id)
    if row:
        return {
            **(result or {}),
            "instrument_id": row.id,
            "generated_at": format_utc(utcnow()),
            "instrument": registry.record(row),
            **dashboard.card(row),
        }
    if result is None:
        raise HTTPException(404, "No observations for this instrument.")
    return result


@router.get("", dependencies=[Depends(require_staff)])
def instruments(core: Session = Depends(get_session_dep)):
    rows = core.query(InstrumentRegistry).order_by(InstrumentRegistry.id)
    return {
        "generated_at": format_utc(utcnow()),
        "instruments": [
            {**registry.record(row), **dashboard.card(row)} for row in rows
        ],
    }


@router.get("/{instrument_id}/runs", dependencies=[Depends(require_staff)])
def instrument_runs(
    instrument_id: str,
    limit: int = Query(25, ge=1, le=100),
    cursor: str | None = Query(None, max_length=512),
    core: Session = Depends(get_session_dep),
):
    row = core.get(InstrumentRegistry, instrument_id)
    if row is None:
        raise HTTPException(404, "Unknown instrument")
    try:
        page = dashboard.read_archive(row, limit=limit, cursor=cursor, runs=True)
    except dashboard.InvalidCursor as exc:
        raise HTTPException(422, str(exc)) from None
    except dashboard.ArchiveError as exc:
        raise HTTPException(503, str(exc)) from None
    return {"instrument_id": row.id, "generated_at": format_utc(utcnow()), **page}


@router.post("", status_code=201, dependencies=[Depends(require_staff)])
def create_instrument(entry: registry.Entry, core: Session = Depends(get_session_dep)):
    try:
        return registry.record(registry.add_entry(core, entry))
    except IntegrityError:
        raise HTTPException(409, "Instrument ID already exists") from None


@router.patch("/{instrument_id}", dependencies=[Depends(require_staff)])
def edit_instrument(
    instrument_id: str, changes: dict, core: Session = Depends(get_session_dep)
):
    row = core.get(InstrumentRegistry, instrument_id)
    if not row:
        raise HTTPException(404, "Unknown instrument")
    if changes.pop("revision", row.revision) != row.revision:
        raise HTTPException(409, "Registry entry changed; reload before saving")
    try:
        return registry.record(registry.change_entry(core, row, changes))
    except ValidationError:
        raise HTTPException(422, "Invalid instrument settings or source path") from None
    except StaleDataError:
        raise HTTPException(
            409, "Registry entry changed; reload before saving"
        ) from None

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ispec.api.security import require_access, require_admin
from ispec.schedule.connect import get_schedule_session_dep
from ispec.schedule.models import ScheduleRequest, ScheduleRequestSlot, ScheduleSlot

router = APIRouter(prefix="/schedule", tags=["Schedule"])

CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

SLOT_STATUSES = {"available", "booked", "closed"}
REQUEST_STATUSES = {"requested", "confirmed", "declined", "cancelled"}


class SlotResponse(BaseModel):
    id: int
    start_at: datetime
    end_at: datetime
    status: str

    model_config = {"from_attributes": True}


class SlotCreate(BaseModel):
    start_at: datetime
    end_at: datetime
    status: str = Field(default="available")


class SlotUpdate(BaseModel):
    start_at: datetime | None = None
    end_at: datetime | None = None
    status: str | None = None


class SlotBulkCreate(BaseModel):
    start_date: date
    end_date: date
    start_time: str = Field(default="09:15")
    end_time: str = Field(default="16:15")
    slot_minutes: int = Field(default=45, ge=15, le=180)
    interval_minutes: int = Field(default=60, ge=15, le=240)
    days: list[int] | None = None
    status: str = Field(default="available")


class ScheduleRequestCreate(BaseModel):
    requester_name: str = Field(min_length=1, max_length=200)
    requester_email: str = Field(min_length=3, max_length=320)
    requester_org: str | None = Field(default=None, max_length=200)
    requester_phone: str | None = Field(default=None, max_length=80)
    project_title: str | None = Field(default=None, max_length=200)
    project_description: str = Field(min_length=1, max_length=5000)
    cancer_related: bool = Field(default=False)
    slot_ids: list[int] = Field(min_length=1, max_length=3)


class ScheduleRequestResponse(BaseModel):
    id: int
    status: str
    created_at: datetime
    slot_ids: list[int]

    model_config = {"from_attributes": True}


class SlotBulkResponse(BaseModel):
    inserted: int
    skipped: int


class SlotListResponse(BaseModel):
    items: list[SlotResponse]


class ScheduleRequestListResponse(BaseModel):
    id: int
    requester_name: str
    requester_email: str
    status: str
    created_at: datetime
    slot_ids: list[int]

    model_config = {"from_attributes": True}


def _normalize_status(value: str, *, allowed: set[str]) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")
    return normalized


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=CENTRAL_TZ)
    return value.astimezone(UTC_TZ).replace(tzinfo=None)


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC_TZ)
    return value.astimezone(UTC_TZ)


def _parse_time(value: str) -> time:
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError("time must be in HH:MM 24h format") from exc


def _range_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(start, time.min, tzinfo=CENTRAL_TZ)
    end_local = datetime.combine(end, time.max, tzinfo=CENTRAL_TZ)
    return _as_utc_naive(start_local), _as_utc_naive(end_local)


def _serialize_slot(slot: ScheduleSlot) -> SlotResponse:
    return SlotResponse(
        id=slot.id,
        start_at=_as_utc_aware(slot.start_at),
        end_at=_as_utc_aware(slot.end_at),
        status=slot.status,
    )


def _default_range() -> tuple[date, date]:
    today_local = datetime.now(CENTRAL_TZ).date()
    return today_local, today_local + timedelta(days=28)


@router.get("/slots", response_model=SlotListResponse)
def list_slots(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    db: Session = Depends(get_schedule_session_dep),
):
    if start is None or end is None:
        start, end = _default_range()
    if end < start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    start_utc, end_utc = _range_bounds(start, end)
    rows = (
        db.query(ScheduleSlot)
        .filter(ScheduleSlot.start_at >= start_utc)
        .filter(ScheduleSlot.start_at <= end_utc)
        .order_by(ScheduleSlot.start_at.asc())
        .all()
    )
    return {"items": [_serialize_slot(row) for row in rows]}


@router.post(
    "/slots",
    response_model=SlotResponse,
    dependencies=[Depends(require_access), Depends(require_admin)],
)
def create_slot(payload: SlotCreate, db: Session = Depends(get_schedule_session_dep)):
    try:
        status = _normalize_status(payload.status, allowed=SLOT_STATUSES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_at = _as_utc_naive(payload.start_at)
    end_at = _as_utc_naive(payload.end_at)
    if end_at <= start_at:
        raise HTTPException(status_code=400, detail="end_at must be after start_at")

    slot = ScheduleSlot(
        start_at=start_at,
        end_at=end_at,
        status=status,
    )
    db.add(slot)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="slot already exists") from exc
    return _serialize_slot(slot)


@router.put(
    "/slots/{slot_id}",
    response_model=SlotResponse,
    dependencies=[Depends(require_access), Depends(require_admin)],
)
def update_slot(
    slot_id: int,
    payload: SlotUpdate,
    db: Session = Depends(get_schedule_session_dep),
):
    slot = db.query(ScheduleSlot).filter(ScheduleSlot.id == slot_id).first()
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")

    updated_start_at = slot.start_at
    updated_end_at = slot.end_at

    if payload.status is not None:
        try:
            slot.status = _normalize_status(payload.status, allowed=SLOT_STATUSES)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.start_at is not None:
        updated_start_at = _as_utc_naive(payload.start_at)
    if payload.end_at is not None:
        updated_end_at = _as_utc_naive(payload.end_at)

    if updated_end_at <= updated_start_at:
        raise HTTPException(status_code=400, detail="end_at must be after start_at")

    slot.start_at = updated_start_at
    slot.end_at = updated_end_at

    db.add(slot)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="slot time conflicts with existing slot") from exc
    return _serialize_slot(slot)


@router.post(
    "/slots/bulk",
    response_model=SlotBulkResponse,
    dependencies=[Depends(require_access), Depends(require_admin)],
)
def bulk_create_slots(
    payload: SlotBulkCreate,
    db: Session = Depends(get_schedule_session_dep),
):
    if payload.end_date < payload.start_date:
        raise HTTPException(status_code=400, detail="end_date must be on or after start_date")

    try:
        status = _normalize_status(payload.status, allowed=SLOT_STATUSES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = _parse_time(payload.start_time)
    end_time = _parse_time(payload.end_time)
    if end_time < start_time:
        raise HTTPException(status_code=400, detail="end_time must be on or after start_time")

    if payload.interval_minutes < payload.slot_minutes:
        raise HTTPException(status_code=400, detail="interval_minutes must be >= slot_minutes")

    days = payload.days
    if days is not None:
        invalid_days = [d for d in days if d < 0 or d > 6]
        if invalid_days:
            raise HTTPException(status_code=400, detail="days must be between 0 and 6")

    start_utc, end_utc = _range_bounds(payload.start_date, payload.end_date)
    existing = (
        db.query(ScheduleSlot.start_at, ScheduleSlot.end_at)
        .filter(ScheduleSlot.start_at >= start_utc)
        .filter(ScheduleSlot.start_at <= end_utc)
        .all()
    )
    existing_pairs = {(row[0], row[1]) for row in existing}

    inserted = 0
    skipped = 0

    current = payload.start_date
    while current <= payload.end_date:
        if days is not None and current.weekday() not in days:
            current += timedelta(days=1)
            continue

        first_start = datetime.combine(current, start_time, tzinfo=CENTRAL_TZ)
        last_start = datetime.combine(current, end_time, tzinfo=CENTRAL_TZ)
        slot_start = first_start

        while slot_start <= last_start:
            slot_end = slot_start + timedelta(minutes=payload.slot_minutes)
            start_utc = _as_utc_naive(slot_start)
            end_utc = _as_utc_naive(slot_end)
            if (start_utc, end_utc) in existing_pairs:
                skipped += 1
            else:
                db.add(
                    ScheduleSlot(
                        start_at=start_utc,
                        end_at=end_utc,
                        status=status,
                    )
                )
                existing_pairs.add((start_utc, end_utc))
                inserted += 1
            slot_start += timedelta(minutes=payload.interval_minutes)

        current += timedelta(days=1)

    db.flush()
    return SlotBulkResponse(inserted=inserted, skipped=skipped)


@router.post("/requests", response_model=ScheduleRequestResponse)
def create_request(
    payload: ScheduleRequestCreate,
    db: Session = Depends(get_schedule_session_dep),
):
    slot_ids = []
    for slot_id in payload.slot_ids:
        if slot_id not in slot_ids:
            slot_ids.append(slot_id)
    if len(slot_ids) > 3:
        raise HTTPException(status_code=400, detail="up to 3 slots can be selected")

    slots = db.query(ScheduleSlot).filter(ScheduleSlot.id.in_(slot_ids)).all()
    if len(slots) != len(slot_ids):
        raise HTTPException(status_code=404, detail="one or more slots not found")

    unavailable = [slot.id for slot in slots if slot.status != "available"]
    if unavailable:
        raise HTTPException(
            status_code=409,
            detail=f"slots not available: {', '.join(str(s) for s in unavailable)}",
        )

    request = ScheduleRequest(
        requester_name=payload.requester_name,
        requester_email=payload.requester_email,
        requester_org=payload.requester_org,
        requester_phone=payload.requester_phone,
        project_title=payload.project_title,
        project_description=payload.project_description,
        cancer_related=payload.cancer_related,
        status="requested",
    )
    db.add(request)
    db.flush()

    for idx, slot_id in enumerate(slot_ids, start=1):
        db.add(
            ScheduleRequestSlot(
                request_id=request.id,
                slot_id=slot_id,
                rank=idx,
            )
        )

    for slot in slots:
        slot.status = "booked"
        db.add(slot)

    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="one or more slots were already booked; refresh and try again",
        ) from exc
    return ScheduleRequestResponse(
        id=request.id,
        status=request.status,
        created_at=_as_utc_aware(request.created_at),
        slot_ids=slot_ids,
    )


@router.get(
    "/requests",
    response_model=list[ScheduleRequestListResponse],
    dependencies=[Depends(require_access), Depends(require_admin)],
)
def list_requests(db: Session = Depends(get_schedule_session_dep)):
    rows = db.query(ScheduleRequest).order_by(ScheduleRequest.created_at.desc()).all()
    payload = []
    for row in rows:
        slot_ids = [link.slot_id for link in sorted(row.slots, key=lambda s: s.rank)]
        payload.append(
            ScheduleRequestListResponse(
                id=row.id,
                requester_name=row.requester_name,
                requester_email=row.requester_email,
                status=row.status,
                created_at=_as_utc_aware(row.created_at),
                slot_ids=slot_ids,
            )
        )
    return payload


@router.delete(
    "/slots/{slot_id}",
    status_code=204,
    dependencies=[Depends(require_access), Depends(require_admin)],
)
def delete_slot(slot_id: int, db: Session = Depends(get_schedule_session_dep)):
    slot = db.query(ScheduleSlot).filter(ScheduleSlot.id == slot_id).first()
    if slot is None:
        raise HTTPException(status_code=404, detail="slot not found")

    if slot.status == "booked":
        raise HTTPException(status_code=409, detail="booked slots cannot be deleted")

    linked = db.query(ScheduleRequestSlot).filter(ScheduleRequestSlot.slot_id == slot_id).first()
    if linked is not None:
        raise HTTPException(
            status_code=409,
            detail="slot is linked to a request and cannot be deleted",
        )

    db.delete(slot)
    db.flush()

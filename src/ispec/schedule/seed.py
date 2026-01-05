from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from ispec.schedule.connect import get_schedule_session
from ispec.schedule.models import ScheduleSlot


CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

SLOT_STATUSES = {"available", "booked", "closed"}
BUSINESS_START = time(9, 0)
BUSINESS_END = time(17, 0)


def _normalize_status(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in SLOT_STATUSES:
        raise ValueError(f"status must be one of {sorted(SLOT_STATUSES)}")
    return normalized


def _parse_time(value: str) -> time:
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError("time must be in HH:MM 24h format") from exc


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=CENTRAL_TZ)
    return value.astimezone(UTC_TZ).replace(tzinfo=None)


def _range_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(start, time.min, tzinfo=CENTRAL_TZ)
    end_local = datetime.combine(end, time.max, tzinfo=CENTRAL_TZ)
    return _as_utc_naive(start_local), _as_utc_naive(end_local)


def seed_slots(
    db: Session,
    *,
    start_date: date,
    end_date: date,
    start_time: str,
    end_time: str,
    slot_minutes: int,
    interval_minutes: int,
    days: list[int] | None,
    status: str,
) -> tuple[int, int]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    status_value = _normalize_status(status)

    start_t = _parse_time(start_time)
    end_t = _parse_time(end_time)
    if end_t < start_t:
        raise ValueError("end_time must be on or after start_time")

    if interval_minutes < slot_minutes:
        raise ValueError("interval_minutes must be >= slot_minutes")

    if days is not None:
        invalid_days = [d for d in days if d < 0 or d > 6]
        if invalid_days:
            raise ValueError("days must be between 0 and 6")

    start_utc, end_utc = _range_bounds(start_date, end_date)
    existing = (
        db.query(ScheduleSlot.start_at, ScheduleSlot.end_at)
        .filter(ScheduleSlot.start_at >= start_utc)
        .filter(ScheduleSlot.start_at <= end_utc)
        .all()
    )
    existing_pairs = {(row[0], row[1]) for row in existing}

    inserted = 0
    skipped = 0

    current = start_date
    while current <= end_date:
        if days is not None and current.weekday() not in days:
            current += timedelta(days=1)
            continue

        first_start = datetime.combine(current, start_t, tzinfo=CENTRAL_TZ)
        last_start = datetime.combine(current, end_t, tzinfo=CENTRAL_TZ)
        business_first = datetime.combine(current, BUSINESS_START, tzinfo=CENTRAL_TZ)
        business_last_start = datetime.combine(current, BUSINESS_END, tzinfo=CENTRAL_TZ) - timedelta(
            minutes=slot_minutes
        )

        slot_start = max(first_start, business_first)
        last_start = min(last_start, business_last_start)
        if last_start < slot_start:
            current += timedelta(days=1)
            continue

        while slot_start <= last_start:
            slot_end = slot_start + timedelta(minutes=slot_minutes)
            start_utc = _as_utc_naive(slot_start)
            end_utc = _as_utc_naive(slot_end)
            if (start_utc, end_utc) in existing_pairs:
                skipped += 1
            else:
                db.add(
                    ScheduleSlot(
                        start_at=start_utc,
                        end_at=end_utc,
                        status=status_value,
                    )
                )
                existing_pairs.add((start_utc, end_utc))
                inserted += 1
            slot_start += timedelta(minutes=interval_minutes)

        current += timedelta(days=1)

    db.flush()
    return inserted, skipped


def _default_range() -> tuple[date, date]:
    today_local = datetime.now(CENTRAL_TZ).date()
    return today_local, today_local + timedelta(days=28)


def _parse_days(value: str) -> list[int]:
    raw = (value or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    days: list[int] = []
    for part in parts:
        try:
            days.append(int(part))
        except ValueError as exc:
            raise ValueError("days must be a comma-separated list of integers (0=Mon .. 6=Sun)") from exc
    return days


def build_parser() -> argparse.ArgumentParser:
    start_default, end_default = _default_range()
    parser = argparse.ArgumentParser(
        prog="python -m ispec.schedule.seed",
        description="Seed fake schedule slots into the iSPEC schedule database.",
    )
    parser.add_argument("--db", dest="db_path", default=None, help="Optional schedule DB path/URI.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=start_default)
    parser.add_argument("--end-date", type=date.fromisoformat, default=end_default)
    parser.add_argument("--start-time", default="09:15", help="Local start time (HH:MM).")
    parser.add_argument("--end-time", default="16:15", help="Local last slot start time (HH:MM).")
    parser.add_argument("--slot-minutes", type=int, default=45)
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument(
        "--days",
        default="0,1,2,3,4",
        help="Comma-separated weekdays (0=Mon .. 6=Sun). Default: weekdays.",
    )
    parser.add_argument("--status", default="available", choices=sorted(SLOT_STATUSES))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        days_raw = _parse_days(args.days)
        days = days_raw or None
        with get_schedule_session(args.db_path) as db:
            inserted, skipped = seed_slots(
                db,
                start_date=args.start_date,
                end_date=args.end_date,
                start_time=args.start_time,
                end_time=args.end_time,
                slot_minutes=args.slot_minutes,
                interval_minutes=args.interval_minutes,
                days=days,
                status=args.status,
            )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    print(f"Seeded schedule slots: inserted={inserted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""UTC clock and sortable timestamps for new storage and portable receipts."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("A source timezone is required")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(
        value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    )
    if parsed.utcoffset() is None:
        raise ValueError("A source timezone is required")
    return parsed.astimezone(timezone.utc)


def compact_utc(value: datetime) -> str:
    return parse_utc(format_utc(value)).strftime("%Y%m%dT%H%M%SZ")

"""Short page transactions for legacy sync; never hold a transaction over HTTP."""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from ispec.db.models import LegacySyncState


def _utc_naive(value: datetime | None) -> datetime | None:
    # SQLAlchemy defaults can be aware, but SQLite reloads DateTime as naive UTC.
    if value is not None and value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


@dataclass(frozen=True, kw_only=True)
class LegacyCursorCheckpoint:
    since: datetime | None
    since_pk: int | None
    since_pk_json: str | None
    modified_at: datetime | None

    @classmethod
    def from_row(cls, row: LegacySyncState | None) -> "LegacyCursorCheckpoint | None":
        if row is None:
            return None
        return cls(
            since=_utc_naive(row.since),
            since_pk=row.since_pk,
            since_pk_json=row.since_pk_json,
            modified_at=_utc_naive(row.legsync_ModificationTS),
        )


def read_sync_cursor(
    session: Session, table: str | None
) -> LegacyCursorCheckpoint | None:
    row = session.get(LegacySyncState, table) if table else None
    checkpoint = LegacyCursorCheckpoint.from_row(row)
    session.rollback()  # End even the initial read transaction before fetching.
    return checkpoint


def begin_sync_page(
    session: Session,
    *,
    table: str | None,
    expected: LegacyCursorCheckpoint | None,
    dry_run: bool,
) -> LegacySyncState | None:
    if session.in_transaction():
        raise RuntimeError("Legacy sync page started with an open transaction")
    if not dry_run:
        # Serialize only the local reconciliation/commit, not the remote fetch.
        session.execute(text("BEGIN IMMEDIATE"))
    row = session.get(LegacySyncState, table, populate_existing=True) if table else None
    if table and LegacyCursorCheckpoint.from_row(row) != expected:
        raise RuntimeError(
            f"Legacy sync cursor for {table} changed during fetch; retry from the saved cursor"
        )
    return row


def finish_sync_page(
    session: Session,
    *,
    cursor: LegacySyncState | None,
    dry_run: bool,
) -> LegacyCursorCheckpoint | None:
    if not dry_run:
        session.flush()
    checkpoint = LegacyCursorCheckpoint.from_row(cursor)
    if dry_run:
        session.rollback()
    else:
        session.commit()
    return checkpoint

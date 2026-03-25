from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.agent.connect import get_agent_db_uri, get_agent_session
from ispec.agent.models import AgentBase, AgentCommand, AgentEvent, AgentRun, AgentStep
from ispec.db.models import sqlite_engine
from ispec.logging import get_logger

logger = get_logger(__file__)

_ALLOWED_SQLITE_JOURNAL_MODES = {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}
_TERMINAL_COMMAND_STATUSES = {"succeeded", "failed"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _normalize_sqlite_path_or_uri(value: str | Path) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Expected a non-empty SQLite path or URI.")
    if raw.startswith("sqlite"):
        return raw
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str(path)


def _sqlite_path_from_uri(db_uri: str) -> Path | None:
    raw = (db_uri or "").strip()
    if raw.startswith("sqlite:///"):
        return Path(raw.removeprefix("sqlite:///")).expanduser()
    if raw.startswith("sqlite://") and not raw.startswith("sqlite:////"):
        return Path(raw.removeprefix("sqlite://")).expanduser()
    return None


def _resolve_archive_db_uri(path_or_uri: str | Path | None, *, dry_run: bool) -> str | None:
    candidate = str(path_or_uri or os.getenv("ISPEC_AGENT_ARCHIVE_DB_PATH") or "").strip()
    if not candidate:
        if dry_run:
            return None
        raise ValueError(
            "Missing archive database path. Pass --archive-database or set ISPEC_AGENT_ARCHIVE_DB_PATH."
        )
    return _normalize_sqlite_path_or_uri(candidate)


def get_agent_archive_db_uri(
    file_path: str | Path | None = None,
    *,
    required: bool = False,
) -> str | None:
    try:
        return _resolve_archive_db_uri(file_path, dry_run=not required)
    except ValueError:
        if required:
            raise
        return None


def _archive_sqlite_journal_mode(raw: str | None = None) -> str | None:
    value = str(raw or os.getenv("ISPEC_AGENT_ARCHIVE_SQLITE_JOURNAL_MODE") or "").strip().upper()
    if not value:
        return None
    if value in _ALLOWED_SQLITE_JOURNAL_MODES:
        return value
    logger.warning("Ignoring unsupported ISPEC_AGENT_ARCHIVE_SQLITE_JOURNAL_MODE=%s", value)
    return None


@lru_cache(maxsize=None)
def _get_archive_engine(db_uri: str, journal_mode: str | None) -> Engine:
    engine = sqlite_engine(db_uri)
    AgentBase.metadata.create_all(bind=engine)
    if journal_mode:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(f"PRAGMA journal_mode={journal_mode}")
                if journal_mode == "WAL":
                    conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        except Exception:
            logger.debug("Unable to set %s journal mode for agent archive DB.", journal_mode)
    return engine


@contextmanager
def get_agent_archive_session(
    file_path: str | Path | None = None,
    *,
    journal_mode: str | None = None,
) -> Iterator[Session]:
    db_uri = get_agent_archive_db_uri(file_path, required=True)
    assert db_uri is not None
    engine = _get_archive_engine(db_uri, _archive_sqlite_journal_mode(journal_mode))
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_agent_archive_session_if_available(
    file_path: str | Path | None = None,
    *,
    journal_mode: str | None = None,
) -> Iterator[Session | None]:
    db_uri = get_agent_archive_db_uri(file_path, required=False)
    if not db_uri:
        yield None
        return
    db_path = _sqlite_path_from_uri(db_uri)
    if db_path is not None and not db_path.exists():
        yield None
        return
    engine = _get_archive_engine(db_uri, _archive_sqlite_journal_mode(journal_mode))
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _clone_row(model_cls: type[Any], row: Any) -> Any:
    payload = {column.name: getattr(row, column.name) for column in model_cls.__table__.columns}
    return model_cls(**payload)


def _same_sqlite_target(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    a_path = _sqlite_path_from_uri(a)
    b_path = _sqlite_path_from_uri(b)
    if a_path is not None and b_path is not None:
        try:
            return a_path.resolve() == b_path.resolve()
        except Exception:
            return str(a_path) == str(b_path)
    return False


def _empty_summary(*, selected: bool) -> dict[str, Any]:
    return {
        "selected": bool(selected),
        "matched": 0,
        "archived": 0,
        "pruned": 0,
        "batches": 0,
    }


def archive_agent_logs(
    *,
    agent_db_file_path: str | None = None,
    archive_db_file_path: str | Path | None = None,
    older_than_days: int = 14,
    batch_size: int = 500,
    max_batches: int | None = 20,
    dry_run: bool = False,
    prune_live: bool = True,
    archive_steps: bool = True,
    archive_events: bool = True,
    archive_commands: bool = True,
    archive_journal_mode: str | None = None,
) -> dict[str, Any]:
    normalized_batch_size = max(1, int(batch_size))
    normalized_max_batches = None if max_batches is None else max(1, int(max_batches))
    normalized_days = max(1, int(older_than_days))
    cutoff = _utcnow() - timedelta(days=normalized_days)

    live_db_uri = get_agent_db_uri(agent_db_file_path)
    archive_db_uri = get_agent_archive_db_uri(archive_db_file_path, required=not bool(dry_run))
    if archive_db_uri is not None and _same_sqlite_target(live_db_uri, archive_db_uri):
        raise ValueError("Archive database must differ from the live agent database.")

    journal_mode = _archive_sqlite_journal_mode(archive_journal_mode)

    summary: dict[str, Any] = {
        "ok": True,
        "dry_run": bool(dry_run),
        "prune_live": bool(prune_live),
        "cutoff": cutoff.isoformat(),
        "older_than_days": int(normalized_days),
        "batch_size": int(normalized_batch_size),
        "max_batches": normalized_max_batches,
        "live_database": live_db_uri,
        "archive_database": archive_db_uri,
        "archive_journal_mode": journal_mode,
        "steps": _empty_summary(selected=archive_steps),
        "events": _empty_summary(selected=archive_events),
        "commands": _empty_summary(selected=archive_commands),
        "runs_archived": 0,
    }

    archived_run_ids: set[int] = set()

    with get_agent_session(agent_db_file_path) as live_db:
        archive_db_cm = (
            get_agent_archive_session(archive_db_file_path, journal_mode=journal_mode)
            if not dry_run
            else nullcontext(None)
        )
        with archive_db_cm as archive_db:
            if archive_steps:
                last_id = 0
                while True:
                    if normalized_max_batches is not None and summary["steps"]["batches"] >= normalized_max_batches:
                        break
                    rows = (
                        live_db.query(AgentStep)
                        .filter(AgentStep.ended_at.is_not(None))
                        .filter(AgentStep.ended_at < cutoff)
                        .filter(AgentStep.id > int(last_id))
                        .order_by(AgentStep.id.asc())
                        .limit(normalized_batch_size)
                        .all()
                    )
                    if not rows:
                        break
                    summary["steps"]["batches"] += 1
                    summary["steps"]["matched"] += len(rows)
                    last_id = int(rows[-1].id)

                    if dry_run:
                        continue

                    run_ids = sorted({int(row.run_pk) for row in rows})
                    run_rows = live_db.query(AgentRun).filter(AgentRun.id.in_(run_ids)).all() if run_ids else []
                    for run in run_rows:
                        archive_db.merge(_clone_row(AgentRun, run))
                        archived_run_ids.add(int(run.id))
                    for row in rows:
                        archive_db.merge(_clone_row(AgentStep, row))
                    archive_db.commit()
                    summary["steps"]["archived"] += len(rows)

                    if prune_live:
                        ids = [int(row.id) for row in rows]
                        live_db.query(AgentStep).filter(AgentStep.id.in_(ids)).delete(synchronize_session=False)
                        live_db.commit()
                        summary["steps"]["pruned"] += len(ids)

            if archive_events:
                last_id = 0
                while True:
                    if normalized_max_batches is not None and summary["events"]["batches"] >= normalized_max_batches:
                        break
                    rows = (
                        live_db.query(AgentEvent)
                        .filter(AgentEvent.received_at < cutoff)
                        .filter(AgentEvent.id > int(last_id))
                        .order_by(AgentEvent.id.asc())
                        .limit(normalized_batch_size)
                        .all()
                    )
                    if not rows:
                        break
                    summary["events"]["batches"] += 1
                    summary["events"]["matched"] += len(rows)
                    last_id = int(rows[-1].id)

                    if dry_run:
                        continue

                    for row in rows:
                        archive_db.merge(_clone_row(AgentEvent, row))
                    archive_db.commit()
                    summary["events"]["archived"] += len(rows)

                    if prune_live:
                        ids = [int(row.id) for row in rows]
                        live_db.query(AgentEvent).filter(AgentEvent.id.in_(ids)).delete(synchronize_session=False)
                        live_db.commit()
                        summary["events"]["pruned"] += len(ids)

            if archive_commands:
                last_id = 0
                while True:
                    if normalized_max_batches is not None and summary["commands"]["batches"] >= normalized_max_batches:
                        break
                    rows = (
                        live_db.query(AgentCommand)
                        .filter(AgentCommand.status.in_(sorted(_TERMINAL_COMMAND_STATUSES)))
                        .filter(AgentCommand.ended_at.is_not(None))
                        .filter(AgentCommand.ended_at < cutoff)
                        .filter(AgentCommand.id > int(last_id))
                        .order_by(AgentCommand.id.asc())
                        .limit(normalized_batch_size)
                        .all()
                    )
                    if not rows:
                        break
                    summary["commands"]["batches"] += 1
                    summary["commands"]["matched"] += len(rows)
                    last_id = int(rows[-1].id)

                    if dry_run:
                        continue

                    for row in rows:
                        archive_db.merge(_clone_row(AgentCommand, row))
                    archive_db.commit()
                    summary["commands"]["archived"] += len(rows)

                    if prune_live:
                        ids = [int(row.id) for row in rows]
                        live_db.query(AgentCommand).filter(AgentCommand.id.in_(ids)).delete(synchronize_session=False)
                        live_db.commit()
                        summary["commands"]["pruned"] += len(ids)

    summary["runs_archived"] = len(archived_run_ids)
    return summary

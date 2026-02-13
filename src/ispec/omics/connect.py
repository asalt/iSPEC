from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.db.connect import get_db_dir
from ispec.db.models import OmicsDatabaseRegistry, sqlite_engine
from ispec.logging import get_logger
from ispec.omics.models import OmicsBase

logger = get_logger(__file__)

DEFAULT_OMICS_LOGICAL_NAME = "primary"


class OmicsDatabaseUnavailableError(RuntimeError):
    """Raised when a previously-known omics DB is no longer available."""


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _sqlite_uri(db_path: str | Path) -> str:
    raw = str(db_path).strip()
    if raw.startswith("sqlite"):
        return raw
    path = Path(raw).expanduser()
    return "sqlite:///" + str(path)


def get_omics_db_uri(file: str | Path | None = None) -> str:
    """Return a SQLite URI string for the omics DB.

    Resolution order:
      1) explicit ``file`` argument
      2) env ``ISPEC_OMICS_DB_PATH``
      3) alongside ``ISPEC_DB_PATH`` (same dir, ``ispec-omics.db``)
      4) fallback to ``ISPEC_DB_DIR`` (via :func:`ispec.db.connect.get_db_dir`)
    """

    if file is not None:
        return _sqlite_uri(file)

    env_path = (os.getenv("ISPEC_OMICS_DB_PATH") or "").strip()
    if env_path:
        return _sqlite_uri(env_path)

    main_path = (os.getenv("ISPEC_DB_PATH") or "").strip()
    candidate: Path | None = None
    if main_path:
        if main_path.startswith("sqlite:///"):
            candidate = Path(main_path.removeprefix("sqlite:///"))
        elif "://" not in main_path:
            candidate = Path(main_path)

    if candidate is not None:
        omics_file = candidate.expanduser().resolve().parent / "ispec-omics.db"
        return _sqlite_uri(omics_file)

    return _sqlite_uri(get_db_dir() / "ispec-omics.db")


def _sqlite_path_from_uri(db_uri: str) -> Path | None:
    raw = (db_uri or "").strip()
    if raw.startswith("sqlite:///"):
        return Path(raw.removeprefix("sqlite:///")).expanduser()
    if raw.startswith("sqlite://") and not raw.startswith("sqlite:////"):
        # sqlite://relative/path.db form (rare in this codebase, but valid)
        return Path(raw.removeprefix("sqlite://")).expanduser()
    return None


def _get_registry_row(
    core_session: Session,
    *,
    logical_name: str,
) -> OmicsDatabaseRegistry | None:
    stmt = select(OmicsDatabaseRegistry).where(
        OmicsDatabaseRegistry.omdb_LogicalName == logical_name
    )
    return core_session.execute(stmt).scalar_one_or_none()


@lru_cache(maxsize=None)
def _get_engine(db_uri: str) -> Engine:
    engine = sqlite_engine(db_uri)
    OmicsBase.metadata.create_all(bind=engine)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
    except Exception:
        logger.debug("Unable to set WAL pragmas for omics DB.")
    return engine


@contextmanager
def get_omics_session(
    file_path: str | Path | None = None,
    *,
    core_session: Session | None = None,
    logical_name: str = DEFAULT_OMICS_LOGICAL_NAME,
    allow_recreate_missing: bool = False,
) -> Iterator[Session]:
    """Context-managed SQLAlchemy session for the omics DB."""

    db_uri = get_omics_db_uri(file_path)
    sqlite_path = _sqlite_path_from_uri(db_uri)
    registry_row: OmicsDatabaseRegistry | None = None

    if core_session is not None:
        registry_row = _get_registry_row(core_session, logical_name=logical_name)
        if registry_row is None:
            registry_row = OmicsDatabaseRegistry(
                omdb_LogicalName=logical_name,
                omdb_DBURI=db_uri,
                omdb_DBPath=str(sqlite_path) if sqlite_path is not None else None,
                omdb_Status="unknown",
            )
            core_session.add(registry_row)
            core_session.flush()
        else:
            registry_row.omdb_DBURI = db_uri
            registry_row.omdb_DBPath = str(sqlite_path) if sqlite_path is not None else None

        if (
            sqlite_path is not None
            and registry_row.omdb_LastAvailableTS is not None
            and not sqlite_path.exists()
            and not allow_recreate_missing
        ):
            msg = (
                f"Omics DB '{logical_name}' is unavailable at {sqlite_path}. "
                "Refusing to auto-create a new database at that location."
            )
            logger.warning(msg)
            raise OmicsDatabaseUnavailableError(msg)

    if sqlite_path is not None and not sqlite_path.exists():
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    engine = _get_engine(db_uri)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        if registry_row is not None:
            now = _utcnow_naive()
            registry_row.omdb_Status = "available"
            registry_row.omdb_LastCheckedTS = now
            registry_row.omdb_LastAvailableTS = now
            registry_row.omdb_LastError = None
            core_session.flush()
        yield session
        session.commit()
    except Exception:
        session.rollback()
        if registry_row is not None:
            registry_row.omdb_Status = "unavailable"
            registry_row.omdb_LastCheckedTS = _utcnow_naive()
            registry_row.omdb_LastError = "connect_or_session_error"
            core_session.flush()
        raise
    finally:
        session.close()


def get_omics_session_dep() -> Iterator[Session]:
    """FastAPI dependency yielding an omics DB session."""

    with get_omics_session() as session:
        yield session

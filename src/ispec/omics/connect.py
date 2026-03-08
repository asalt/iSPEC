from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.config.paths import resolve_db_location
from ispec.db.models import OmicsDatabaseRegistry, sqlite_engine
from ispec.logging import get_logger
from ispec.omics.models import OmicsBase

logger = get_logger(__file__)

DEFAULT_OMICS_LOGICAL_NAME = "analysis"
LEGACY_OMICS_LOGICAL_NAME = "primary"
PSM_OMICS_LOGICAL_NAME = "psm"


class OmicsDatabaseUnavailableError(RuntimeError):
    """Raised when a previously-known omics DB is no longer available."""


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _normalize_logical_name(logical_name: str) -> str:
    normalized = (logical_name or DEFAULT_OMICS_LOGICAL_NAME).strip().lower()
    if normalized == LEGACY_OMICS_LOGICAL_NAME:
        return DEFAULT_OMICS_LOGICAL_NAME
    return normalized or DEFAULT_OMICS_LOGICAL_NAME


def get_omics_db_uri(
    file: str | Path | None = None,
    *,
    logical_name: str = DEFAULT_OMICS_LOGICAL_NAME,
) -> str:
    """Return a SQLite URI string for the omics DB.

    Resolution order:
      1) explicit ``file`` argument
      2) env for the requested logical DB (analysis or psm)
      3) compatibility alias (analysis only: ``ISPEC_OMICS_DB_PATH``)
      4) sibling file next to ``ISPEC_DB_PATH``
    """

    resolved = resolve_db_location(_normalize_logical_name(logical_name), file=file)
    return resolved.uri or str(resolved.value)


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
    normalized_name = _normalize_logical_name(logical_name)
    stmt = select(OmicsDatabaseRegistry).where(
        OmicsDatabaseRegistry.omdb_LogicalName == normalized_name
    )
    row = core_session.execute(stmt).scalar_one_or_none()
    if row is not None:
        return row
    if normalized_name != DEFAULT_OMICS_LOGICAL_NAME:
        return None
    legacy_stmt = select(OmicsDatabaseRegistry).where(
        OmicsDatabaseRegistry.omdb_LogicalName == LEGACY_OMICS_LOGICAL_NAME
    )
    return core_session.execute(legacy_stmt).scalar_one_or_none()


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

    normalized_name = _normalize_logical_name(logical_name)
    db_uri = get_omics_db_uri(file_path, logical_name=normalized_name)
    sqlite_path = _sqlite_path_from_uri(db_uri)
    registry_row: OmicsDatabaseRegistry | None = None
    reuse_core_session = False

    if core_session is not None:
        core_db_path = _sqlite_path_from_uri(str(core_session.bind.url))
        if (
            sqlite_path is not None
            and core_db_path is not None
            and core_db_path.resolve() == sqlite_path.resolve()
        ):
            reuse_core_session = True

        registry_row = _get_registry_row(core_session, logical_name=normalized_name)
        if registry_row is None:
            registry_row = OmicsDatabaseRegistry(
                omdb_LogicalName=normalized_name,
                omdb_DBURI=db_uri,
                omdb_DBPath=str(sqlite_path) if sqlite_path is not None else None,
                omdb_Status="unknown",
            )
            core_session.add(registry_row)
            core_session.flush()
        else:
            if registry_row.omdb_LogicalName != normalized_name:
                registry_row.omdb_LogicalName = normalized_name
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

    if reuse_core_session and core_session is not None:
        OmicsBase.metadata.create_all(bind=core_session.connection())
        if registry_row is not None:
            now = _utcnow_naive()
            registry_row.omdb_Status = "available"
            registry_row.omdb_LastCheckedTS = now
            registry_row.omdb_LastAvailableTS = now
            registry_row.omdb_LastError = None
            core_session.flush()
        yield core_session
        return

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

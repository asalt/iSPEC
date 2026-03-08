# ispec/db/connect.py

from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine


from ispec.db.models import sqlite_engine, initialize_db
from ispec.config.paths import resolve_db_dir, resolve_db_location
from ispec.logging import get_logger

logger = get_logger(__file__)

@lru_cache(maxsize=None)
def get_db_dir() -> Path:
    db_dir = Path(resolve_db_dir().path or (Path.home() / "ispec"))
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("setting db_dir to %s", str(db_dir))
    return db_dir


@lru_cache(maxsize=None)
def get_db_path(file: str | Path | None = None) -> str:
    """Return a SQLite database URI string.

    Parameters
    ----------
    file:
        Optional path to a SQLite database file. When ``None`` the default
        directory from :func:`get_db_dir` and the filename ``ispec.db`` are
        used.

    Returns
    -------
    str
        SQLite URI pointing to the database file.
    """

    resolved = resolve_db_location("core", file=file)
    db_file = Path(resolved.path) if resolved.path is not None else None
    if db_file is not None:
        db_file.parent.mkdir(parents=True, exist_ok=True)
    db_uri = resolved.uri or str(resolved.value)
    logger.info("setting db_path to %s", db_uri)
    return db_uri


"""
Legacy helpers for reading raw SQL initialization scripts have been removed.
Database initialization now relies solely on SQLAlchemy models
via ispec.db.models.initialize_db.
"""




def make_session_factory(engine: Engine):
    SessionLocal = sessionmaker(bind=engine)
    initialize_db(engine=engine)

    @contextmanager
    def get_session():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return get_session


# Session Context Manager
@contextmanager
def get_session(file_path: str | Path | None = None) -> Session:
    """Provide a transactional scope around a series of operations.

    Parameters
    ----------
    file_path:
        Optional path or URI to the SQLite database. If not provided, the
        ``ISPEC_DB_PATH`` environment variable or the default URI from
        :func:`get_db_path` is used.
    """

    resolved = resolve_db_location("core", file=file_path)
    if resolved.path is not None:
        Path(resolved.path).parent.mkdir(parents=True, exist_ok=True)
    db_uri = resolved.uri or str(resolved.value)

    engine = sqlite_engine(db_uri)
    initialize_db(engine=engine)

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


def get_session_dep() -> Iterator[Session]:
    """FastAPI dependency that yields a SQLAlchemy session.

    ``get_session`` is a ``contextmanager`` for CLI/scripts. FastAPI expects a
    generator dependency (``yield``) so it can manage teardown after the
    request. This wrapper bridges the two.
    """

    with get_session() as session:
        yield session


# def ensure_db_dir():
#    logger.debug("ensuring db dir")
#    get_db_dir().mkdir(parents=True, exist_ok=True)

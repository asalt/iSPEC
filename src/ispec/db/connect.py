# ispec/db/connect.py

import os
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine


from ispec.db.models import sqlite_engine, initialize_db
from ispec.logging import get_logger

logger = get_logger(__file__)

@lru_cache(maxsize=None)
def get_db_dir() -> Path:
    db_dir = Path(os.environ.get("ISPEC_DB_DIR", Path.home() / "ispec"))
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

    if file is None:
        db_path = get_db_dir()
        db_file = db_path / "ispec.db"
    else:
        db_file = Path(file)
    db_uri = "sqlite:///" + str(db_file)
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

    db_path = os.getenv("ISPEC_DB_PATH") if file_path is None else file_path
    if db_path is None:
        db_path = get_db_path()
    # ensure sqlite URI prefix
    db_uri = str(db_path)
    if not str(db_uri).startswith("sqlite"):
        db_uri = "sqlite:///" + str(db_uri)

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

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.db.models import sqlite_engine
from ispec.logging import get_logger

from .models import AgentBase

logger = get_logger(__file__)


def _sqlite_uri(db_path: str | Path) -> str:
    raw = str(db_path).strip()
    if raw.startswith("sqlite"):
        return raw
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str(path)


def get_agent_db_uri(file: str | Path | None = None) -> str:
    """Return a SQLite URI string for the agent DB.

    Resolution order:
      1) explicit ``file`` argument
      2) env ``ISPEC_AGENT_DB_PATH``
      3) default to the support assistant DB (see :func:`ispec.assistant.connect.get_assistant_db_uri`)
    """

    if file is not None:
        return _sqlite_uri(file)

    env_path = (os.getenv("ISPEC_AGENT_DB_PATH") or "").strip()
    if env_path:
        return _sqlite_uri(env_path)

    from ispec.assistant.connect import get_assistant_db_uri

    return get_assistant_db_uri()


@lru_cache(maxsize=None)
def _get_engine(db_uri: str) -> Engine:
    engine = sqlite_engine(db_uri)
    AgentBase.metadata.create_all(bind=engine)
    return engine


@contextmanager
def get_agent_session(file_path: str | Path | None = None) -> Iterator[Session]:
    """Context-managed SQLAlchemy session for the agent DB."""

    db_uri = get_agent_db_uri(file_path)
    engine = _get_engine(db_uri)
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


def get_agent_session_dep() -> Iterator[Session]:
    """FastAPI dependency yielding an agent DB session."""

    with get_agent_session() as session:
        yield session

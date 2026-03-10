from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.config.paths import resolve_db_location
from ispec.db.models import sqlite_engine

from .models import AgentStateBase


def get_agent_state_db_uri(file: str | Path | None = None) -> str:
    resolved = resolve_db_location("agent_state", file=file)
    if resolved.path is not None:
        Path(resolved.path).parent.mkdir(parents=True, exist_ok=True)
    return resolved.uri or str(resolved.value)


@lru_cache(maxsize=None)
def _get_engine(db_uri: str) -> Engine:
    engine = sqlite_engine(db_uri)
    AgentStateBase.metadata.create_all(bind=engine)
    return engine


@contextmanager
def get_agent_state_session(file_path: str | Path | None = None) -> Iterator[Session]:
    db_uri = get_agent_state_db_uri(file_path)
    engine = _get_engine(db_uri)
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_agent_state_session_dep() -> Iterator[Session]:
    with get_agent_state_session() as session:
        yield session

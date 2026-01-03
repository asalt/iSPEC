from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.db.models import sqlite_engine
from ispec.logging import get_logger

from .models import AssistantBase


logger = get_logger(__file__)


def _sqlite_uri(db_path: str | Path) -> str:
    raw = str(db_path).strip()
    if raw.startswith("sqlite"):
        return raw
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str(path)


def get_assistant_db_uri(file: str | Path | None = None) -> str:
    """Return a SQLite URI string for the assistant DB.

    Resolution order:
      1) explicit ``file`` argument
      2) env ``ISPEC_ASSISTANT_DB_PATH``
      3) alongside ``ISPEC_DB_PATH`` (same dir, ``ispec-assistant.db``)
      4) fallback to ``ISPEC_DB_DIR`` (via :func:`ispec.db.connect.get_db_dir`)
    """

    if file is not None:
        return _sqlite_uri(file)

    env_path = (os.getenv("ISPEC_ASSISTANT_DB_PATH") or "").strip()
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
        assistant_file = candidate.expanduser().resolve().parent / "ispec-assistant.db"
        return _sqlite_uri(assistant_file)

    from ispec.db.connect import get_db_dir

    return _sqlite_uri(get_db_dir() / "ispec-assistant.db")


@lru_cache(maxsize=None)
def _get_engine(db_uri: str) -> Engine:
    engine = sqlite_engine(db_uri)
    AssistantBase.metadata.create_all(bind=engine)
    _ensure_support_session_columns(engine)
    _ensure_support_message_columns(engine)
    return engine


def _ensure_support_session_columns(engine: Engine) -> None:
    """Best-effort SQLite schema upgrades for the assistant DB.

    The assistant DB is expected to evolve quickly during development, so we
    perform lightweight ``ALTER TABLE`` operations when columns are missing.
    """

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("support_session")}
    except Exception:
        return

    if "state_json" in columns:
        return

    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE support_session ADD COLUMN "state_json" TEXT'))
    logger.info("Added missing column support_session.state_json")


def _ensure_support_message_columns(engine: Engine) -> None:
    """Best-effort SQLite upgrades for support_message feedback fields."""

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("support_message")}
    except Exception:
        return

    missing: list[tuple[str, str]] = []
    if "feedback_note" not in columns:
        missing.append(("feedback_note", "TEXT"))
    if "feedback_meta_json" not in columns:
        missing.append(("feedback_meta_json", "TEXT"))

    if not missing:
        return

    with engine.begin() as conn:
        for name, col_type in missing:
            conn.execute(text(f'ALTER TABLE support_message ADD COLUMN "{name}" {col_type}'))

    logger.info(
        "Added missing columns support_message.%s",
        ", ".join(name for name, _ in missing),
    )


@contextmanager
def get_assistant_session(file_path: str | Path | None = None) -> Iterator[Session]:
    """Context-managed SQLAlchemy session for the assistant DB."""

    db_uri = get_assistant_db_uri(file_path)
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


def get_assistant_session_dep() -> Iterator[Session]:
    """FastAPI dependency yielding an assistant DB session."""

    with get_assistant_session() as session:
        yield session

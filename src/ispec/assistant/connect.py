from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ispec.db.models import sqlite_engine
from ispec.logging import get_logger

from .models import AssistantBase, SupportSession, SupportSessionReview


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
    try:
        from ispec.agent.models import AgentBase
    except Exception:
        AgentBase = None
    if AgentBase is not None:
        AgentBase.metadata.create_all(bind=engine)
    _ensure_support_session_columns(engine)
    _ensure_support_message_columns(engine)
    _migrate_support_session_reviews_from_state(engine)
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


def _migrate_support_session_reviews_from_state(engine: Engine) -> None:
    """Best-effort migration of legacy conversation reviews stored in state_json."""

    try:
        tables = set(inspect(engine).get_table_names())
    except Exception:
        return
    if "support_session_review" not in tables:
        return

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        rows = db.query(SupportSession).filter(SupportSession.state_json.isnot(None)).all()
        migrated = 0
        for session in rows:
            raw_state = getattr(session, "state_json", None)
            if not raw_state:
                continue
            try:
                state = json.loads(raw_state)
            except Exception:
                continue
            if not isinstance(state, dict):
                continue
            review = state.get("conversation_review")
            if not isinstance(review, dict):
                continue

            target_id = state.get("conversation_review_up_to_id")
            if not isinstance(target_id, int):
                target_id = review.get("target_message_id")
            if not isinstance(target_id, int) or target_id <= 0:
                continue

            existing = (
                db.query(SupportSessionReview)
                .filter(SupportSessionReview.session_pk == int(session.id))
                .filter(SupportSessionReview.target_message_id == int(target_id))
                .first()
            )
            if existing is not None:
                continue

            record = SupportSessionReview(
                session_pk=int(session.id),
                target_message_id=int(target_id),
                schema_version=int(review.get("schema_version") or 1),
                review_json=review,
            )
            db.add(record)
            try:
                db.commit()
                migrated += 1
            except IntegrityError:
                db.rollback()

        if migrated:
            logger.info("Migrated %s support session review(s) from support_session.state_json", migrated)
    finally:
        db.close()


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

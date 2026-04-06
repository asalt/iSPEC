from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ispec.config.paths import resolve_db_location

from .models import PromptVersionInfo


_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prompt_family (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    title TEXT,
    notes TEXT,
    active_version_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(active_version_id) REFERENCES prompt_version(id)
);

CREATE TABLE IF NOT EXISTS prompt_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id INTEGER NOT NULL,
    version_num INTEGER NOT NULL,
    body_sha256 TEXT NOT NULL,
    body_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(family_id, version_num),
    UNIQUE(family_id, body_sha256),
    FOREIGN KEY(family_id) REFERENCES prompt_family(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prompt_binding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id INTEGER NOT NULL,
    module TEXT NOT NULL,
    qualname TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_line INTEGER,
    binding_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(family_id, module, qualname),
    FOREIGN KEY(family_id) REFERENCES prompt_family(id) ON DELETE CASCADE
);
"""


def get_prompts_db_path(file: str | Path | None = None) -> Path | None:
    resolved = resolve_db_location("prompts", file=file)
    if resolved.path is None:
        return None
    return Path(resolved.path).expanduser()


@contextmanager
def connect_prompts_db(file: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = get_prompts_db_path(file=file)
    if db_path is None:
        raise ValueError("Prompt DB path could not be resolved.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def ensure_prompts_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def lookup_prompt_version(*, family: str, body_sha256: str, file: str | Path | None = None) -> PromptVersionInfo:
    db_path = get_prompts_db_path(file=file)
    if db_path is None or not db_path.exists():
        return PromptVersionInfo()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT pv.id, pv.version_num
            FROM prompt_version pv
            JOIN prompt_family pf ON pf.id = pv.family_id
            WHERE pf.family = ? AND pv.body_sha256 = ?
            LIMIT 1
            """,
            (family, body_sha256),
        ).fetchone()
    except sqlite3.DatabaseError:
        return PromptVersionInfo()
    finally:
        conn.close()
    if row is None:
        return PromptVersionInfo()
    return PromptVersionInfo(
        version_id=int(row["id"]) if row["id"] is not None else None,
        version_num=int(row["version_num"]) if row["version_num"] is not None else None,
    )

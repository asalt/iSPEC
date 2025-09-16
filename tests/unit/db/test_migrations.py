"""Tests that exercise Alembic migrations end-to-end."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from ispec.db.models import Base

REPO_ROOT_SENTINEL = "alembic.ini"


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / REPO_ROOT_SENTINEL).exists():
            return parent
    msg = "Unable to locate project root with alembic.ini"
    raise RuntimeError(msg)


def _make_config(project_root: Path) -> Config:
    config = Config(str(project_root / REPO_ROOT_SENTINEL))
    config.set_main_option("script_location", str(project_root / "alembic"))
    return config


@contextmanager
def _engine_connection():
    """Provide a shared in-memory SQLite engine and transaction."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
        with engine.begin() as conn:
            yield engine, conn
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def alembic_cfg() -> Config:
    return _make_config(_project_root())


@pytest.fixture(scope="function")
def live_conn(alembic_cfg: Config):
    with _engine_connection() as (engine, conn):
        alembic_cfg.attributes["connection"] = conn
        yield engine, conn


def _script_dir(cfg: Config) -> ScriptDirectory:
    return ScriptDirectory.from_config(cfg)


def test_single_head(alembic_cfg: Config) -> None:
    script = _script_dir(alembic_cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Multiple heads found: {heads}"
    assert script.get_current_head() == heads[0]


def test_upgrade_head_creates_core_tables(alembic_cfg: Config, live_conn) -> None:
    engine, _ = live_conn
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected = {
        "person",
        "project",
        "project_comment",
        "project_person",
        "letter_of_support",
    }

    assert expected.issubset(tables)


def test_model_metadata_matches_database(alembic_cfg: Config, live_conn) -> None:
    engine, _ = live_conn
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(engine)
    db_tables = set(inspector.get_table_names())
    model_tables = {table.name for table in Base.metadata.sorted_tables}

    assert model_tables.issubset(db_tables)


def test_upgrade_is_idempotent(alembic_cfg: Config, live_conn) -> None:
    command.upgrade(alembic_cfg, "head")
    command.upgrade(alembic_cfg, "head")


def test_roundtrip_each_revision(alembic_cfg: Config, live_conn) -> None:
    engine, _ = live_conn
    script = _script_dir(alembic_cfg)
    revisions = list(script.walk_revisions(base="base", head="heads"))[::-1]

    for revision in revisions:
        command.upgrade(alembic_cfg, revision.revision)
        inspect(engine).get_table_names()

    for down_revision in (rev.down_revision for rev in revisions[::-1]):
        target = down_revision or "base"
        command.downgrade(alembic_cfg, target)


def test_fk_enforcement_present(alembic_cfg: Config, live_conn) -> None:
    _, conn = live_conn
    command.upgrade(alembic_cfg, "head")

    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                """
                INSERT INTO project_comment (id, project_id, person_id, com_Comment)
                VALUES (1, 9999, 9999, 'fk test')
                """
            )
        )


def test_foreign_key_metadata(alembic_cfg: Config, live_conn) -> None:
    engine, _ = live_conn
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(engine)

    comment_fks = {
        fk["constrained_columns"][0]: fk["referred_table"]
        for fk in inspector.get_foreign_keys("project_comment")
    }
    assert comment_fks == {"project_id": "project", "person_id": "person"}

    mapping_fks = {
        fk["constrained_columns"][0]: fk["referred_table"]
        for fk in inspector.get_foreign_keys("project_person")
    }
    assert mapping_fks == {"project_id": "project", "person_id": "person"}

    for table in ("person", "project", "project_comment", "project_person"):
        pk = inspector.get_pk_constraint(table)
        assert pk["constrained_columns"] == ["id"], f"Missing PK for {table}"

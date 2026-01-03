"""Alembic environment configuration for iSPEC."""

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection, Engine

from ispec.db.connect import get_db_path
from ispec.db.models import Base

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name and Path(config.config_file_name).exists():
    fileConfig(config.config_file_name)

# target_metadata is used for 'autogenerate' support.
target_metadata = Base.metadata


def _coerce_sqlite_url(raw: str) -> str:
    """Ensure a SQLite URL has the proper prefix."""

    if raw.startswith("sqlite"):
        return raw
    return "sqlite:///" + raw


def _get_database_url() -> str:
    """Resolve the database URL for Alembic to use."""

    env_url = os.getenv("ISPEC_DB_PATH")
    if env_url:
        return _coerce_sqlite_url(env_url)

    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url:
        return configured_url

    return get_db_path()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""

    url = _get_database_url()
    config.set_main_option("sqlalchemy.url", url)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection: Connection, *, external: bool = False) -> None:
    """Configure Alembic to run migrations using ``connection``."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()

    # When Alembic is supplied an existing Connection (common in unit tests with
    # an in-memory SQLite StaticPool), inspecting the engine can check out the
    # same underlying DBAPI connection and issue an implicit ROLLBACK, which can
    # wipe the uncommitted alembic_version row. We commit at the DBAPI level so
    # the surrounding SQLAlchemy transaction context manager remains usable.
    if external and connection.dialect.name == "sqlite":
        try:
            dbapi_conn = connection.connection
            dbapi_conn.commit()
        except Exception:  # pragma: no cover - best effort
            pass


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    existing = config.attributes.get("connection")

    if isinstance(existing, Engine):
        with existing.connect() as connection:
            _run_with_connection(connection, external=True)
        return

    if isinstance(existing, Connection):
        _run_with_connection(existing, external=True)
        return

    url = _get_database_url()
    config.set_main_option("sqlalchemy.url", url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        _run_with_connection(connection, external=False)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

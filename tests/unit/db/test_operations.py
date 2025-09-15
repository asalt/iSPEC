import logging

import pytest

from ispec.db import operations


def test_check_status_logs_version(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    logger = operations.logger
    orig_prop = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO):
            operations.check_status()
    finally:
        logger.propagate = orig_prop
    assert "sqlite version" in caplog.text


def test_show_tables_logs_tables(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))
    logger = operations.logger
    orig_prop = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO):
            operations.show_tables()
    finally:
        logger.propagate = orig_prop
    assert "person" in caplog.text

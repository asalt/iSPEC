"""Tests for logging utilities."""

import logging

from ispec.logging import get_logger, reset_logger


def test_reset_logger_allows_reconfiguration(tmp_path):
    log1 = tmp_path / "first.log"
    log2 = tmp_path / "second.log"

    # Initial configuration writes to the first file
    logger = get_logger("test", log_file=log1, console=False)
    logger.info("first message")
    for handler in logger.handlers:
        handler.flush()

    assert "first message" in log1.read_text()

    # Reset and ensure logger has no handlers
    reset_logger()
    assert logging.getLogger("test").handlers == []

    # Reconfigure to write to the second file
    logger2 = get_logger("test", log_file=log2, console=False)
    logger2.info("second message")
    for handler in logger2.handlers:
        handler.flush()

    assert "second message" in log2.read_text()
    # Ensure the old file did not receive the new message
    assert "second message" not in log1.read_text()


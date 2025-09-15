"""Tests for the threaded task queue."""

import time
from threading import Thread
from unittest.mock import patch

from ispec.ai.task_queue import TaskQueue


def test_task_queue_executes_tasks():
    queue = TaskQueue()
    queue.start()
    results = []
    queue.add_task(results.append, 1)
    queue.add_task(results.append, 2)
    queue.join()
    queue.stop()
    assert results == [1, 2]


def test_start_called_only_once():
    """Starting the queue twice should not spawn extra threads."""

    queue = TaskQueue(max_workers=2)
    with patch("ispec.ai.task_queue.Thread.start") as mock_start:
        queue.start()
        queue.start()
        queue.stop()
    assert mock_start.call_count == 2


def test_tasks_accept_kwargs():
    queue = TaskQueue()
    queue.start()

    results = []

    def record(a, b=None):
        results.append((a, b))

    queue.add_task(record, 1, b=2)
    queue.join()
    queue.stop()

    assert results == [(1, 2)]


def test_exception_does_not_stop_queue():
    queue = TaskQueue()
    queue.start()

    results = []

    def boom():
        results.append("boom")
        raise ValueError("boom")

    def ok():
        results.append("ok")

    queue.add_task(boom)
    queue.add_task(ok)

    queue.join()
    queue.stop()

    assert results == ["boom", "ok"]


def test_task_error_reporting():
    queue = TaskQueue()
    with patch("ispec.ai.task_queue.logger.exception") as mock_log:
        queue.start()

        def boom():
            raise ValueError("boom")

        task = queue.add_task(boom)
        queue.join()
        queue.stop()

        mock_log.assert_called_once()

    assert isinstance(task.error, ValueError)
    errors = queue.get_errors()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_get_errors_clear():
    queue = TaskQueue()
    queue.start()

    def boom():
        raise ValueError("boom")

    queue.add_task(boom)
    queue.join()
    queue.stop()

    assert queue.get_errors()  # initial error present
    cleared = queue.get_errors(clear=True)
    assert len(cleared) == 1
    assert isinstance(cleared[0], ValueError)
    assert queue.get_errors() == []


def test_concurrency_controls():
    queue = TaskQueue(max_workers=2)
    queue.start()

    start = time.time()
    for _ in range(2):
        queue.add_task(time.sleep, 0.2)
    queue.join()
    queue.stop()
    elapsed = time.time() - start
    assert elapsed < 0.35


def test_stop_before_start():
    queue = TaskQueue()
    # Should not raise even though the queue hasn't been started
    queue.stop()

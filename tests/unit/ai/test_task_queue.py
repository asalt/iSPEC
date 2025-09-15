"""Tests for the threaded task queue."""

from unittest.mock import MagicMock

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

    queue = TaskQueue()
    original_start = queue._thread.start
    queue._thread.start = MagicMock(side_effect=original_start)

    queue.start()
    queue.start()
    queue.stop()

    assert queue._thread.start.call_count == 1


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


def test_stop_before_start():
    queue = TaskQueue()
    # Should not raise even though the queue hasn't been started
    queue.stop()

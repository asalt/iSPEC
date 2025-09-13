"""Tests for the threaded task queue."""

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

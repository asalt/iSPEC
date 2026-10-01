"""Unit tests for the thread-based task queue."""

from __future__ import annotations

import sys
import time
import types


if "sqlalchemy" not in sys.modules:
    sqlalchemy_stub = types.ModuleType("sqlalchemy")
    orm_stub = types.ModuleType("sqlalchemy.orm")
    engine_stub = types.ModuleType("sqlalchemy.engine")

    class _Session:
        def commit(self) -> None:  # pragma: no cover - simple stub
            pass

        def rollback(self) -> None:  # pragma: no cover - simple stub
            pass

        def close(self) -> None:  # pragma: no cover - simple stub
            pass

    def _sessionmaker(*args: object, **kwargs: object):  # pragma: no cover - simple stub
        class _SessionFactory:
            def __call__(self, *args: object, **kwargs: object) -> _Session:
                return _Session()

        return _SessionFactory()

    class _Engine:  # pragma: no cover - simple stub
        pass

    orm_stub.Session = _Session
    orm_stub.sessionmaker = _sessionmaker
    engine_stub.Engine = _Engine
    sqlalchemy_stub.orm = orm_stub
    sqlalchemy_stub.engine = engine_stub

    sys.modules.setdefault("sqlalchemy", sqlalchemy_stub)
    sys.modules.setdefault("sqlalchemy.orm", orm_stub)
    sys.modules.setdefault("sqlalchemy.engine", engine_stub)

if "ispec.db" not in sys.modules:
    db_stub = types.ModuleType("ispec.db")

    def _get_session_stub(*args: object, **kwargs: object) -> None:  # pragma: no cover - stub
        raise RuntimeError("database access not available in unit tests")

    db_stub.get_session = _get_session_stub  # type: ignore[attr-defined]
    db_stub.__all__ = ["get_session"]
    db_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["ispec.db"] = db_stub

from ispec.ai.task_queue import TaskQueue


def test_task_queue_executes_tasks_in_order() -> None:
    """Tasks submitted to the queue should execute and preserve order."""

    queue = TaskQueue(max_workers=1)
    results: list[str] = []

    try:
        queue.add_task(results.append, "first")
        queue.add_task(results.append, "second")
        queue.join()

        assert results == ["first", "second"]
    finally:
        queue.stop()


def test_task_queue_allows_concurrent_execution() -> None:
    """Multiple workers should run tasks in parallel when available."""

    queue = TaskQueue(max_workers=2)

    try:
        start = time.monotonic()
        for _ in range(2):
            queue.add_task(time.sleep, 0.2)

        queue.join()
        elapsed = time.monotonic() - start

        # With two workers the total runtime should be roughly the duration of
        # a single sleep, not the sum of both sleeps.
        assert elapsed < 0.35
    finally:
        queue.stop()


def test_task_queue_collects_and_clears_errors() -> None:
    """Exceptions raised by tasks should be tracked and retrievable."""

    queue = TaskQueue()

    try:
        def boom() -> None:
            raise RuntimeError("boom")

        task = queue.add_task(boom)
        queue.join()

        errors = queue.get_errors()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert isinstance(task.error, RuntimeError)
        assert task.error is errors[0]

        cleared = queue.get_errors(clear=True)
        assert cleared == errors
        assert queue.get_errors() == []
    finally:
        queue.stop()

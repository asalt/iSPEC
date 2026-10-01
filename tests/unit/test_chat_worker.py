"""Unit tests for the chat worker using mocked dependencies."""

from __future__ import annotations

import sys
import types
from types import MethodType


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

import ispec.ai.worker as worker_module


def test_chat_worker_enqueues_and_processes_messages(monkeypatch) -> None:
    """Messages should be enqueued and processed with mocked dependencies."""

    llm_inputs = []
    backend_calls = []

    def fake_generate_response(session):
        llm_inputs.append(session)
        return session.add_ai_message("mocked reply")

    def fake_put_response(url, data):
        backend_calls.append((url, data))

    monkeypatch.setattr(worker_module, "generate_response", fake_generate_response)
    monkeypatch.setattr(worker_module, "put_response", fake_put_response)

    queue = worker_module.TaskQueue(max_workers=1)
    original_add_task = queue.add_task
    scheduled = []

    def spy_add_task(self, func, *args, **kwargs):
        scheduled.append(func)
        return original_add_task(func, *args, **kwargs)

    queue.add_task = MethodType(spy_add_task, queue)
    worker = worker_module.ChatWorker(backend_url="http://backend", queue=queue)

    try:
        worker.enqueue("hello world")
        worker.queue.join()
    finally:
        worker.stop()

    assert len(scheduled) == 1
    scheduled_call = scheduled[0]
    assert getattr(scheduled_call, "__self__", None) is worker
    assert (
        getattr(getattr(scheduled_call, "__func__", None), "__name__", None)
        == "_process_message"
    )

    assert len(llm_inputs) == 1
    llm_session = llm_inputs[0]
    assert llm_session.messages[-1].sender == "user"
    assert llm_session.messages[-1].content == "hello world"

    conversation = [(msg.sender, msg.content) for msg in worker.session.messages]
    assert conversation == [("user", "hello world"), ("ai", "mocked reply")]

    assert backend_calls == [("http://backend", {"response": "mocked reply"})]
    assert worker.queue.get_errors() == []


def test_chat_worker_ignores_backend_when_url_missing(monkeypatch) -> None:
    """No backend call should be issued when ``backend_url`` is ``None``."""

    llm_inputs = []
    backend_calls = []

    def fake_generate_response(session):
        llm_inputs.append(session)
        return session.add_ai_message("no backend reply")

    def fake_put_response(url, data):
        backend_calls.append((url, data))

    monkeypatch.setattr(worker_module, "generate_response", fake_generate_response)
    monkeypatch.setattr(worker_module, "put_response", fake_put_response)

    worker = worker_module.ChatWorker(backend_url=None)

    try:
        worker.enqueue("ping")
        worker.queue.join()
    finally:
        worker.stop()

    assert len(llm_inputs) == 1
    llm_session = llm_inputs[0]
    assert llm_session.messages[-1].sender == "user"
    assert llm_session.messages[-1].content == "ping"

    conversation = [(msg.sender, msg.content) for msg in worker.session.messages]
    assert conversation == [("user", "ping"), ("ai", "no backend reply")]

    assert backend_calls == []
    assert worker.queue.get_errors() == []

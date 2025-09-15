"""Tests for the chat worker and backend integration."""

import ispec.ai.worker as worker_module
from ispec.ai import ChatWorker


def test_chat_worker_sends_backend(monkeypatch):
    sent = []

    def fake_put(url, data):
        sent.append((url, data))

    monkeypatch.setattr(worker_module, "put_response", fake_put)
    worker = ChatWorker(backend_url="http://example.com/api")
    worker.start()
    worker.enqueue("hi")
    worker.queue.join()
    worker.stop()

    assert sent and sent[0][0] == "http://example.com/api"
    # ensure AI message generated
    assert worker.session.messages[-1].sender == "ai"


def test_chat_worker_no_backend_no_call(monkeypatch):
    called = False

    def fake_put(url, data):
        nonlocal called
        called = True

    monkeypatch.setattr(worker_module, "put_response", fake_put)
    worker = ChatWorker(backend_url=None)
    worker.start()
    worker.enqueue("hi")
    worker.queue.join()
    worker.stop()

    assert called is False
    assert worker.session.messages[-1].sender == "ai"


def test_chat_worker_alternating_messages(monkeypatch):
    # Stub out network interaction just in case
    monkeypatch.setattr(worker_module, "put_response", lambda *a, **k: None)

    worker = ChatWorker()
    worker.start()
    messages = ["one", "two", "three"]
    for m in messages:
        worker.enqueue(m)
    worker.queue.join()
    worker.stop()

    senders = [msg.sender for msg in worker.session.messages]
    assert senders == ["user", "ai"] * len(messages)

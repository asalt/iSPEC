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

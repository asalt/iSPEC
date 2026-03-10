from __future__ import annotations

import queue
import threading

from ispec.concurrency.thread_context import assert_main_thread, is_main_thread, set_main_thread
from ispec.concurrency.worker_wrapper import run_as_worker


def test_assert_main_thread_raises_on_worker_thread() -> None:
    set_main_thread(owner="pytest")
    assert is_main_thread() is True

    out: "queue.Queue[Exception | None]" = queue.Queue()

    def _worker() -> None:
        try:
            assert_main_thread("test.worker")
            out.put(None)
        except Exception as exc:
            out.put(exc)

    thread = threading.Thread(target=_worker, name="pytest-worker", daemon=True)
    thread.start()
    thread.join(timeout=2.0)

    result = out.get(timeout=1.0)
    assert isinstance(result, RuntimeError)


def test_run_as_worker_attaches_thread_identity() -> None:
    out: "queue.Queue[dict]" = queue.Queue()

    def _worker() -> None:
        result = run_as_worker("example-worker", lambda: {"ok": True, "value": 1})
        out.put(result)

    thread = threading.Thread(target=_worker, name="pytest-worker", daemon=True)
    thread.start()
    thread.join(timeout=2.0)

    result = out.get(timeout=1.0)
    assert result["ok"] is True
    assert result["value"] == 1
    assert isinstance(result.get("_thread"), dict)
    assert result["_thread"]["role"] == "worker"
    assert result["_thread"]["name"] == "example-worker"
    assert isinstance(result["_thread"]["ident"], int)

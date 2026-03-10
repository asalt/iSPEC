"""Helpers for running spun-off work in worker threads.

The main rule: worker threads do not write to the agent DB. They return small
result dicts which the main thread aggregates and persists.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from ispec.concurrency.thread_context import current_thread_info


def run_as_worker(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``fn`` in a worker context, attaching thread identity to the result."""

    prev_name = threading.current_thread().name
    try:
        if isinstance(name, str) and name.strip():
            try:
                threading.current_thread().name = str(name)
            except Exception:
                pass

        result: Any = fn()
        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": "Worker returned non-dict result.",
                "_thread": current_thread_info(role="worker"),
            }

        if "_thread" not in result:
            result = {**result, "_thread": current_thread_info(role="worker")}
        return result
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "_thread": current_thread_info(role="worker"),
        }
    finally:
        try:
            threading.current_thread().name = prev_name
        except Exception:
            pass


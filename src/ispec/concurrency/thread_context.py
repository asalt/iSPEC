"""Thread identity tracking and main-thread assertions.

The supervisor/orchestrator code relies on a strict "main thread owns writes"
invariant. This module provides a tiny runtime contract:

- The supervisor sets the main thread once at startup via ``set_main_thread``.
- Any code path that mutates agent DB state should call ``assert_main_thread``.
- Spun-off work should tag itself via ``current_thread_info(role="worker")``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

from ispec.logging import get_logger

logger = get_logger(__file__)


@dataclass(frozen=True)
class ThreadIdentity:
    ident: int
    native_id: int | None
    name: str

    def as_dict(self, *, role: str, owner: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": role,
            "ident": int(self.ident),
            "native_id": int(self.native_id) if self.native_id is not None else None,
            "name": str(self.name or ""),
        }
        if owner:
            payload["owner"] = str(owner)
        return payload


_MAIN_LOCK = threading.Lock()
_MAIN_THREAD: ThreadIdentity | None = None
_MAIN_OWNER: str | None = None


def _current_native_id() -> int | None:
    # Python 3.8+ provides threading.get_native_id(); keep a fallback for safety.
    getter = getattr(threading, "get_native_id", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return None
    native_id = getattr(threading.current_thread(), "native_id", None)
    if isinstance(native_id, int):
        return int(native_id)
    return None


def set_main_thread(*, owner: str) -> dict[str, Any]:
    """Declare the current thread as the main supervisor/orchestrator thread."""

    global _MAIN_THREAD, _MAIN_OWNER
    with _MAIN_LOCK:
        ident = int(threading.get_ident())
        native_id = _current_native_id()
        name = str(threading.current_thread().name or "")

        if _MAIN_THREAD is not None and _MAIN_THREAD.ident != ident:
            logger.warning(
                "Main thread identity changed old_ident=%s old_name=%s new_ident=%s new_name=%s owner=%s",
                _MAIN_THREAD.ident,
                _MAIN_THREAD.name,
                ident,
                name,
                owner,
            )

        _MAIN_THREAD = ThreadIdentity(ident=ident, native_id=native_id, name=name)
        _MAIN_OWNER = str(owner or "").strip() or None

        return _MAIN_THREAD.as_dict(role="main", owner=_MAIN_OWNER)


def main_thread_info() -> dict[str, Any] | None:
    with _MAIN_LOCK:
        if _MAIN_THREAD is None:
            return None
        return _MAIN_THREAD.as_dict(role="main", owner=_MAIN_OWNER)


def current_thread_info(*, role: Literal["main", "worker", "unknown"]) -> dict[str, Any]:
    ident = int(threading.get_ident())
    native_id = _current_native_id()
    name = str(threading.current_thread().name or "")
    return {
        "role": str(role),
        "ident": ident,
        "native_id": native_id,
        "name": name,
    }


def is_main_thread() -> bool:
    with _MAIN_LOCK:
        main = _MAIN_THREAD
    if main is None:
        return False
    return int(threading.get_ident()) == int(main.ident)


def assert_main_thread(context: str, *, mode: Literal["raise", "log"] = "raise") -> None:
    """Assert that the caller is running on the configured main thread.

    This is intentionally strict. Code that wants to write to shared state or
    the agent DB must run on the main thread.
    """

    with _MAIN_LOCK:
        main = _MAIN_THREAD
        owner = _MAIN_OWNER

    current = current_thread_info(role="unknown")
    if main is None:
        msg = f"Main thread not set (context={context}, current_ident={current.get('ident')}, current_name={current.get('name')})."
        if mode == "log":
            logger.warning(msg)
            return
        raise RuntimeError(msg)

    if int(current.get("ident") or 0) != int(main.ident):
        msg = (
            "Main-thread assertion failed "
            f"(context={context}, owner={owner}, main_ident={main.ident}, main_name={main.name}, "
            f"current_ident={current.get('ident')}, current_name={current.get('name')})."
        )
        if mode == "log":
            logger.warning(msg)
            return
        raise RuntimeError(msg)


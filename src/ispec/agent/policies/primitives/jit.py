from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, TypeVar, cast

_TRUTHY = {"1", "true", "yes", "y", "on"}

F = TypeVar("F", bound=Callable[..., Any])


def _numba_enabled() -> bool:
    raw = (os.getenv("ISPEC_AGENT_ENABLE_NUMBA") or "").strip()
    return raw.lower() in _TRUTHY


@lru_cache(maxsize=None)
def _get_numba_njit() -> Any | None:
    if not _numba_enabled():
        return None
    try:
        from numba import njit  # type: ignore[import-not-found]
    except Exception:
        return None
    return njit


def jit_if_available(
    fn: F,
    *,
    cache: bool = False,
    fastmath: bool = False,
) -> F:
    """Optionally JIT-compile a pure numeric function if Numba is available.

    This is a dev/perf escape hatch: it is disabled by default and only enabled
    when ``ISPEC_AGENT_ENABLE_NUMBA`` is truthy *and* Numba is importable.
    """

    njit = _get_numba_njit()
    if njit is None:
        return fn

    try:
        compiled = njit(cache=cache, fastmath=fastmath)(fn)
    except Exception:
        return fn

    return cast(F, compiled)


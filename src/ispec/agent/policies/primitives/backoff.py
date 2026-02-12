from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExponentialBackoffParams(BaseModel):
    """Parameters for a simple exponential backoff family.

    Delay: ``base_seconds * factor ** clamp(step - start_step, 0, max_exp)``

    Notes:
    - ``start_step`` lets you control whether the first "step" uses exponent 0
      (e.g. idle_streak=1 -> base_seconds when start_step=1).
    - Keep this deterministic by default (no randomness/jitter).
    """

    base_seconds: float = Field(..., gt=0)
    factor: float = Field(2.0, gt=1e-9)
    start_step: int = Field(0, ge=0)
    max_exp: int = Field(6, ge=0, le=60)
    cap_seconds: float | None = Field(default=None, gt=0)


def backoff_exponential_current(
    step: int,
    *,
    base_seconds: float,
    factor: float = 2.0,
    start_step: int = 0,
    max_exp: int = 6,
    cap_seconds: float | None = None,
) -> float:
    """Return an exponential backoff delay in seconds (deterministic).

    Intended usage patterns:
    - error streak: ``start_step=0`` so streak=1 yields base*2.
    - idle streak: ``start_step=1`` so streak=1 yields base.
    """

    if step < 0:
        raise ValueError("step must be >= 0")
    if base_seconds <= 0:
        raise ValueError("base_seconds must be > 0")
    if factor <= 0:
        raise ValueError("factor must be > 0")
    if start_step < 0:
        raise ValueError("start_step must be >= 0")
    if max_exp < 0:
        raise ValueError("max_exp must be >= 0")
    if cap_seconds is not None and cap_seconds <= 0:
        raise ValueError("cap_seconds must be > 0 when provided")

    effective = int(step) - int(start_step)
    if effective < 0:
        effective = 0
    exp = min(int(max_exp), effective)
    delay = float(base_seconds) * (float(factor) ** float(exp))

    if cap_seconds is not None:
        delay = min(float(cap_seconds), delay)
    return float(delay)


def apply_backoff_current(
    step: int,
    *,
    params: ExponentialBackoffParams | dict[str, Any],
) -> float:
    parsed = params if isinstance(params, ExponentialBackoffParams) else ExponentialBackoffParams.model_validate(params)
    return backoff_exponential_current(
        int(step),
        base_seconds=float(parsed.base_seconds),
        factor=float(parsed.factor),
        start_step=int(parsed.start_step),
        max_exp=int(parsed.max_exp),
        cap_seconds=parsed.cap_seconds,
    )


from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .expr import ExprKernelSpec, compile_expr_kernel_current


class StretchedExponentialParams(BaseModel):
    """Parameters for a stretched exponential (Weibull survival) decay kernel.

    Kernel: ``m(dt) = exp(-ln(2) * (dt / half_life) ** shape)``

    Properties:
    - ``m(0) == 1``
    - ``m(half_life) == 0.5`` for any positive ``shape``
    - ``shape == 1`` reduces to classic exponential decay
    """

    half_life_seconds: float = Field(..., gt=0)
    shape: float = Field(1.0, gt=0)


class ValueBounds(BaseModel):
    """Optional bounds applied to the *value* after a transformation."""

    floor: float | None = Field(default=None)
    cap: float | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ValueBounds":
        floor = self.floor
        cap = self.cap
        if floor is not None and cap is not None and floor > cap:
            raise ValueError("floor must be <= cap")
        return self


_KERNEL_STRETCHED_EXPONENTIAL_SPEC_CURRENT = ExprKernelSpec(
    expr="exp(-ln2 * (dt_seconds / half_life_seconds) ** shape)",
    args=("dt_seconds", "half_life_seconds", "shape"),
    tier="basic",
)

_kernel_stretched_exponential_impl = compile_expr_kernel_current(
    _KERNEL_STRETCHED_EXPONENTIAL_SPEC_CURRENT,
    backend="auto",
)


def kernel_stretched_exponential_current(
    dt_seconds: float,
    *,
    half_life_seconds: float,
    shape: float = 1.0,
) -> float:
    """Return a decay multiplier in ``[0, 1]`` for elapsed time ``dt_seconds``."""

    if dt_seconds < 0:
        raise ValueError("dt_seconds must be >= 0")
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    if shape <= 0:
        raise ValueError("shape must be > 0")
    if dt_seconds == 0:
        return 1.0

    multiplier = _kernel_stretched_exponential_impl(float(dt_seconds), float(half_life_seconds), float(shape))

    if multiplier <= 0:
        return 0.0
    if multiplier >= 1:
        return 1.0
    return float(multiplier)


def apply_value_bounds_current(
    value: float,
    *,
    floor: float | None = None,
    cap: float | None = None,
) -> float:
    """Apply optional (floor, cap) bounds to a value."""

    bounded = float(value)
    if floor is not None:
        bounded = max(float(floor), bounded)
    if cap is not None:
        bounded = min(float(cap), bounded)
    return float(bounded)


def apply_decay_current(
    value: float,
    dt_seconds: float,
    *,
    params: StretchedExponentialParams | dict[str, Any],
    bounds: ValueBounds | dict[str, Any] | None = None,
) -> float:
    """Apply stretched-exponential decay to a value, then apply optional bounds."""

    kernel_params = (
        params if isinstance(params, StretchedExponentialParams) else StretchedExponentialParams.model_validate(params)
    )
    bound_params = None
    if bounds is not None:
        bound_params = bounds if isinstance(bounds, ValueBounds) else ValueBounds.model_validate(bounds)

    multiplier = kernel_stretched_exponential_current(
        dt_seconds,
        half_life_seconds=float(kernel_params.half_life_seconds),
        shape=float(kernel_params.shape),
    )
    decayed = float(value) * float(multiplier)

    if bound_params is None:
        return float(decayed)
    return apply_value_bounds_current(decayed, floor=bound_params.floor, cap=bound_params.cap)

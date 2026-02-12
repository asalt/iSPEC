from __future__ import annotations

import math

import pytest

from ispec.agent.policies.primitives.decay import (
    StretchedExponentialParams,
    ValueBounds,
    apply_decay_current,
    apply_value_bounds_current,
    kernel_stretched_exponential_current,
)


def test_kernel_returns_expected_values_for_exponential_case() -> None:
    half_life = 10.0
    assert kernel_stretched_exponential_current(0.0, half_life_seconds=half_life, shape=1.0) == 1.0
    assert kernel_stretched_exponential_current(half_life, half_life_seconds=half_life, shape=1.0) == pytest.approx(
        0.5
    )
    assert kernel_stretched_exponential_current(2 * half_life, half_life_seconds=half_life, shape=1.0) == pytest.approx(
        0.25
    )


def test_kernel_shape_parameter_controls_tail_behavior() -> None:
    half_life = 10.0
    dt = 2 * half_life

    heavier_tail = kernel_stretched_exponential_current(dt, half_life_seconds=half_life, shape=0.5)
    exponential = kernel_stretched_exponential_current(dt, half_life_seconds=half_life, shape=1.0)
    faster_drop = kernel_stretched_exponential_current(dt, half_life_seconds=half_life, shape=2.0)

    assert heavier_tail > exponential > faster_drop

    expected_faster_drop = math.exp(-math.log(2.0) * (2.0**2))
    assert faster_drop == pytest.approx(expected_faster_drop)


def test_apply_value_bounds_current_bounds_value_not_multiplier() -> None:
    assert apply_value_bounds_current(0.2, floor=0.3, cap=0.9) == 0.3
    assert apply_value_bounds_current(1.2, floor=0.3, cap=0.9) == 0.9
    assert apply_value_bounds_current(0.5, floor=0.3, cap=0.9) == 0.5


def test_apply_decay_current_applies_kernel_then_value_bounds() -> None:
    params = StretchedExponentialParams(half_life_seconds=10.0, shape=1.0)
    bounds = ValueBounds(floor=0.3, cap=0.9)

    # multiplier at half-life is 0.5; 0.4 * 0.5 floors to 0.3
    assert apply_decay_current(0.4, 10.0, params=params, bounds=bounds) == pytest.approx(0.3)

    # multiplier at half-life is 0.5; 2.0 * 0.5 caps to 0.9
    assert apply_decay_current(2.0, 10.0, params=params, bounds=bounds) == pytest.approx(0.9)

    # without bounds, return the raw decayed value
    assert apply_decay_current(1.0, 10.0, params=params, bounds=None) == pytest.approx(0.5)


def test_apply_decay_current_accepts_dict_params() -> None:
    assert apply_decay_current(
        0.4,
        10.0,
        params={"half_life_seconds": 10.0, "shape": 1.0},
        bounds={"floor": 0.3, "cap": 0.9},
    ) == pytest.approx(0.3)


def test_value_bounds_validation_rejects_floor_greater_than_cap() -> None:
    with pytest.raises(ValueError, match="floor must be <= cap"):
        ValueBounds(floor=1.0, cap=0.0)


def test_kernel_rejects_negative_elapsed_time() -> None:
    with pytest.raises(ValueError, match="dt_seconds must be >= 0"):
        kernel_stretched_exponential_current(-1.0, half_life_seconds=10.0, shape=1.0)

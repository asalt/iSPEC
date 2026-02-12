from __future__ import annotations

import pytest

from ispec.agent.policies.primitives.backoff import backoff_exponential_current


def test_backoff_exponential_current_defaults_match_error_streak_semantics() -> None:
    # start_step=0 => streak=1 doubles the base.
    assert backoff_exponential_current(0, base_seconds=60.0) == pytest.approx(60.0)
    assert backoff_exponential_current(1, base_seconds=60.0) == pytest.approx(120.0)
    assert backoff_exponential_current(2, base_seconds=60.0) == pytest.approx(240.0)


def test_backoff_exponential_current_supports_idle_streak_semantics_via_start_step() -> None:
    # start_step=1 => idle_streak=1 returns base.
    assert backoff_exponential_current(1, base_seconds=60.0, start_step=1) == pytest.approx(60.0)
    assert backoff_exponential_current(2, base_seconds=60.0, start_step=1) == pytest.approx(120.0)


def test_backoff_exponential_current_applies_max_exp_and_cap_seconds() -> None:
    assert backoff_exponential_current(10, base_seconds=60.0, max_exp=2) == pytest.approx(240.0)
    assert backoff_exponential_current(10, base_seconds=60.0, max_exp=2, cap_seconds=100.0) == pytest.approx(100.0)


def test_backoff_exponential_current_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="step must be >= 0"):
        backoff_exponential_current(-1, base_seconds=60.0)


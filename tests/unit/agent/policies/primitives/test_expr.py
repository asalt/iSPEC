from __future__ import annotations

import pytest

from ispec.agent.policies.primitives.expr import ExprKernelSpec, compile_expr_kernel_current


def test_compile_basic_kernel_returns_expected_values() -> None:
    spec = ExprKernelSpec(
        expr="exp(-ln2 * (dt_seconds / half_life_seconds) ** shape)",
        args=("dt_seconds", "half_life_seconds", "shape"),
        tier="basic",
    )
    fn = compile_expr_kernel_current(spec, backend="python")

    assert fn(0.0, 10.0, 1.0) == pytest.approx(1.0)
    assert fn(10.0, 10.0, 1.0) == pytest.approx(0.5)
    assert fn(20.0, 10.0, 1.0) == pytest.approx(0.25)


def test_compile_rejects_attribute_access_even_on_allowed_names() -> None:
    with pytest.raises(ValueError, match="unsupported expression node: Attribute"):
        ExprKernelSpec(expr="exp.__globals__", args=("x",), tier="basic")


def test_compile_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown name: os"):
        ExprKernelSpec(expr="os", args=("x",), tier="basic")


def test_basic_tier_rejects_comparisons() -> None:
    with pytest.raises(ValueError, match="comparisons are not allowed"):
        ExprKernelSpec(expr="x < 0", args=("x",), tier="basic")


def test_extended_tier_supports_ifelse_and_clamp() -> None:
    spec = ExprKernelSpec(
        expr="ifelse(x < 0, 0.0, clamp(x, 0.0, 1.0))",
        args=("x",),
        tier="extended",
    )
    fn = compile_expr_kernel_current(spec, backend="python")

    assert fn(-1.0) == pytest.approx(0.0)
    assert fn(0.5) == pytest.approx(0.5)
    assert fn(2.0) == pytest.approx(1.0)


def test_extended_tier_supports_piecewise() -> None:
    spec = ExprKernelSpec(
        expr="piecewise(x < 0, 0.0, x < 1, x, 1.0)",
        args=("x",),
        tier="extended",
    )
    fn = compile_expr_kernel_current(spec, backend="python")

    assert fn(-1.0) == pytest.approx(0.0)
    assert fn(0.25) == pytest.approx(0.25)
    assert fn(2.0) == pytest.approx(1.0)


def test_compile_rejects_keyword_arguments() -> None:
    with pytest.raises(ValueError, match="keyword arguments are not supported"):
        ExprKernelSpec(
            expr="clamp(x, lo=0.0, hi=1.0)",
            args=("x",),
            tier="extended",
        )


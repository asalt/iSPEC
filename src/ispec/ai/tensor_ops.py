"""Minimal tensor utilities using only Python built-ins."""

from __future__ import annotations

from typing import Iterable, Sequence, List

Matrix = Sequence[Sequence[float]]


def transpose(matrix: Matrix) -> List[List[float]]:
    """Return the transpose of ``matrix``.

    Parameters
    ----------
    matrix:
        Matrix represented as a sequence of sequences.
    """

    return [list(row) for row in zip(*matrix)]


def matmul(a: Matrix, b: Matrix) -> List[List[float]]:
    """Multiply two matrices ``a`` and ``b``.

    The implementation operates purely on Python lists so it can be used
    without additional numerical dependencies.
    """

    if not a or not b:
        raise ValueError("Input matrices must be non-empty")
    if len(a[0]) != len(b):
        raise ValueError("Incompatible shapes for matrix multiplication")

    result: List[List[float]] = []
    for row in a:
        new_row = []
        for col in zip(*b):
            new_row.append(sum(x * y for x, y in zip(row, col)))
        result.append(new_row)
    return result

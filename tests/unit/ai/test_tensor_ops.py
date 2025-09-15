"""Tests for simple tensor operations."""

import pytest

from ispec.ai.tensor_ops import matmul, transpose


def test_transpose():
    matrix = [[1, 2, 3], [4, 5, 6]]
    assert transpose(matrix) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_single_row():
    matrix = [[1, 2, 3]]
    assert transpose(matrix) == [[1], [2], [3]]


def test_transpose_single_element():
    matrix = [[42]]
    assert transpose(matrix) == [[42]]


def test_matmul():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert matmul(a, b) == [[19, 22], [43, 50]]


def test_matmul_raises_on_empty_input():
    with pytest.raises(ValueError):
        matmul([], [])


def test_matmul_raises_on_shape_mismatch():
    a = [[1, 2], [3, 4]]
    b = [[1], [2], [3]]
    with pytest.raises(ValueError):
        matmul(a, b)

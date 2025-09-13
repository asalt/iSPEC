"""Tests for simple tensor operations."""

from ispec.ai import matmul, transpose


def test_transpose():
    matrix = [[1, 2, 3], [4, 5, 6]]
    assert transpose(matrix) == [[1, 4], [2, 5], [3, 6]]


def test_matmul():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert matmul(a, b) == [[19, 22], [43, 50]]

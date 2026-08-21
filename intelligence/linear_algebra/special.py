from __future__ import annotations

import math

from .matrix import Matrix


def is_symmetric(A: Matrix, tol: float = 1e-10) -> bool:
    if A.rows != A.cols:
        return False
    for i in range(A.rows):
        for j in range(i + 1, A.cols):
            if abs(A._data[i][j] - A._data[j][i]) > tol:
                return False
    return True


def is_orthogonal(A: Matrix, tol: float = 1e-10) -> bool:
    if A.rows != A.cols:
        return False
    AtA = (A.transpose() @ A)
    n = A.rows
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(AtA._data[i][j] - expected) > tol:
                return False
    return True


def is_positive_definite(A: Matrix) -> bool:
    if A.rows != A.cols:
        return False
    n = A.rows
    for i in range(n):
        det_val = Matrix([row[:i + 1] for row in A._data[:i + 1]]).determinant()
        if det_val <= 1e-14:
            return False
    return True


def is_diagonal(A: Matrix) -> bool:
    for i in range(A.rows):
        for j in range(A.cols):
            if i != j and abs(A._data[i][j]) > 1e-14:
                return False
    return True


def is_triangular(A: Matrix) -> bool:
    if A.rows != A.cols:
        return False
    is_upper = all(abs(A._data[i][j]) < 1e-14 for i in range(A.rows) for j in range(i))
    is_lower = all(abs(A._data[i][j]) < 1e-14 for i in range(A.rows) for j in range(i + 1, A.cols))
    return is_upper or is_lower


def vandermonde(x: list[float]) -> Matrix:
    n = len(x)
    return Matrix([[xi ** (n - 1 - j) for j in range(n)] for xi in x])


def rotation_2d(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return Matrix([[c, -s], [s, c]])


def hilbert(n: int) -> Matrix:
    return Matrix([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def toeplitz(row: list[float], col: list[float]) -> Matrix:
    n = len(row)
    return Matrix([[row[i - j] if i >= j else col[j - i] for j in range(n)] for i in range(n)])

from __future__ import annotations

import math

from .matrix import DimensionError, Matrix


def dot_product(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        raise DimensionError("Vectors must have same length")
    return sum(a * b for a, b in zip(v1, v2))


def cross_product(v1: list[float], v2: list[float]) -> list[float]:
    if len(v1) != 3 or len(v2) != 3:
        raise DimensionError("Cross product requires 3D vectors")
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def vector_norm(v: list[float], p: int | float = 2) -> float:
    if p == 1:
        return sum(abs(x) for x in v)
    if p == 2:
        return math.sqrt(sum(x * x for x in v))
    return sum(abs(x) ** p for x in v) ** (1.0 / p)


def normalize(v: list[float]) -> list[float]:
    n = vector_norm(v)
    if n < 1e-15:
        raise DimensionError("Cannot normalize zero vector")
    return [x / n for x in v]


def angle_between(v1: list[float], v2: list[float]) -> float:
    d = dot_product(v1, v2)
    n1 = vector_norm(v1)
    n2 = vector_norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return 0.0
    cos_a = max(-1.0, min(1.0, d / (n1 * n2)))
    return math.acos(cos_a)


def outer_product(v1: list[float], v2: list[float]) -> Matrix:
    return Matrix([[a * b for b in v2] for a in v1])


def kronecker_product(A: Matrix, B: Matrix) -> Matrix:
    result = []
    for i in range(A.rows):
        for k in range(B.rows):
            row = []
            for j in range(A.cols):
                for l in range(B.cols):
                    row.append(A._data[i][j] * B._data[k][l])
            result.append(row)
    return Matrix(result)


def hadamard_product(A: Matrix, B: Matrix) -> Matrix:
    if A.rows != B.rows or A.cols != B.cols:
        raise DimensionError("Dimension mismatch for Hadamard product")
    return Matrix([[A._data[i][j] * B._data[i][j] for j in range(A.cols)] for i in range(A.rows)])

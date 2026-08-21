from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass


class DimensionError(Exception):
    """Raised when matrix dimensions are incompatible."""


class SingularMatrixError(Exception):
    """Raised when matrix is singular and cannot be inverted."""


class NonSquareMatrixError(Exception):
    """Raised when operation requires a square matrix."""


@dataclass
class Matrix:
    _data: list[list[float]]
    rows: int
    cols: int

    def __init__(self, data: list[list[float]]) -> None:
        if not data or not data[0]:
            self._data = [[]]
            self.rows = 0
            self.cols = 0
            return
        self.rows = len(data)
        self.cols = len(data[0])
        for row in data:
            if len(row) != self.cols:
                raise DimensionError("All rows must have equal length")
        self._data = [list(row) for row in data]

    @classmethod
    def from_rows(cls, rows: list[list[float]]) -> Matrix:
        return cls(rows)

    @classmethod
    def from_cols(cls, cols: list[list[float]]) -> Matrix:
        if not cols:
            return cls([[]])
        rows = len(cols[0])
        for c in cols:
            if len(c) != rows:
                raise DimensionError("All columns must have equal length")
        return cls([[cols[r][i] for r in range(rows)] for i in range(len(cols))])

    @classmethod
    def identity(cls, n: int) -> Matrix:
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, r: int, c: int) -> Matrix:
        return cls([[0.0] * c for _ in range(r)])

    @classmethod
    def ones(cls, r: int, c: int) -> Matrix:
        return cls([[1.0] * c for _ in range(r)])

    @classmethod
    def diagonal(cls, values: list[float]) -> Matrix:
        n = len(values)
        return cls([[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)])

    @classmethod
    def random(cls, r: int, c: int, low: float = 0.0, high: float = 1.0, seed: int | None = None) -> Matrix:
        rng = random.Random(seed)
        return cls([[rng.uniform(low, high) for _ in range(c)] for _ in range(r)])

    def __getitem__(self, idx: tuple[int, int]) -> float:
        return self._data[idx[0]][idx[1]]

    def __setitem__(self, idx: tuple[int, int], val: float) -> None:
        self._data[idx[0]][idx[1]] = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self._data[i][j] - other._data[i][j]) > 1e-10:
                    return False
        return True

    def __repr__(self) -> str:
        lines = []
        for row in self._data:
            lines.append("[" + ", ".join(f"{v:8.4f}" for v in row) + "]")
        return "Matrix([\n" + "\n".join(lines) + "\n])"

    def copy(self) -> Matrix:
        return Matrix([row[:] for row in self._data])

    def transpose(self) -> Matrix:
        return Matrix([[self._data[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def trace(self) -> float:
        if self.rows != self.cols:
            raise NonSquareMatrixError("Trace requires square matrix")
        return sum(self._data[i][i] for i in range(self.rows))

    def flatten(self) -> list[float]:
        return [v for row in self._data for v in row]

    def reshape(self, r: int, c: int) -> Matrix:
        flat = self.flatten()
        if r * c != len(flat):
            raise DimensionError("Cannot reshape to different total size")
        return Matrix([flat[i * c:(i + 1) * c] for i in range(r)])

    def map(self, fn: Callable[[float], float]) -> Matrix:
        return Matrix([[fn(v) for v in row] for row in self._data])

    def norm(self, kind: str = "frobenius") -> float:
        if kind == "frobenius":
            return math.sqrt(sum(v * v for row in self._data for v in row))
        raise ValueError(f"Unknown norm: {kind}")

    def __add__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionError("Dimension mismatch for addition")
        return Matrix([
            [self._data[i][j] + other._data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionError("Dimension mismatch for subtraction")
        return Matrix([
            [self._data[i][j] - other._data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __mul__(self, other: Matrix | float) -> Matrix:
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise DimensionError("Dimension mismatch for element-wise multiply")
            return Matrix([
                [self._data[i][j] * other._data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ])
        return Matrix([[v * other for v in row] for row in self._data])

    def __rmul__(self, other: float) -> Matrix:
        return self.__mul__(other)

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.cols != other.rows:
            raise DimensionError(
                f"Cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}"
            )
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self._data[i][k] * other._data[k][j]
                result[i][j] = s
        return Matrix(result)

    def determinant(self) -> float:
        if self.rows != self.cols:
            raise NonSquareMatrixError("Determinant requires square matrix")
        n = self.rows
        if n == 0:
            return 1.0
        if n == 1:
            return self._data[0][0]
        if n == 2:
            return self._data[0][0] * self._data[1][1] - self._data[0][1] * self._data[1][0]
        mat = [row[:] for row in self._data]
        det = 1.0
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(mat[row][col]) > abs(mat[max_row][col]):
                    max_row = row
            if abs(mat[max_row][col]) < 1e-14:
                return 0.0
            if max_row != col:
                mat[col], mat[max_row] = mat[max_row], mat[col]
                det *= -1.0
            det *= mat[col][col]
            for row in range(col + 1, n):
                factor = mat[row][col] / mat[col][col]
                for c in range(col, n):
                    mat[row][c] -= factor * mat[col][c]
        return det

    def inverse(self) -> Matrix:
        if self.rows != self.cols:
            raise NonSquareMatrixError("Inverse requires square matrix")
        n = self.rows
        aug = [self._data[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            if abs(aug[max_row][col]) < 1e-14:
                raise SingularMatrixError("Matrix is singular")
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            aug[col] = [v / pivot for v in aug[col]]
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
        return Matrix([row[n:] for row in aug])

    def rank(self) -> int:
        mat = [row[:] for row in self._data]
        r = 0
        for col in range(self.cols):
            max_row = r
            for row in range(r + 1, self.rows):
                if abs(mat[row][col]) > abs(mat[max_row][col]):
                    max_row = row
            if abs(mat[max_row][col]) < 1e-12:
                continue
            mat[r], mat[max_row] = mat[max_row], mat[r]
            pivot = mat[r][col]
            for j in range(self.cols):
                mat[r][j] /= pivot
            for row in range(self.rows):
                if row == r:
                    continue
                factor = mat[row][col]
                for j in range(self.cols):
                    mat[row][j] -= factor * mat[r][j]
            r += 1
        return r

    def slice_rows(self, start: int, end: int) -> Matrix:
        return Matrix(self._data[start:end])

    def slice_cols(self, start: int, end: int) -> Matrix:
        return Matrix([row[start:end] for row in self._data])

    def concat_horizontal(self, other: Matrix) -> Matrix:
        if self.rows != other.rows:
            raise DimensionError("Horizontal concat requires same row count")
        return Matrix([self._data[i][:] + other._data[i][:] for i in range(self.rows)])

    def concat_vertical(self, other: Matrix) -> Matrix:
        if self.cols != other.cols:
            raise DimensionError("Vertical concat requires same column count")
        return Matrix(self._data + other._data)

    def to_list(self) -> list[list[float]]:
        return [row[:] for row in self._data]

    def is_square(self) -> bool:
        return self.rows == self.cols

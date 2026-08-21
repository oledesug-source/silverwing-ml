from __future__ import annotations

from dataclasses import dataclass

from .matrix import DimensionError, Matrix


@dataclass
class SparseMatrix:
    row_indices: list[int]
    col_indices: list[int]
    values: list[float]
    rows: int
    cols: int

    @classmethod
    def from_dense(cls, mat: Matrix) -> SparseMatrix:
        rows_list = []
        cols_list = []
        vals = []
        for i in range(mat.rows):
            for j in range(mat.cols):
                if abs(mat._data[i][j]) > 1e-15:
                    rows_list.append(i)
                    cols_list.append(j)
                    vals.append(mat._data[i][j])
        return cls(rows_list, cols_list, vals, mat.rows, mat.cols)

    def to_dense(self) -> Matrix:
        data = [[0.0] * self.cols for _ in range(self.rows)]
        for r, c, v in zip(self.row_indices, self.col_indices, self.values):
            data[r][c] = v
        return Matrix(data)

    def get(self, i: int, j: int) -> float:
        for r, c, v in zip(self.row_indices, self.col_indices, self.values):
            if r == i and c == j:
                return v
        return 0.0

    def transpose(self) -> SparseMatrix:
        return SparseMatrix(
            self.col_indices[:], self.row_indices[:], self.values[:],
            self.cols, self.rows,
        )

    def matvec(self, vector: list[float]) -> list[float]:
        if len(vector) != self.cols:
            raise DimensionError("Vector length must match matrix columns")
        result = [0.0] * self.rows
        for r, c, v in zip(self.row_indices, self.col_indices, self.values):
            result[r] += v * vector[c]
        return result

    def nnz(self) -> int:
        return len(self.values)

    def density(self) -> float:
        total = self.rows * self.cols
        return self.nnz() / total if total > 0 else 0.0

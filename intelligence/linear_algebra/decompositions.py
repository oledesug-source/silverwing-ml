from __future__ import annotations

import math

from .matrix import Matrix, NonSquareMatrixError


def lu_decomposition(A: Matrix) -> tuple[Matrix, Matrix, list[int]]:
    n = A.rows
    if n != A.cols:
        raise NonSquareMatrixError("LU requires square matrix")
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in A._data]
    pivots = list(range(n))
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(U[row][col]) > abs(U[max_row][col]):
                max_row = row
        if max_row != col:
            U[col], U[max_row] = U[max_row], U[col]
            pivots[col], pivots[max_row] = pivots[max_row], pivots[col]
            for r in range(col):
                L[col][r], L[max_row][r] = L[max_row][r], L[col][r]
        if abs(U[col][col]) < 1e-14:
            continue
        for row in range(col + 1, n):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for c in range(col, n):
                U[row][c] -= factor * U[col][c]
    for i in range(n):
        L[i][i] = 1.0
    return Matrix(L), Matrix(U), pivots


def qr_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    m, n = A.rows, A.cols
    Q = [A._data[i][:] for i in range(m)]
    R = [[0.0] * n for _ in range(n)]
    for j in range(min(m, n)):
        for i in range(j):
            dot = sum(Q[k][j] * Q[k][i] for k in range(m))
            R[i][j] = dot
            for k in range(m):
                Q[k][j] -= dot * Q[k][i]
        norm_val = math.sqrt(sum(Q[k][j] * Q[k][j] for k in range(m)))
        R[j][j] = norm_val
        if norm_val < 1e-14:
            continue
        for k in range(m):
            Q[k][j] /= norm_val
    return Matrix(Q), Matrix(R)


def cholesky(A: Matrix) -> Matrix:
    n = A.rows
    if n != A.cols:
        raise NonSquareMatrixError("Cholesky requires square matrix")
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A._data[i][i] - s
                if val < 1e-14:
                    raise NonSquareMatrixError("Matrix is not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                if abs(L[j][j]) < 1e-14:
                    raise NonSquareMatrixError("Matrix is not positive definite")
                L[i][j] = (A._data[i][j] - s) / L[j][j]
    return Matrix(L)


def eigen_decomposition(A: Matrix, max_iter: int = 300) -> tuple[list[float], Matrix]:
    n = A.rows
    if n != A.cols:
        raise NonSquareMatrixError("Eigen requires square matrix")
    mat = [row[:] for row in A._data]
    eigvecs = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_iter):
        Q, R = qr_decomposition(Matrix(mat))
        mat_new = (R @ Q)._data
        eigvecs_new = (Matrix(eigvecs) @ Q)._data
        mat = mat_new
        eigvecs = eigvecs_new
        off_diag = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag += abs(mat[i][j])
        if off_diag < 1e-10:
            break
    eigenvalues = [mat[i][i] for i in range(n)]
    return eigenvalues, Matrix(eigvecs)


def LDL_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    n = A.rows
    L = [[0.0] * n for _ in range(n)]
    D = [0.0] * n
    for i in range(n):
        D[i] = A._data[i][i] - sum(L[i][k] * L[i][k] * D[k] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (A._data[j][i] - sum(L[j][k] * L[i][k] * D[k] for k in range(i))) / D[i]
    for i in range(n):
        L[i][i] = 1.0
    return Matrix(L), Matrix.diagonal(D)

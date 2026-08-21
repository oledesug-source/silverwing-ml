from __future__ import annotations

from .decompositions import eigen_decomposition
from .matrix import Matrix


def solve_lower_triangular(L: Matrix, b: list[float]) -> list[float]:
    n = L.rows
    x = [0.0] * n
    for i in range(n):
        s = sum(L._data[i][j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / L._data[i][i]
    return x


def solve_upper_triangular(U: Matrix, b: list[float]) -> list[float]:
    n = U.rows
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U._data[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / U._data[i][i]
    return x


def solve(A: Matrix, b: list[float]) -> list[float]:
    from .decompositions import lu_decomposition
    L, U, pivots = lu_decomposition(A)
    pb = [b[pivots[i]] for i in range(len(b))]
    y = solve_lower_triangular(L, pb)
    return solve_upper_triangular(U, y)


def solve_iterative(A: Matrix, b: list[float], method: str = "jacobi",
                    tol: float = 1e-10, max_iter: int = 1000) -> list[float]:
    n = A.rows
    x = [0.0] * n
    for _ in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            if abs(A._data[i][i]) < 1e-14:
                continue
            s = sum(A._data[i][j] * (x[j] if method == "gauss_seidel" else x_new[j])
                    for j in range(i))
            s += sum(A._data[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s) / A._data[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new
        x = x_new
    return x


def least_squares(A: Matrix, b: list[float]) -> list[float]:
    At = A.transpose()
    AtA = At @ A
    Atb = [sum(At._data[i][j] * b[j] for j in range(len(b))) for i in range(AtA.rows)]
    return solve(AtA, Atb)


def pseudo_inverse(A: Matrix) -> Matrix:
    At = A.transpose()
    AtA = At @ A
    n = AtA.cols
    eigenvalues, eigenvectors = eigen_decomposition(AtA)
    inv_eigenvalues = [1.0 / ev if abs(ev) > 1e-14 else 0.0 for ev in eigenvalues]
    inv_eig = Matrix([[eigenvectors._data[i][j] * inv_eigenvalues[j]
                       for j in range(n)] for i in range(n)])
    AtA_inv = inv_eig @ eigenvectors.transpose()
    return AtA_inv @ At


def eigenvalues(A: Matrix, max_iter: int = 300) -> list[float]:
    vals, _ = eigen_decomposition(A, max_iter)
    return vals

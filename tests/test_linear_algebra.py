import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.linear_algebra import (
    DimensionError,
    Matrix,
    SingularMatrixError,
    SparseMatrix,
    angle_between,
    cholesky,
    cross_product,
    dot_product,
    eigen_decomposition,
    eigenvalues,
    hadamard_product,
    hilbert,
    is_diagonal,
    is_positive_definite,
    is_symmetric,
    is_triangular,
    kronecker_product,
    least_squares,
    lu_decomposition,
    normalize,
    outer_product,
    qr_decomposition,
    rotation_2d,
    solve,
    solve_lower_triangular,
    solve_upper_triangular,
    toeplitz,
    vandermonde,
    vector_norm,
)
from intelligence.linear_algebra.solvers import solve_iterative


class TestMatrixCreation:
    def test_zeros(self):
        m = Matrix.zeros(3, 4)
        assert m.rows == 3
        assert m.cols == 4
        assert all(m[i, j] == 0.0 for i in range(3) for j in range(4))

    def test_identity(self):
        I = Matrix.identity(3)
        assert I.rows == 3
        assert I.cols == 3
        assert I[0, 0] == 1.0
        assert I[1, 1] == 1.0
        assert I[0, 1] == 0.0

    def test_diagonal(self):
        D = Matrix.diagonal([1, 2, 3])
        assert D[0, 0] == 1
        assert D[1, 1] == 2
        assert D[2, 2] == 3
        assert D[0, 1] == 0

    def test_from_rows(self):
        m = Matrix.from_rows([[1, 2], [3, 4]])
        assert m.rows == 2
        assert m[1, 0] == 3

    def test_from_cols(self):
        m = Matrix.from_cols([[1, 3], [2, 4]])
        assert m[0, 0] == 1
        assert m[0, 1] == 2
        assert m[1, 0] == 3

    def test_ones(self):
        m = Matrix.ones(2, 3)
        assert all(m[i, j] == 1.0 for i in range(2) for j in range(3))

    def test_random(self):
        m = Matrix.random(3, 3, seed=42)
        assert m.rows == 3

    def test_non_rectangular(self):
        with pytest.raises(DimensionError):
            Matrix([[1, 2], [3]])


class TestMatrixOperations:
    def test_add(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = A + B
        assert C[0, 0] == 6
        assert C[1, 1] == 12

    def test_sub(self):
        A = Matrix([[5, 6], [7, 8]])
        B = Matrix([[1, 2], [3, 4]])
        C = A - B
        assert C[0, 0] == 4
        assert C[1, 1] == 4

    def test_elementwise_mul(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[2, 3], [4, 5]])
        C = A * B
        assert C[0, 0] == 2
        assert C[1, 1] == 20

    def test_matmul(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = A @ B
        assert C[0, 0] == 19
        assert C[0, 1] == 22
        assert C[1, 0] == 43
        assert C[1, 1] == 50

    def test_scalar_mul(self):
        A = Matrix([[1, 2], [3, 4]])
        C = A * 2.0
        assert C[0, 0] == 2
        assert C[1, 1] == 8

    def test_rmul(self):
        A = Matrix([[1, 2], [3, 4]])
        C = 3.0 * A
        assert C[0, 0] == 3

    def test_transpose(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        At = A.transpose()
        assert At.rows == 3
        assert At.cols == 2
        assert At[0, 1] == 4

    def test_trace(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.trace() == 5

    def test_determinant_2x2(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.determinant() == -2

    def test_determinant_3x3(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert abs(A.determinant()) < 1e-10

    def test_determinant_identity(self):
        assert Matrix.identity(3).determinant() == 1.0

    def test_inverse(self):
        A = Matrix([[1, 2], [3, 4]])
        Ainv = A.inverse()
        I = A @ Ainv
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                assert abs(I[i, j] - expected) < 1e-10

    def test_inverse_3x3(self):
        A = Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
        Ainv = A.inverse()
        I = A @ Ainv
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(I[i, j] - expected) < 1e-10

    def test_singular_inverse(self):
        A = Matrix([[1, 2], [2, 4]])
        with pytest.raises(SingularMatrixError):
            A.inverse()

    def test_rank_full(self):
        assert Matrix([[1, 0], [0, 1]]).rank() == 2

    def test_rank_deficient(self):
        assert Matrix([[1, 2], [2, 4]]).rank() == 1

    def test_norm(self):
        A = Matrix([[3, 4]])
        assert abs(A.norm() - 5.0) < 1e-10

    def test_equality(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[1, 2], [3, 4]])
        assert A == B

    def test_inequality(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[1, 2], [3, 5]])
        assert A != B

    def test_flatten(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.flatten() == [1, 2, 3, 4]

    def test_reshape(self):
        A = Matrix([[1, 2, 3, 4]])
        B = A.reshape(2, 2)
        assert B.rows == 2
        assert B.cols == 2

    def test_map(self):
        A = Matrix([[1, 2], [3, 4]])
        B = A.map(lambda x: x * 2)
        assert B[0, 0] == 2
        assert B[1, 1] == 8


class TestVectorOps:
    def test_dot(self):
        assert dot_product([1, 2, 3], [4, 5, 6]) == 32

    def test_cross(self):
        r = cross_product([1, 0, 0], [0, 1, 0])
        assert r == [0, 0, 1]

    def test_norm(self):
        assert abs(vector_norm([3, 4]) - 5.0) < 1e-10

    def test_norm_l1(self):
        assert vector_norm([1, -2, 3], p=1) == 6

    def test_normalize(self):
        n = normalize([3, 4])
        assert abs(vector_norm(n) - 1.0) < 1e-10

    def test_angle(self):
        a = angle_between([1, 0], [0, 1])
        assert abs(a - math.pi / 2) < 1e-10

    def test_outer(self):
        m = outer_product([1, 2], [3, 4])
        assert m.rows == 2
        assert m.cols == 2
        assert m[0, 0] == 3
        assert m[1, 1] == 8

    def test_hadamard(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[2, 3], [4, 5]])
        C = hadamard_product(A, B)
        assert C[0, 0] == 2
        assert C[1, 1] == 20

    def test_kronecker(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[0, 1], [1, 0]])
        C = kronecker_product(A, B)
        assert C.rows == 4
        assert C.cols == 4


class TestDecompositions:
    def test_lu(self):
        A = Matrix([[2, 1], [5, 3]])
        L, U, pivots = lu_decomposition(A)
        P_inv = Matrix.zeros(2, 2)
        for i in range(2):
            P_inv[pivots[i], i] = 1.0
        PA = P_inv @ A
        LU = L @ U
        for i in range(2):
            for j in range(2):
                assert abs(PA[i, j] - LU[i, j]) < 1e-10

    def test_qr(self):
        A = Matrix([[1, 0], [0, 2]])
        Q, R = qr_decomposition(A)
        I_est = Q @ Q.transpose()
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                assert abs(I_est[i, j] - expected) < 1e-10

    def test_cholesky(self):
        A = Matrix([[4, 2], [2, 3]])
        L = cholesky(A)
        C = L @ L.transpose()
        assert abs(C[0, 0] - 4) < 1e-10
        assert abs(C[0, 1] - 2) < 1e-10

    def test_eigen(self):
        A = Matrix([[2, 1], [1, 2]])
        vals, vecs = eigen_decomposition(A)
        vals.sort()
        assert abs(vals[0] - 1.0) < 1e-6
        assert abs(vals[1] - 3.0) < 1e-6


class TestSolvers:
    def test_lower_triangular(self):
        L = Matrix([[2, 0], [1, 3]])
        b = [4, 8]
        x = solve_lower_triangular(L, b)
        assert abs(x[0] - 2) < 1e-10
        assert abs(x[1] - 2) < 1e-10

    def test_upper_triangular(self):
        U = Matrix([[2, 1], [0, 3]])
        b = [5, 6]
        x = solve_upper_triangular(U, b)
        assert abs(x[0] - 1.5) < 1e-10
        assert abs(x[1] - 2.0) < 1e-10

    def test_solve(self):
        A = Matrix([[2, 1], [1, 3]])
        b = [5, 7]
        x = solve(A, b)
        assert abs(A[0, 0] * x[0] + A[0, 1] * x[1] - 5) < 1e-10
        assert abs(A[1, 0] * x[0] + A[1, 1] * x[1] - 7) < 1e-10

    def test_solve_iterative_jacobi(self):
        A = Matrix([[4, 1], [1, 3]])
        b = [5, 4]
        x = solve_iterative(A, b, method="jacobi")
        assert abs(A[0, 0] * x[0] + A[0, 1] * x[1] - 5) < 1e-6

    def test_solve_iterative_gauss_seidel(self):
        A = Matrix([[4, 1], [1, 3]])
        b = [5, 4]
        x = solve_iterative(A, b, method="gauss_seidel")
        assert abs(A[0, 0] * x[0] + A[0, 1] * x[1] - 5) < 1e-6

    def test_least_squares(self):
        A = Matrix([[1, 0], [1, 1], [1, 2]])
        b = [1, 2, 3]
        x = least_squares(A, b)
        assert abs(x[0] - 1.0) < 1e-6
        assert abs(x[1] - 1.0) < 1e-6

    def test_eigenvalues(self):
        A = Matrix([[1, 0], [0, 2]])
        vals = eigenvalues(A)
        vals.sort()
        assert abs(vals[0] - 1.0) < 1e-6
        assert abs(vals[1] - 2.0) < 1e-6


class TestSpecialMatrices:
    def test_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [2, 1]]))
        assert not is_symmetric(Matrix([[1, 2], [3, 4]]))

    def test_positive_definite(self):
        assert is_positive_definite(Matrix([[2, 1], [1, 3]]))

    def test_diagonal(self):
        assert is_diagonal(Matrix.diagonal([1, 2, 3]))
        assert not is_diagonal(Matrix([[1, 2], [3, 4]]))

    def test_triangular(self):
        assert is_triangular(Matrix([[1, 2], [0, 3]]))
        assert is_triangular(Matrix([[1, 0], [2, 3]]))

    def test_vandermonde(self):
        V = vandermonde([1, 2, 3])
        assert V.rows == 3
        assert V.cols == 3
        assert V[0, 0] == 1
        assert V[2, 0] == 9

    def test_rotation_2d(self):
        R = rotation_2d(0)
        assert abs(R[0, 0] - 1.0) < 1e-10
        assert abs(R[1, 1] - 1.0) < 1e-10
        assert abs(R[0, 1]) < 1e-10

    def test_hilbert(self):
        H = hilbert(3)
        assert H.rows == 3
        assert abs(H[0, 0] - 1.0) < 1e-10
        assert abs(H[0, 1] - 0.5) < 1e-10

    def test_toeplitz(self):
        T = toeplitz([1, 2, 3], [1, 0, -1])
        assert T[0, 0] == 1
        assert T[1, 0] == 2
        assert T[0, 1] == 0


class TestSparseMatrix:
    def test_from_dense(self):
        A = Matrix([[1, 0, 2], [0, 0, 3], [4, 0, 0]])
        S = SparseMatrix.from_dense(A)
        assert S.nnz() == 4

    def test_to_dense(self):
        A = Matrix([[1, 0], [0, 2]])
        S = SparseMatrix.from_dense(A)
        B = S.to_dense()
        assert B == A

    def test_get(self):
        A = Matrix([[1, 0], [0, 2]])
        S = SparseMatrix.from_dense(A)
        assert S.get(0, 0) == 1
        assert S.get(0, 1) == 0
        assert S.get(1, 1) == 2

    def test_transpose(self):
        A = Matrix([[1, 0, 3], [0, 2, 0]])
        S = SparseMatrix.from_dense(A)
        St = S.transpose()
        assert St.rows == 3
        assert St.cols == 2
        assert St.get(2, 0) == 3

    def test_matvec(self):
        A = Matrix([[1, 2], [3, 4]])
        S = SparseMatrix.from_dense(A)
        r = S.matvec([1, 1])
        assert r == [3.0, 7.0]

    def test_density(self):
        A = Matrix([[0, 0], [0, 1]])
        S = SparseMatrix.from_dense(A)
        assert abs(S.density() - 0.25) < 1e-10

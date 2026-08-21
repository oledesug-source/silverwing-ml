from .decompositions import (
    LDL_decomposition,
    cholesky,
    eigen_decomposition,
    lu_decomposition,
    qr_decomposition,
)
from .matrix import DimensionError, Matrix, NonSquareMatrixError, SingularMatrixError
from .operations import (
    angle_between,
    cross_product,
    dot_product,
    hadamard_product,
    kronecker_product,
    normalize,
    outer_product,
    vector_norm,
)
from .solvers import (
    eigenvalues,
    least_squares,
    pseudo_inverse,
    solve,
    solve_iterative,
    solve_lower_triangular,
    solve_upper_triangular,
)
from .sparse import SparseMatrix
from .special import (
    hilbert,
    is_diagonal,
    is_orthogonal,
    is_positive_definite,
    is_symmetric,
    is_triangular,
    rotation_2d,
    toeplitz,
    vandermonde,
)

__all__ = [
    "Matrix", "DimensionError", "SingularMatrixError", "NonSquareMatrixError",
    "dot_product", "cross_product", "vector_norm", "normalize",
    "angle_between", "outer_product", "kronecker_product", "hadamard_product",
    "lu_decomposition", "qr_decomposition", "cholesky", "eigen_decomposition",
    "LDL_decomposition",
    "solve_lower_triangular", "solve_upper_triangular", "solve",
    "solve_iterative", "least_squares", "pseudo_inverse", "eigenvalues",
    "is_symmetric", "is_orthogonal", "is_positive_definite",
    "is_diagonal", "is_triangular", "vandermonde", "rotation_2d",
    "hilbert", "toeplitz",
    "SparseMatrix",
]

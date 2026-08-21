"""Linear models for regression and classification."""

from __future__ import annotations

import math

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "LogisticRegression",
    "MultiClassLogisticRegression",
]


def _matmul_transpose(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Compute A^T * B."""
    rows_a = len(A[0])
    cols_b = len(B[0])
    rows_b = len(B)
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(rows_b):
                s += A[k][i] * B[k][j]
            result[i][j] = s
    return result


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication."""
    rows_a = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def _transpose(A: list[list[float]]) -> list[list[float]]:
    """Matrix transpose."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _mat_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector multiplication."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    """Vector subtraction."""
    return [x - y for x, y in zip(a, b)]


def _inverse(M: list[list[float]]) -> list[list[float]]:
    """Matrix inverse via Gauss-Jordan elimination."""
    n = len(M)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n:] for i in range(n)]


def _sigmoid(z: float) -> float:
    """Sigmoid activation."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class LinearRegression:
    """Ordinary least squares linear regression via normal equation."""

    def __init__(self):
        self.coefficients: list[float] = []
        self.intercept: float = 0.0
        self.residuals: list[float] = []

    def fit(self, X: list[list[float]], y: list[float]) -> LinearRegression:
        """Fit model using the normal equation."""
        n = len(X)
        p = len(X[0])
        ones = [[1.0] for _ in range(n)]
        Xb = [ones[i] + list(X[i]) for i in range(n)]
        Xt = _transpose(Xb)
        XtX = _matmul(Xt, Xb)
        Xty = _mat_vec_mul(Xt, y)
        try:
            beta = _mat_vec_mul(_inverse(XtX), Xty)
        except Exception:
            beta = [0.0] * (p + 1)
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        y_pred = self.predict(X)
        self.residuals = [y[i] - y_pred[i] for i in range(n)]
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values."""
        return [
            self.intercept + sum(x[j] * self.coefficients[j] for j in range(len(x)))
            for x in X
        ]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    def summary(self) -> str:
        """Return string summary of the model."""
        lines = ["LinearRegression Summary", "=" * 30, f"Intercept: {self.intercept:.6f}"]
        for i, c in enumerate(self.coefficients):
            lines.append(f"  Feature {i}: {c:.6f}")
        if self.residuals:
            rm = sum(self.residuals) / len(self.residuals)
            lines.append(f"Mean Residual: {rm:.6f}")
        return "\n".join(lines)


class RidgeRegression:
    """Linear regression with L2 regularization."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def fit(self, X: list[list[float]], y: list[float], alpha: float | None = None) -> RidgeRegression:
        """Fit ridge regression."""
        if alpha is not None:
            self.alpha = alpha
        n = len(X)
        p = len(X[0])
        ones = [[1.0] for _ in range(n)]
        Xb = [ones[i] + list(X[i]) for i in range(n)]
        Xt = _transpose(Xb)
        XtX = _matmul(Xt, Xb)
        reg = [[self.alpha if i == j else 0.0 for j in range(len(XtX[0]))] for i in range(len(XtX))]
        XtX_reg = [[XtX[i][j] + reg[i][j] for j in range(len(XtX[0]))] for i in range(len(XtX))]
        Xty = _mat_vec_mul(Xt, y)
        try:
            beta = _mat_vec_mul(_inverse(XtX_reg), Xty)
        except Exception:
            beta = [0.0] * (p + 1)
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values."""
        return [
            self.intercept + sum(x[j] * self.coefficients[j] for j in range(len(x)))
            for x in X
        ]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {"alpha": self.alpha, "coefficients": list(self.coefficients), "intercept": self.intercept}

    def summary(self) -> str:
        """Return string summary."""
        lines = [f"RidgeRegression (alpha={self.alpha})", f"Intercept: {self.intercept:.6f}"]
        for i, c in enumerate(self.coefficients):
            lines.append(f"  Feature {i}: {c:.6f}")
        return "\n".join(lines)


class LassoRegression:
    """Linear regression with L1 regularization via coordinate descent."""

    def __init__(self, alpha: float = 1.0, lr: float = 0.01, epochs: int = 1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def fit(self, X: list[list[float]], y: list[float], alpha: float | None = None, lr: float | None = None, epochs: int | None = None) -> LassoRegression:
        """Fit lasso regression via gradient descent with L1 subgradient."""
        if alpha is not None:
            self.alpha = alpha
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs
        n = len(X)
        p = len(X[0])
        mean_y = sum(y) / n
        self.intercept = mean_y
        coeffs = [0.0] * p
        for _ in range(self.epochs):
            gradient = [0.0] * p
            for i in range(n):
                pred = self.intercept + sum(X[i][j] * coeffs[j] for j in range(p))
                err = pred - y[i]
                for j in range(p):
                    gradient[j] += err * X[i][j]
            for j in range(p):
                gradient[j] /= n
                grad = gradient[j]
                if coeffs[j] > 0:
                    grad += self.alpha
                elif coeffs[j] < 0:
                    grad -= self.alpha
                else:
                    if grad > self.alpha:
                        grad -= self.alpha
                    elif grad < -self.alpha:
                        grad += self.alpha
                    else:
                        grad = 0.0
                coeffs[j] -= self.lr * grad
        self.coefficients = coeffs
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values."""
        return [
            self.intercept + sum(x[j] * self.coefficients[j] for j in range(len(x)))
            for x in X
        ]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {"alpha": self.alpha, "lr": self.lr, "epochs": self.epochs, "coefficients": list(self.coefficients)}

    def summary(self) -> str:
        """Return string summary."""
        lines = [f"LassoRegression (alpha={self.alpha})", f"Intercept: {self.intercept:.6f}"]
        for i, c in enumerate(self.coefficients):
            lines.append(f"  Feature {i}: {c:.6f}")
        return "\n".join(lines)


class ElasticNet:
    """Linear regression with L1 + L2 regularization."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, lr: float = 0.01, epochs: int = 1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.epochs = epochs
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def fit(self, X: list[list[float]], y: list[float], alpha: float | None = None, l1_ratio: float | None = None, lr: float | None = None, epochs: int | None = None) -> ElasticNet:
        """Fit elastic net via gradient descent."""
        if alpha is not None:
            self.alpha = alpha
        if l1_ratio is not None:
            self.l1_ratio = l1_ratio
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs
        n = len(X)
        p = len(X[0])
        self.intercept = sum(y) / n
        coeffs = [0.0] * p
        l1_coeff = self.alpha * self.l1_ratio
        l2_coeff = self.alpha * (1.0 - self.l1_ratio)
        for _ in range(self.epochs):
            gradient = [0.0] * p
            for i in range(n):
                pred = self.intercept + sum(X[i][j] * coeffs[j] for j in range(p))
                err = pred - y[i]
                for j in range(p):
                    gradient[j] += err * X[i][j]
            for j in range(p):
                gradient[j] /= n
                grad = gradient[j] + l2_coeff * coeffs[j]
                if coeffs[j] > 0:
                    grad += l1_coeff
                elif coeffs[j] < 0:
                    grad -= l1_coeff
                else:
                    if grad > l1_coeff:
                        grad -= l1_coeff
                    elif grad < -l1_coeff:
                        grad += l1_coeff
                    else:
                        grad = 0.0
                coeffs[j] -= self.lr * grad
        self.coefficients = coeffs
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values."""
        return [
            self.intercept + sum(x[j] * self.coefficients[j] for j in range(len(x)))
            for x in X
        ]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "lr": self.lr,
            "epochs": self.epochs,
            "coefficients": list(self.coefficients),
        }

    def summary(self) -> str:
        """Return string summary."""
        lines = [f"ElasticNet (alpha={self.alpha}, l1_ratio={self.l1_ratio})", f"Intercept: {self.intercept:.6f}"]
        for i, c in enumerate(self.coefficients):
            lines.append(f"  Feature {i}: {c:.6f}")
        return "\n".join(lines)


class LogisticRegression:
    """Binary logistic regression trained via gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def fit(self, X: list[list[float]], y: list[int | float], lr: float | None = None, epochs: int | None = None) -> LogisticRegression:
        """Fit logistic regression via gradient descent."""
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs
        n = len(X)
        p = len(X[0])
        self.coefficients = [0.0] * p
        self.intercept = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * p
            grad_b = 0.0
            for i in range(n):
                z = self.intercept + sum(X[i][j] * self.coefficients[j] for j in range(p))
                pred = _sigmoid(z)
                err = pred - y[i]
                grad_b += err
                for j in range(p):
                    grad_w[j] += err * X[i][j]
            for j in range(p):
                self.coefficients[j] -= self.lr * grad_w[j] / n
            self.intercept -= self.lr * grad_b / n
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Predict probability of the positive class."""
        return [
            _sigmoid(self.intercept + sum(x[j] * self.coefficients[j] for j in range(len(x))))
            for x in X
        ]

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict binary class labels."""
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {"lr": self.lr, "epochs": self.epochs, "coefficients": list(self.coefficients), "intercept": self.intercept}

    def summary(self) -> str:
        """Return string summary."""
        lines = [f"LogisticRegression (lr={self.lr}, epochs={self.epochs})", f"Intercept: {self.intercept:.6f}"]
        for i, c in enumerate(self.coefficients):
            lines.append(f"  Feature {i}: {c:.6f}")
        return "\n".join(lines)


class MultiClassLogisticRegression:
    """Multi-class logistic regression using one-vs-rest strategy."""

    def __init__(self, lr: float = 0.1, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.classes_: list = []
        self._models: list[LogisticRegression] = []

    def fit(self, X: list[list[float]], y: list, lr: float | None = None, epochs: int | None = None) -> MultiClassLogisticRegression:
        """Fit multi-class logistic regression using one-vs-rest."""
        if lr is not None:
            self.lr = lr
        if epochs is not None:
            self.epochs = epochs
        self.classes_ = sorted(set(y))
        self._models = []
        for cls in self.classes_:
            binary_y = [1 if v == cls else 0 for v in y]
            model = LogisticRegression(lr=self.lr, epochs=self.epochs)
            model.fit(X, binary_y)
            self._models.append(model)
        return self

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return [self.classes_[max(range(len(p)), key=lambda c: p[c])] for p in probas]

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Predict probabilities for each class."""
        raw = [m.predict_proba(X) for m in self._models]
        n = len(X)
        k = len(self.classes_)
        result = [[raw[j][i] for j in range(k)] for i in range(n)]
        return result

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0

    def get_params(self) -> dict:
        """Return model parameters."""
        return {"lr": self.lr, "epochs": self.epochs, "classes": list(self.classes_)}

    def summary(self) -> str:
        """Return string summary."""
        return f"MultiClassLogisticRegression (classes={self.classes_}, lr={self.lr}, epochs={self.epochs})"

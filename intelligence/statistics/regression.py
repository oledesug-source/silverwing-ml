"""Regression analysis: linear, multiple, polynomial, logistic, ridge, lasso, residual diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "RegressionResult",
    "LogisticResult",
    "linear_regression",
    "multiple_regression",
    "polynomial_regression",
    "logistic_regression",
    "ridge_regression",
    "lasso_regression",
    "residual_analysis",
    "predict",
]


def _mean(data: list[float]) -> float:
    return sum(data) / len(data)


def _variance(data: list[float], ddof: int = 1) -> float:
    n = len(data)
    mu = _mean(data)
    return sum((x - mu) ** 2 for x in data) / (n - ddof)


@dataclass
class RegressionResult:
    """Result container for linear / polynomial / ridge / lasso regression."""

    coefficients: list[float]
    intercept: float
    r_squared: float
    residuals: list[float]
    std_error: float
    mse: float
    x_data: list[list[float]] = field(default_factory=list)
    y_data: list[float] = field(default_factory=list)

    @property
    def slope(self) -> float:
        """Return the first coefficient (slope for simple linear regression)."""
        return self.coefficients[0] if self.coefficients else 0.0


@dataclass
class LogisticResult:
    """Result container for logistic regression."""

    coefficients: list[float]
    intercept: float
    loss_history: list[float]
    accuracy: float

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        """Return predicted probabilities for each row in *x*."""
        return [_sigmoid(self.intercept + sum(c * v for c, v in zip(self.coefficients, row))) for row in x]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ex = math.exp(z)
    return ex / (1.0 + ex)


def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(row[i] * vec[i] for i in range(len(vec))) for row in mat]


def _transpose(mat: list[list[float]]) -> list[list[float]]:
    return [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat_inverse(mat: list[list[float]]) -> list[list[float]]:
    n = len(mat)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-14:
            pivot = 1e-14
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n:] for i in range(n)]


def linear_regression(x: list[float], y: list[float]) -> RegressionResult:
    """Perform simple OLS linear regression and return a RegressionResult."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length >= 2")
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        slope = 0.0
    else:
        slope = sxy / sxx
    intercept = my - slope * mx
    y_pred = [slope * xi + intercept for xi in x]
    residuals = [yi - pi for yi, pi in zip(y, y_pred)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mse = ss_res / (n - 2) if n > 2 else 0.0
    std_error = math.sqrt(mse) if mse > 0 else 0.0
    return RegressionResult(
        coefficients=[slope],
        intercept=intercept,
        r_squared=r_squared,
        residuals=residuals,
        std_error=std_error,
        mse=mse,
        x_data=[[xi] for xi in x],
        y_data=list(y),
    )


def multiple_regression(X: list[list[float]], y: list[float]) -> RegressionResult:
    """Perform OLS multiple regression: β = (XᵀX)⁻¹ Xᵀy."""
    n = len(y)
    if n == 0:
        raise ValueError("y must not be empty")
    k = len(X[0])
    Xt = _transpose(X)
    XtX = _mat_mul(Xt, X)
    Xty = _mat_vec_mul(Xt, y)
    inv_XtX = _mat_inverse(XtX)
    coefficients = _mat_vec_mul(inv_XtX, Xty)
    intercept = 0.0
    y_pred = _mat_vec_mul(X, coefficients)
    residuals = [yi - pi for yi, pi in zip(y, y_pred)]
    ss_res = sum(r ** 2 for r in residuals)
    my = _mean(y)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    p = k + 1
    mse = ss_res / (n - p) if n > p else 0.0
    std_error = math.sqrt(mse) if mse > 0 else 0.0
    return RegressionResult(
        coefficients=coefficients,
        intercept=intercept,
        r_squared=r_squared,
        residuals=residuals,
        std_error=std_error,
        mse=mse,
        x_data=[list(row) for row in X],
        y_data=list(y),
    )


def polynomial_regression(x: list[float], y: list[float], degree: int) -> RegressionResult:
    """Perform polynomial regression of given *degree* by expanding features."""
    if len(x) != len(y) or len(x) < degree + 1:
        raise ValueError("insufficient data for the given degree")
    X: list[list[float]] = []
    for xi in x:
        row = [xi ** d for d in range(1, degree + 1)]
        X.append(row)
    return multiple_regression(X, y)


def logistic_regression(x: list[list[float]], y: list[float], lr: float = 0.01, epochs: int = 1000) -> LogisticResult:
    """Perform binary logistic regression via gradient descent."""
    n = len(y)
    if n == 0:
        raise ValueError("data must not be empty")
    k = len(x[0]) if x else 0
    weights = [0.0] * k
    intercept = 0.0
    loss_history: list[float] = []
    for _ in range(epochs):
        grad_w = [0.0] * k
        grad_b = 0.0
        total_loss = 0.0
        for i in range(n):
            z = intercept + sum(weights[j] * x[i][j] for j in range(k))
            p = _sigmoid(z)
            err = p - y[i]
            for j in range(k):
                grad_w[j] += err * x[i][j]
            grad_b += err
            if y[i] == 1:
                total_loss -= math.log(max(p, 1e-15))
            else:
                total_loss -= math.log(max(1.0 - p, 1e-15))
        for j in range(k):
            weights[j] -= lr * grad_w[j] / n
        intercept -= lr * grad_b / n
        loss_history.append(total_loss / n)
    correct = 0
    for i in range(n):
        z = intercept + sum(weights[j] * x[i][j] for j in range(k))
        pred = 1 if _sigmoid(z) >= 0.5 else 0
        if pred == int(y[i]):
            correct += 1
    return LogisticResult(
        coefficients=weights,
        intercept=intercept,
        loss_history=loss_history,
        accuracy=correct / n,
    )


def ridge_regression(x: list[float], y: list[float], alpha: float) -> RegressionResult:
    """Perform ridge regression (OLS + L2 penalty on coefficients)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length >= 2")
    n = len(x)
    mx, my = _mean(x), _mean(y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    denom = sxx + alpha
    if denom == 0:
        slope = 0.0
    else:
        slope = sxy / denom
    intercept = my - slope * mx
    y_pred = [slope * xi + intercept for xi in x]
    residuals = [yi - pi for yi, pi in zip(y, y_pred)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mse = ss_res / (n - 2) if n > 2 else 0.0
    std_error = math.sqrt(mse) if mse > 0 else 0.0
    return RegressionResult(
        coefficients=[slope],
        intercept=intercept,
        r_squared=r_squared,
        residuals=residuals,
        std_error=std_error,
        mse=mse,
        x_data=[[xi] for xi in x],
        y_data=list(y),
    )


def lasso_regression(x: list[float], y: list[float], alpha: float, lr: float = 0.01, epochs: int = 1000) -> RegressionResult:
    """Perform LASSO regression (OLS + L1 penalty) via coordinate descent."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length >= 2")
    n = len(x)
    mx, my = _mean(x), _mean(y)
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / max(sum((xi - mx) ** 2 for xi in x), 1e-14)
    intercept = my - slope * mx
    for _ in range(epochs):
        y_pred = [slope * xi + intercept for xi in x]
        grad_slope = sum(-2.0 * xi * (yi - pi) for xi, yi, pi in zip(x, y, y_pred)) / n
        slope -= lr * (grad_slope + alpha * (1.0 if slope > 0 else (-1.0 if slope < 0 else 0.0)))
        grad_intercept = sum(-2.0 * (yi - pi) for yi, pi in zip(y, y_pred)) / n
        intercept -= lr * grad_intercept
    y_pred = [slope * xi + intercept for xi in x]
    residuals = [yi - pi for yi, pi in zip(y, y_pred)]
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mse = ss_res / (n - 2) if n > 2 else 0.0
    std_error = math.sqrt(mse) if mse > 0 else 0.0
    return RegressionResult(
        coefficients=[slope],
        intercept=intercept,
        r_squared=r_squared,
        residuals=residuals,
        std_error=std_error,
        mse=mse,
        x_data=[[xi] for xi in x],
        y_data=list(y),
    )


def residual_analysis(result: RegressionResult) -> dict:
    """Perform residual diagnostics: Durbin-Watson, normality check, heteroscedasticity hint."""
    r = result.residuals
    n = len(r)
    dw_num = sum((r[i] - r[i - 1]) ** 2 for i in range(1, n))
    dw_den = sum(ri ** 2 for ri in r)
    durbin_watson = dw_num / dw_den if dw_den != 0 else 0.0
    mu_r = _mean(r)
    m3 = sum((ri - mu_r) ** 3 for ri in r) / n
    m4 = sum((ri - mu_r) ** 4 for ri in r) / n
    m2 = sum((ri - mu_r) ** 2 for ri in r) / n
    skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurt = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0
    nrm = abs(skew) < 1.0 and abs(kurt) < 3.0
    abs_r = [abs(ri) for ri in r]
    if n >= 4:
        half = n // 2
        first_half_var = _variance(abs_r[:half]) if half > 1 else 0.0
        second_half_var = _variance(abs_r[half:]) if (n - half) > 1 else 0.0
        het_ratio = second_half_var / first_half_var if first_half_var > 0 else 1.0
    else:
        het_ratio = 1.0
    heteroscedastic = 0.5 < het_ratio < 2.0
    return {
        "durbin_watson": durbin_watson,
        "skewness": skew,
        "kurtosis": kurt,
        "normality_approximate": nrm,
        "heteroscedastic_ratio": het_ratio,
        "homoscedastic_approximate": heteroscedastic,
    }


def predict(model: RegressionResult | LogisticResult, x_new: list[list[float]]) -> list[float]:
    """Make predictions using a fitted model."""
    if isinstance(model, LogisticResult):
        return model.predict_proba(x_new)
    coefs = model.coefficients
    intercept = model.intercept
    return [intercept + sum(c * v for c, v in zip(coefs, row)) for row in x_new]

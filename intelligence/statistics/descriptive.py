"""Descriptive statistics: measures of central tendency, dispersion, position, and shape."""

from __future__ import annotations

import math

__all__ = [
    "mean",
    "median",
    "mode",
    "weighted_mean",
    "geometric_mean",
    "harmonic_mean",
    "trimmed_mean",
    "percentile",
    "quartiles",
    "iqr",
    "range_stats",
    "variance",
    "std_dev",
    "covariance",
    "correlation",
    "spearman_correlation",
    "skewness",
    "kurtosis",
    "moment",
    "central_moment",
    "frequency_distribution",
    "summary",
]


def mean(data: list[float]) -> float:
    """Return the arithmetic mean of *data*."""
    if not data:
        raise ValueError("data must not be empty")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Return the median of *data*."""
    if not data:
        raise ValueError("data must not be empty")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def mode(data: list) -> list:
    """Return a list of the most frequently occurring values in *data*."""
    if not data:
        raise ValueError("data must not be empty")
    counts: dict = {}
    for v in data:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    return [k for k, c in counts.items() if c == max_count]


def weighted_mean(values: list[float], weights: list[float]) -> float:
    """Return the weighted arithmetic mean."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        raise ValueError("data must not be empty")
    w_sum = sum(weights)
    if w_sum == 0.0:
        raise ValueError("sum of weights must not be zero")
    return sum(v * w for v, w in zip(values, weights)) / w_sum


def geometric_mean(data: list[float]) -> float:
    """Return the geometric mean of *data*."""
    if not data:
        raise ValueError("data must not be empty")
    if any(x <= 0 for x in data):
        raise ValueError("all values must be positive")
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


def harmonic_mean(data: list[float]) -> float:
    """Return the harmonic mean of *data*."""
    if not data:
        raise ValueError("data must not be empty")
    if any(x <= 0 for x in data):
        raise ValueError("all values must be positive")
    recip_sum = sum(1.0 / x for x in data)
    return len(data) / recip_sum


def trimmed_mean(data: list[float], proportion: float = 0.1) -> float:
    """Return the mean after trimming *proportion* fraction from each tail."""
    if not data:
        raise ValueError("data must not be empty")
    if not (0.0 <= proportion < 0.5):
        raise ValueError("proportion must be in [0, 0.5)")
    s = sorted(data)
    n = len(s)
    trim_count = int(n * proportion)
    trimmed = s[trim_count : n - trim_count] if trim_count > 0 else s
    return sum(trimmed) / len(trimmed)


def percentile(data: list[float], p: float) -> float:
    """Return the *p*-th percentile (0–100) using linear interpolation."""
    if not data:
        raise ValueError("data must not be empty")
    s = sorted(data)
    n = len(s)
    k = (p / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def quartiles(data: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) of *data*."""
    return (percentile(data, 25), percentile(data, 50), percentile(data, 75))


def iqr(data: list[float]) -> float:
    """Return the inter-quartile range Q3 - Q1."""
    q1, _, q3 = quartiles(data)
    return q3 - q1


def range_stats(data: list[float]) -> tuple[float, float]:
    """Return (min, max) of *data*."""
    if not data:
        raise ValueError("data must not be empty")
    return (min(data), max(data))


def variance(data: list[float], population: bool = False) -> float:
    """Return the variance of *data*.

    Use *population=True* for population variance, *False* (default) for sample variance (Bessel's correction).
    """
    if not data:
        raise ValueError("data must not be empty")
    mu = sum(data) / len(data)
    ss = sum((x - mu) ** 2 for x in data)
    denom = len(data) if population else len(data) - 1
    if denom == 0:
        return 0.0
    return ss / denom


def std_dev(data: list[float], population: bool = False) -> float:
    """Return the standard deviation of *data*."""
    return math.sqrt(variance(data, population))


def covariance(data_x: list[float], data_y: list[float]) -> float:
    """Return the sample covariance of two equal-length series."""
    if len(data_x) != len(data_y):
        raise ValueError("data_x and data_y must have the same length")
    n = len(data_x)
    if n < 2:
        raise ValueError("need at least two data points")
    mx = sum(data_x) / n
    my = sum(data_y) / n
    return sum((x - mx) * (y - my) for x, y in zip(data_x, data_y)) / (n - 1)


def correlation(data_x: list[float], data_y: list[float]) -> float:
    """Return the Pearson correlation coefficient."""
    if len(data_x) != len(data_y):
        raise ValueError("data_x and data_y must have the same length")
    n = len(data_x)
    if n < 2:
        raise ValueError("need at least two data points")
    mx = sum(data_x) / n
    my = sum(data_y) / n
    sxx = sum((x - mx) ** 2 for x in data_x)
    syy = sum((y - my) ** 2 for y in data_y)
    sxy = sum((x - mx) * (y - my) for x, y in zip(data_x, data_y))
    denom = math.sqrt(sxx * syy)
    if denom == 0.0:
        return 0.0
    return sxy / denom


def _rank(data: list[float]) -> list[float]:
    n = len(data)
    indexed = sorted(enumerate(data), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(data_x: list[float], data_y: list[float]) -> float:
    """Return the Spearman rank correlation coefficient."""
    return correlation(_rank(data_x), _rank(data_y))


def skewness(data: list[float]) -> float:
    """Return the sample skewness of *data*."""
    n = len(data)
    if n < 3:
        raise ValueError("need at least three data points for skewness")
    mu = sum(data) / n
    m3 = sum((x - mu) ** 3 for x in data) / n
    m2 = sum((x - mu) ** 2 for x in data) / n
    if m2 == 0.0:
        return 0.0
    s = math.sqrt(m2)
    return m3 / (s ** 3)


def kurtosis(data: list[float]) -> float:
    """Return the excess kurtosis of *data* (Fisher's definition, normal = 0)."""
    n = len(data)
    if n < 4:
        raise ValueError("need at least four data points for kurtosis")
    mu = sum(data) / n
    m4 = sum((x - mu) ** 4 for x in data) / n
    m2 = sum((x - mu) ** 2 for x in data) / n
    if m2 == 0.0:
        return 0.0
    return m4 / (m2 ** 2) - 3.0


def moment(data: list[float], order: int) -> float:
    """Return the *order*-th raw moment E[X^order]."""
    if order < 0:
        raise ValueError("order must be non-negative")
    n = len(data)
    if n == 0:
        raise ValueError("data must not be empty")
    return sum(x ** order for x in data) / n


def central_moment(data: list[float], order: int) -> float:
    """Return the *order*-th central moment E[(X - μ)^order]."""
    if order < 0:
        raise ValueError("order must be non-negative")
    n = len(data)
    if n == 0:
        raise ValueError("data must not be empty")
    mu = sum(data) / n
    return sum((x - mu) ** order for x in data) / n


def frequency_distribution(data: list) -> dict:
    """Return a dict mapping each value to its count."""
    freq: dict = {}
    for v in data:
        freq[v] = freq.get(v, 0) + 1
    return freq


def summary(data: list[float]) -> dict:
    """Return a dict of basic descriptive statistics for *data*."""
    if not data:
        raise ValueError("data must not be empty")
    s = sorted(data)
    n = len(s)
    mu = sum(s) / n
    ss = sum((x - mu) ** 2 for x in s)
    var_sample = ss / (n - 1) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mu,
        "median": median(data),
        "std": math.sqrt(var_sample),
        "var": var_sample,
        "min": s[0],
        "max": s[-1],
        "range": s[-1] - s[0],
        "q1": percentile(data, 25),
        "q3": percentile(data, 75),
        "iqr": percentile(data, 75) - percentile(data, 25),
    }

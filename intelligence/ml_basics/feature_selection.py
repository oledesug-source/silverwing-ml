"""Feature selection methods."""

from __future__ import annotations

import math
import random
from collections import defaultdict

__all__ = [
    "VarianceThreshold",
    "CorrelationFilter",
    "mutual_information",
    "chi_squared",
    "RecursiveFeatureElimination",
    "SelectKBest",
]


class VarianceThreshold:
    """Remove features with variance below a threshold."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._selected: list[int] = []

    def fit(self, X: list[list[float]]) -> VarianceThreshold:
        """Identify features with variance above the threshold."""
        n = len(X)
        p = len(X[0])
        self._selected = []
        for j in range(p):
            col = [X[i][j] for i in range(n)]
            mean = sum(col) / n
            var = sum((v - mean) ** 2 for v in col) / n
            if var > self.threshold:
                self._selected.append(j)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Select features that passed the variance threshold."""
        return [[row[j] for j in self._selected] for row in X]


class CorrelationFilter:
    """Remove features that are highly correlated with each other."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._selected: list[int] = []

    def _correlation(self, a: list[float], b: list[float]) -> float:
        n = len(a)
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = math.sqrt(sum((v - mean_a) ** 2 for v in a) / n)
        std_b = math.sqrt(sum((v - mean_b) ** 2 for v in b) / n)
        if std_a == 0 or std_b == 0:
            return 0.0
        return cov / (std_a * std_b)

    def fit(self, X: list[list[float]]) -> CorrelationFilter:
        """Identify features to keep after removing highly correlated ones."""
        n_cols = len(X[0])
        cols = [[X[i][j] for i in range(len(X))] for j in range(n_cols)]
        removed = set()
        for i in range(n_cols):
            if i in removed:
                continue
            for j in range(i + 1, n_cols):
                if j in removed:
                    continue
                if abs(self._correlation(cols[i], cols[j])) > self.threshold:
                    removed.add(j)
        self._selected = [j for j in range(n_cols) if j not in removed]
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Select uncorrelated features."""
        return [[row[j] for j in self._selected] for row in X]


def mutual_information(X: list[list[float]], y: list[int | str]) -> list[float]:
    """Compute mutual information between each feature and the target."""
    n = len(X)
    p = len(X[0])
    classes = sorted(set(y))
    class_counts = defaultdict(int)
    for val in y:
        class_counts[val] += 1
    result = []
    for j in range(p):
        feat_vals = [X[i][j] for i in range(n)]
        fmin, fmax = min(feat_vals), max(feat_vals)
        if fmin == fmax:
            result.append(0.0)
            continue
        n_bins = min(10, n)
        width = (fmax - fmin) / n_bins
        [fmin + k * width for k in range(n_bins + 1)]
        bin_counts = defaultdict(int)
        joint_counts = defaultdict(int)
        for i in range(n):
            b = min(int((feat_vals[i] - fmin) / width), n_bins - 1)
            bin_counts[b] += 1
            joint_counts[(b, y[i])] += 1
        mi = 0.0
        for b, b_count in bin_counts.items():
            for cls in classes:
                joint = joint_counts.get((b, cls), 0)
                if joint == 0:
                    continue
                p_xy = joint / n
                p_x = b_count / n
                p_y = class_counts[cls] / n
                mi += p_xy * math.log(p_xy / (p_x * p_y))
        result.append(mi)
    return result


def chi_squared(X: list[list[float]], y: list[int | str]) -> list[float]:
    """Compute chi-squared scores between features and target."""
    n = len(X)
    p = len(X[0])
    classes = sorted(set(y))
    class_counts = defaultdict(int)
    for val in y:
        class_counts[val] += 1
    result = []
    for j in range(p):
        feat_vals = [X[i][j] for i in range(n)]
        fmin, fmax = min(feat_vals), max(feat_vals)
        if fmin == fmax:
            result.append(0.0)
            continue
        n_bins = min(10, len(set(feat_vals)))
        width = (fmax - fmin) / n_bins if n_bins > 0 else 1.0
        if width == 0:
            result.append(0.0)
            continue
        observed = defaultdict(int)
        feat_bin_counts = defaultdict(int)
        for i in range(n):
            b = min(int((feat_vals[i] - fmin) / width), n_bins - 1)
            observed[(b, y[i])] += 1
            feat_bin_counts[b] += 1
        chi2 = 0.0
        for b in range(n_bins):
            for cls in classes:
                o = observed.get((b, cls), 0)
                e = feat_bin_counts[b] * class_counts[cls] / n
                if e > 0:
                    chi2 += (o - e) ** 2 / e
        result.append(chi2)
    return result


class RecursiveFeatureElimination:
    """Recursively remove least important features based on coefficient magnitude."""

    def __init__(self, estimator, n_features: int = 1):
        self.estimator = estimator
        self.n_features = n_features
        self._selected: list[int] = []

    def fit(self, X: list[list[float]], y: list) -> RecursiveFeatureElimination:
        """Fit RFE by iteratively removing least important features."""
        n_cols = len(X[0])
        selected = list(range(n_cols))
        current_X = [list(row) for row in X]
        while len(selected) > self.n_features:
            self.estimator.fit(current_X, y)
            if hasattr(self.estimator, "coefficients"):
                importances = list(self.estimator.coefficients)
            else:
                importances = [abs(random.random()) for _ in range(len(selected))]
            abs_imp = [abs(v) for v in importances]
            min_idx = abs_imp.index(min(abs_imp))
            selected.pop(min_idx)
            current_X = [[row[j] for j in range(len(row)) if j != min_idx] for row in current_X]
        self._selected = selected
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Select the remaining features."""
        return [[row[j] for j in self._selected] for row in X]


class SelectKBest:
    """Select the k best features based on a scoring function."""

    def __init__(self, score_func=None, k: int = 1):
        self.score_func = score_func or mutual_information
        self.k = k
        self._selected: list[int] = []

    def fit(self, X: list[list[float]], y: list) -> SelectKBest:
        """Identify the k best features."""
        scores = self.score_func(X, y)
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        self._selected = [idx for idx, _ in indexed[:self.k]]
        self._selected.sort()
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Select the k best features."""
        return [[row[j] for j in self._selected] for row in X]

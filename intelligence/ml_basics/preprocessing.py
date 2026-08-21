"""Data preprocessing utilities including scalers, encoders, and splitting functions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
    "RobustScaler",
    "Normalizer",
    "OneHotEncoder",
    "LabelEncoder",
    "PolynomialFeatures",
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "cross_val_score",
]


class MinMaxScaler:
    """Scale features to a given range, default [0, 1]."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min: list[float] = []
        self.data_max: list[float] = []
        self.scale_: list[float] = []
        self.min_: list[float] = []

    def fit(self, X: list[list[float]]) -> MinMaxScaler:
        n_cols = len(X[0])
        self.data_min = [min(row[j] for row in X) for j in range(n_cols)]
        self.data_max = [max(row[j] for row in X) for j in range(n_cols)]
        fr_min, fr_max = self.feature_range
        self.scale_ = []
        self.min_ = []
        for j in range(n_cols):
            diff = self.data_max[j] - self.data_min[j]
            if diff == 0:
                self.scale_.append(1.0)
                self.min_.append(fr_min)
            else:
                self.scale_.append((fr_max - fr_min) / diff)
                self.min_.append(fr_min - self.data_min[j] * self.scale_[-1])
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [row[j] * self.scale_[j] + self.min_[j] for j in range(len(row))]
            for row in X
        ]

    def inverse_transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [(row[j] - self.min_[j]) / self.scale_[j] for j in range(len(row))]
            for row in X
        ]


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: list[float] = []
        self.std_: list[float] = []

    def fit(self, X: list[list[float]]) -> StandardScaler:
        n_cols = len(X[0])
        n_rows = len(X)
        self.mean_ = [sum(row[j] for row in X) / n_rows for j in range(n_cols)]
        self.std_ = []
        for j in range(n_cols):
            var = sum((row[j] - self.mean_[j]) ** 2 for row in X) / n_rows
            self.std_.append(math.sqrt(var) if var > 0 else 1.0)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [(row[j] - self.mean_[j]) / self.std_[j] for j in range(len(row))]
            for row in X
        ]

    def inverse_transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [row[j] * self.std_[j] + self.mean_[j] for j in range(len(row))]
            for row in X
        ]


class RobustScaler:
    """Scale features using statistics that are robust to outliers (IQR-based)."""

    def __init__(self):
        self.median_: list[float] = []
        self.iqr_: list[float] = []

    def _percentile(self, values: list[float], p: float) -> float:
        s = sorted(values)
        n = len(s)
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    def fit(self, X: list[list[float]]) -> RobustScaler:
        n_cols = len(X[0])
        self.median_ = [self._percentile([row[j] for row in X], 0.5) for j in range(n_cols)]
        q1 = [self._percentile([row[j] for row in X], 0.25) for j in range(n_cols)]
        q3 = [self._percentile([row[j] for row in X], 0.75) for j in range(n_cols)]
        self.iqr_ = [q3[j] - q1[j] for j in range(n_cols)]
        self.iqr_ = [v if v > 0 else 1.0 for v in self.iqr_]
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [(row[j] - self.median_[j]) / self.iqr_[j] for j in range(len(row))]
            for row in X
        ]

    def inverse_transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [row[j] * self.iqr_[j] + self.median_[j] for j in range(len(row))]
            for row in X
        ]


class Normalizer:
    """Normalize samples individually to unit norm (L1 or L2)."""

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    def fit(self, X: list[list[float]]) -> Normalizer:
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        result = []
        for row in X:
            if self.norm == "l1":
                norm_val = sum(abs(v) for v in row)
            elif self.norm == "l2":
                norm_val = math.sqrt(sum(v * v for v in row))
            else:
                norm_val = max(abs(v) for v in row) if row else 1.0
            if norm_val == 0:
                result.append(list(row))
            else:
                result.append([v / norm_val for v in row])
        return result


class OneHotEncoder:
    """One-hot encode categorical features."""

    def __init__(self):
        self.categories_: list[list] = []
        self._cat_idx: list[dict] = []

    def fit(self, X: list[list]) -> OneHotEncoder:
        n_cols = len(X[0])
        self.categories_ = []
        self._cat_idx = []
        for j in range(n_cols):
            unique = []
            seen = set()
            for row in X:
                if row[j] not in seen:
                    unique.append(row[j])
                    seen.add(row[j])
            self.categories_.append(unique)
            self._cat_idx.append({c: i for i, c in enumerate(unique)})
        return self

    def transform(self, X: list[list]) -> list[list[int]]:
        result = []
        for row in X:
            encoded = []
            for j in range(len(row)):
                vec = [0] * len(self.categories_[j])
                idx = self._cat_idx[j].get(row[j])
                if idx is not None:
                    vec[idx] = 1
                encoded.extend(vec)
            result.append(encoded)
        return result

    def inverse_transform(self, X: list[list[int]]) -> list[list]:
        result = []
        offset = 0
        for row in X:
            original = []
            for _j, cats in enumerate(self.categories_):
                segment = row[offset : offset + len(cats)]
                idx = segment.index(1) if 1 in segment else 0
                original.append(cats[idx])
                offset += len(cats)
            result.append(original)
            offset = 0
        return result


class LabelEncoder:
    """Encode target labels with values between 0 and n_classes - 1."""

    def __init__(self):
        self.classes_: list = []
        self._class_idx: dict = {}

    def fit(self, y: list) -> LabelEncoder:
        seen = set()
        self.classes_ = []
        for val in y:
            if val not in seen:
                self.classes_.append(val)
                seen.add(val)
        self._class_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y: list) -> list[int]:
        return [self._class_idx[v] for v in y]

    def inverse_transform(self, y: list[int]) -> list:
        return [self.classes_[i] for i in y]


class PolynomialFeatures:
    """Generate polynomial and interaction features up to given degree."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self._n_features: int = 0
        self._combinations: list[tuple] = []

    def _generate_combinations(self, n: int) -> list[tuple]:
        from itertools import combinations_with_replacement
        combos = []
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n), d):
                combos.append(combo)
        return combos

    def fit(self, X: list[list[float]]) -> PolynomialFeatures:
        self._n_features = len(X[0])
        self._combinations = self._generate_combinations(self._n_features)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        result = []
        for row in X:
            new_row = []
            for combo in self._combinations:
                val = 1.0
                for idx in combo:
                    val *= row[idx]
                new_row.append(val)
            result.append(new_row)
        return result


def train_test_split(
    X: list,
    y: list,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[list, list, list, list]:
    """Split arrays into random train and test subsets."""
    n = len(X)
    rng = random.Random(random_state)
    indices = list(range(n))
    if shuffle:
        rng.shuffle(indices)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


@dataclass
class KFold:
    """K-Fold cross-validation splitter."""

    n_splits: int = 5

    def split(self, X: list) -> list[tuple[list[int], list[int]]]:
        """Generate train/test index pairs."""
        n = len(X)
        indices = list(range(n))
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        folds = []
        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current : current + fold_size]
            train_idx = indices[:current] + indices[current + fold_size :]
            folds.append((train_idx, test_idx))
            current += fold_size
        return folds


@dataclass
class StratifiedKFold:
    """Stratified K-Fold cross-validation splitter."""

    n_splits: int = 5

    def split(self, X: list, y: list) -> list[tuple[list[int], list[int]]]:
        """Generate train/test index pairs preserving class distribution."""
        from collections import defaultdict

        label_indices: dict = defaultdict(list)
        for i, label in enumerate(y):
            label_indices[label].append(i)
        folds: list[list[int]] = [[] for _ in range(self.n_splits)]
        for label in sorted(label_indices.keys()):
            indices = list(label_indices[label])
            rng = random.Random(0)
            rng.shuffle(indices)
            for i, idx in enumerate(indices):
                folds[i % self.n_splits].append(idx)
        result = []
        for k in range(self.n_splits):
            test_idx = sorted(folds[k])
            train_idx = sorted(idx for i, f in enumerate(folds) if i != k for idx in f)
            result.append((train_idx, test_idx))
        return result


def cross_val_score(estimator, X: list, y: list, cv: KFold | StratifiedKFold) -> list[float]:
    """Compute cross-validated scores."""
    if isinstance(cv, StratifiedKFold):
        splits = cv.split(X, y)
    else:
        splits = cv.split(X)
    scores = []
    for train_idx, test_idx in splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        correct = sum(1 for a, b in zip(y_test, y_pred) if a == b)
        scores.append(correct / len(y_test) if y_test else 0.0)
    return scores

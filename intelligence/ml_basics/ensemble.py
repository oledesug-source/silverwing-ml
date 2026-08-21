"""Ensemble methods: bagging, voting, stacking, and boosting."""

from __future__ import annotations

import math
import random
from collections import defaultdict

__all__ = [
    "BaggingClassifier",
    "VotingClassifier",
    "StackingClassifier",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class BaggingClassifier:
    """Bootstrap aggregating classifier."""

    def __init__(self, base_estimator=None, n_estimators: int = 10, max_samples: int | None = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self._estimators: list = []
        self._oob_indices: list[set[int]] = []

    def fit(self, X: list[list[float]], y: list) -> BaggingClassifier:
        """Fit bagging ensemble with bootstrap samples."""
        n = len(X)
        ms = self.max_samples or n
        self._estimators = []
        self._oob_indices = []
        for _ in range(self.n_estimators):
            rng = random.Random()
            indices = [rng.randint(0, n - 1) for _ in range(ms)]
            boot_X = [X[i] for i in indices]
            boot_y = [y[i] for i in indices]
            est = _clone_estimator(self.base_estimator, self._detect_type(y))
            est.fit(boot_X, boot_y)
            self._estimators.append(est)
            self._oob_indices.append(set(range(n)) - set(indices))
        return self

    def _detect_type(self, y: list) -> str:
        if all(isinstance(v, (int, float)) for v in y) and len(set(y)) > 10:
            return "regression"
        return "classification"

    def predict(self, X: list[list[float]]) -> list:
        """Predict by majority vote."""
        all_preds = [est.predict(X) for est in self._estimators]
        n = len(X)
        result = []
        for i in range(n):
            votes = defaultdict(int)
            for preds in all_preds:
                votes[preds[i]] += 1
            result.append(max(votes, key=votes.get))
        return result


class VotingClassifier:
    """Ensemble that combines predictions from multiple estimators."""

    def __init__(self, estimators: list[tuple[str, object]] | None = None, strategy: str = "hard"):
        self.estimators = estimators or []
        self.strategy = strategy
        self._fitted_estimators: list[tuple[str, object]] = []

    def fit(self, X: list[list[float]], y: list) -> VotingClassifier:
        """Fit all base estimators."""
        self._fitted_estimators = []
        for name, est in self.estimators:
            clone = _clone_estimator(est, self._detect_type(y))
            clone.fit(X, y)
            self._fitted_estimators.append((name, clone))
        return self

    def _detect_type(self, y: list) -> str:
        if all(isinstance(v, (int, float)) for v in y) and len(set(y)) > 10:
            return "regression"
        return "classification"

    def predict(self, X: list[list[float]]) -> list:
        """Predict by hard or soft voting."""
        if self.strategy == "hard" or not hasattr(self._fitted_estimators[0][1], "predict_proba"):
            all_preds = [est.predict(X) for _, est in self._fitted_estimators]
            n = len(X)
            result = []
            for i in range(n):
                votes = defaultdict(int)
                for preds in all_preds:
                    votes[preds[i]] += 1
                result.append(max(votes, key=votes.get))
            return result
        else:
            probas = [est.predict_proba(X) for _, est in self._fitted_estimators]
            n = len(X)
            k = len(probas[0][0])
            result = []
            for i in range(n):
                avg = [sum(p[i][c] for p in probas) / len(probas) for c in range(k)]
                result.append(avg.index(max(avg)))
            return result


class StackingClassifier:
    """Stacking ensemble that uses a meta-estimator on base predictions."""

    def __init__(self, base_estimators: list[tuple[str, object]] | None = None, meta_estimator=None):
        self.base_estimators = base_estimators or []
        self.meta_estimator = meta_estimator
        self._fitted_base: list[tuple[str, object]] = []
        self._fitted_meta = None

    def fit(self, X: list[list[float]], y: list) -> StackingClassifier:
        """Fit base estimators and the meta-estimator."""
        self._fitted_base = []
        for name, est in self.base_estimators:
            clone = _clone_estimator(est, self._detect_type(y))
            clone.fit(X, y)
            self._fitted_base.append((name, clone))
        meta_X = self._get_meta_features(X)
        self._fitted_meta = _clone_estimator(self.meta_estimator, self._detect_type(y))
        self._fitted_meta.fit(meta_X, y)
        return self

    def _detect_type(self, y: list) -> str:
        if all(isinstance(v, (int, float)) for v in y) and len(set(y)) > 10:
            return "regression"
        return "classification"

    def _get_meta_features(self, X: list[list[float]]) -> list[list[float]]:
        preds = [est.predict(X) for _, est in self._fitted_base]
        n = len(X)
        meta = []
        for i in range(n):
            meta.append([preds[j][i] for j in range(len(preds))])
        return meta

    def predict(self, X: list[list[float]]) -> list:
        """Predict using the stacking ensemble."""
        meta_X = self._get_meta_features(X)
        return self._fitted_meta.predict(meta_X)


class AdaBoostClassifier:
    """AdaBoost classifier using decision stumps as weak learners."""

    def __init__(self, base_estimator=None, n_estimators: int = 50, lr: float = 1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.lr = lr
        self._stumps: list[dict] = []
        self._alphas: list[float] = []

    def fit(self, X: list[list[float]], y: list[int], n_estimators: int | None = None, lr: float | None = None) -> AdaBoostClassifier:
        """Fit AdaBoost by sequentially adding stumps."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if lr is not None:
            self.lr = lr
        n = len(X)
        weights = [1.0 / n] * n
        labels = sorted(set(y))
        y_mapped = [1 if v == labels[-1] else -1 for v in y] if len(labels) == 2 else list(y)
        self._stumps = []
        self._alphas = []
        for _ in range(self.n_estimators):
            stump = self._fit_stump(X, y_mapped, weights)
            preds = [self._predict_stump(stump, x) for x in X]
            err = sum(weights[i] for i in range(n) if preds[i] != y_mapped[i])
            err = max(err, 1e-10)
            alpha = self.lr * 0.5 * math.log((1 - err) / err)
            for i in range(n):
                if preds[i] == y_mapped[i]:
                    weights[i] *= math.exp(-alpha)
                else:
                    weights[i] *= math.exp(alpha)
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]
            self._stumps.append(stump)
            self._alphas.append(alpha)
        self._labels = labels
        return self

    def _fit_stump(self, X: list[list[float]], y: list[int], weights: list[float]) -> dict:
        """Fit a single decision stump."""
        best_err = float("inf")
        best_stump = {"feature": 0, "threshold": 0.0, "direction": 1}
        p = len(X[0])
        for j in range(p):
            values = sorted({X[i][j] for i in range(len(X))})
            for t in values:
                for direction in [1, -1]:
                    err = 0.0
                    for i in range(len(X)):
                        pred = direction if X[i][j] <= t else -direction
                        if pred != y[i]:
                            err += weights[i]
                    if err < best_err:
                        best_err = err
                        best_stump = {"feature": j, "threshold": t, "direction": direction}
        return best_stump

    def _predict_stump(self, stump: dict, x: list[float]) -> int:
        """Predict with a single stump."""
        if x[stump["feature"]] <= stump["threshold"]:
            return stump["direction"]
        return -stump["direction"]

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels using weighted vote of stumps."""
        n = len(X)
        raw = [0.0] * n
        for stump, alpha in zip(self._stumps, self._alphas):
            for i in range(n):
                raw[i] += alpha * self._predict_stump(stump, X[i])
        if hasattr(self, "_labels") and len(self._labels) == 2:
            return [self._labels[1] if r >= 0 else self._labels[0] for r in raw]
        return [1 if r >= 0 else 0 for r in raw]

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0


class AdaBoostRegressor:
    """AdaBoost regressor using decision stumps."""

    def __init__(self, base_estimator=None, n_estimators: int = 50, lr: float = 1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.lr = lr
        self._stumps: list[dict] = []
        self._alphas: list[float] = []

    def fit(self, X: list[list[float]], y: list[float], n_estimators: int | None = None, lr: float | None = None) -> AdaBoostRegressor:
        """Fit AdaBoost regressor sequentially."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if lr is not None:
            self.lr = lr
        n = len(X)
        weights = [1.0 / n] * n
        current_y = list(y)
        self._stumps = []
        self._alphas = []
        for _ in range(self.n_estimators):
            stump = self._fit_stump(X, current_y, weights)
            preds = [self._predict_stump(stump, x) for x in X]
            errors = [abs(current_y[i] - preds[i]) for i in range(n)]
            max_err = max(errors) if errors else 1.0
            max_err = max(max_err, 1e-10)
            weighted_err = sum(weights[i] * errors[i] / max_err for i in range(n))
            weighted_err = max(min(weighted_err, 1 - 1e-10), 1e-10)
            alpha = self.lr * (1 - weighted_err) / max(weighted_err, 1e-10)
            for i in range(n):
                weights[i] *= math.exp(alpha * errors[i] / max_err)
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]
            self._stumps.append(stump)
            self._alphas.append(alpha)
        return self

    def _fit_stump(self, X: list[list[float]], y: list[float], weights: list[float]) -> dict:
        """Fit a single regression stump."""
        best_err = float("inf")
        best_stump = {"feature": 0, "threshold": 0.0, "value": 0.0}
        p = len(X[0])
        for j in range(p):
            values = sorted({X[i][j] for i in range(len(X))})
            for t in values:
                left_y = [y[i] for i in range(len(X)) if X[i][j] <= t]
                right_y = [y[i] for i in range(len(X)) if X[i][j] > t]
                if left_y:
                    left_val = sum(left_y) / len(left_y)
                else:
                    left_val = 0.0
                if right_y:
                    right_val = sum(right_y) / len(right_y)
                else:
                    right_val = 0.0
                err = 0.0
                for i in range(len(X)):
                    pred = left_val if X[i][j] <= t else right_val
                    err += weights[i] * (y[i] - pred) ** 2
                if err < best_err:
                    best_err = err
                    best_stump = {"feature": j, "threshold": t, "left_val": left_val, "right_val": right_val}
        return best_stump

    def _predict_stump(self, stump: dict, x: list[float]) -> float:
        if x[stump["feature"]] <= stump["threshold"]:
            return stump["left_val"]
        return stump["right_val"]

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict using weighted sum of stump predictions."""
        n = len(X)
        raw = [0.0] * n
        total_alpha = sum(self._alphas) if self._alphas else 1.0
        for stump, alpha in zip(self._stumps, self._alphas):
            for i in range(n):
                raw[i] += alpha * self._predict_stump(stump, X[i])
        return [r / total_alpha if total_alpha > 0 else 0.0 for r in raw]


def _clone_estimator(est, task_type: str):
    """Create a fresh copy of an estimator with default parameters."""
    from .linear_models import LinearRegression, LogisticRegression
    from .tree_models import DecisionTreeClassifier, DecisionTreeRegressor
    if est is None:
        if task_type == "regression":
            return LinearRegression()
        return LogisticRegression()
    name = type(est).__name__
    params = {}
    if hasattr(est, "get_params"):
        try:
            params = est.get_params()
        except Exception:
            pass
    if name == "LogisticRegression":
        return LogisticRegression(lr=params.get("lr", 0.1), epochs=params.get("epochs", 100))
    elif name == "LinearRegression":
        return LinearRegression()
    elif name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
        )
    elif name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
        )
    return LogisticRegression()

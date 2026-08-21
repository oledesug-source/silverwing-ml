"""Decision trees, random forests, and gradient boosting models."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

__all__ = [
    "DecisionNode",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "gini_impurity",
    "entropy",
    "mse",
]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def gini_impurity(labels: list) -> float:
    """Compute Gini impurity for a list of labels."""
    if not labels:
        return 0.0
    n = len(labels)
    counts: dict = defaultdict(int)
    for l in labels:
        counts[l] += 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels: list) -> float:
    """Compute entropy for a list of labels."""
    if not labels:
        return 0.0
    n = len(labels)
    counts: dict = defaultdict(int)
    for l in labels:
        counts[l] += 1
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def mse(values: list[float]) -> float:
    """Compute mean squared error."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


@dataclass
class DecisionNode:
    """Node in a decision tree."""

    feature_index: int = 0
    threshold: float = 0.0
    left: DecisionNode | None = None
    right: DecisionNode | None = None
    value: float | None = None
    class_label: object = None


class DecisionTreeClassifier:
    """Classification and regression tree for classification."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root: DecisionNode | None = None
        self.feature_importances_: list[float] = []
        self._n_features: int = 0
        self._importance_sums: list[float] = []

    def fit(self, X: list[list[float]], y: list, max_depth: int | None = None, min_samples_split: int | None = None, criterion: str | None = None) -> DecisionTreeClassifier:
        """Fit the decision tree."""
        if max_depth is not None:
            self.max_depth = max_depth
        if min_samples_split is not None:
            self.min_samples_split = min_samples_split
        if criterion is not None:
            self.criterion = criterion
        self._n_features = len(X[0])
        self._importance_sums = [0.0] * self._n_features
        self.root = self._build(X, y, 0)
        total = sum(self._importance_sums) if sum(self._importance_sums) > 0 else 1.0
        self.feature_importances_ = [imp / total for imp in self._importance_sums]
        return self

    def _build(self, X: list[list[float]], y: list, depth: int) -> DecisionNode:
        if len(set(y)) == 1:
            return DecisionNode(value=None, class_label=y[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionNode(value=None, class_label=self._majority(y))
        if len(y) < self.min_samples_split:
            return DecisionNode(value=None, class_label=self._majority(y))
        best_feature, best_threshold, best_score = self._best_split(X, y)
        if best_score == float("inf"):
            return DecisionNode(value=None, class_label=self._majority(y))
        self._importance_sums[best_feature] += len(y) * best_score
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        node = DecisionNode(feature_index=best_feature, threshold=best_threshold)
        node.left = self._build(left_X, left_y, depth + 1)
        node.right = self._build(right_X, right_y, depth + 1)
        return node

    def _best_split(self, X: list[list[float]], y: list) -> tuple[int, float, float]:
        n = len(y)
        best_feature = 0
        best_threshold = 0.0
        best_score = float("inf")
        parent_score = gini_impurity(y) if self.criterion == "gini" else entropy(y)
        for j in range(self._n_features):
            values = sorted({X[i][j] for i in range(n)})
            for k in range(len(values) - 1):
                threshold = (values[k] + values[k + 1]) / 2.0
                left_y, right_y = [], []
                for i in range(n):
                    if X[i][j] <= threshold:
                        left_y.append(y[i])
                    else:
                        right_y.append(y[i])
                if not left_y or not right_y:
                    continue
                if self.criterion == "gini":
                    imp_left = gini_impurity(left_y)
                    imp_right = gini_impurity(right_y)
                else:
                    imp_left = entropy(left_y)
                    imp_right = entropy(right_y)
                weighted = (len(left_y) * imp_left + len(right_y) * imp_right) / n
                gain = parent_score - weighted
                if gain < best_score:
                    best_score = gain
                    best_feature = j
                    best_threshold = threshold
        return best_feature, best_threshold, best_score

    def _majority(self, y: list) -> object:
        counts: dict = defaultdict(int)
        for v in y:
            counts[v] += 1
        return max(counts, key=counts.get)

    def _predict_one(self, x: list[float], node: DecisionNode) -> object:
        if node.class_label is not None:
            return node.class_label
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels for X."""
        return [self._predict_one(x, self.root) for x in X]

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0

    def get_params(self) -> dict:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "criterion": self.criterion}

    def summary(self) -> str:
        lines = [f"DecisionTreeClassifier (max_depth={self.max_depth}, criterion={self.criterion})"]
        lines.append(f"Feature importances: {[round(v, 4) for v in self.feature_importances_]}")
        return "\n".join(lines)


class DecisionTreeRegressor:
    """Regression tree using MSE as splitting criterion."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: DecisionNode | None = None
        self.feature_importances_: list[float] = []
        self._n_features: int = 0
        self._importance_sums: list[float] = []

    def fit(self, X: list[list[float]], y: list[float], max_depth: int | None = None, min_samples_split: int | None = None) -> DecisionTreeRegressor:
        """Fit the regression tree."""
        if max_depth is not None:
            self.max_depth = max_depth
        if min_samples_split is not None:
            self.min_samples_split = min_samples_split
        self._n_features = len(X[0])
        self._importance_sums = [0.0] * self._n_features
        self.root = self._build(X, y, 0)
        total = sum(self._importance_sums) if sum(self._importance_sums) > 0 else 1.0
        self.feature_importances_ = [imp / total for imp in self._importance_sums]
        return self

    def _build(self, X: list[list[float]], y: list[float], depth: int) -> DecisionNode:
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return DecisionNode(value=sum(y) / len(y))
        best_feature, best_threshold, best_reduction = self._best_split(X, y)
        if best_reduction <= 0:
            return DecisionNode(value=sum(y) / len(y))
        self._importance_sums[best_feature] += len(y) * best_reduction
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        node = DecisionNode(feature_index=best_feature, threshold=best_threshold)
        node.left = self._build(left_X, left_y, depth + 1)
        node.right = self._build(right_X, right_y, depth + 1)
        return node

    def _best_split(self, X: list[list[float]], y: list[float]) -> tuple[int, float, float]:
        n = len(y)
        best_feature = 0
        best_threshold = 0.0
        best_reduction = -float("inf")
        parent_mse = mse(y)
        for j in range(self._n_features):
            values = sorted({X[i][j] for i in range(n)})
            for k in range(len(values) - 1):
                threshold = (values[k] + values[k + 1]) / 2.0
                left_y, right_y = [], []
                for i in range(n):
                    if X[i][j] <= threshold:
                        left_y.append(y[i])
                    else:
                        right_y.append(y[i])
                if not left_y or not right_y:
                    continue
                weighted = (len(left_y) * mse(left_y) + len(right_y) * mse(right_y)) / n
                reduction = parent_mse - weighted
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = j
                    best_threshold = threshold
        return best_feature, best_threshold, best_reduction

    def _predict_one(self, x: list[float], node: DecisionNode) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values for X."""
        return [self._predict_one(x, self.root) for x in X]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}

    def summary(self) -> str:
        lines = [f"DecisionTreeRegressor (max_depth={self.max_depth})"]
        lines.append(f"Feature importances: {[round(v, 4) for v in self.feature_importances_]}")
        return "\n".join(lines)


class RandomForestClassifier:
    """Random forest ensemble of classification trees."""

    def __init__(self, n_trees: int = 100, max_depth: int = 10, feature_subset: int | None = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_subset = feature_subset
        self._trees: list[DecisionTreeClassifier] = []
        self.feature_importances_: list[float] = []
        self._n_features: int = 0

    def fit(self, X: list[list[float]], y: list, n_trees: int | None = None, max_depth: int | None = None, feature_subset: int | None = None) -> RandomForestClassifier:
        """Fit the random forest."""
        if n_trees is not None:
            self.n_trees = n_trees
        if max_depth is not None:
            self.max_depth = max_depth
        if feature_subset is not None:
            self.feature_subset = feature_subset
        n = len(X)
        self._n_features = len(X[0])
        fs = self.feature_subset or max(1, int(math.sqrt(self._n_features)))
        self._trees = []
        self.feature_importances_ = [0.0] * self._n_features
        rng = random.Random()
        for _ in range(self.n_trees):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            boot_X = [X[i] for i in indices]
            boot_y = [y[i] for i in indices]
            feat_indices = rng.sample(range(self._n_features), min(fs, self._n_features))
            sub_X = [[row[j] for j in feat_indices] for row in boot_X]
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(sub_X, boot_y)
            self._trees.append((tree, feat_indices))
            for fi, idx in enumerate(feat_indices):
                if idx < len(tree.feature_importances_):
                    self.feature_importances_[idx] += tree.feature_importances_[fi]
        total = sum(self.feature_importances_) if sum(self.feature_importances_) > 0 else 1.0
        self.feature_importances_ = [v / total for v in self.feature_importances_]
        return self

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels by majority vote."""
        n = len(X)
        all_preds = []
        for tree, feat_indices in self._trees:
            sub_X = [[row[j] for j in feat_indices] for row in X]
            all_preds.append(tree.predict(sub_X))
        result = []
        for i in range(n):
            votes: dict = defaultdict(int)
            for preds in all_preds:
                votes[preds[i]] += 1
            result.append(max(votes, key=votes.get))
        return result

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0

    def get_params(self) -> dict:
        return {"n_trees": self.n_trees, "max_depth": self.max_depth, "feature_subset": self.feature_subset}

    def summary(self) -> str:
        lines = [f"RandomForestClassifier (n_trees={self.n_trees}, max_depth={self.max_depth})"]
        lines.append(f"Feature importances: {[round(v, 4) for v in self.feature_importances_]}")
        return "\n".join(lines)


class RandomForestRegressor:
    """Random forest ensemble of regression trees."""

    def __init__(self, n_trees: int = 100, max_depth: int = 10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self._trees: list[tuple[DecisionTreeRegressor, list[int]]] = []
        self.feature_importances_: list[float] = []

    def fit(self, X: list[list[float]], y: list[float], n_trees: int | None = None, max_depth: int | None = None) -> RandomForestRegressor:
        """Fit the random forest regressor."""
        if n_trees is not None:
            self.n_trees = n_trees
        if max_depth is not None:
            self.max_depth = max_depth
        n = len(X)
        nf = len(X[0])
        fs = max(1, int(math.sqrt(nf)))
        self._trees = []
        self.feature_importances_ = [0.0] * nf
        rng = random.Random()
        for _ in range(self.n_trees):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            boot_X = [X[i] for i in indices]
            boot_y = [y[i] for i in indices]
            feat_indices = rng.sample(range(nf), min(fs, nf))
            sub_X = [[row[j] for j in feat_indices] for row in boot_X]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(sub_X, boot_y)
            self._trees.append((tree, feat_indices))
            for fi, idx in enumerate(feat_indices):
                if idx < len(tree.feature_importances_):
                    self.feature_importances_[idx] += tree.feature_importances_[fi]
        total = sum(self.feature_importances_) if sum(self.feature_importances_) > 0 else 1.0
        self.feature_importances_ = [v / total for v in self.feature_importances_]
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values by averaging tree predictions."""
        n = len(X)
        all_preds = []
        for tree, feat_indices in self._trees:
            sub_X = [[row[j] for j in feat_indices] for row in X]
            all_preds.append(tree.predict(sub_X))
        return [sum(preds[i] for preds in all_preds) / len(all_preds) for i in range(n)]

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        return {"n_trees": self.n_trees, "max_depth": self.max_depth}

    def summary(self) -> str:
        return f"RandomForestRegressor (n_trees={self.n_trees}, max_depth={self.max_depth})"


class GradientBoostingClassifier:
    """Gradient boosting ensemble for classification using decision stumps."""

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self._trees: list[DecisionTreeRegressor] = []
        self._initial_pred: float = 0.0

    def fit(self, X: list[list[float]], y: list[int], n_estimators: int | None = None, lr: float | None = None, max_depth: int | None = None) -> GradientBoostingClassifier:
        """Fit the gradient boosting classifier."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if lr is not None:
            self.lr = lr
        if max_depth is not None:
            self.max_depth = max_depth
        labels = sorted(set(y))
        self._pos_label = labels[-1] if len(labels) == 2 else 1
        y_float = [1.0 if v == self._pos_label else 0.0 for v in y]
        pos_count = sum(y_float)
        neg_count = len(y_float) - pos_count
        self._initial_pred = math.log(max(pos_count, 1) / max(neg_count, 1)) if neg_count > 0 else 0.0
        current_pred = [self._initial_pred] * len(X)
        self._trees = []
        for _ in range(self.n_estimators):
            probs = [_sigmoid(p) for p in current_pred]
            residuals = [y_float[i] - probs[i] for i in range(len(X))]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self._trees.append(tree)
            preds = tree.predict(X)
            for i in range(len(X)):
                current_pred[i] += self.lr * preds[i]
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Predict probability of the positive class."""
        preds = [self._initial_pred] * len(X)
        for tree in self._trees:
            tree_preds = tree.predict(X)
            preds = [preds[i] + self.lr * tree_preds[i] for i in range(len(X))]
        return [_sigmoid(p) for p in preds]

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return [self._pos_label if p >= 0.5 else 0 for p in probas]

    def score(self, X: list[list[float]], y: list) -> float:
        """Compute accuracy."""
        preds = self.predict(X)
        return sum(1 for a, b in zip(y, preds) if a == b) / len(y) if y else 0.0

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators, "lr": self.lr, "max_depth": self.max_depth}

    def summary(self) -> str:
        return f"GradientBoostingClassifier (n_estimators={self.n_estimators}, lr={self.lr}, max_depth={self.max_depth})"


class GradientBoostingRegressor:
    """Gradient boosting ensemble for regression."""

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self._trees: list[DecisionTreeRegressor] = []
        self._initial_pred: float = 0.0

    def fit(self, X: list[list[float]], y: list[float], n_estimators: int | None = None, lr: float | None = None, max_depth: int | None = None) -> GradientBoostingRegressor:
        """Fit the gradient boosting regressor."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if lr is not None:
            self.lr = lr
        if max_depth is not None:
            self.max_depth = max_depth
        self._initial_pred = sum(y) / len(y)
        current_pred = [self._initial_pred] * len(X)
        self._trees = []
        for _ in range(self.n_estimators):
            residuals = [y[i] - current_pred[i] for i in range(len(X))]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self._trees.append(tree)
            preds = tree.predict(X)
            for i in range(len(X)):
                current_pred[i] += self.lr * preds[i]
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values."""
        preds = [self._initial_pred] * len(X)
        for tree in self._trees:
            tree_preds = tree.predict(X)
            preds = [preds[i] + self.lr * tree_preds[i] for i in range(len(X))]
        return preds

    def score(self, X: list[list[float]], y: list[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators, "lr": self.lr, "max_depth": self.max_depth}

    def summary(self) -> str:
        return f"GradientBoostingRegressor (n_estimators={self.n_estimators}, lr={self.lr}, max_depth={self.max_depth})"

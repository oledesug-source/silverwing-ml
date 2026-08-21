"""Nearest neighbors algorithms for classification and regression."""

from __future__ import annotations

import math

__all__ = [
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "RadiusNeighborsClassifier",
    "NearestNeighbors",
    "euclidean_distance",
    "manhattan_distance",
    "minkowski_distance",
    "cosine_distance",
    "BallTree",
]


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute Manhattan distance between two points."""
    return sum(abs(x - y) for x, y in zip(a, b))


def minkowski_distance(a: list[float], b: list[float], p: float = 2.0) -> float:
    """Compute Minkowski distance between two points."""
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1.0 / p)


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two points."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


class BallTree:
    """Ball tree data structure for efficient nearest neighbor queries."""

    def __init__(self, X: list[list[float]], leaf_size: int = 30):
        self._X = [list(x) for x in X]
        self._indices = list(range(len(X)))
        self._leaf_size = leaf_size
        self._root = self._build(self._indices)

    def _centroid(self, indices: list[int]) -> list[float]:
        n = len(indices)
        dim = len(self._X[0])
        return [sum(self._X[i][j] for i in indices) / n for j in range(dim)]

    def _radius(self, indices: list[int], centroid: list[float]) -> float:
        return max(euclidean_distance(self._X[i], centroid) for i in indices)

    def _build(self, indices: list[int]) -> dict:
        if len(indices) <= self._leaf_size:
            return {"leaf": True, "indices": indices}
        centroid = self._centroid(indices)
        r = self._radius(indices, centroid)
        dim = len(self._X[0])
        max_spread = -1.0
        split_dim = 0
        for j in range(dim):
            vals = [self._X[i][j] for i in indices]
            spread = max(vals) - min(vals)
            if spread > max_spread:
                max_spread = spread
                split_dim = j
        vals_sorted = sorted(indices, key=lambda i: self._X[i][split_dim])
        mid = len(vals_sorted) // 2
        left = self._build(vals_sorted[:mid])
        right = self._build(vals_sorted[mid:])
        return {
            "leaf": False,
            "centroid": centroid,
            "radius": r,
            "split_dim": split_dim,
            "split_val": self._X[vals_sorted[mid]][split_dim],
            "left": left,
            "right": right,
        }

    def _knn(self, node: dict, query: list[float], k: int, best: list) -> list:
        if node["leaf"]:
            for i in node["indices"]:
                d = euclidean_distance(query, self._X[i])
                if len(best) < k:
                    best.append((d, i))
                    best.sort(key=lambda x: x[0])
                elif d < best[-1][0]:
                    best[-1] = (d, i)
                    best.sort(key=lambda x: x[0])
            return best
        left, right = node["left"], node["right"]
        for child in [left, right]:
            if child["leaf"]:
                best = self._knn(child, query, k, best)
            else:
                d = euclidean_distance(query, child["centroid"])
                if len(best) < k or d - child["radius"] < best[-1][0]:
                    best = self._knn(child, query, k, best)
        return best

    def knn_query(self, query: list[float], k: int) -> list[tuple[float, int]]:
        """Return k nearest neighbors as list of (distance, index)."""
        results = self._knn(self._root, query, k, [])
        return results

    def radius_query(self, query: list[float], radius: float) -> list[tuple[float, int]]:
        """Return all neighbors within radius as list of (distance, index)."""
        results: list[tuple[float, int]] = []
        self._radius_search(self._root, query, radius, results)
        return results

    def _radius_search(self, node: dict, query: list[float], radius: float, results: list) -> None:
        if node["leaf"]:
            for i in node["indices"]:
                d = euclidean_distance(query, self._X[i])
                if d <= radius:
                    results.append((d, i))
            return
        d = euclidean_distance(query, node["centroid"])
        if d - node["radius"] <= radius:
            self._radius_search(node["left"], query, radius, results)
            self._radius_search(node["right"], query, radius, results)


class NearestNeighbors:
    """Unsupervised nearest neighbors fit and query."""

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric
        self._X: list[list[float]] = []
        self._tree: BallTree | None = None

    def fit(self, X: list[list[float]]) -> NearestNeighbors:
        self._X = [list(x) for x in X]
        self._tree = BallTree(X)
        return self

    def kneighbors(self, query: list[float], k: int = 5) -> tuple[list[float], list[int]]:
        """Find k-nearest neighbors."""
        results = self._tree.knn_query(query, k)
        distances = [r[0] for r in results]
        indices = [r[1] for r in results]
        return distances, indices

    def radius_neighbors(self, query: list[float], radius: float) -> tuple[list[float], list[int]]:
        """Find all neighbors within radius."""
        results = self._tree.radius_query(query, radius)
        results.sort(key=lambda x: x[0])
        distances = [r[0] for r in results]
        indices = [r[1] for r in results]
        return distances, indices


class KNeighborsClassifier:
    """K-nearest neighbors classifier."""

    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = k
        self.weights = weights
        self._X: list[list[float]] = []
        self._y: list = []
        self._tree: BallTree | None = None

    def fit(self, X: list[list[float]], y: list) -> KNeighborsClassifier:
        self._X = [list(x) for x in X]
        self._y = list(y)
        self._tree = BallTree(X)
        return self

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels for X."""
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x: list[float]) -> object:
        results = self._tree.knn_query(x, self.k)
        votes: dict = {}
        for dist, idx in results:
            label = self._y[idx]
            if self.weights == "distance":
                w = 1.0 / (dist + 1e-10)
            else:
                w = 1.0
            votes[label] = votes.get(label, 0.0) + w
        return max(votes, key=lambda k: votes[k])


class KNeighborsRegressor:
    """K-nearest neighbors regressor."""

    def __init__(self, k: int = 5):
        self.k = k
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._tree: BallTree | None = None

    def fit(self, X: list[list[float]], y: list[float]) -> KNeighborsRegressor:
        self._X = [list(x) for x in X]
        self._y = list(y)
        self._tree = BallTree(X)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values for X."""
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x: list[float]) -> float:
        results = self._tree.knn_query(x, self.k)
        total_w = 0.0
        total_v = 0.0
        for dist, idx in results:
            w = 1.0 / (dist + 1e-10)
            total_w += w
            total_v += w * self._y[idx]
        return total_v / total_w if total_w > 0 else 0.0


class RadiusNeighborsClassifier:
    """Radius-based neighbors classifier."""

    def __init__(self, radius: float = 1.0, weights: str = "uniform"):
        self.radius = radius
        self.weights = weights
        self._X: list[list[float]] = []
        self._y: list = []
        self._tree: BallTree | None = None

    def fit(self, X: list[list[float]], y: list) -> RadiusNeighborsClassifier:
        self._X = [list(x) for x in X]
        self._y = list(y)
        self._tree = BallTree(X)
        return self

    def predict(self, X: list[list[float]]) -> list:
        """Predict class labels for X."""
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x: list[float]) -> object:
        results = self._tree.radius_query(x, self.radius)
        if not results:
            return self._y[0] if self._y else 0
        votes: dict = {}
        for dist, idx in results:
            label = self._y[idx]
            if self.weights == "distance":
                w = 1.0 / (dist + 1e-10)
            else:
                w = 1.0
            votes[label] = votes.get(label, 0.0) + w
        return max(votes, key=lambda k: votes[k])

"""Clustering algorithms: KMeans, DBSCAN, hierarchical, GMM, spectral."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "KMeans",
    "KMeansPlusPlus",
    "DBSCAN",
    "HierarchicalClustering",
    "AgglomerativeClustering",
    "GMM",
    "SpectralClustering",
]


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KMeans:
    """K-Means clustering algorithm."""

    def __init__(self, k: int = 3, max_iter: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: list[list[float]] = []
        self.labels: list[int] = []
        self.inertia: float = 0.0

    def fit(self, X: list[list[float]], max_iter: int | None = None, tol: float | None = None) -> KMeans:
        """Fit K-Means to data."""
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        n = len(X)
        dim = len(X[0])
        rng = random.Random(0)
        indices = rng.sample(range(n), min(self.k, n))
        self.centroids = [list(X[i]) for i in indices]
        for _ in range(self.max_iter):
            self.labels = [self._assign(x) for x in X]
            new_centroids = []
            for c in range(self.k):
                members = [X[i] for i in range(n) if self.labels[i] == c]
                if members:
                    nc = [sum(m[j] for m in members) / len(members) for j in range(dim)]
                else:
                    nc = list(self.centroids[c])
                new_centroids.append(nc)
            shift = sum(_dist(new_centroids[c], self.centroids[c]) for c in range(self.k))
            self.centroids = new_centroids
            if shift < self.tol:
                break
        self.inertia = sum(
            _dist(X[i], self.centroids[self.labels[i]]) ** 2
            for i in range(n)
        )
        return self

    def _assign(self, x: list[float]) -> int:
        return min(range(self.k), key=lambda c: _dist(x, self.centroids[c]))

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict cluster labels for new data."""
        return [self._assign(x) for x in X]

    def get_params(self) -> dict:
        return {"k": self.k, "max_iter": self.max_iter, "tol": self.tol}


class KMeansPlusPlus(KMeans):
    """K-Means with k-means++ initialization."""

    def fit(self, X: list[list[float]], max_iter: int | None = None, tol: float | None = None) -> KMeansPlusPlus:
        """Fit K-Means with k-means++ initialization."""
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        n = len(X)
        rng = random.Random(0)
        self.centroids = [list(X[rng.randint(0, n - 1)])]
        for _ in range(1, self.k):
            dists = []
            for i in range(n):
                min_d = min(_dist(X[i], c) ** 2 for c in self.centroids)
                dists.append(min_d)
            total = sum(dists)
            if total == 0:
                idx = rng.randint(0, n - 1)
            else:
                r = rng.random() * total
                cumsum = 0.0
                idx = 0
                for i in range(n):
                    cumsum += dists[i]
                    if cumsum >= r:
                        idx = i
                        break
            self.centroids.append(list(X[idx]))
        self.labels = [self._assign(x) for x in X]
        for _ in range(self.max_iter):
            self.labels = [self._assign(x) for x in X]
            new_centroids = []
            dim = len(X[0])
            for c in range(self.k):
                members = [X[i] for i in range(n) if self.labels[i] == c]
                if members:
                    nc = [sum(m[j] for m in members) / len(members) for j in range(dim)]
                else:
                    nc = list(self.centroids[c])
                new_centroids.append(nc)
            shift = sum(_dist(new_centroids[c], self.centroids[c]) for c in range(self.k))
            self.centroids = new_centroids
            if shift < self.tol:
                break
        self.inertia = sum(
            _dist(X[i], self.centroids[self.labels[i]]) ** 2
            for i in range(n)
        )
        return self


class DBSCAN:
    """Density-based spatial clustering of applications with noise."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels: list[int] = []
        self.core_points: list[int] = []

    def fit(self, X: list[list[float]], eps: float | None = None, min_samples: int | None = None) -> DBSCAN:
        """Fit DBSCAN clustering."""
        if eps is not None:
            self.eps = eps
        if min_samples is not None:
            self.min_samples = min_samples
        n = len(X)
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _dist(X[i], X[j])
                if d <= self.eps:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        self.core_points = [i for i in range(n) if len(neighbors[i]) >= self.min_samples]
        self.labels = [-1] * n
        cluster_id = 0
        for i in range(n):
            if self.labels[i] != -1:
                continue
            if i not in self.core_points:
                continue
            self._expand(X, neighbors, i, cluster_id)
            cluster_id += 1
        return self

    def _expand(self, X: list[list[float]], neighbors: list[list[int]], point: int, cluster_id: int) -> None:
        self.labels[point] = cluster_id
        queue = list(neighbors[point])
        while queue:
            j = queue.pop()
            if self.labels[j] == -1:
                self.labels[j] = cluster_id
            if j in self.core_points and self.labels[j] == cluster_id:
                for nb in neighbors[j]:
                    if self.labels[nb] == -1:
                        queue.append(nb)

    def predict(self, X: list[list[float]]) -> list[int]:
        """Return fitted labels."""
        return self.labels

    def get_params(self) -> dict:
        return {"eps": self.eps, "min_samples": self.min_samples}


@dataclass
class _DendrogramNode:
    left: int
    right: int
    distance: float
    n_members: int


class HierarchicalClustering:
    """Agglomerative hierarchical clustering."""

    def __init__(self, linkage: str = "single"):
        self.linkage = linkage
        self.dendrogram_data: list[_DendrogramNode] = []
        self._labels: list[int] = []

    def fit(self, X: list[list[float]], linkage: str | None = None) -> HierarchicalClustering:
        """Fit hierarchical clustering."""
        if linkage is not None:
            self.linkage = linkage
        n = len(X)
        self._labels = list(range(n))
        dists = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _dist(X[i], X[j])
                dists[i][j] = d
                dists[j][i] = d
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        self.dendrogram_data = []
        next_id = n
        active = list(range(n))
        merge_dist = {}
        for i in range(n):
            for j in range(n):
                merge_dist[(i, j)] = self._linkage_dist(dists, clusters[i], clusters[j])
        while len(active) > 1:
            best_d = float("inf")
            best_i, best_j = -1, -1
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    ci, cj = active[ii], active[jj]
                    d = merge_dist.get((ci, cj), float("inf"))
                    if d < best_d:
                        best_d = d
                        best_i, best_j = ii, jj
            ci, cj = active[best_i], active[best_j]
            new_cluster = clusters[ci] + clusters[cj]
            self.dendrogram_data.append(_DendrogramNode(ci, cj, best_d, len(new_cluster)))
            clusters[next_id] = new_cluster
            for k in active:
                if k != ci and k != cj:
                    d = self._linkage_dist(dists, clusters[k], new_cluster)
                    merge_dist[(k, next_id)] = d
                    merge_dist[(next_id, k)] = d
            del clusters[ci]
            del clusters[cj]
            active.pop(best_j)
            active.pop(best_i)
            active.append(next_id)
            next_id += 1
        return self

    def _linkage_dist(self, full_dists: list[list[float]], c1: list[int], c2: list[int]) -> float:
        if self.linkage == "single":
            return min(full_dists[i][j] for i in c1 for j in c2)
        elif self.linkage == "complete":
            return max(full_dists[i][j] for i in c1 for j in c2)
        else:
            total = sum(full_dists[i][j] for i in c1 for j in c2)
            return total / (len(c1) * len(c2))

    def cut(self, n_clusters: int) -> list[int]:
        """Cut the dendrogram to get n_clusters clusters."""
        n = self.dendrogram_data[-1].n_members if self.dendrogram_data else 0
        if n == 0:
            return []
        labels = list(range(n))
        merges_needed = n - n_clusters
        for node in self.dendrogram_data[:merges_needed]:
            new_label = node.left
            old_label = node.right
            for i in range(len(labels)):
                if labels[i] == old_label:
                    labels[i] = new_label
        unique = sorted(set(labels))
        remap = {old: i for i, old in enumerate(unique)}
        return [remap[l] for l in labels]


class AgglomerativeClustering:
    """Agglomerative clustering producing labels and children."""

    def __init__(self, n_clusters: int = 2, linkage: str = "single"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels: list[int] = []
        self.children_: list[tuple[int, int]] = []

    def fit(self, X: list[list[float]], n_clusters: int | None = None, linkage: str | None = None) -> AgglomerativeClustering:
        """Fit agglomerative clustering."""
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if linkage is not None:
            self.linkage = linkage
        hc = HierarchicalClustering(linkage=self.linkage)
        hc.fit(X)
        self.children_ = [(node.left, node.right) for node in hc.dendrogram_data]
        self.labels = hc.cut(self.n_clusters)
        return self

    def get_params(self) -> dict:
        return {"n_clusters": self.n_clusters, "linkage": self.linkage}


def _multivariate_gauss(x: list[float], mean: list[float], cov: list[list[float]]) -> float:
    """Evaluate multivariate Gaussian density."""
    n = len(x)
    diff = [x[i] - mean[i] for i in range(n)]
    det = _det3(cov)
    if det <= 1e-10:
        det = 1e-10
    inv = _inv3(cov)
    exp_arg = sum(diff[i] * sum(inv[i][j] * diff[j] for j in range(n)) for i in range(n))
    return math.exp(-0.5 * exp_arg) / math.sqrt((2 * math.pi) ** n * det)


def _det3(M: list[list[float]]) -> float:
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1 :] for row in M[1:]]
        sign = 1 if j % 2 == 0 else -1
        det += sign * M[0][j] * _det3(minor)
    return det


def _inv3(M: list[list[float]]) -> list[list[float]]:
    n = len(M)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            for r in range(col + 1, n):
                if abs(aug[r][col]) > 1e-12:
                    aug[col], aug[r] = aug[r], aug[col]
                    pivot = aug[col][col]
                    break
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


class GMM:
    """Gaussian Mixture Model using the EM algorithm."""

    def __init__(self, k: int = 3, max_iter: int = 100):
        self.k = k
        self.max_iter = max_iter
        self.means: list[list[float]] = []
        self.covariances: list[list[list[float]]] = []
        self.weights: list[float] = []
        self.log_likelihood: float = 0.0
        self.labels: list[int] = []

    def fit(self, X: list[list[float]], k: int | None = None, max_iter: int | None = None) -> GMM:
        """Fit GMM using the EM algorithm."""
        if k is not None:
            self.k = k
        if max_iter is not None:
            self.max_iter = max_iter
        n = len(X)
        dim = len(X[0])
        rng = random.Random(0)
        indices = rng.sample(range(n), min(self.k, n))
        self.means = [list(X[i]) for i in indices]
        self.covariances = []
        for _ in range(self.k):
            cov = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                cov[i][i] = 1.0
            self.covariances.append(cov)
        self.weights = [1.0 / self.k] * self.k
        for _iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            self.log_likelihood = self._compute_log_likelihood(X)
        all_resp = self._e_step(X)
        self.labels = [max(range(self.k), key=lambda c: all_resp[i][c]) for i in range(n)]
        return self

    def _e_step(self, X: list[list[float]]) -> list[list[float]]:
        n = len(X)
        responsibilities = []
        for i in range(n):
            probs = []
            for c in range(self.k):
                p = self.weights[c] * _multivariate_gauss(X[i], self.means[c], self.covariances[c])
                probs.append(p)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / self.k] * self.k
            responsibilities.append(probs)
        return responsibilities

    def _m_step(self, X: list[list[float]], responsibilities: list[list[float]]) -> None:
        n = len(X)
        dim = len(X[0])
        for c in range(self.k):
            resp_sum = sum(responsibilities[i][c] for i in range(n))
            if resp_sum < 1e-10:
                resp_sum = 1e-10
            self.weights[c] = resp_sum / n
            self.means[c] = [
                sum(responsibilities[i][c] * X[i][j] for i in range(n)) / resp_sum
                for j in range(dim)
            ]
            cov = [[0.0] * dim for _ in range(dim)]
            for i in range(n):
                diff = [X[i][j] - self.means[c][j] for j in range(dim)]
                for j in range(dim):
                    for l in range(dim):
                        cov[j][l] += responsibilities[i][c] * diff[j] * diff[l]
            for j in range(dim):
                for l in range(dim):
                    cov[j][l] /= resp_sum
                cov[j][j] += 1e-6
            self.covariances[c] = cov

    def _compute_log_likelihood(self, X: list[list[float]]) -> float:
        ll = 0.0
        for x in X:
            total = sum(
                self.weights[c] * _multivariate_gauss(x, self.means[c], self.covariances[c])
                for c in range(self.k)
            )
            if total > 0:
                ll += math.log(total)
        return ll

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict cluster labels."""
        resp = self._e_step(X)
        return [max(range(self.k), key=lambda c: r[c]) for r in resp]

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Predict soft cluster assignments."""
        return self._e_step(X)

    def get_params(self) -> dict:
        return {"k": self.k, "max_iter": self.max_iter}


class SpectralClustering:
    """Spectral clustering using Laplacian eigenmaps."""

    def __init__(self, n_clusters: int = 3, affinity: str = "rbf", gamma: float = 1.0):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.labels: list[int] = []

    def fit(self, X: list[list[float]], n_clusters: int | None = None, affinity: str | None = None) -> SpectralClustering:
        """Fit spectral clustering."""
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if affinity is not None:
            self.affinity = affinity
        n = len(X)
        W = self._affinity_matrix(X)
        D = [sum(W[i]) for i in range(n)]
        L = [[D[i] - W[i][j] if i == j else -W[i][j] for j in range(n)] for i in range(n)]
        eigenvectors = _eigen_decomposition(L, self.n_clusters)
        km = KMeans(k=self.n_clusters, max_iter=100)
        km.fit(eigenvectors)
        self.labels = km.labels
        return self

    def _affinity_matrix(self, X: list[list[float]]) -> list[list[float]]:
        n = len(X)
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _dist(X[i], X[j])
                if self.affinity == "rbf":
                    w = math.exp(-self.gamma * d * d)
                elif self.affinity == "precomputed":
                    w = d
                else:
                    w = max(0, 1 - d)
                W[i][j] = w
                W[j][i] = w
        return W

    def predict(self, X: list[list[float]]) -> list[int]:
        """Return fitted labels."""
        return self.labels

    def get_params(self) -> dict:
        return {"n_clusters": self.n_clusters, "affinity": self.affinity}


def _eigen_decomposition(M: list[list[float]], k: int) -> list[list[float]]:
    """Compute the k smallest eigenvectors of symmetric matrix M via power iteration."""
    n = len(M)
    rng = random.Random(42)
    eigenvectors = []
    A = [row[:] for row in M]
    for _ in range(k):
        v = [rng.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]
        for _ in range(200):
            Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            for ev in eigenvectors:
                dot = sum(Av[i] * ev[i] for i in range(n))
                Av = [Av[i] - dot * ev[i] for i in range(n)]
            norm = math.sqrt(sum(x * x for x in Av))
            if norm < 1e-10:
                break
            v = [x / norm for x in Av]
        eigenvectors.append(v)
        eigenval = sum(sum(A[i][j] * v[j] for j in range(n)) * v[i] for i in range(n))
        for i in range(n):
            for j in range(n):
                A[i][j] -= eigenval * v[i] * v[j]
    return [[eigenvectors[j][i] for j in range(k)] for i in range(n)]

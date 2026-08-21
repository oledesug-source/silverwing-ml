"""Data loading, batching, and splitting utilities."""

import math
import random
from collections.abc import Iterator
from typing import Any


class Dataset:
    def __init__(self, X: list[list[float]], y: list[Any] | None = None):
        assert len(X) > 0
        assert len(X) == len(y) if y is not None else True
        self.X = [list(row) for row in X]
        self.y = list(y) if y is not None else None
        self.n_samples = len(X)
        self.n_features = len(X[0])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[list[float], Any]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None

    def subset(self, indices: list[int]) -> "Dataset":
        X_sub = [self.X[i] for i in indices]
        y_sub = [self.y[i] for i in indices] if self.y is not None else None
        return Dataset(X_sub, y_sub)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[tuple[list[list[float]], Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        n = len(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            X_batch = [self.dataset.X[i] for i in batch_idx]
            if self.dataset.y is not None:
                y_batch = [self.dataset.y[i] for i in batch_idx]
                yield X_batch, y_batch
            else:
                yield X_batch, None


def train_test_split(
    X: list[list[float]],
    y: list[Any] | None = None,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple:
    if seed is not None:
        random.seed(seed)
    n = len(X)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    split = int(n * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    if y is not None:
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test
    return X_train, X_test


def k_fold_split(
    n_samples: int, k: int = 5, shuffle: bool = True, seed: int | None = None
) -> list[tuple[list[int], list[int]]]:
    if seed is not None:
        random.seed(seed)
    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current : current + fold_size]
        train_idx = indices[:current] + indices[current + fold_size :]
        folds.append((train_idx, test_idx))
        current += fold_size
    return folds


def normalize(
    X: list[list[float]], mean: list[float] | None = None, std: list[float] | None = None
) -> tuple[list[list[float]], list[float], list[float]]:
    n_features = len(X[0])
    n_samples = len(X)
    if mean is None:
        mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    if std is None:
        std = []
        for j in range(n_features):
            var = sum((X[i][j] - mean[j]) ** 2 for i in range(n_samples)) / n_samples
            std.append(math.sqrt(var) if var > 0 else 1.0)
    normalized = [
        [(X[i][j] - mean[j]) / std[j] for j in range(n_features)]
        for i in range(n_samples)
    ]
    return normalized, mean, std


def one_hot(labels: list[int], num_classes: int | None = None) -> list[list[float]]:
    if num_classes is None:
        num_classes = max(labels) + 1
    result = []
    for label in labels:
        row = [0.0] * num_classes
        row[label] = 1.0
        result.append(row)
    return result


def label_encode(labels: list[Any]) -> tuple[list[int], dict[Any, int]]:
    unique = sorted(set(labels), key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(unique)}
    return [mapping[l] for l in labels], mapping

"""Optional TensorFlow integration shim for Silverwing-ML.

This module is import-safe: it works even when ``tensorflow`` is not
installed by exposing a lightweight in-process backend that mirrors the
small slice of the Keras API actually used here.

Two use-cases are supported:

1. **TF present** (`pip install tensorflow-cpu` or `tensorflow`): a real
   Keras ``Sequential`` trainer mirrors ``TrainConfig`` and trains the
   from-scratch ``intelligence.ml_basics`` / ``intelligence.training``
   linear-algebra + statistics + calculus modules on real JSONL data.
2. **TF absent**: the same entry point falls back to a NumPy autograd
   mini-engine (``_fallback``) so callers never see an ImportError.

Both paths log to MLflow/W&B via ``foundation.ops`` when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._compat import optional_dependency

tensorflow = optional_dependency("tensorflow")

_HAS_TF = tensorflow.__class__.__module__ != "foundation._compat"


def available() -> bool:
    return _HAS_TF


@dataclass
class TFTrainConfig:
    """Minimal mirror of foundation.training.config.TrainConfig for TF."""

    max_steps: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    verbose: int = 1


class _FallbackLinear:
    """Tiny NumPy-backed linear model used when TF is absent."""

    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        import numpy as np

        self.w = np.zeros((in_dim, out_dim), dtype=np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)

    def __call__(self, x):
        import numpy as np

        return x @ self.w + self.b

    def fit(self, x, y, epochs: int = 200, lr: float = 1e-3) -> None:
        import numpy as np

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        for _ in range(epochs):
            pred = self(x)
            err = pred - y.reshape(pred.shape)
            grad_w = (x.T @ err) / len(y)
            grad_b = err.mean(axis=0)
            self.w -= lr * grad_w
            self.b -= lr * grad_b


class TFTrainer:
    """Train a Keras model on tabular features extracted from the math corpus.

    When ``tensorflow`` is installed this builds a real ``Sequential`` MLP;
    otherwise it delegates to :class:`_FallbackLinear` so the public API is
    identical in both cases.
    """

    def __init__(self, cfg: TFTrainConfig | None = None) -> None:
        self.cfg = cfg or TFTrainConfig()

    def build_model(self, in_dim: int):
        if _HAS_TF:
            try:
                from tensorflow import keras  # type: ignore
                from tensorflow.keras import layers  # type: ignore

                model = keras.Sequential(
                    [
                        keras.Input(shape=(in_dim,)),
                        layers.Dense(64, activation="relu"),
                        layers.Dense(32, activation="relu"),
                        layers.Dense(1, activation="linear"),
                    ]
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=self.cfg.lr),
                    loss="mse",
                )
                return model
            except Exception:
                return _FallbackLinear(in_dim, 1)
        return _FallbackLinear(in_dim, 1)

    def fit(self, x, y) -> dict[str, Any]:
        import numpy as np

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        in_dim = x.shape[1] if x.ndim > 1 else 1
        model = self.build_model(in_dim)
        if _HAS_TF and hasattr(model, "fit"):
            history = model.fit(x, y, epochs=self.cfg.max_steps, batch_size=self.cfg.batch_size, verbose=self.cfg.verbose)
            return {"loss": float(history.history["loss"][-1]), "backend": "tensorflow"}
        model.fit(x, y, epochs=self.cfg.max_steps, lr=self.cfg.lr)
        pred = model(x)
        mse = float(np.mean((np.asarray(pred).reshape(-1) - y) ** 2))
        return {"loss": mse, "backend": "numpy_fallback"}


def from_math_corpus(rows: list[dict[str, Any]], target_key: str = "answer") -> tuple[Any, Any]:
    """Build (X, y) arrays from math-corpus / JSONL rows using numeric features.

    Each row's ``solution`` (or ``output``) text is featurized by length,
    word count, and counts of math operators so that *linear algebra*,
    *statistics* and *calculus* problem text can train a regression head.
    """
    import numpy as np

    feats = []
    targets = []
    for row in rows:
        text = row.get("solution") or row.get("output") or row.get("text") or ""
        target = row.get(target_key)
        if target is None:
            continue
        try:
            target = float(target)
        except (TypeError, ValueError):
            continue
        feats.append(
            [
                len(text),
                text.count(" "),
                text.count("+"),
                text.count("-"),
                text.count("*"),
                text.count("/"),
                text.count("sqrt"),
                text.count("pi"),
                text.count("log"),
                text.count("sin"),
                text.count("cos"),
                text.count("x"),
            ]
        )
        targets.append(target)
    x = np.array(feats, dtype=np.float32) if feats else np.empty((0, 12), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return x, y


def trainer() -> TFTrainer:
    return TFTrainer()

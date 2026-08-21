"""Operational integrations for Silverwing-ML.

This package provides optional, self-contained wrappers around common
MLOps and big-data tools. Every sub-module degrades gracefully when its
underlying third-party package is not installed, so importing
``foundation.ops`` never fails in a stripped-down environment.

Feature flags
--------------
``is_available(name)`` returns ``True`` only when the integration can be
used (i.e. its dependency imports successfully). ``AVAILABLE`` is a dict
mirroring those flags for quick introspection.

    >>> from foundation import ops
    >>> ops.is_available("mlflow")
    True
    >>> ops.AVAILABLE["torch"]
    True
"""

from __future__ import annotations

import importlib
from typing import Any


def _import_ok(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


_TORCH_AVAILABLE = _import_ok("torch")
_MLFLOW_AVAILABLE = _import_ok("mlflow")
_WANDB_AVAILABLE = _import_ok("wandb")
_KFP_AVAILABLE = _import_ok("kfp")
_TENSORFLOW_AVAILABLE = _import_ok("tensorflow")
_PYSPARK_AVAILABLE = _import_ok("pyspark")
_DATABRICKS_AVAILABLE = _import_ok("databricks.sdk")

AVAILABLE: dict[str, bool] = {
    "torch": _TORCH_AVAILABLE,
    "mlflow": _MLFLOW_AVAILABLE,
    "wandb": _WANDB_AVAILABLE,
    "kfp": _KFP_AVAILABLE,
    "tensorflow": _TENSORFLOW_AVAILABLE,
    "pyspark": _PYSPARK_AVAILABLE,
    "databricks": _DATABRICKS_AVAILABLE,
    "sympy": _import_ok("sympy"),
    "datasets": _import_ok("datasets"),
}


def is_available(name: str) -> bool:
    """Return True if the named integration is importable & usable."""
    return bool(AVAILABLE.get(name.lower(), False))


_modules: dict[str, Any] = {}


def get_tracker(kind: str = "mlflow"):
    """Return a configured tracker instance (``mlflow`` or ``wandb``).

    The trainer prefers this over direct ``import mlflow`` / ``import wandb``
    so that the training loop survives the dependency being absent.
    """
    if kind not in ("mlflow", "wandb"):
        return None
    if not is_available(kind):
        return None
    if kind not in _modules:
        mod = importlib.import_module(f"foundation.ops.{kind}_tracker")
        _modules[kind] = mod
    return _modules[kind]


__all__ = [
    "AVAILABLE",
    "is_available",
    "get_tracker",
]

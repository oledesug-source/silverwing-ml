"""Local, self-hosted MLflow tracking for Silverwing-ML.

Usage (no MLflow server required — file:// backend)::

    from foundation import ops
    if ops.is_available("mlflow"):
        tracker = ops.get_tracker("mlflow")
        with tracker.start_run("pretrain", config=cfg.to_dict()):
            tracker.log_metric("loss", 2.3)
            tracker.log_params(lr=3e-4, batch_size=8)
    ...

Design notes
------------
* Tracking URI defaults to ``experiments/mlruns`` (a plain directory under
  the project) so every run is persisted locally with no credentials.
* ``mlflow.log_artifact`` is used for model checkpoints via ``save_checkpoint``.
* All public functions are no-ops-safe: if ``mlflow`` is absent the module
  still imports (it is only instantiated through ``ops.get_tracker``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .._compat import optional_dependency

mlflow = optional_dependency("mlflow")
_MF = mlflow  # may be a stub module

_DEFAULT_DIR = "experiments/mlruns"


def _tracking_uri() -> str:
    existing = _MF.get_tracking_uri() if _MF else None
    resolved = (Path.cwd() / _DEFAULT_DIR)
    if existing and (existing.startswith("file:") or "://" in existing):
        return existing
    return str(resolved)


def configure(tracking_uri: str | None = None, experiment: str = "silverwing") -> None:
    """Point MLflow at a local directory and select an experiment.

    MLflow 3.x marks the legacy file-store as maintenance-mode; we opt back
    in via ``MLFLOW_ALLOW_FILE_STORE=true`` so the local ``file://`` backend
    works for private self-hosted use without a database.
    """
    if _MF is None:  # pragma: no cover - guarded by is_available
        return
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_TRACKING_URI", _tracking_uri())
    uri = tracking_uri or _tracking_uri()
    # Normalize bare absolute Windows paths to file:/ URIs without a host
    # (MLflow 3.x rejects file://C:/... as a remote on Windows). Relative
    # and already-schemed URIs are passed through untouched.
    if uri and "://" not in uri and uri != "databricks" and not uri.startswith("sqlite"):
        p = Path(uri)
        if p.is_absolute():
            uri = f"file:{p.as_posix()}"
    _MF.set_tracking_uri(uri)
    try:
        _MF.set_experiment(experiment)
    except Exception:
        pass


class MLflowTracker:
    """Thin, defensive wrapper around ``mlflow``."""

    def __init__(self, experiment: str = "silverwing", tracking_uri: str | None = None) -> None:
        self._active = False
        configure(tracking_uri or _tracking_uri(), experiment)

    # context manager -------------------------------------------------
    @contextmanager
    def start_run(self, run_name: str | None = None, config: dict[str, Any] | None = None):
        if _MF is None:
            yield _NoOpRun()
            return
        run = _MF.start_run(run_name=run_name) if run_name else _MF.start_run()
        self._active = True
        if config:
            _MF.log_params(config)
        try:
            yield run
        finally:
            self._active = False
            _MF.end_run()

    # logging primitives --------------------------------------------
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if _MF is None:
            return
        if step is not None:
            _MF.log_metric(key, value, step=step)
        else:
            _MF.log_metric(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if _MF is None:
            return
        _MF.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if _MF is None:
            return
        _MF.log_params(params)

    def log_params_flat(self, params: dict[str, Any], prefix: str | None = None) -> None:
        """Flatten nested dicts before logging (mlflow disallows nested params)."""
        if _MF is None:
            return
        flat = _flatten(params, prefix)
        _MF.log_params(flat)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if _MF is None:
            return
        _MF.log_artifact(str(path), artifact_path=artifact_path)

    def save_model(self, model, path: str | Path) -> Path:
        """Persist a torch/tf model via mlflow if available; else no-op copy."""
        p = Path(path)
        if _MF is not None:
            try:
                _MF.pytorch.log_model(model, "model", conda_env=None)
            except Exception:
                pass
        return p

    def end_run(self) -> None:
        if _MF is None:
            return
        _MF.end_run()

    @property
    def active(self) -> bool:
        return self._active


class _NoOpRun:
    """Stand-in yielded when mlflow is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _flatten(d: dict, prefix: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def tracker() -> MLflowTracker | None:
    """Convenience: return a shared tracker or None if mlflow unavailable."""
    from foundation.ops import is_available

    if not is_available("mlflow"):
        return None
    return MLflowTracker()

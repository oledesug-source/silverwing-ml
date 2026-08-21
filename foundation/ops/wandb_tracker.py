"""Offline-only Weights & Biases tracking for Silverwing-ML.

Designed for private, self-hosted, no-account usage. Every run is persisted
locally under ``experiments/wandb`` and never syncs to the network.

    from foundation import ops
    if ops.is_available("wandb"):
        from foundation.ops.wandb_tracker import W&BTracker
        with WnBTracker.start_run("pretrain", config=cfg.to_dict()) as t:
            t.log_metric("loss", 2.3, step=epoch)
            t.log_params(lr=3e-4)

The trainer wires this in automatically when ``wandb`` is importable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .._compat import optional_dependency

wandb = optional_dependency("wandb")

_DEFAULT_DIR = Path("experiments/wandb")


def _offline_settings(dir: str | Path) -> Any:
    if not _has_wandb():
        return None
    try:
        return wandb.Settings(mode="offline", dir=str(dir), _save_raw_settings=True)
    except TypeError:
        return wandb.Settings(mode="offline", dir=str(dir))


def _has_wandb() -> bool:
    try:
        import wandb as _real  # noqa: F401
    except Exception:
        return False
    return True


class WnBTracker:  # note: "Wn" keeps the symbol short and typo-safe from `wandb`
    """Defensive offline wrapper around the ``wandb`` SDK."""

    def __init__(self, project: str = "silverwing-training", dir: str | Path | None = None) -> None:
        self._project = project
        self._dir = str((Path(dir) if dir else _DEFAULT_DIR).resolve())
        self._run = None

    @classmethod
    def start_run(
        cls,
        run_name: str | None = None,
        project: str = "silverwing-training",
        config: dict[str, Any] | None = None,
        dir: str | Path | None = None,
    ):
        tracker = cls(project=project, dir=dir)
        if not _has_wandb():
            return tracker
        os.makedirs(tracker._dir, exist_ok=True)
        tracker._run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            dir=tracker._dir,
            mode="offline",
        )

        @contextmanager
        def _cm():
            try:
                yield tracker
            finally:
                tracker.finish()

        return _cm()

    # instance logging primitives ------------------------------------
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._run is None:
            return
        payload = {key: value}
        if step is not None:
            payload["_step"] = step
        wandb.log(payload)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        payload = dict(metrics)
        if step is not None:
            payload["_step"] = step
        wandb.log(payload)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run is None:
            return
        wandb.config.update(_flatten(params), allow_val_change=True)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            wandb.finish()
        except Exception:
            pass
        self._run = None


# Re-exported alias so callers can write `WandBTracker` cleanly if desired.
WandBTracker = WnBTracker


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


def tracker(project: str = "silverwing-training", dir: str | Path | None = None) -> WnBTracker | None:
    from foundation.ops import is_available

    if not is_available("wandb"):
        return None
    return WnBTracker(project=project, dir=dir)

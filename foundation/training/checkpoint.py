"""Checkpoint save/load with a metadata manifest for reproducible runs."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture framework RNG state needed for an exact resumed run."""
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore state produced by :func:`capture_rng_state`."""
    if "torch" not in state:
        raise ValueError("checkpoint is missing torch RNG state")
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    run_id: str,
    config_digest: str,
    tokenizer_hash: str,
    dataset_hash: str | None,
    git_commit: str,
    model_config_digest: str | None = None,
    resume_config_digest: str | None = None,
    data_state: dict | None = None,
    rng_state: dict[str, Any] | None = None,
    best_eval_loss: float | None = None,
    eval_loss: float | None = None,
    filename: str | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"step-{step:08d}.pt"
    path = checkpoint_dir / filename
    payload = {
        "run_id": run_id,
        "step": step,
        "config_digest": config_digest,
        "tokenizer_hash": tokenizer_hash,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "model_config_digest": model_config_digest,
        "resume_config_digest": resume_config_digest,
        "data_state": data_state,
        "rng_state": rng_state,
        "best_eval_loss": best_eval_loss,
        "eval_loss": eval_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    # A checkpoint is either fully visible or not visible at all; this avoids
    # a power interruption leaving a valid-looking, truncated checkpoint.
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".tmp", prefix=f".{path.name}.", dir=checkpoint_dir, delete=False
        ) as temporary:
            temp_name = temporary.name
        torch.save(payload, temp_name)
        Path(temp_name).replace(path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

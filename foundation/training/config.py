"""Authoritative training config for Silverwing Decoder V2.

The YAML at configs/training.yaml is the single source of truth for a run:
model, corpus, tokenizer, optimizer, schedule, checkpoints and the
reproducibility guard (M01 rule: every run reproducible from committed
config + commit hash).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

TRAINING_VERSION = "training-v1"


def _default_dict() -> dict[str, Any]:
    return {f.name: f.default for f in fields(TrainConfig) if not f.name.startswith("_")}


_COERCERS: dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "tuple[float, float]": lambda v: tuple(v) if not isinstance(v, tuple) else v,
}


def _coerce(value: Any, annotation: str) -> Any:
    if value is None:
        return None
    inner = annotation
    if annotation.startswith("Optional[") and annotation.endswith("]"):
        inner = annotation[len("Optional[") : -1]
    if inner == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    coercer = _COERCERS.get(inner)
    if coercer is None:
        return value
    if isinstance(coercer, type) and isinstance(value, coercer):
        return value
    return coercer(value) if callable(coercer) else value


@dataclass(frozen=True)
class TrainConfig:
    version: str = TRAINING_VERSION
    model_config_path: str = "configs/model.yaml"
    corpus_dir: str = "experiments/corpus"
    tokenizer_dir: str = "experiments/tokenizer"
    tokenizer_version: str | None = None
    checkpoint_dir: str = "experiments/checkpoints"
    resume_from: str | None = None
    init_from: str | None = None
    batch_size: int = 8
    grad_accum_steps: int = 1
    block_size: int = 512
    max_steps: int = 2000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 100
    eval_sequences: int = 8
    save_steps: int = 500
    max_tokens: int | None = None
    verify_dataset: bool = True
    expected_dataset_hash: str | None = None
    require_validation: bool = True
    require_clean_repo: bool = True
    device: str = "cpu"
    amp: bool = False
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.block_size <= 0:
            raise ValueError("batch_size, grad_accum_steps and block_size must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0 or None")
        if self.eval_steps < 0 or self.save_steps < 0 or self.log_steps < 0:
            raise ValueError("log_steps, eval_steps and save_steps must be >= 0")
        if self.eval_sequences <= 0:
            raise ValueError("eval_sequences must be positive")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        if self.amp and not self.device.startswith("cuda"):
            raise ValueError("amp requires a cuda device")

    def to_dict(self) -> dict[str, Any]:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["betas"] = list(self.betas)
        return values

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resume_digest(self) -> str:
        """Digest fields that must stay fixed for an exact continuation.

        Checkpoint location and the resume path are operational choices, not
        training dynamics, so they are deliberately excluded.
        """
        values = self.to_dict()
        values.pop("checkpoint_dir", None)
        values.pop("resume_from", None)
        payload = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        defaults = _default_dict()
        filtered: dict[str, Any] = {}
        for field in fields(cls):
            key = field.name
            if key in values and values[key] is not None:
                filtered[key] = _coerce(values[key], field.type)
            else:
                filtered[key] = defaults[key]
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("training", raw)
        return cls.from_dict(section)

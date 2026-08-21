"""Curriculum learning configuration (M14).

Defines a sequence of training stages with increasing difficulty. Each stage
has its own dataset, learning rate, steps, and checkpoint directory. The
curriculum runs them sequentially, loading from the previous stage's best
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    """Single stage in a curriculum."""
    name: str
    dataset_path: str
    max_steps: int = 500
    lr: float = 1e-4
    warmup_steps: int = 50
    checkpoint_dir: str = "experiments/checkpoints/curriculum"
    log_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    grad_clip: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class CurriculumConfig:
    """Full curriculum: base settings + ordered stages."""
    version: str = "curriculum-v1"
    model_config_path: str = "configs/model.yaml"
    tokenizer_dir: str = "experiments/tokenizer"
    init_from: str = "experiments/checkpoints/best.pt"
    block_size: int = 512
    batch_size: int = 1
    grad_accum_steps: int = 1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 42
    log_steps: int = 10
    eval_fraction: float = 0.1
    eval_examples: int = 32
    require_clean_repo: bool = True
    device: str = "cpu"
    stages: list[StageConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "stages"}
        d["betas"] = list(self.betas)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path) -> CurriculumConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("curriculum", raw)
        stages_raw = section.get("stages", [])
        stages = [StageConfig(**s) for s in stages_raw]
        defaults = {f.name: f.default for f in fields(cls) if f.name != "stages"}
        defaults["betas"] = (0.9, 0.95)
        values: dict[str, Any] = {}
        for field_def in fields(cls):
            key = field_def.name
            if key == "stages":
                values["stages"] = stages
            elif key in section and section[key] is not None:
                values[key] = section[key]
            else:
                values[key] = defaults[key]
        return cls(**values)

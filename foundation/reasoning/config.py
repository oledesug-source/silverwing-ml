"""Configuration for reasoning-chain training (M13).

The YAML at ``configs/reasoning.yaml`` is the single source of truth for a
run, with the same reproducibility discipline as pretraining and SFT: model,
tokenizer, dataset, init checkpoint, optimizer, schedule, checkpoints and the
clean-repo guard (M01 rule).

Reasoning training produces a model that emits *chain-of-thought* style
responses: a problem prompt, followed by numbered reasoning steps, then a
final answer.  Only the reasoning and answer tokens are supervised; the
problem prompt is masked (same masked-CE approach as M11 SFT).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

REASONING_VERSION = "reasoning-v1"

# Dataset field names
FIELD_ID = "id"
FIELD_REASONING_TYPE = "reasoning_type"
FIELD_DOMAIN = "domain"
FIELD_PROBLEM = "problem"
FIELD_REASONING_STEPS = "reasoning_steps"
FIELD_FINAL_ANSWER = "final_answer"
FIELD_DIFFICULTY = "difficulty"
FIELD_QUALITY = "quality_score"

# Formatting constants for serialized examples
PROBLEM_HEADER = "Problem:\n"
REASONING_HEADER = "Reasoning:\n"
STEP_PREFIX = "{i}. "
FINAL_ANSWER_HEADER = "\nFinal Answer:\n"
EXAMPLE_SEPARATOR = "<|endoftext|>"

# Default reasoning types (mirrors M08 legacy 79R taxonomy)
DEFAULT_REASONING_TYPES = [
    "numerical_reasoning",
    "multi_step",
    "constraint_reasoning",
    "deduction",
    "induction",
    "error_analysis",
    "comparison",
    "diagnosis",
    "causal_reasoning",
    "decision_reasoning",
]

# Default domains
DEFAULT_DOMAINS = [
    "arithmetic",
    "algebra",
    "geometry",
    "probability",
    "number_theory",
    "trigonometry",
    "differentiation",
    "integration",
    "functions",
    "linear_equations",
]


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
    if inner == "int":
        return int(value)
    if inner == "float":
        return float(value)
    if inner == "str":
        return str(value)
    if inner == "tuple[float, float]":
        return tuple(value) if not isinstance(value, tuple) else value
    if inner.startswith(("list[str]", "list[str")):
        return list(value)
    return value


@dataclass(frozen=True)
class ReasoningConfig:
    version: str = REASONING_VERSION
    model_config_path: str = "configs/model.yaml"
    tokenizer_dir: str = "experiments/tokenizer"
    init_from: str = "experiments/checkpoints/best.pt"
    dataset_path: str = "experiments/reasoning/reasoning-v1.jsonl"
    checkpoint_dir: str = "experiments/checkpoints/reasoning"
    batch_size: int = 1
    block_size: int = 512
    max_steps: int = 200
    warmup_steps: int = 20
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 50
    eval_examples: int = 32
    save_steps: int = 100
    eval_fraction: float = 0.1
    require_clean_repo: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.block_size <= 0:
            raise ValueError("batch_size and block_size must be positive")
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
        if self.eval_examples <= 0:
            raise ValueError("eval_examples must be positive")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["betas"] = list(self.betas)
        return values

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ReasoningConfig:
        defaults = {f.name: f.default for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for field in fields(cls):
            key = field.name
            if key in values and values[key] is not None:
                filtered[key] = _coerce(values[key], field.type)
            else:
                filtered[key] = defaults[key]
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReasoningConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("reasoning", raw)
        return cls.from_dict(section)


@dataclass(frozen=True)
class ReasoningDatasetConfig:
    version: str = "reasoning-dataset-v1"
    seed: int = 42
    per_topic: int = 100
    topics: list[str] = None
    output_path: str = "experiments/reasoning/reasoning-v1.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "per_topic": self.per_topic,
            "topics": self.topics,
            "output_path": self.output_path,
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReasoningDatasetConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("reasoning_dataset", raw)
        return cls(
            version=section.get("version", "reasoning-dataset-v1"),
            seed=int(section.get("seed", 42)),
            per_topic=int(section.get("per_topic", 100)),
            topics=list(section["topics"]) if section.get("topics") else None,
            output_path=section.get(
                "output_path", "experiments/reasoning/reasoning-v1.jsonl"
            ),
        )

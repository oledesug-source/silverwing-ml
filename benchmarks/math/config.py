"""Versioned configuration for the M09 mathematics benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "math_benchmark.yaml"
DEFAULT_TOPICS = (
    "arithmetic",
    "algebra",
    "linear_equations",
    "functions",
    "differentiation",
    "integration",
    "geometry",
    "probability",
    "number_theory",
    "trigonometry",
)


@dataclass(frozen=True)
class MathBenchmarkConfig:
    """All inputs that define a reproducible math benchmark release."""

    version: str = "math-benchmark-v1"
    seed: int = 20260816
    items_per_topic: int = 20
    topics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TOPICS)
    output_path: str = "benchmarks/math/math-v1.jsonl"
    corpus_dir: str = "experiments/corpus"
    contamination_ngram: int = 8
    contamination_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.items_per_topic < 1:
            raise ValueError("items_per_topic must be positive")
        if not self.topics:
            raise ValueError("at least one benchmark topic is required")
        unknown = set(self.topics).difference(DEFAULT_TOPICS)
        if unknown:
            raise ValueError(f"unknown benchmark topics: {sorted(unknown)}")
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("benchmark topics must be unique")
        if self.contamination_ngram < 1:
            raise ValueError("contamination_ngram must be positive")
        if not 0.0 <= self.contamination_threshold <= 1.0:
            raise ValueError("contamination_threshold must be in [0, 1]")

    @property
    def total_items(self) -> int:
        return len(self.topics) * self.items_per_topic

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> MathBenchmarkConfig:
        import yaml

        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"Math benchmark config not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        section = raw.get("math_benchmark", raw) or {}
        topics = tuple(section.get("topics", DEFAULT_TOPICS))
        return cls(
            version=str(section.get("version", "math-benchmark-v1")),
            seed=int(section.get("seed", 20260816)),
            items_per_topic=int(section.get("items_per_topic", 20)),
            topics=topics,
            output_path=str(section.get("output_path", "benchmarks/math/math-v1.jsonl")),
            corpus_dir=str(section.get("corpus_dir", "experiments/corpus")),
            contamination_ngram=int(section.get("contamination_ngram", 8)),
            contamination_threshold=float(section.get("contamination_threshold", 0.6)),
        )

"""Math corpus configuration (M08).

``configs/math_corpus.yaml`` is the single authoritative source for the
mathematical training corpus: which topics, how many documents per topic,
how many examples/exercises per document, the seed, and where the raw
staging output and the final pipeline corpus live.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .problems import PROBLEM_GENERATORS

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "math_corpus.yaml"

DEFAULT_CURRICULUM: dict[str, int] = {
    "arithmetic": 350,
    "algebra": 350,
    "linear_equations": 300,
    "functions": 300,
    "differentiation": 350,
    "integration": 350,
    "geometry": 300,
    "probability": 300,
    "number_theory": 300,
    "trigonometry": 300,
    # M18 advanced STEM
    "linear_algebra": 250,
    "advanced_probability": 250,
    "statistics": 200,
}


@dataclass(frozen=True)
class MathCorpusConfig:
    version: str = "math-corpus-v1"
    seed: int = 42
    curriculum: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CURRICULUM))
    examples_per_document: int = 2
    exercises_per_document: int = 6
    staging_dir: str = "experiments/raw-math"
    corpus_dir: str = "experiments/corpus"
    corpus_config_path: str = "configs/corpus.yaml"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.examples_per_document < 1 or self.exercises_per_document < 1:
            raise ValueError("examples_per_document and exercises_per_document must be positive")
        for topic, count in self.curriculum.items():
            if topic not in PROBLEM_GENERATORS:
                raise ValueError(f"unknown curriculum topic: {topic}")
            if count < 1:
                raise ValueError(f"curriculum[{topic}] must be positive, got {count}")

    @property
    def total_documents(self) -> int:
        return sum(self.curriculum.values())

    def digest(self) -> str:
        return _sha256(asdict(self))

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> MathCorpusConfig:
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"Math corpus config not found: {config_path}")
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        section = data.get("math_corpus", {}) or {}
        curriculum = dict(DEFAULT_CURRICULUM)
        curriculum.update({k: int(v) for k, v in (section.get("curriculum") or {}).items()})
        return cls(
            version=section.get("version", "math-corpus-v1"),
            seed=int(section.get("seed", 42)),
            curriculum=curriculum,
            examples_per_document=int(section.get("examples_per_document", 2)),
            exercises_per_document=int(section.get("exercises_per_document", 6)),
            staging_dir=section.get("staging_dir", "experiments/raw-math"),
            corpus_dir=section.get("corpus_dir", "experiments/corpus"),
            corpus_config_path=section.get("corpus_config_path", "configs/corpus.yaml"),
        )


def _sha256(payload: dict) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

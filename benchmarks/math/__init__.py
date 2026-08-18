"""Deterministic, contamination-checked mathematics benchmark (M09)."""

from .config import DEFAULT_CONFIG_PATH, DEFAULT_TOPICS, MathBenchmarkConfig
from .generator import (
    MathBenchmarkRecord,
    flag_corpus_overlap,
    generate_math_benchmark,
    write_math_benchmark,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TOPICS",
    "MathBenchmarkConfig",
    "MathBenchmarkRecord",
    "generate_math_benchmark",
    "write_math_benchmark",
    "flag_corpus_overlap",
]

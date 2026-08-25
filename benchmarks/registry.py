"""Benchmark registry.

Maps benchmark names to data files + task types, so runs are reproducible from
a name alone and evaluation reports can pin the exact spec used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import TASK_TYPES


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    path: Path
    task_type: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task_type '{self.task_type}'")
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Benchmark data not found: {self.path}")


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._benchmarks: dict[str, BenchmarkSpec] = {}

    def register(self, name: str, path: str | Path, task_type: str, description: str = "") -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Benchmark name must be a non-empty string")
        self._benchmarks[name] = BenchmarkSpec(name=name, path=Path(path), task_type=task_type, description=description)

    def get(self, name: str) -> BenchmarkSpec:
        if name not in self._benchmarks:
            raise KeyError(f"Unknown benchmark '{name}'; registered: {sorted(self._benchmarks)}")
        return self._benchmarks[name]

    def names(self) -> list[str]:
        return sorted(self._benchmarks)


def default_registry() -> BenchmarkRegistry:
    """Registry seeded with bundled and released local benchmarks."""
    registry = BenchmarkRegistry()
    data_dir = Path(__file__).resolve().parent / "data"
    sample_math = data_dir / "sample_arithmetic.jsonl"
    if sample_math.exists():
        registry.register("sample_arithmetic", sample_math, "numeric", "bundled sample arithmetic benchmark")
    math_v1 = Path(__file__).resolve().parent / "math" / "math-v1.jsonl"
    if math_v1.exists():
        registry.register("math-benchmark-v1", math_v1, "numeric", "M09 held-out mathematics benchmark")
    unified_v2 = Path(__file__).resolve().parent / "unified" / "unified-v2.jsonl"
    if unified_v2.exists():
        registry.register(
            "unified-benchmark-v2", unified_v2, "numeric",
            "M20 held-out evaluation across all 16 lesson-plan topics",
        )
    return registry

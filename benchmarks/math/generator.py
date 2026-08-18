"""Independent deterministic problem families for M09 evaluation.

These question families intentionally differ from the M08 curriculum's
training-document templates.  They only emit short final numeric answers so
the generic numeric benchmark metrics remain meaningful and unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from benchmarks.dataset import BenchmarkItem
from benchmarks.guard import flag_contaminated

from .config import MathBenchmarkConfig


@dataclass(frozen=True)
class MathBenchmarkRecord:
    item_id: str
    prompt: str
    reference: str
    category: str
    task_type: str = "numeric"

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.item_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "category": self.category,
            "task_type": self.task_type,
        }

    def as_benchmark_item(self) -> BenchmarkItem:
        return BenchmarkItem(
            item_id=self.item_id,
            prompt=self.prompt,
            reference=self.reference,
            category=self.category,
            task_type=self.task_type,
        )


def _signed(value: int) -> str:
    return f"+ {value}" if value >= 0 else f"- {abs(value)}"


def _arithmetic(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    multiplier, subtractor = rng.randint(2, 9), rng.randint(3, 40)
    answer = (a + b) * multiplier - subtractor
    return (
        f"Evaluate ({a} + {b}) × {multiplier} − {subtractor}. Give only the final integer.",
        answer,
    )


def _algebra(rng: random.Random) -> tuple[str, int]:
    coefficient = rng.randint(2, 11)
    solution = rng.randint(-18, 18)
    offset = rng.randint(-30, 30)
    rhs = coefficient * solution + offset
    return (
        f"Determine the integer n satisfying {coefficient}n {_signed(offset)} = {rhs}. Give only n.",
        solution,
    )


def _linear_equations(rng: random.Random) -> tuple[str, int]:
    x, y = rng.randint(-20, 20), rng.randint(-20, 20)
    total, difference = x + y, x - y
    return (
        f"Two integers p and q satisfy p + q = {total} and p − q = {difference}. What is p? Give only the integer.",
        x,
    )


def _functions(rng: random.Random) -> tuple[str, int]:
    a, b, c = rng.randint(1, 7), rng.randint(-12, 12), rng.randint(-20, 20)
    value = rng.randint(-8, 8)
    answer = a * value * value + b * value + c
    return (
        f"Let h(t) = {a}t² {_signed(b)}t {_signed(c)}. Compute h({value}). Give only the integer.",
        answer,
    )


def _differentiation(rng: random.Random) -> tuple[str, int]:
    a, b, c = rng.randint(1, 6), rng.randint(-8, 8), rng.randint(-10, 10)
    point = rng.randint(-5, 5)
    answer = 3 * a * point * point + 2 * b * point + c
    return (
        f"For f(x) = {a}x³ {_signed(b)}x² {_signed(c)}x, evaluate f′({point}). Give only the integer.",
        answer,
    )


def _integration(rng: random.Random) -> tuple[str, int]:
    slope, intercept = rng.randint(1, 8), rng.randint(-10, 10)
    upper = rng.choice((2, 4, 6, 8))
    answer = slope * upper * upper // 2 + intercept * upper
    return (
        f"Compute the definite integral from 0 to {upper} of ({slope}x {_signed(intercept)}) dx. Give only the integer.",
        answer,
    )


def _geometry(rng: random.Random) -> tuple[str, int]:
    length, width, height = (rng.randint(2, 15) for _ in range(3))
    answer = length * width * height
    return (
        f"A rectangular prism is {length} units long, {width} units wide, and {height} units high. What is its volume? Give only the integer.",
        answer,
    )


def _probability(rng: random.Random) -> tuple[str, int]:
    n = rng.randint(6, 15)
    k = rng.randint(2, min(5, n - 1))
    answer = math.comb(n, k)
    return (
        f"How many distinct teams of {k} can be selected from {n} people? Give only the integer.",
        answer,
    )


def _number_theory(rng: random.Random) -> tuple[str, int]:
    base, offset, modulus = rng.randint(11, 99), rng.randint(1, 99), rng.randint(7, 29)
    answer = (base * base + offset) % modulus
    return (
        f"Find the remainder when {base}² + {offset} is divided by {modulus}. Give only the integer.",
        answer,
    )


_PRIMITIVE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
)


def _trigonometry(rng: random.Random) -> tuple[str, int]:
    leg1, leg2, hypotenuse = rng.choice(_PRIMITIVE_TRIPLES)
    scale = rng.randint(1, 9)
    first, second, answer = leg1 * scale, leg2 * scale, hypotenuse * scale
    return (
        f"A right triangle has perpendicular side lengths {first} and {second}. What is the hypotenuse length? Give only the integer.",
        answer,
    )


_GENERATORS: dict[str, Callable[[random.Random], tuple[str, int]]] = {
    "arithmetic": _arithmetic,
    "algebra": _algebra,
    "linear_equations": _linear_equations,
    "functions": _functions,
    "differentiation": _differentiation,
    "integration": _integration,
    "geometry": _geometry,
    "probability": _probability,
    "number_theory": _number_theory,
    "trigonometry": _trigonometry,
}


def generate_math_benchmark(config: MathBenchmarkConfig) -> list[MathBenchmarkRecord]:
    """Generate unique, answer-verified benchmark items from ``config``."""
    rng = random.Random(config.seed)
    records: list[MathBenchmarkRecord] = []
    all_prompts: set[str] = set()
    for topic in config.topics:
        seen_prompts: set[str] = set()
        attempts = 0
        while len(seen_prompts) < config.items_per_topic:
            attempts += 1
            if attempts > config.items_per_topic * 100:
                raise RuntimeError(f"could not produce enough unique {topic} benchmark items")
            prompt, answer = _GENERATORS[topic](rng)
            if prompt in seen_prompts or prompt in all_prompts:
                continue
            seen_prompts.add(prompt)
            all_prompts.add(prompt)
            index = len(seen_prompts)
            records.append(
                MathBenchmarkRecord(
                    item_id=f"{config.version}-{topic}-{index:04d}",
                    prompt=prompt,
                    reference=str(answer),
                    category=topic,
                )
            )
    if len({record.item_id for record in records}) != len(records):
        raise RuntimeError("generated duplicate benchmark item IDs")
    return records


def _canonical_jsonl(records: list[MathBenchmarkRecord]) -> str:
    return "\n".join(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in records) + "\n"


def flag_corpus_overlap(
    records: list[MathBenchmarkRecord],
    corpus_dir: str | Path,
    *,
    n: int = 8,
    threshold: float = 0.6,
) -> dict[str, float]:
    """Return benchmark items that overlap the training split above threshold."""
    return flag_contaminated(
        [record.as_benchmark_item() for record in records],
        corpus_dir,
        n=n,
        threshold=threshold,
    )


def write_math_benchmark(
    config: MathBenchmarkConfig,
    output_path: str | Path | None = None,
    records: list[MathBenchmarkRecord] | None = None,
) -> dict:
    """Write a deterministic JSONL benchmark and a provenance manifest."""
    records = records if records is not None else generate_math_benchmark(config)
    output_path = Path(output_path or config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_jsonl(records)
    output_path.write_text(content, encoding="utf-8")
    benchmark_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    counts = dict(sorted(Counter(record.category for record in records).items()))
    manifest = {
        "benchmark": config.version,
        "task_type": "numeric",
        "benchmark_hash": benchmark_hash,
        "config_digest": config.digest(),
        "seed": config.seed,
        "items": len(records),
        "items_by_category": counts,
        "data_file": output_path.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "m09-independent-math-families-v1",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": output_path, "manifest_path": manifest_path, "manifest": manifest, "records": records}

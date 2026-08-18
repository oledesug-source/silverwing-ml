"""Build a deterministic preference dataset for DPO (M12).

Each record is {"id", "instruction", "chosen", "rejected"} where:
  - ``instruction`` is a math problem from the M08 generators
  - ``chosen``  is the correct, code-verified answer
  - ``rejected`` is a plausible but incorrect answer (wrong sign, off-by-one,
                  truncated, or a distractor drawn from a nearby topic)

The dataset is reproducible from the committed config + seed (M01 rule).

Next step (M12 run): `python scripts/train_alignment.py --config configs/alignment.yaml`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.math_corpus import PROBLEM_GENERATORS
from foundation.math_corpus.problems import Problem

REJECTION_TEMPLATES = [
    "I don't know.",
    "0",
    "1",
    "42",
    "The answer is unknown.",
    "Cannot be determined.",
    "Let me think about this differently.",
    "I'm not sure.",
]


def _parse_answer(answer: str) -> str | int | float | None:
    """Try to extract a numeric value from an answer string."""
    answer = answer.strip()
    if "/" in answer:
        parts = answer.replace(" ", "").split("/")
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) != 0:
            try:
                return fraction(parts[0], parts[1])
            except ValueError:
                pass
    try:
        return int(answer)
    except ValueError:
        pass
    try:
        return float(answer)
    except ValueError:
        pass
    return None


def _frac(num: str, den: str) -> str:
    """Render a simplified fraction string."""
    from math import gcd

    n, d = int(num), int(den)
    g = gcd(abs(n), abs(d))
    n, d = n // g, d // g
    if d == 1:
        return str(n)
    return f"{n}/{d}"


def fraction(num: str, den: str) -> str | int:
    """Return a simplified fraction as int or string."""
    from math import gcd

    n, d = int(num), int(den)
    g = gcd(abs(n), abs(d))
    n, d = n // g, d // g
    if d == 1:
        return n
    return f"{n}/{d}"


def _reject_numeric(correct: str, rng: random.Random) -> str:
    """Generate a plausible but wrong numeric answer."""
    val = _parse_answer(correct)
    if val is None:
        return correct
    if isinstance(val, int):
        candidates = [val + 1, val - 1, -val, val * 2, val * 3]
        if val % 2 == 0:
            candidates.append(val // 2)
        return str(rng.choice(candidates))
    if isinstance(val, float):
        candidates = [val + 0.5, val - 0.5, -val, val * 2, val / 2]
        wrong = rng.choice(candidates)
        return str(wrong)
    if "/" in correct:
        parts = correct.replace(" ", "").split("/")
        n, d = int(parts[0]), int(parts[1])
        candidates = [
            _frac(str(n + 1), str(d)),
            _frac(str(n), str(d + 1)),
            _frac(str(d), str(n)) if n != 0 else _frac(str(n + 1), str(d)),
            _frac(str(n + 1), str(d + 1)),
        ]
        return str(rng.choice(candidates))
    return correct


def make_rejection(rng: random.Random, correct: str) -> str:
    """Produce a plausible-but-wrong answer for a given correct answer."""
    val = _parse_answer(correct)
    if val is not None:
        return _reject_numeric(correct, rng)
    return rng.choice(REJECTION_TEMPLATES)


def generate_preference_records(
    topics: list[str],
    per_topic: int,
    seed: int,
) -> list[dict]:
    """Generate ``per_topic`` preference pairs for each topic."""
    rng = random.Random(seed)
    records: list[dict] = []
    for topic in topics:
        if topic not in PROBLEM_GENERATORS:
            raise ValueError(f"unknown topic: {topic}")
        gen = PROBLEM_GENERATORS[topic]
        for i in range(per_topic):
            problem: Problem = gen(rng)
            rejected = make_rejection(rng, problem.answer)
            attempts = 0
            while rejected == problem.answer and attempts < 10:
                rejected = make_rejection(rng, problem.answer)
                attempts += 1
            records.append(
                {
                    "id": f"dpo-v1-{topic}-{i:04d}",
                    "instruction": problem.question,
                    "chosen": problem.answer,
                    "rejected": rejected,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic DPO preference dataset (M12)"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="subset of topics (default: all)",
    )
    parser.add_argument("--per-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/alignment/dpo-v1.jsonl")
    args = parser.parse_args()

    if args.per_topic < 1:
        raise SystemExit("--per-topic must be positive")

    topics = args.topics or list(PROBLEM_GENERATORS.keys())
    records = generate_preference_records(topics, args.per_topic, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "silverwing-dpo",
        "version": "dpo-v1",
        "seed": args.seed,
        "per_topic": args.per_topic,
        "topics": topics,
        "records": len(records),
        "output_path": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a deterministic reasoning-chain dataset (M13).

Each record is a structured reasoning trace:

    {"id", "reasoning_type", "domain", "problem", "reasoning_steps": [str, ...],
     "final_answer", "difficulty", "quality_score"}

Generated from the M08 math problem generators: the problem question becomes
the prompt, the solution field is split into numbered reasoning steps, and the
answer is the final answer.  The dataset is reproducible from the committed
config + seed (M01 rule).

Next step (M13 run): `python scripts/train_reasoning.py --config configs/reasoning.yaml`.
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
from foundation.reasoning.config import ReasoningDatasetConfig
from foundation.reasoning.dataset import split_into_steps


def _select_reasoning_type(topic: str) -> str:
    """Map a math topic to a reasoning type for the legacy 79R taxonomy."""
    mapping = {
        "arithmetic": "numerical_reasoning",
        "algebra": "multi_step",
        "linear_equations": "multi_step",
        "functions": "multi_step",
        "differentiation": "multi_step",
        "integration": "multi_step",
        "geometry": "constraint_reasoning",
        "probability": "numerical_reasoning",
        "number_theory": "deduction",
        "trigonometry": "multi_step",
    }
    return mapping.get(topic, "multi_step")


def _compute_difficulty(problem_obj) -> float:
    """Estimate difficulty from answer length and token complexity."""
    answer_len = len(problem_obj.answer)
    solution_len = len(problem_obj.solution)
    # Normalize to [0.0, 1.0]
    raw = (answer_len / 20.0 + solution_len / 200.0) / 2.0
    return round(min(max(raw, 0.1), 1.0), 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic reasoning-chain dataset (M13)"
    )
    parser.add_argument("--config", default="configs/reasoning_dataset.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-topic", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = ReasoningDatasetConfig.from_yaml(args.config)
    if args.output:
        cfg = ReasoningDatasetConfig(
            version=cfg.version,
            seed=cfg.seed,
            per_topic=cfg.per_topic,
            topics=cfg.topics,
            output_path=args.output,
        )
    if args.per_topic is not None:
        cfg = ReasoningDatasetConfig(
            version=cfg.version,
            seed=cfg.seed,
            per_topic=args.per_topic,
            topics=cfg.topics,
            output_path=cfg.output_path,
        )
    if args.seed is not None:
        cfg = ReasoningDatasetConfig(
            version=cfg.version,
            seed=args.seed,
            per_topic=cfg.per_topic,
            topics=cfg.topics,
            output_path=cfg.output_path,
        )

    if cfg.per_topic < 1:
        raise SystemExit("--per-topic must be positive")

    topics = cfg.topics or list(PROBLEM_GENERATORS.keys())
    unknown = [t for t in topics if t not in PROBLEM_GENERATORS]
    if unknown:
        raise SystemExit(f"unknown topics: {unknown}")

    rng = random.Random(cfg.seed)
    records: list[dict] = []
    for topic in topics:
        gen = PROBLEM_GENERATORS[topic]
        for i in range(cfg.per_topic):
            problem = gen(rng)
            steps = split_into_steps(problem.solution)
            if not steps:
                steps = [problem.solution]
            records.append(
                {
                    "id": f"reasoning-v1-{topic}-{i:04d}",
                    "reasoning_type": _select_reasoning_type(topic),
                    "domain": topic,
                    "problem": problem.question,
                    "reasoning_steps": steps,
                    "final_answer": problem.answer,
                    "difficulty": _compute_difficulty(problem),
                    "quality_score": 1.0,
                }
            )

    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "silverwing-reasoning",
        "version": cfg.version,
        "config_digest": cfg.digest(),
        "seed": cfg.seed,
        "per_topic": cfg.per_topic,
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

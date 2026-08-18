"""Build curriculum stage datasets (M14).

Generates 3 SFT datasets with increasing difficulty:
  Stage 1 (Basic): arithmetic, geometry, probability
  Stage 2 (Intermediate): algebra, linear_equations, functions, number_theory
  Stage 3 (Advanced): differentiation, integration, trigonometry

Each stage has 1000 examples. Run:
    python scripts/build_curriculum_datasets.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.math_corpus import PROBLEM_GENERATORS  # noqa: E402

STAGES = {
    "stage1-basic": {
        "topics": ["arithmetic", "geometry", "probability"],
        "description": "Basic arithmetic, geometry, probability",
        "examples_per_topic": 334,
    },
    "stage2-intermediate": {
        "topics": ["algebra", "linear_equations", "functions", "number_theory"],
        "description": "Algebra, linear equations, functions, number theory",
        "examples_per_topic": 250,
    },
    "stage3-advanced": {
        "topics": ["differentiation", "integration", "trigonometry"],
        "description": "Differentiation, integration, trigonometry",
        "examples_per_topic": 334,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curriculum stage datasets (M14)")
    parser.add_argument("--output-dir", default="experiments/curriculum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-topic", type=int, default=None, help="override examples per topic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for stage_name, stage_cfg in STAGES.items():
        topics = stage_cfg["topics"]
        per_topic = args.per_topic or stage_cfg["examples_per_topic"]

        records = []
        for topic in topics:
            gen = PROBLEM_GENERATORS[topic]
            for i in range(per_topic):
                problem = gen(rng)
                records.append({
                    "id": f"{stage_name}-{topic}-{i:04d}",
                    "instruction": problem.question,
                    "response": problem.answer,
                })

        output_path = output_dir / f"{stage_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest = {
            "stage": stage_name,
            "description": stage_cfg["description"],
            "seed": args.seed,
            "topics": topics,
            "examples_per_topic": per_topic,
            "total_examples": len(records),
            "output_path": str(output_path),
        }
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {stage_name}: {len(records)} examples -> {output_path}")

    print(f"\nAll 3 stages built in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

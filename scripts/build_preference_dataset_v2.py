"""Build the DPO preference dataset v2 (M17): math + general chat.

Math pairs reuse the M12 generator (correct answer vs plausible wrong one).
General pairs come from foundation/general_corpus: chosen = the bank's true,
helpful response; rejected = a realistic bad output of the math-only SFT
model (evasive, garbled math scaffold, or a distractor answer).

Writes:
    experiments/alignment/dpo-v2-all.jsonl
    experiments/alignment/dpo-v2-all.manifest.json

Reproducible from seed alone (M01 rule).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.general_corpus import general_preference_pairs, write_jsonl  # noqa: E402
from scripts.build_preference_dataset import generate_preference_records  # noqa: E402

DEFAULT_OUTPUT = "experiments/alignment/dpo-v2-all.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DPO v2 preference dataset")
    parser.add_argument("--math-per-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from foundation.math_corpus import PROBLEM_GENERATORS

    topics = list(PROBLEM_GENERATORS.keys())

    records: list[dict] = []
    seen_ids: set[str] = set()

    for record in generate_preference_records(topics, args.math_per_topic, args.seed):
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)
    n_math = len(records)

    for record in general_preference_pairs(seed=args.seed):
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)

    rng = random.Random(args.seed + 1)
    rng.shuffle(records)

    output_path = ROOT / args.output
    write_jsonl(output_path, records)

    manifest = {
        "dataset_name": "silverwing-dpo",
        "version": "dpo-v2-all",
        "seed": args.seed,
        "sources": {
            "math_pairs": n_math,
            "general_pairs": len(records) - n_math,
            "topics": topics,
        },
        "total": len(records),
        "output_path": str(output_path),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

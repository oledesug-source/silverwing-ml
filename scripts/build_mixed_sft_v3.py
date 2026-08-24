"""Build the mixed SFT dataset v3 (M17/M18): math + reasoning + general chat
+ advanced STEM.

Combines the existing committed corpora (sft-v2-all.jsonl = 4,000 math +
1,000 chain-of-thought) with the deterministic general-conversation bank
(foundation/general_corpus) AND the M18 advanced generators (linear algebra,
advanced probability, statistics - each emitted twice: bare-answer and
chain-of-thought form), shuffles with a fixed seed, and writes:

    experiments/sft/sft-v3-all.jsonl
    experiments/sft/sft-v3-all.manifest.json

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

from foundation.general_corpus import expand_bank, write_jsonl  # noqa: E402
from foundation.math_corpus import PROBLEM_GENERATORS  # noqa: E402

DEFAULT_BASE = "experiments/sft/sft-v2-all.jsonl"
DEFAULT_OUTPUT = "experiments/sft/sft-v3-all.jsonl"
ADVANCED_TOPICS = ("linear_algebra", "advanced_probability", "statistics")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mixed SFT v3 dataset")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-topic-advanced", type=int, default=150,
        help="problems per advanced topic (each also emitted as chain-of-thought)",
    )
    args = parser.parse_args()

    base_path = ROOT / args.base
    if not base_path.exists():
        raise SystemExit(f"base dataset missing: {base_path}")

    records: list[dict] = []
    seen_ids: set[str] = set()

    def add(record: dict) -> None:
        if record["id"] in seen_ids:
            return
        seen_ids.add(record["id"])
        records.append(record)

    for line in base_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            add(json.loads(line))
    n_base = len(records)

    # ---- advanced STEM topics (bare answer + chain-of-thought variants) ----
    rng = random.Random(args.seed)
    n_advanced = 0
    for topic in ADVANCED_TOPICS:
        gen = PROBLEM_GENERATORS[topic]
        for i in range(args.per_topic_advanced):
            problem = gen(rng)
            add({
                "id": f"sft-v3-adv-{topic}-{i:04d}",
                "instruction": problem.question,
                "response": problem.answer,
            })
            add({
                "id": f"cot-v3-{topic}-{i:04d}",
                "instruction": problem.question,
                "response": problem.solution,
            })
            n_advanced += 2

    for record in expand_bank(seed=args.seed):
        add(record)

    random.Random(args.seed + 7).shuffle(records)

    output_path = ROOT / args.output
    write_jsonl(output_path, records)

    by_prefix: dict[str, int] = {}
    for record in records:
        parts = record["id"].split("-")
        prefix = "-".join(parts[:2]) if record["id"].startswith(("sft-v3", "cot-v3")) else parts[1] if len(parts) > 1 else "?"
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    manifest = {
        "dataset_name": "silverwing-sft",
        "version": "sft-v3-all",
        "seed": args.seed,
        "sources": {
            "base": str(args.base),
            "base_records": n_base,
            "advanced_records": n_advanced,
            "advanced_topics": list(ADVANCED_TOPICS),
            "general_bank_records": len(records) - n_base - n_advanced,
        },
        "records_by_kind": by_prefix,
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

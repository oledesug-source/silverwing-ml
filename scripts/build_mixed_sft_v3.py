"""Build the mixed SFT dataset v3 (M17-M19): unified lesson track + general.

Layers:
  1. base corpus      - sft-v2-all.jsonl (4,000 math + 1,000 CoT)
  2. UNIFIED LESSONS  - foundation/lesson_plan.py: one graded track from
                        lesson 1 (arithmetic) to lesson 16 (networking),
                        every problem emitted as bare-answer AND
                        chain-of-thought records
  3. general chat     - foundation/general_corpus bank

Writes experiments/sft/sft-v3-all.jsonl (+ .manifest.json). Reproducible
from seed alone (M01 rule).
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
from foundation.lesson_plan import lesson_records, manifest as lesson_manifest  # noqa: E402

DEFAULT_BASE = "experiments/sft/sft-v2-all.jsonl"
DEFAULT_OUTPUT = "experiments/sft/sft-v3-all.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mixed SFT v3 dataset")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
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

    # ---- unified lesson track (lesson 1 .. last) ----
    rng = random.Random(args.seed)
    lessons = lesson_records(rng)
    for record in lessons:
        add(record)
    n_lessons = len(lessons)

    for record in expand_bank(seed=args.seed):
        add(record)

    # ---- integrated tool-use traces (M21): model learns to call tools ----
    for extra in ("tool-traces-v1.jsonl", "coding-v1.jsonl"):
        trace_path = ROOT / "experiments" / "sft" / extra
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    add(json.loads(line))

    random.Random(args.seed + 7).shuffle(records)

    output_path = ROOT / args.output
    write_jsonl(output_path, records)

    by_prefix: dict[str, int] = {}
    for record in records:
        parts = record["id"].split("-")
        prefix = parts[0] if parts[0].startswith("U") else (
            "-".join(parts[:2]) if record["id"].startswith(("sft-v3", "cot-v3")) else parts[1] if len(parts) > 1 else "?")
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    manifest = {
        "dataset_name": "silverwing-sft",
        "version": "sft-v3-all",
        "seed": args.seed,
        "sources": {
            "base": str(args.base),
            "base_records": n_base,
            "unified_lessons": n_lessons,
            "lesson_plan": lesson_manifest(),
            "general_bank_records": len(records) - n_base - n_lessons,
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

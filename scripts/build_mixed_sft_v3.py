"""Build the mixed SFT dataset v3 (M17): math + reasoning + general chat.

Combines the existing committed corpora (sft-v2-all.jsonl = 4,000 math +
1,000 chain-of-thought) with the deterministic general-conversation bank
(foundation/general_corpus), shuffles with a fixed seed, and writes:

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

    for line in base_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)

    n_base = len(records)

    general = expand_bank(seed=args.seed)
    for record in general:
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    output_path = ROOT / args.output
    write_jsonl(output_path, records)

    by_prefix: dict[str, int] = {}
    for record in records:
        prefix = record["id"].split("-")[1] if "-" in record["id"] else "?"
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    manifest = {
        "dataset_name": "silverwing-sft",
        "version": "sft-v3-all",
        "seed": args.seed,
        "sources": {
            "base": str(args.base),
            "base_records": n_base,
            "general_bank_records": len(records) - n_base,
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

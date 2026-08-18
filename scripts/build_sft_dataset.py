"""Build the deterministic SFT dataset (M11) from the M08 problem generators.

Each record is {"id", "instruction", "response"} where the response is the
code-computed answer from the generator, so the supervised text is exact and
reproducible (M01 rule) from the committed config + seed.

Next step (M11 run): `python scripts/train_sft.py --config configs/sft.yaml`.
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
from foundation.sft import SftDatasetConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic SFT dataset (M11)")
    parser.add_argument("--config", default="configs/sft_dataset.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-topic", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = SftDatasetConfig.from_yaml(args.config)
    if args.output:
        cfg = SftDatasetConfig(
            version=cfg.version,
            per_topic=cfg.per_topic,
            seed=cfg.seed,
            topics=cfg.topics,
            output_path=args.output,
        )
    if args.per_topic:
        cfg = SftDatasetConfig(
            version=cfg.version,
            per_topic=args.per_topic,
            seed=cfg.seed,
            topics=cfg.topics,
            output_path=cfg.output_path,
        )
    if args.seed is not None:
        cfg = SftDatasetConfig(
            version=cfg.version,
            per_topic=cfg.per_topic,
            seed=args.seed,
            topics=cfg.topics,
            output_path=cfg.output_path,
        )

    topics = list(cfg.topics or PROBLEM_GENERATORS.keys())
    unknown = [t for t in topics if t not in PROBLEM_GENERATORS]
    if unknown:
        raise SystemExit(f"unknown topics: {unknown}")

    rng = random.Random(cfg.seed)
    records: list[dict] = []
    for topic in topics:
        gen = PROBLEM_GENERATORS[topic]
        for i in range(cfg.per_topic):
            problem = gen(rng)
            records.append(
                {
                    "id": f"sft-v1-{topic}-{i:04d}",
                    "instruction": problem.question,
                    "response": problem.answer,
                }
            )

    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "silverwing-sft",
        "version": cfg.version,
        "config_digest": cfg.digest(),
        "seed": cfg.seed,
        "per_topic": cfg.per_topic,
        "topics": topics,
        "records": len(records),
        "output_path": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

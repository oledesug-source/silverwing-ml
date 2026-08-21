"""Generate and release the contamination-checked M09 math benchmark.

The benchmark is generated from independent evaluation-only problem families.
By default it verifies the selected training corpus and refuses to write a
benchmark if any item violates the configured n-gram contamination threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.math import (  # noqa: E402
    MathBenchmarkConfig,
    flag_corpus_overlap,
    generate_math_benchmark,
    write_math_benchmark,
)
from foundation.corpus import verify_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the held-out Silverwing mathematics benchmark (M09)")
    parser.add_argument("--config", default=None, help="math benchmark YAML (default: configs/math_benchmark.yaml)")
    parser.add_argument("--output-path", default=None, help="override the JSONL output path")
    parser.add_argument("--corpus-dir", default=None, help="override the corpus to check for contamination")
    parser.add_argument("--items-per-topic", type=int, default=None, help="override item count per configured topic")
    parser.add_argument("--force", action="store_true", help="allow replacing an existing benchmark JSONL")
    parser.add_argument(
        "--skip-contamination-check",
        action="store_true",
        help="development-only: do not verify a corpus or run the overlap guard",
    )
    args = parser.parse_args()

    cfg = MathBenchmarkConfig.from_yaml(args.config)
    cfg = replace(
        cfg,
        output_path=args.output_path or cfg.output_path,
        corpus_dir=args.corpus_dir or cfg.corpus_dir,
        items_per_topic=args.items_per_topic if args.items_per_topic is not None else cfg.items_per_topic,
    )
    output_path = Path(cfg.output_path)
    if output_path.exists() and not args.force:
        parser.error(f"benchmark file already exists: {output_path}; use --force to replace it")

    records = generate_math_benchmark(cfg)
    if args.skip_contamination_check:
        print("warning: contamination check skipped; this output is not a publishable benchmark", file=sys.stderr)
    else:
        verification = verify_dataset(cfg.corpus_dir)
        if not verification.ok:
            print(json.dumps(verification.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
            print("refusing benchmark release: training corpus integrity verification failed", file=sys.stderr)
            return 2
        flagged = flag_corpus_overlap(
            records,
            cfg.corpus_dir,
            n=cfg.contamination_ngram,
            threshold=cfg.contamination_threshold,
        )
        if flagged:
            print(json.dumps({"contaminated_items": flagged}, ensure_ascii=False, indent=2), file=sys.stderr)
            print("refusing benchmark release: corpus overlap threshold exceeded", file=sys.stderr)
            return 1

    release = write_math_benchmark(cfg, records=records)
    manifest = release["manifest"]
    print(f"benchmark={manifest['benchmark']} items={manifest['items']} hash={manifest['benchmark_hash']}")
    print(f"data={release['path']}")
    print(f"manifest={release['manifest_path']}")
    print(f"evaluate with: python scripts/run_benchmark.py --benchmark {manifest['benchmark']} --model dummy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

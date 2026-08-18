"""Verify the integrity of a built Silverwing corpus (M02/M03).

Recomputes the dataset root digest from the shard files on disk and compares
it against the hash recorded in manifest.json (and optionally a pinned
expected hash), detecting corruption, truncation, missing shards or release
substitution. Exits 0 on success, 1 on any integrity failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.corpus import load_corpus_config, verify_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Silverwing corpus integrity")
    parser.add_argument("--config", default=None, help="corpus config (for output_dir / pinned hash)")
    parser.add_argument("--output-dir", default=None, help="dataset directory (default experiments/corpus)")
    parser.add_argument("--expected-hash", default=None, help="pinned dataset hash to require")
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    args = parser.parse_args()

    config = load_corpus_config(args.config) if args.config else {}
    output_dir = args.output_dir or config.get("output_dir", "experiments/corpus")

    expected = args.expected_hash
    if expected is None:
        pinned = config.get("integrity", {}).get("dataset_hash")
        if pinned not in (None, "", "recomputed"):
            expected = pinned

    result = verify_dataset(output_dir, expected_dataset_hash=expected)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"manifest: {result.manifest_path}")
        print(f"ok: {result.ok}")
        print(f"dataset_hash recorded={result.recorded_dataset_hash or 'n/a'} computed={result.computed_dataset_hash or 'n/a'}")
        for missing in result.missing_shards:
            print(f"missing shard: {missing}")
        for error in result.split_errors:
            print(f"error: {error}")

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()

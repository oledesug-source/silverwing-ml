"""Generate the deterministic math training corpus (M08).

Writes the seeded math curriculum documents to the configured staging dir
(experiments/raw-math by default) plus a generation_report.json pinning the
config digest, git commit, per-topic counts and content digest.

Next step (M02/M03 release):
    scripts/build_corpus.py --source math=<staging_dir> --output-dir <corpus_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.math_corpus import MathCorpusConfig, generate_math_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic math training corpus (M08).")
    parser.add_argument("--config", default=None, help="Path to math corpus config (default: configs/math_corpus.yaml)")
    parser.add_argument("--cap", type=int, default=None, help="Optional cap on total documents (debugging)")
    args = parser.parse_args()

    cfg = MathCorpusConfig.from_yaml(args.config)
    report = generate_math_corpus(cfg, cap_documents=args.cap)
    print(f"math corpus version={cfg.version} seed={cfg.seed} "
          f"documents={report['total_documents']} content_digest={report['content_digest']}")
    print(f"staging dir: {report['staging_dir']}")
    print("release with: scripts/build_corpus.py "
          f"--source math={cfg.staging_dir} --output-dir {cfg.corpus_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

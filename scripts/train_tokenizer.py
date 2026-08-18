"""Train Tokenizer V2 from the sharded corpus.

Usage:
    python scripts/train_tokenizer.py --corpus-dir experiments/corpus
    python scripts/train_tokenizer.py --vocab-size 8192 --max-documents 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.tokenizer import train_tokenizer_from_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Silverwing Tokenizer V2 on a sharded corpus")
    parser.add_argument("--corpus-dir", default=None, help="sharded corpus directory (default from tokenizer.yaml)")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    defaults = {}
    config_path = Path("configs") / "tokenizer.yaml"
    if config_path.exists():
        import yaml

        defaults = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("tokenizer", {})

    def pick(cli, key, default):
        return cli if cli is not None else defaults.get(key, default)

    report = train_tokenizer_from_corpus(
        corpus_dir=pick(args.corpus_dir, "corpus_dir", "experiments/corpus"),
        vocab_size=pick(args.vocab_size, "vocab_size", 16384),
        output_dir=pick(args.output_dir, "output_dir", "experiments/tokenizer"),
        min_frequency=pick(args.min_frequency, "min_frequency", 2),
        max_documents=args.max_documents if args.max_documents is not None else defaults.get("max_documents"),
        max_bytes=defaults.get("max_bytes"),
    )
    print(f"vocab_size={report['vocab_size']} merges={report['corpus_stats']['produced_merges']}")
    print(f"early_stopped={report['corpus_stats']['early_stopped']}")
    print(f"tokenizer_hash={report['tokenizer_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark-vs-corpus contamination guard.

Verifies that a benchmark's items do not overlap a training corpus (which would
inflate eval scores). Uses the same normalized-token n-gram matching as the
corpus pipeline so the two share one contamination definition.
"""

from __future__ import annotations

import json
from pathlib import Path

from foundation.corpus.contamination import ngrams, normalize_tokens

from .dataset import BenchmarkItem


def build_corpus_fingerprint(corpus_dir: str | Path, split: str = "train", n: int = 8, max_items: int | None = None) -> set:
    """Fingerprint (normalized n-gram set) of a sharded corpus split."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_dir}")
    fingerprint: set = set()
    items = 0
    for shard in sorted(corpus_dir.glob(f"{split}.*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            fingerprint.update(ngrams(normalize_tokens(text), n))
            items += 1
            if max_items is not None and items >= max_items:
                return fingerprint
    return fingerprint


def item_contamination_ratio(item: BenchmarkItem, fingerprint: set, n: int = 8, max_ngrams: int = 5000) -> float:
    """Fraction of the item's n-grams found in the corpus fingerprint."""
    grams = ngrams(normalize_tokens(item.prompt + " " + item.reference), n)
    if not grams:
        return 0.0
    sampled = grams[:max_ngrams]
    hits = sum(1 for gram in sampled if gram in fingerprint)
    return hits / len(sampled)


def flag_contaminated(
    items: list[BenchmarkItem],
    corpus_dir: str | Path,
    n: int = 8,
    threshold: float = 0.6,
) -> dict[str, float]:
    """Return {item_id: ratio} for items overlapping the corpus above threshold."""
    fingerprint = build_corpus_fingerprint(corpus_dir, n=n)
    flagged = {}
    for item in items:
        ratio = item_contamination_ratio(item, fingerprint, n=n)
        if ratio >= threshold:
            flagged[item.item_id] = ratio
    return flagged

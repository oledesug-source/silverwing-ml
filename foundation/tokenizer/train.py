"""Train Tokenizer V2 from a sharded corpus.

Consumes the corpus produced by the M02/M03 pipeline (sharded JSONL + manifest)
and writes a tokenizer release (vocab, merges, config, hash) plus a training
report that pins the corpus, config and git commit for reproducibility.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .bpe import train_bpe
from .tokenizer import TokenizerV2


def iter_corpus_texts(
    corpus_dir: str | Path,
    split: str = "train",
    max_documents: int | None = None,
    max_bytes: int | None = None,
) -> Iterable[str]:
    """Yield document texts from a sharded corpus split, optionally capped."""
    corpus_dir = Path(corpus_dir)
    documents = 0
    total_bytes = 0
    for shard in sorted(corpus_dir.glob(f"{split}.*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if max_documents is not None and documents >= max_documents:
                return
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue
            if max_bytes is not None and total_bytes + len(text.encode("utf-8")) > max_bytes:
                return
            documents += 1
            total_bytes += len(text.encode("utf-8"))
            yield text


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def train_tokenizer_from_corpus(
    corpus_dir: str | Path,
    vocab_size: int,
    output_dir: str | Path,
    min_frequency: int = 2,
    max_documents: int | None = None,
    max_bytes: int | None = None,
    split: str = "train",
) -> dict:
    """Train and save a tokenizer; returns a training report dict."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_dir}")
    texts = list(iter_corpus_texts(corpus_dir, split=split, max_documents=max_documents, max_bytes=max_bytes))
    merges, stats = train_bpe(texts, vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer = TokenizerV2(merges=merges)
    tokenizer.save(output_dir)

    manifest_path = corpus_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "tokenizer_version": tokenizer.version,
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "corpus_dir": str(corpus_dir),
        "corpus_split": split,
        "corpus_dataset_hash": manifest.get("dataset_hash"),
        "corpus_stats": stats,
        "tokenizer_hash": tokenizer.digest(),
        "git_commit": _git_commit(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = Path(output_dir) / "tokenizer_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

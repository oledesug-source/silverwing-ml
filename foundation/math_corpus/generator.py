"""Deterministic math corpus generator (M08).

Reads a :class:`MathCorpusConfig`, seeds a ``random.Random``, and writes one
``.txt`` document per curriculum item into ``staging_dir``.  The output is
fully reproducible from the committed config + seed: the same config always
produces byte-identical files, and a ``generation_report.json`` pins the
config digest, git commit and content digest so downstream tokenizer/model
releases can trace back to it.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

from .config import MathCorpusConfig
from .documents import build_document
from .problems import PROBLEM_GENERATORS, REFERENCES


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def generate_math_corpus(cfg: MathCorpusConfig, cap_documents: int | None = None) -> dict:
    rng = random.Random(cfg.seed)
    staging = Path(cfg.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    written = 0
    remaining_global = cap_documents if cap_documents is not None else cfg.total_documents
    for topic, count in cfg.curriculum.items():
        gen = PROBLEM_GENERATORS[topic]
        reference = REFERENCES[topic]
        n_docs = min(count, remaining_global)
        if n_docs <= 0:
            break
        for i in range(n_docs):
            examples = [gen(rng) for _ in range(cfg.examples_per_document)]
            exercises = [gen(rng) for _ in range(cfg.exercises_per_document)]
            text = build_document(topic, reference, examples, exercises, i)
            (staging / f"{topic}_{i:05d}.txt").write_text(text, encoding="utf-8")
        written += n_docs
        remaining_global -= n_docs

    content_digest = _content_digest(staging)
    report = {
        "version": cfg.version,
        "seed": cfg.seed,
        "config_digest": cfg.digest(),
        "git_commit": _git_commit(),
        "total_documents": written,
        "documents_per_topic": _docs_per_topic(cfg, cap_documents),
        "content_digest": content_digest,
        "staging_dir": str(staging),
    }
    (staging / "generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _docs_per_topic(cfg: MathCorpusConfig, cap_documents: int | None) -> dict[str, int]:
    remaining = cap_documents if cap_documents is not None else cfg.total_documents
    out: dict[str, int] = {}
    for topic, count in cfg.curriculum.items():
        n = min(count, remaining)
        out[topic] = n
        remaining -= n
    return out


def _content_digest(staging: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(staging.glob("*.txt")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()

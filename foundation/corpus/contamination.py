"""Contamination detection against held-out benchmark material.

A document is flagged as contaminated when an unusually large fraction of its
n-grams also appear in any benchmark string. Matching uses normalized tokens
(lowercased, punctuation stripped) so casing or formatting differences between
a document and benchmark text do not hide contamination. Contaminated documents
are removed from the training split so held-out evaluation stays honest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .schema import DocumentRecord

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


_DEFAULT_BENCHMARKS = (
    "GSM8K MATH ARC MMLU HellaSwag Winograd WinoGrande OpenBookQA "
    "TriviaQA NaturalQuestions HumanEval APPS"
)


@dataclass
class ContaminationDetector:
    n: int = 8
    threshold: float = 0.6
    max_document_ngrams: int = 5000
    _benchmarks: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)

    def add_benchmark(self, text: str, name: str = "benchmark") -> None:
        tokens = normalize_tokens(text)
        self._benchmarks[name] = self._benchmarks.get(name, set()) | set(ngrams(tokens, self.n))

    @property
    def benchmark_names(self) -> list[str]:
        return list(self._benchmarks)

    def _union(self) -> set[tuple[str, ...]]:
        merged: set[tuple[str, ...]] = set()
        for grams in self._benchmarks.values():
            merged |= grams
        return merged

    def contamination_ratio(self, text: str) -> float:
        grams = ngrams(normalize_tokens(text), self.n)
        if not grams:
            return 0.0
        sampled = grams[: self.max_document_ngrams]
        hits = sum(1 for gram in sampled if gram in self._union())
        return hits / len(sampled)

    def per_benchmark_ratios(self, text: str) -> dict[str, float]:
        grams = ngrams(normalize_tokens(text), self.n)
        if not grams:
            return dict.fromkeys(self._benchmarks, 0.0)
        sampled = grams[: self.max_document_ngrams]
        return {
            name: sum(1 for gram in sampled if gram in grams) / len(sampled)
            for name, grams in self._benchmarks.items()
        }

    def is_contaminated(self, record: DocumentRecord) -> bool:
        if not self._benchmarks:
            return False
        return self.contamination_ratio(record.text) >= self.threshold

    def filter(self, records: Iterable[DocumentRecord]) -> tuple[list[DocumentRecord], int, list[DocumentRecord]]:
        kept, dropped, flagged = [], 0, []
        for record in records:
            if self.is_contaminated(record):
                dropped += 1
                flagged.append(record)
            else:
                kept.append(record)
        return kept, dropped, flagged

    def report(self, records: Iterable[DocumentRecord]) -> dict:
        stats = {name: {"checked": 0, "max_ratio": 0.0, "over_threshold": 0} for name in self._benchmarks}
        for record in records:
            ratios = self.per_benchmark_ratios(record.text)
            for name, ratio in ratios.items():
                entry = stats[name]
                entry["checked"] += 1
                entry["max_ratio"] = max(entry["max_ratio"], ratio)
                if ratio >= self.threshold:
                    entry["over_threshold"] += 1
        return {"n": self.n, "threshold": self.threshold, "benchmarks": stats}

    def write_report(self, records: Iterable[DocumentRecord], path: Path | str) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.report(records), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def build_default_detector() -> ContaminationDetector:
    detector = ContaminationDetector()
    detector.add_benchmark(_DEFAULT_BENCHMARKS, name="heldout-suite")
    return detector

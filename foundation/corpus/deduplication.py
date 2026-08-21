"""Exact and near-duplicate detection.

Exact duplicates are removed by content hash. Near-duplicates are found with a
dependency-free MinHash (random-hash ensemble) plus LSH banding. All hashing is
seeded so results are reproducible.

Performance note: MinHash complexity is O(num_hashes * num_ngrams) per document
and LSH lookups are O(bands * candidates). For large corpora (>100K docs)
reduce num_hashes and bands to trade slight accuracy for major speed gains.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field

from .schema import DocumentRecord

logger = logging.getLogger(__name__)

_PRIME = 2**61 - 1


def _w_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    words = text.split()
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


@dataclass
class MinHash:
    num_hashes: int = 64
    seed: int = 42
    normalize: bool = True
    _coeffs: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self._coeffs = [
            (rng.randrange(1, _PRIME - 1), rng.randrange(0, _PRIME - 1)) for _ in range(self.num_hashes)
        ]

    @staticmethod
    def _digest(token: str) -> int:
        return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")

    def signature(self, text: str, ngram: int = 5) -> list[int]:
        sig = [_PRIME] * self.num_hashes
        payload = text.lower() if self.normalize else text
        for gram in _w_ngrams(payload, ngram):
            value = self._digest(" ".join(gram))
            for i, (a, b) in enumerate(self._coeffs):
                h = (a * value + b) % _PRIME
                if h < sig[i]:
                    sig[i] = h
        return sig

    @staticmethod
    def similarity(a: list[int], b: list[int]) -> float:
        if not a or len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)


class LSHBands:
    """LSH index over MinHash signatures for candidate near-dup detection."""

    def __init__(self, signature: MinHash, bands: int = 8) -> None:
        if signature.num_hashes % bands != 0:
            raise ValueError("num_hashes must be divisible by bands")
        self.rows = signature.num_hashes // bands
        self._index: dict[tuple[int, ...], list[str]] = {}

    def add(self, document_id: str, sig: list[int]) -> None:
        for i in range(0, len(sig), self.rows):
            key = tuple(sig[i : i + self.rows])
            self._index.setdefault(key, []).append(document_id)

    def candidates(self, sig: list[int]) -> set[str]:
        found: set[str] = set()
        for i in range(0, len(sig), self.rows):
            key = tuple(sig[i : i + self.rows])
            found.update(self._index.get(key, ()))
        return found


@dataclass
class Deduplicator:
    seed: int = 42
    num_hashes: int = 64
    bands: int = 16
    similarity_threshold: float = 0.85

    def __post_init__(self) -> None:
        self._minhash = MinHash(num_hashes=self.num_hashes, seed=self.seed)
        self._lsh = LSHBands(self._minhash, bands=self.bands)
        self._exact: set[str] = set()
        self._lsh_sigs: dict[str, list[int]] = {}
        self._count = 0
    def _keep(self, record: DocumentRecord) -> bool:
        if record.content_hash in self._exact:
            return False
        self._exact.add(record.content_hash)
        sig = self._minhash.signature(record.text)
        for candidate in self._lsh.candidates(sig):
            if MinHash.similarity(self._lsh_sigs[candidate], sig) >= self.similarity_threshold:
                return False
        self._lsh.add(record.document_id, sig)
        self._lsh_sigs[record.document_id] = sig
        return True

    def apply(self, records) -> tuple[list[DocumentRecord], int]:
        kept, dropped = [], 0
        for record in records:
            if self._keep(record):
                kept.append(record)
            else:
                dropped += 1
            self._count += 1
            if self._count % 5000 == 0:
                logger.info("  dedup progress: %d processed, %d kept, %d dropped", self._count, len(kept), dropped)
        return kept, dropped

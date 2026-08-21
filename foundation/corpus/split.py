"""Deterministic train/validation/test splitting.

Splits are derived from a salted SHA-256 hash of the document_id, so splits are
stable across runs and new documents never silently change the membership of
existing documents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .schema import DocumentRecord, Split, SplitOptions


class Splitter:
    def __init__(self, options: SplitOptions | None = None, salt: str = "silverwing-split-v1") -> None:
        self.options = options or SplitOptions()
        self.salt = salt

    def _bucket(self, document_id: str) -> int:
        digest = hashlib.sha256((self.salt + document_id).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % 1_000_000

    def split(self, records: Iterable[DocumentRecord]) -> dict[str, list[DocumentRecord]]:
        buckets: dict[str, list[DocumentRecord]] = {Split.TRAIN.value: [], Split.VALIDATION.value: [], Split.TEST.value: []}
        val_start = int(self.options.train * 1_000_000)
        test_start = int((self.options.train + self.options.validation) * 1_000_000)
        for record in records:
            bucket = self._bucket(record.document_id)
            if bucket < val_start:
                name = Split.TRAIN.value
            elif bucket < test_start:
                name = Split.VALIDATION.value
            else:
                name = Split.TEST.value
            record.split = name
            buckets[name].append(record)
        return buckets

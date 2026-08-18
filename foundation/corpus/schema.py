"""Provenance schema for the Silverwing corpus.

Every document that enters the training pipeline carries a DocumentRecord
holding full provenance so that any sample can be traced back to its raw
source, processing history, quality score and dataset split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

PROCESSING_VERSION = "corpus-v1"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SourceType(str, Enum):
    BOOK = "book"
    ARTICLE = "article"
    WEB = "web"
    WIKI = "wiki"
    CODE = "code"
    MATH = "math"
    MANUAL = "manual"
    OTHER = "other"


@dataclass
class SplitOptions:
    train: float = 0.96
    validation: float = 0.02
    test: float = 0.02

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        for name, value in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} fraction out of range: {value}")


@dataclass
class Provenance:
    source_id: str
    source_type: str = SourceType.OTHER.value
    domain: str = "unknown"
    language: str = "unknown"
    collection_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_document: Optional[str] = None
    processing_version: str = PROCESSING_VERSION


@dataclass
class DocumentRecord:
    document_id: str
    text: str
    provenance: Provenance
    content_hash: str
    split: Optional[str] = None
    token_count: int = 0
    quality_score: float = 1.0
    flags: dict = field(default_factory=dict)

    @classmethod
    def build(cls, document_id: str, text: str, provenance: Provenance, normalized_text: Optional[str] = None) -> "DocumentRecord":
        payload = (normalized_text or text).strip()
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(document_id=document_id, text=payload, provenance=provenance, content_hash=content_hash)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "provenance": {
                "source_id": self.provenance.source_id,
                "source_type": self.provenance.source_type,
                "domain": self.provenance.domain,
                "language": self.provenance.language,
                "collection_timestamp": self.provenance.collection_timestamp,
                "parent_document": self.provenance.parent_document,
                "processing_version": self.provenance.processing_version,
            },
            "content_hash": self.content_hash,
            "split": self.split,
            "token_count": self.token_count,
            "quality_score": self.quality_score,
            "flags": dict(self.flags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        prov = data["provenance"]
        return cls(
            document_id=data["document_id"],
            text=data["text"],
            provenance=Provenance(**prov),
            content_hash=data["content_hash"],
            split=data.get("split"),
            token_count=data.get("token_count", 0),
            quality_score=data.get("quality_score", 1.0),
            flags=data.get("flags", {}),
        )

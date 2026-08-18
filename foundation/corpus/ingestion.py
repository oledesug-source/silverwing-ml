"""Raw source ingestion.

Walks configured raw-source paths, reads plain-text and JSONL inputs, and
produces normalized DocumentRecords with provenance attached. Each source
yields records tagged with source_id / source_type / domain / language.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .normalization import normalize_document_id
from .schema import DocumentRecord, Provenance, SourceType


@dataclass
class SourceConfig:
    source_id: str
    path: str
    source_type: str = SourceType.OTHER.value
    domain: str = "unknown"
    language: str = "unknown"
    kind: str = "text"  # text | jsonl | directory


def _iter_text(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    yield normalize_document_id(f"{source_id}-{path.stem}"), text


def _iter_jsonl(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text") or obj.get("content") or obj.get("source") or ""
        if not text:
            continue
        text = text if isinstance(text, str) else str(text)
        did = normalize_document_id(f"{source_id}-{obj.get('id', i)}")
        yield did, text


def _iter_directory(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    for file in sorted(path.rglob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        rel = file.relative_to(path)
        did = normalize_document_id(f"{source_id}-{rel}")
        yield did, text


_READERS = {"text": _iter_text, "jsonl": _iter_jsonl, "directory": _iter_directory}


class Ingestor:
    """Ingests configured raw sources into normalized DocumentRecords."""

    def __init__(self, sources: list[SourceConfig]) -> None:
        if not sources:
            raise ValueError("At least one source is required")
        self.sources = sources

    def ingest(self) -> Iterator[DocumentRecord]:
        for source in self.sources:
            path = Path(source.path)
            if not path.exists():
                raise FileNotFoundError(f"Source path not found: {source.path}")
            if source.kind not in _READERS:
                raise ValueError(f"Unsupported source kind: {source.kind}")
            base = Provenance(
                source_id=source.source_id,
                source_type=source.source_type,
                domain=source.domain,
                language=source.language,
            )
            for document_id, raw in _READERS[source.kind](path, source.source_id):
                if not raw.strip():
                    continue
                record = DocumentRecord.build(document_id=document_id, text=raw, provenance=base)
                yield record

"""Raw source ingestion.

Walks configured raw-source paths, reads plain-text, JSONL, Markdown, and
HuggingFace dataset inputs, and produces normalized DocumentRecords with
provenance attached. Each source yields records tagged with
source_id / source_type / domain / language.

Supported source kinds:
  text      -- single plain-text file
  jsonl     -- JSONL file (text from ``text`` / ``content`` / ``source`` field)
  markdown  -- single Markdown file (treated as plain text)
  directory -- recursively reads ``*.txt`` and ``*.md`` files
  huggingface -- streaming download from HuggingFace ``datasets`` library
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalization import normalize_document_id
from .schema import DocumentRecord, Provenance, SourceType

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    source_id: str
    path: str = ""
    source_type: str = SourceType.OTHER.value
    domain: str = "unknown"
    language: str = "unknown"
    kind: str = "text"  # text | jsonl | markdown | directory | huggingface
    # HuggingFace-specific fields
    hf_dataset: str = ""
    hf_split: str = "train"
    hf_text_column: str = "text"
    hf_name_column: str = ""
    hf_subset: str = ""
    hf_streaming: bool = True
    hf_max_samples: int = 0  # 0 = unlimited
    hf_seed: int = 42
    # Generic extras
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plain-text reader (streaming for large files)
# ---------------------------------------------------------------------------

def _iter_text(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    """Yield (document_id, text) from a single text file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    yield normalize_document_id(f"{source_id}-{path.stem}"), text


# ---------------------------------------------------------------------------
# JSONL reader (streaming line-by-line)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    """Yield (document_id, text) from a JSONL file, one record at a time."""
    idx = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("content") or obj.get("source") or ""
            if not text:
                continue
            text = text if isinstance(text, str) else str(text)
            did = normalize_document_id(f"{source_id}-{obj.get('id', idx)}")
            idx += 1
            yield did, text


# ---------------------------------------------------------------------------
# Markdown reader
# ---------------------------------------------------------------------------

def _iter_markdown(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    """Yield (document_id, text) from a Markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    yield normalize_document_id(f"{source_id}-{path.stem}"), text


# ---------------------------------------------------------------------------
# Directory reader (recursive, supports .txt and .md)
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".md"}


def _iter_directory(path: Path, source_id: str) -> Iterator[tuple[str, str]]:
    """Recursively yield (document_id, text) from .txt and .md files."""
    files = sorted(
        f for f in path.rglob("*") if f.suffix.lower() in _TEXT_EXTENSIONS and f.is_file()
    )
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        rel = file.relative_to(path)
        did = normalize_document_id(f"{source_id}-{rel}")
        yield did, text


# ---------------------------------------------------------------------------
# HuggingFace datasets reader (streaming)
# ---------------------------------------------------------------------------

def _iter_huggingface(source: SourceConfig) -> Iterator[tuple[str, str]]:
    """Stream documents from a HuggingFace dataset.

    Requires the ``datasets`` package.  Supports streaming mode to avoid
    downloading the full dataset into memory.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace ingestion. "
            "Install it with: pip install datasets"
        ) from None

    load_kwargs: dict[str, Any] = {
        "streaming": source.hf_streaming,
    }
    if source.hf_subset:
        load_kwargs["name"] = source.hf_subset

    logger.info(
        "Loading HuggingFace dataset %s (split=%s, streaming=%s)",
        source.hf_dataset, source.hf_split, source.hf_streaming,
    )
    ds = load_dataset(source.hf_dataset, split=source.hf_split, **load_kwargs)

    count = 0
    for idx, row in enumerate(ds):
        if source.hf_max_samples and idx >= source.hf_max_samples:
            logger.info("Reached hf_max_samples limit (%d)", source.hf_max_samples)
            break

        text = row.get(source.hf_text_column, "")
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        name = ""
        if source.hf_name_column and source.hf_name_column in row:
            name = str(row[source.hf_name_column])
        elif "id" in row:
            name = str(row["id"])
        else:
            name = str(idx)

        did = normalize_document_id(f"{source.source_id}-{name}")
        count += 1
        if count % 100_000 == 0:
            logger.info("Ingested %d documents from %s", count, source.hf_dataset)
        yield did, text

    logger.info("Finished HuggingFace dataset %s: %d documents", source.hf_dataset, count)


# ---------------------------------------------------------------------------
# Reader registry
# ---------------------------------------------------------------------------

_READERS = {
    "text": _iter_text,
    "jsonl": _iter_jsonl,
    "markdown": _iter_markdown,
    "directory": _iter_directory,
}


class Ingestor:
    """Ingests configured raw sources into normalized DocumentRecords."""

    def __init__(self, sources: list[SourceConfig]) -> None:
        if not sources:
            raise ValueError("At least one source is required")
        self.sources = sources

    def ingest(self) -> Iterator[DocumentRecord]:
        for source in self.sources:
            if source.kind == "huggingface":
                yield from self._ingest_hf(source)
            else:
                yield from self._ingest_file(source)

    def _ingest_file(self, source: SourceConfig) -> Iterator[DocumentRecord]:
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
            record = DocumentRecord.build(
                document_id=document_id, text=raw, provenance=base
            )
            yield record

    def _ingest_hf(self, source: SourceConfig) -> Iterator[DocumentRecord]:
        base = Provenance(
            source_id=source.source_id,
            source_type=source.source_type,
            domain=source.domain,
            language=source.language,
        )
        for document_id, raw in _iter_huggingface(source):
            if not raw.strip():
                continue
            record = DocumentRecord.build(
                document_id=document_id, text=raw, provenance=base
            )
            yield record

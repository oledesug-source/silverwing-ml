"""Corpus pipeline orchestrator.

Chain: ingest -> normalize -> filter -> deduplicate -> chunk -> contamination
check -> split -> shard. Emits a PipelineReport with per-stage counts so every
dataset release has a full audit trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .chunking import Chunker, estimate_tokens
from .contamination import ContaminationDetector
from .deduplication import Deduplicator
from .filtering import FilterChain, LanguageFilter, QualityFilter, detect_domain, detect_language
from .ingestion import Ingestor
from .schema import DocumentRecord, SplitOptions
from .split import Splitter
from .storage import ShardWriter


@dataclass
class PipelineReport:
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    stages: dict = field(default_factory=dict)
    manifest: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"started_at": self.started_at, "completed_at": self.completed_at, "stages": self.stages, "manifest": self.manifest}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class CorpusPipeline:
    def __init__(
        self,
        ingest: Ingestor,
        output_dir: str = "experiments/corpus",
        chunker: Chunker | None = None,
        split_options: SplitOptions | None = None,
        allowed_languages: tuple[str, ...] = ("en",),
        detector: ContaminationDetector | None = None,
        seed: int = 42,
        quality_config: dict | None = None,
        deduplication_config: dict | None = None,
    ) -> None:
        self.ingest = ingest
        self.output_dir = Path(output_dir)
        self.chunker = chunker or Chunker()
        self.split_options = split_options or SplitOptions()
        self.allowed_languages = allowed_languages
        self.detector = detector or ContaminationDetector()
        self.seed = seed
        self.quality_config = quality_config or {}
        self.deduplication_config = deduplication_config or {}
        self.report = PipelineReport()

    def run(self) -> PipelineReport:
        ingested = list(self.ingest.ingest())
        self.report.stages["ingested"] = len(ingested)

        quality_fields = {f.name for f in dc_fields(QualityFilter)}
        quality = QualityFilter(**{k: v for k, v in self.quality_config.items() if k in quality_fields})
        kept, dropped = FilterChain(
            [quality.keep, LanguageFilter(self.allowed_languages).keep]
        ).apply(self._annotate(ingested))
        self.report.stages["quality_and_language_filter"] = {"kept": len(kept), "dropped": dropped}
        dedup_fields = {f.name for f in dc_fields(Deduplicator)}
        dedup = Deduplicator(**{k: v for k, v in self.deduplication_config.items() if k in dedup_fields}, seed=self.seed)
        kept, dropped = dedup.apply(kept)
        self.report.stages["deduplication"] = {"kept": len(kept), "dropped": dropped}

        chunks = []
        for record in kept:
            chunks.extend(self.chunker.chunk(record))
        self.report.stages["chunking"] = {"documents": len(kept), "chunks": len(chunks)}

        chunks, contaminated, flagged = self.detector.filter(chunks)
        self.report.stages["contamination"] = {"kept": len(chunks), "dropped": contaminated, "flagged": [r.document_id for r in flagged]}

        splitter = Splitter(self.split_options)
        split_records = splitter.split(chunks)

        writer = ShardWriter(self.output_dir)
        manifest = writer.write(split_records)
        self.report.manifest = manifest
        self.report.completed_at = datetime.now(timezone.utc).isoformat()
        return self.report

    @staticmethod
    def _annotate(records: Iterable[DocumentRecord]) -> list[DocumentRecord]:
        annotated = []
        for record in records:
            record.provenance.language = detect_language(record.text, record.provenance.language)
            record.provenance.domain = detect_domain(record.text, record.provenance.domain)
            record.token_count = estimate_tokens(record.text)
            annotated.append(record)
        return annotated

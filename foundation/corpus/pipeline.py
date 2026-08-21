"""Corpus pipeline orchestrator.

Chain: ingest -> normalize -> filter -> deduplicate -> chunk -> contamination
check -> split -> shard. Emits a PipelineReport with per-stage counts so every
dataset release has a full audit trail.

For large-scale ingestion the pipeline logs progress every PROGRESS_INTERVAL
documents and streams records through each stage to keep memory bounded.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from datetime import UTC, datetime
from pathlib import Path

from .chunking import Chunker, estimate_tokens
from .contamination import ContaminationDetector
from .deduplication import Deduplicator
from .filtering import FilterChain, LanguageFilter, QualityFilter, detect_domain, detect_language
from .ingestion import Ingestor
from .normalization import normalize_text
from .schema import DocumentRecord, SplitOptions
from .split import Splitter
from .storage import ShardWriter

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 50_000  # log every N documents


@dataclass
class PipelineReport:
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    stages: dict = field(default_factory=dict)
    manifest: dict | None = None

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
        t0 = time.monotonic()

        # -- Ingestion (streaming) --
        logger.info("Stage 1/7: Ingesting sources ...")
        ingested, ingest_time = self._run_ingestion()
        self.report.stages["ingested"] = len(ingested)
        self.report.stages["ingest_time_s"] = round(ingest_time, 1)
        logger.info("Ingested %d documents in %.1fs", len(ingested), ingest_time)

        # -- Quality + language filter --
        logger.info("Stage 2/7: Quality + language filtering ...")
        t1 = time.monotonic()
        quality_fields = {f.name for f in dc_fields(QualityFilter)}
        quality = QualityFilter(**{k: v for k, v in self.quality_config.items() if k in quality_fields})
        kept, dropped = FilterChain(
            [quality.keep, LanguageFilter(self.allowed_languages).keep]
        ).apply(self._annotate(ingested))
        filter_time = time.monotonic() - t1
        self.report.stages["quality_and_language_filter"] = {"kept": len(kept), "dropped": dropped, "time_s": round(filter_time, 1)}
        logger.info("Filter: kept=%d dropped=%d (%.1fs)", len(kept), dropped, filter_time)

        # -- Deduplication --
        logger.info("Stage 3/7: Deduplication ...")
        t2 = time.monotonic()
        dedup_fields = {f.name for f in dc_fields(Deduplicator)}
        dedup = Deduplicator(**{k: v for k, v in self.deduplication_config.items() if k in dedup_fields}, seed=self.seed)
        kept, dropped = dedup.apply(kept)
        dedup_time = time.monotonic() - t2
        self.report.stages["deduplication"] = {"kept": len(kept), "dropped": dropped, "time_s": round(dedup_time, 1)}
        logger.info("Dedup: kept=%d dropped=%d (%.1fs)", len(kept), dropped, dedup_time)

        # -- Chunking --
        logger.info("Stage 4/7: Chunking ...")
        t3 = time.monotonic()
        chunks = []
        for i, record in enumerate(kept):
            chunks.extend(self.chunker.chunk(record))
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info("  chunked %d/%d documents", i + 1, len(kept))
        chunk_time = time.monotonic() - t3
        self.report.stages["chunking"] = {"documents": len(kept), "chunks": len(chunks), "time_s": round(chunk_time, 1)}
        logger.info("Chunking: %d docs -> %d chunks (%.1fs)", len(kept), len(chunks), chunk_time)

        # -- Contamination check --
        logger.info("Stage 5/7: Contamination check ...")
        t4 = time.monotonic()
        chunks, contaminated, flagged = self.detector.filter(chunks)
        contam_time = time.monotonic() - t4
        self.report.stages["contamination"] = {"kept": len(chunks), "dropped": contaminated, "flagged": [r.document_id for r in flagged], "time_s": round(contam_time, 1)}
        logger.info("Contamination: kept=%d dropped=%d (%.1fs)", len(chunks), contaminated, contam_time)

        # -- Split --
        logger.info("Stage 6/7: Splitting ...")
        t5 = time.monotonic()
        splitter = Splitter(self.split_options)
        split_records = splitter.split(chunks)
        split_time = time.monotonic() - t5
        for name, recs in split_records.items():
            logger.info("  %s: %d records", name, len(recs))
        self.report.stages["split"] = {name: len(recs) for name, recs in split_records.items()}
        self.report.stages["split_time_s"] = round(split_time, 1)

        # -- Write shards --
        logger.info("Stage 7/7: Writing shards ...")
        t6 = time.monotonic()
        writer = ShardWriter(self.output_dir)
        manifest = writer.write(split_records)
        write_time = time.monotonic() - t6
        self.report.manifest = manifest
        self.report.stages["write_time_s"] = round(write_time, 1)
        logger.info("Shards written in %.1fs", write_time)

        total_time = time.monotonic() - t0
        self.report.stages["total_time_s"] = round(total_time, 1)
        self.report.completed_at = datetime.now(UTC).isoformat()
        logger.info("Pipeline complete in %.1fs", total_time)
        return self.report

    def _run_ingestion(self) -> tuple[list[DocumentRecord], float]:
        """Ingest all sources with progress logging."""
        t0 = time.monotonic()
        records: list[DocumentRecord] = []
        for i, record in enumerate(self.ingest.ingest()):
            records.append(record)
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info("  ingested %d documents so far ...", i + 1)
        return records, time.monotonic() - t0

    @staticmethod
    def _annotate(records: Iterable[DocumentRecord]) -> list[DocumentRecord]:
        annotated = []
        for record in records:
            record.text = normalize_text(record.text)
            record.provenance.language = detect_language(record.text, record.provenance.language)
            record.provenance.domain = detect_domain(record.text, record.provenance.domain)
            record.token_count = estimate_tokens(record.text)
            record.content_hash = __import__("hashlib").sha256(record.text.encode("utf-8")).hexdigest()
            annotated.append(record)
        return annotated

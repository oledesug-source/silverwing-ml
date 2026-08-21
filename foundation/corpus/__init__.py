"""Silverwing corpus platform.

Raw sources -> ingestion -> normalization -> quality/language/domain filtering
-> deduplication -> chunking -> contamination check -> train/val/test split
-> sharded dataset, with full provenance recorded per document.
"""

from .config import build_pipeline_from_config, load_corpus_config, pipeline_config_digest
from .filtering import PerplexityFilter, QualityFilter
from .ingestion import SourceConfig
from .pipeline import CorpusPipeline, PipelineReport
from .schema import DocumentRecord, Provenance, Split, SplitOptions
from .verify import VerificationResult, verify_dataset

__all__ = [
    "DocumentRecord",
    "Provenance",
    "Split",
    "SplitOptions",
    "CorpusPipeline",
    "PipelineReport",
    "VerificationResult",
    "verify_dataset",
    "load_corpus_config",
    "build_pipeline_from_config",
    "pipeline_config_digest",
    "SourceConfig",
    "QualityFilter",
    "PerplexityFilter",
]

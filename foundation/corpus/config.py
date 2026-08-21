"""Corpus configuration loading.

Loads configs/corpus.yaml (and overrides) and builds a CorpusPipeline from it,
so the YAML is the single source of truth for pipeline parameters (M01 rule:
every run reproducible from committed config + commit hash).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .chunking import Chunker
from .contamination import ContaminationDetector
from .ingestion import Ingestor, SourceConfig
from .pipeline import CorpusPipeline
from .schema import SplitOptions

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "corpus.yaml"


def load_corpus_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Corpus config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    corpus = data.get("corpus", {})
    for key in ("splits",):
        if key in corpus:
            overrides = {k: v for k, v in corpus[key].items() if v is not None}
            corpus[key] = overrides
    return corpus


def source_configs_from_env(config: dict) -> list[SourceConfig]:
    """Build source configs from the YAML `sources` list and env overrides.

    Env var CORPUS_SOURCE_<N> may point to a path for the Nth declared source.
    """
    raw_sources = config.get("sources") or []
    sources = []
    for i, source in enumerate(raw_sources):
        kind = source.get("kind", "text")

        if kind == "huggingface":
            hf_dataset = source.get("hf_dataset", "")
            if not hf_dataset:
                raise ValueError(f"Source {i} ({source.get('source_id')}) has no hf_dataset")
            sources.append(
                SourceConfig(
                    source_id=source["source_id"],
                    path="",
                    source_type=source.get("source_type", "web"),
                    domain=source.get("domain", "web"),
                    language=source.get("language", "en"),
                    kind="huggingface",
                    hf_dataset=hf_dataset,
                    hf_split=source.get("hf_split", "train"),
                    hf_text_column=source.get("hf_text_column", "text"),
                    hf_name_column=source.get("hf_name_column", ""),
                    hf_subset=source.get("hf_subset", ""),
                    hf_streaming=source.get("hf_streaming", True),
                    hf_max_samples=source.get("hf_max_samples", 0),
                    hf_seed=source.get("hf_seed", 42),
                )
            )
        else:
            path = os.environ.get(f"CORPUS_SOURCE_{i}", source.get("path"))
            if not path:
                raise ValueError(f"Source {i} ({source.get('source_id')}) has no path; set CORPUS_SOURCE_{i}")
            sources.append(
                SourceConfig(
                    source_id=source["source_id"],
                    path=str(path),
                    source_type=source.get("source_type", "other"),
                    domain=source.get("domain", "unknown"),
                    language=source.get("language", "unknown"),
                    kind=kind,
                )
            )
    return sources


def build_pipeline_from_config(
    config: dict,
    sources: list[SourceConfig] | None = None,
    output_dir: str | None = None,
) -> CorpusPipeline:
    chunk_cfg = config.get("chunking", {})
    split_cfg = config.get("splits", {})
    filt_cfg = config.get("filtering", {})
    dedup_cfg = config.get("deduplication", {})
    contam_cfg = config.get("contamination", {})

    chunker = Chunker(
        max_tokens=int(chunk_cfg.get("max_tokens", 1024)),
        overlap_tokens=int(chunk_cfg.get("overlap_tokens", 128)),
    )
    split_options = SplitOptions(
        train=float(split_cfg.get("train", 0.96)),
        validation=float(split_cfg.get("validation", 0.02)),
        test=float(split_cfg.get("test", 0.02)),
    )
    allowed_languages = tuple(filt_cfg.get("allowed_languages", ["en"]))
    detector = ContaminationDetector(
        n=int(contam_cfg.get("ngram_n", 8)),
        threshold=float(contam_cfg.get("threshold", 0.6)),
    )

    pipeline = CorpusPipeline(
        ingest=Ingestor(sources or source_configs_from_env(config)),
        output_dir=output_dir or config.get("output_dir", "experiments/corpus"),
        chunker=chunker,
        split_options=split_options,
        allowed_languages=allowed_languages,
        detector=detector,
        seed=int(config.get("seed", 42)),
        quality_config={k: v for k, v in filt_cfg.get("quality", {}).items() if v is not None},
        deduplication_config={k: v for k, v in dedup_cfg.items() if v is not None},
    )
    return pipeline


def pipeline_config_digest(config: dict) -> str:
    """Deterministic digest of the pipeline-relevant config subset."""
    import hashlib

    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

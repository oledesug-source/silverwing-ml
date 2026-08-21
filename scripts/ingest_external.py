"""Ingest external datasets from HuggingFace into the Silverwing corpus pipeline.

Supports two modes:

1. Preset mode (recommended):
   python scripts/ingest_external.py --preset openwebtext
   python scripts/ingest_external.py --preset all

2. Direct mode (for custom datasets):
   python scripts/ingest_external.py \
       --hf-dataset Skylion007/openwebtext \
       --hf-split train \
       --hf-text-column text \
       --hf-max-samples 100000 \
       --source-id openwebtext-100k \
       --output-dir experiments/corpus-openwebtext

The pipeline downloads data in streaming mode (no full dataset in memory),
applies quality/language/dedup filtering, chunks, and writes sharded JSONL
with full provenance tracking.

Requirements: pip install datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.corpus import build_pipeline_from_config, load_corpus_config, pipeline_config_digest
from foundation.corpus.ingestion import SourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_external")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _load_preset(config_path: str, preset_name: str) -> list[dict]:
    """Load a named preset from the external sources config."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("external_sources", {}).get("presets", {})
    if preset_name not in presets:
        available_names = list(presets.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available: {available_names}")
    return presets[preset_name]["sources"]


def _source_config_from_dict(d: dict) -> SourceConfig:
    return SourceConfig(
        source_id=d["source_id"],
        path=d.get("path", ""),
        source_type=d.get("source_type", "other"),
        domain=d.get("domain", "unknown"),
        language=d.get("language", "en"),
        kind=d.get("kind", "huggingface"),
        hf_dataset=d.get("hf_dataset", ""),
        hf_split=d.get("hf_split", "train"),
        hf_text_column=d.get("hf_text_column", "text"),
        hf_name_column=d.get("hf_name_column", ""),
        hf_subset=d.get("hf_subset", ""),
        hf_streaming=d.get("hf_streaming", True),
        hf_max_samples=d.get("hf_max_samples", 0),
        hf_seed=d.get("hf_seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest external HuggingFace datasets into the Silverwing corpus"
    )
    parser.add_argument(
        "--config", default="configs/external_sources.yaml",
        help="external sources config (default: configs/external_sources.yaml)",
    )
    parser.add_argument(
        "--preset",
        help="named preset from config (e.g. openwebtext, wikipedia, arxiv, math, all)",
    )
    parser.add_argument("--hf-dataset", help="HuggingFace dataset ID (direct mode)")
    parser.add_argument("--hf-split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--hf-text-column", default="text", help="Text column name (default: text)")
    parser.add_argument("--hf-subset", default="", help="Dataset subset/config name")
    parser.add_argument("--hf-max-samples", type=int, default=0, help="Max samples (0=unlimited)")
    parser.add_argument("--hf-streaming", action="store_true", default=True, help="Stream mode (default)")
    parser.add_argument("--source-id", default="external", help="Source ID tag")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty corpus release")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build source list
    if args.preset:
        preset_sources = _load_preset(args.config, args.preset)
        sources = [_source_config_from_dict(s) for s in preset_sources]
        logger.info("Loaded preset '%s': %d source(s)", args.preset, len(sources))
    elif args.hf_dataset:
        sources = [
            SourceConfig(
                source_id=args.source_id,
                kind="huggingface",
                hf_dataset=args.hf_dataset,
                hf_split=args.hf_split,
                hf_text_column=args.hf_text_column,
                hf_subset=args.hf_subset,
                hf_streaming=args.hf_streaming,
                hf_max_samples=args.hf_max_samples,
                source_type="web",
                domain="web",
                language="en",
            )
        ]
        logger.info("Direct mode: dataset=%s, split=%s, max_samples=%d",
                     args.hf_dataset, args.hf_split, args.hf_max_samples)
    else:
        raise SystemExit("Provide --preset or --hf-dataset")

    # Load pipeline config from external_sources.yaml
    corpus_config = load_corpus_config(args.config)
    output_dir = args.output_dir or corpus_config.get("output_dir", "experiments/corpus-external")

    # Build and run pipeline
    pipeline = build_pipeline_from_config(corpus_config, sources=sources, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("Starting corpus pipeline: %d source(s) -> %s", len(sources), output_dir)
    logger.info("=" * 60)

    t0 = time.monotonic()
    report = pipeline.run()
    elapsed = time.monotonic() - t0

    # Write pipeline report
    payload = {
        "config_path": args.config,
        "config_digest": pipeline_config_digest(corpus_config),
        "git_commit": _git_commit(),
        "output_dir": output_dir,
        "preset": args.preset or "direct",
        "sources": [
            {
                "source_id": s.source_id,
                "hf_dataset": s.hf_dataset,
                "hf_split": s.hf_split,
                "hf_max_samples": s.hf_max_samples,
            }
            for s in sources
        ],
        "report": report.to_dict(),
    }
    report_path = Path(output_dir) / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    manifest = report.manifest or {}
    splits = manifest.get("splits", {})
    total_records = sum(int(info.get("records", 0)) for info in splits.values())
    total_tokens = sum(int(info.get("tokens", 0)) for info in splits.values())

    logger.info("=" * 60)
    logger.info("CORPUS BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("Dataset hash : %s", manifest.get("dataset_hash"))
    logger.info("Total records: %d", total_records)
    logger.info("Total tokens : ~%s", f"{total_tokens:,}")
    for name, info in sorted(splits.items()):
        logger.info("  %s: %d records, %s tokens", name, info.get("records", 0), f"{info.get('tokens', 0):,}")
    logger.info("Elapsed      : %.1fs", elapsed)
    logger.info("Report       : %s", report_path)
    logger.info("Verify       : python scripts/verify_corpus.py --output-dir %s", output_dir)


if __name__ == "__main__":
    main()

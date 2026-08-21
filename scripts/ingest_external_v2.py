"""Two-phase corpus build: ingest external data to disk, then process.

Phase 1 (ingest): Stream from HuggingFace datasets to a local JSONL file.
Phase 2 (process): Run the JSONL through the full corpus pipeline.

This separation means:
- Ingestion is fast (just streaming + writing JSONL)
- Processing can be re-run with different filter settings without re-downloading
- Large datasets don't need to fit entirely in memory during processing

Usage:
    # Phase 1: Download
    python scripts/ingest_external_v2.py download --preset quickstart

    # Phase 2: Process
    python scripts/ingest_external_v2.py process --input experiments/ingested/quickstart.jsonl

    # Or combined (default):
    python scripts/ingest_external_v2.py --preset quickstart
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
logger = logging.getLogger("ingest_v2")

INGEST_DIR = Path("experiments/ingested")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _load_preset(config_path: str, preset_name: str) -> list[dict]:
    import yaml
    path = Path(config_path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("external_sources", {}).get("presets", {})
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list(presets.keys())}")
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


def cmd_download(args: argparse.Namespace) -> Path:
    """Phase 1: Stream from HuggingFace to local JSONL."""
    from foundation.corpus.ingestion import _iter_huggingface

    if args.preset:
        preset_sources = _load_preset(args.config, args.preset)
        sources = [_source_config_from_dict(s) for s in preset_sources]
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
    else:
        raise SystemExit("Provide --preset or --hf-dataset")

    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    name = args.preset or args.source_id
    out_path = INGEST_DIR / f"{name}.jsonl"

    logger.info("Phase 1: Downloading %d source(s) to %s", len(sources), out_path)

    t0 = time.monotonic()
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for source in sources:
            logger.info("  Streaming from %s (max_samples=%d)", source.hf_dataset, source.hf_max_samples or -1)
            for doc_id, text in _iter_huggingface(source):
                record = {"id": doc_id, "text": text, "source_id": source.source_id, "source_type": source.source_type, "domain": source.domain, "language": source.language}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    logger.info("    ... %d documents downloaded", count)

    elapsed = time.monotonic() - t0
    logger.info("Phase 1 complete: %d documents -> %s (%.1fs)", count, out_path, elapsed)
    return out_path


def cmd_process(args: argparse.Namespace) -> dict:
    """Phase 2: Process a JSONL file through the full corpus pipeline."""
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    corpus_config = load_corpus_config(args.config)
    output_dir = args.output_dir or corpus_config.get("output_dir", f"experiments/corpus-{input_path.stem}")

    source = SourceConfig(
        source_id=input_path.stem,
        path=str(input_path),
        kind="jsonl",
        source_type="web",
        domain="web",
        language="en",
    )

    pipeline = build_pipeline_from_config(corpus_config, sources=[source], output_dir=output_dir)

    logger.info("Phase 2: Processing %s -> %s", input_path, output_dir)
    t0 = time.monotonic()
    report = pipeline.run()
    elapsed = time.monotonic() - t0

    payload = {
        "config_path": args.config,
        "config_digest": pipeline_config_digest(corpus_config),
        "git_commit": _git_commit(),
        "input_file": str(input_path),
        "output_dir": output_dir,
        "report": report.to_dict(),
    }
    report_path = Path(output_dir) / "pipeline_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase corpus builder (download + process)")
    sub = parser.add_subparsers(dest="command")

    # Shared args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/external_sources.yaml")
    common.add_argument("--preset", help="Named preset from config")
    common.add_argument("--hf-dataset", help="HuggingFace dataset ID (direct mode)")
    common.add_argument("--hf-split", default="train")
    common.add_argument("--hf-text-column", default="text")
    common.add_argument("--hf-subset", default="")
    common.add_argument("--hf-max-samples", type=int, default=0)
    common.add_argument("--hf-streaming", action="store_true", default=True)
    common.add_argument("--source-id", default="external")
    common.add_argument("-v", "--verbose", action="store_true")

    # Download subcommand
    dl = sub.add_parser("download", parents=[common], help="Phase 1: Download from HuggingFace to JSONL")
    dl.add_argument("--output", default=None, help="Output JSONL path")

    # Process subcommand
    pr = sub.add_parser("process", parents=[common], help="Phase 2: Process JSONL through pipeline")
    pr.add_argument("--input", required=True, help="Input JSONL file from download phase")
    pr.add_argument("--output-dir", default=None, help="Output directory")

    # Combined (default)
    parser.add_argument("--config", default="configs/external_sources.yaml")
    parser.add_argument("--preset", help="Named preset from config")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset ID")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-column", default="text")
    parser.add_argument("--hf-subset", default="")
    parser.add_argument("--hf-max-samples", type=int, default=0)
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--source-id", default="external")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "process":
        cmd_process(args)
    else:
        # Combined: download then process
        if not args.preset and not args.hf_dataset:
            raise SystemExit("Provide --preset or --hf-dataset (or use download/process subcommands)")
        jsonl_path = cmd_download(args)
        args.input = str(jsonl_path)
        cmd_process(args)


if __name__ == "__main__":
    main()

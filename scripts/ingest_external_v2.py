"""Two-phase corpus build: ingest external data to disk, then process.

Phase 1 (ingest): Stream from HuggingFace datasets to per-source JSONL files.
Phase 2 (process): Run the JSONL files through the full corpus pipeline.

This separation means:
- Ingestion is fast (just streaming + writing JSONL)
- Processing can be re-run with different filter settings without re-downloading
- Large datasets don't need to fit entirely in memory during processing
- Downloads are resumable: each source writes to ``<source>.jsonl.part`` and
  appends on re-run (``--skip-existing`` keeps finished sources), so a network
  drop costs minutes, not hours.

Usage:
    # Phase 1: Download
    python scripts/ingest_external_v2.py download --preset quickstart

    # Phase 2: Process (input may be a JSONL file or a directory of them)
    python scripts/ingest_external_v2.py process --input experiments/ingested/quickstart

    # Or combined (default):
    python scripts/ingest_external_v2.py --preset quickstart
"""

from __future__ import annotations

import argparse
import itertools
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


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _download_source(source: SourceConfig, out_path: Path, skip_existing: bool) -> tuple[Path, int]:
    """Download one source to ``out_path`` with .part append-resume.

    Finished files (non-empty) are kept when ``skip_existing`` is set.  A
    partial ``.part`` file is counted and the stream fast-forwards past the
    records already saved, so re-runs only fetch the remainder.
    """
    from foundation.corpus.ingestion import _iter_huggingface

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        done = _count_lines(out_path)
        logger.info("  %s: already complete (%d documents), skipping", source.source_id, done)
        return out_path, done

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    resume_from = 0
    if part_path.exists():
        resume_from = _count_lines(part_path)
        if resume_from:
            logger.info("  %s: resuming from document %d", source.source_id, resume_from)

    logger.info("  Streaming from %s (max_samples=%d)", source.hf_dataset, source.hf_max_samples or -1)
    t0 = time.monotonic()
    written = 0
    mode = "a" if resume_from else "w"
    with part_path.open(mode, encoding="utf-8") as fh:
        stream = itertools.islice(_iter_huggingface(source), resume_from, None)
        for doc_id, text in stream:
            record = {
                "id": doc_id,
                "text": text,
                "source_id": source.source_id,
                "source_type": source.source_type,
                "domain": source.domain,
                "language": source.language,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            total = resume_from + written
            if total % 10000 == 0:
                rate = written / max(time.monotonic() - t0, 1e-6)
                logger.info("    ... %d documents (%.0f docs/s)", total, rate)

    part_path.replace(out_path)
    elapsed = time.monotonic() - t0
    total = resume_from + written
    logger.info("  %s: %d documents -> %s (%.1fs)", source.source_id, total, out_path.name, elapsed)
    return out_path, total


def cmd_download(args: argparse.Namespace) -> list[Path]:
    """Phase 1: Stream from HuggingFace to per-source JSONL files."""
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

    name = args.preset or args.source_id
    if getattr(args, "output", None):
        out_dir = Path(args.output).parent
        single = True
    else:
        out_dir = INGEST_DIR / name
        single = False
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: Downloading %d source(s) to %s", len(sources), out_dir)

    t0 = time.monotonic()
    out_paths: list[Path] = []
    for source in sources:
        if single and len(sources) == 1:
            out_path = Path(args.output)
        else:
            out_path = out_dir / f"{source.source_id}.jsonl"
        _, _ = _download_source(source, out_path, args.skip_existing)
        out_paths.append(out_path)

    total_docs = sum(_count_lines(p) for p in out_paths)
    logger.info(
        "Phase 1 complete: %d documents across %d file(s) in %s (%.1fs)",
        total_docs, len(out_paths), out_dir, time.monotonic() - t0,
    )
    return out_paths


def cmd_process(args: argparse.Namespace) -> dict:
    """Phase 2: Process JSONL file(s) through the full corpus pipeline."""
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {input_path}")
    elif input_path.exists():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    corpus_config = load_corpus_config(args.config)
    default_name = input_path.stem if input_path.is_file() else input_path.name
    output_dir = args.output_dir or corpus_config.get("output_dir", f"experiments/corpus-{default_name}")

    sources = [
        SourceConfig(
            source_id=f.stem,
            path=str(f),
            kind="jsonl",
            source_type="web",
            domain="web",
            language="en",
        )
        for f in files
    ]

    pipeline = build_pipeline_from_config(corpus_config, sources=sources, output_dir=output_dir)

    logger.info("Phase 2: Processing %d file(s) from %s -> %s", len(files), input_path, output_dir)
    t0 = time.monotonic()
    report = pipeline.run()
    elapsed = time.monotonic() - t0

    payload = {
        "config_path": args.config,
        "config_digest": pipeline_config_digest(corpus_config),
        "git_commit": _git_commit(),
        "input_files": [str(f) for f in files],
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
    dl.add_argument("--skip-existing", action="store_true", help="Skip download if output JSONL already has content")

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
    parser.add_argument("--skip-existing", action="store_true", help="Skip download if output JSONL already has content")
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
        out_paths = cmd_download(args)
        args.input = str(out_paths[0].parent)
        cmd_process(args)


if __name__ == "__main__":
    main()

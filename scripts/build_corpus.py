"""Build the sharded Silverwing corpus from raw sources (M02/M03).

Operational entry point for the corpus pipeline. Sources come from the
config's `sources` list (with CORPUS_SOURCE_<N> env path overrides) or are
provided explicitly via repeated --source <id>=<path> flags, which fully
replace the configured source list. A release is refused when the pipeline
yields zero records (safe non-empty-output guard) unless --allow-empty is
given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.corpus import build_pipeline_from_config, load_corpus_config, pipeline_config_digest
from foundation.corpus.config import source_configs_from_env
from foundation.corpus.ingestion import SourceConfig


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_source(arg: str) -> SourceConfig:
    """Parse an explicit --source <id>=<path> override; kind is auto-detected."""
    if "=" not in arg:
        raise ValueError("--source must be <id>=<path>")
    source_id, path = arg.split("=", 1)
    source_id = source_id.strip()
    if not source_id:
        raise ValueError("source id must be non-empty")
    candidate = Path(path)
    if candidate.is_dir():
        kind = "directory"
    elif candidate.suffix.lower() == ".jsonl":
        kind = "jsonl"
    else:
        kind = "text"
    return SourceConfig(source_id=source_id, path=str(candidate), kind=kind)


def _resolve_sources(config: dict, explicit: list[str] | None) -> list[SourceConfig]:
    if explicit:
        return [_parse_source(arg) for arg in explicit]
    if config.get("sources"):
        return source_configs_from_env(config)
    raise SystemExit(
        "no sources: pass --source <id>=<path> (repeatable) or declare `sources` in the corpus config"
    )


def _non_empty_guard(manifest: dict) -> int:
    total = sum(int(info.get("records", 0)) for info in manifest.get("splits", {}).values())
    if total <= 0:
        raise SystemExit(
            "refusing to release an empty corpus: 0 records survived the pipeline; "
            "check your --source paths, kinds and filtering config"
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sharded Silverwing corpus")
    parser.add_argument("--config", default="configs/corpus.yaml", help="corpus config (source of truth)")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="ID=PATH",
        help="explicit source override (repeatable; kind auto-detected from path)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow releasing an empty corpus (default: refused)",
    )
    args = parser.parse_args()

    config = load_corpus_config(args.config)
    sources = _resolve_sources(config, args.source)
    output_dir = args.output_dir or config.get("output_dir", "experiments/corpus")

    pipeline = build_pipeline_from_config(config, sources=sources, output_dir=output_dir)
    report = pipeline.run()

    payload = {
        "config_path": args.config,
        "config_digest": pipeline_config_digest(config),
        "git_commit": _git_commit(),
        "output_dir": output_dir,
        "sources": [s.__dict__ for s in sources],
        "report": report.to_dict(),
    }
    report_path = Path(output_dir) / "pipeline_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = report.manifest or {}
    total = _non_empty_guard(manifest) if not args.allow_empty else sum(
        int(info.get("records", 0)) for info in manifest.get("splits", {}).values()
    )

    splits = manifest.get("splits", {})
    print(f"corpus built: dataset_hash={manifest.get('dataset_hash')}")
    print(f"records: total={total} " + " ".join(f"{name}={info.get('records', 0)}" for name, info in sorted(splits.items())))
    print(f"pipeline_report: {report_path}")


if __name__ == "__main__":
    main()

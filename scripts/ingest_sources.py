#!/usr/bin/env python3
"""Ingest real, free HuggingFace datasets for the Silverwing math/language corpus.

Pulls small, high-quality slices (free, no account required) in streaming mode:
  - openwebtext     (general web text)
  - wikipedia       (encyclopedia)
  - cc100 / arxiv   (technical text; cc100 en)
  - gsm8k           (grade-school math, with solutions)
  - openmath-instruct (math problem/solution pairs)
  - code_search     (Stack-V1 / bigcode subset: programming code)

Writes sharded, deduplicated JSONL to ``datasets/raw/<source>/``.

Usage::

    python -m scripts.ingest_sources --preset openwebtext --max-samples 5000
    python -m scripts.ingest_sources --preset math --max-samples 2000
    python -m scripts.ingest_sources --preset all --max-samples 2000   # ~12k docs

Each output file is content-hashed (SHA-256 of the JSON line) and deduplicated
within the run so the downstream tokenizer/model pipeline sees no duplicate rows.

Requires: pip install datasets  (already a project dependency)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ingest_sources")

DEFAULT_PRESETS = {
    "openwebtext": {
        "hf_dataset": "Skylion007/openwebtext",
        "hf_split": "train",
        "hf_text_column": "text",
        "hf_streaming": True,
        "text_key": "text",
    },
    "wikipedia": {
        "hf_dataset": "wikipedia",
        "hf_split": "train",
        "hf_text_column": "text",
        "hf_subset": "20220301.en",
        "hf_streaming": True,
        "text_key": "text",
    },
    "arxiv": {
        "hf_dataset": "cc100",
        "hf_split": "train",
        "hf_text_column": "text",
        "hf_subset": "en",
        "hf_streaming": True,
        "text_key": "text",
        "filter_domain": True,
    },
    "gsm8k": {
        "hf_dataset": "gsm8k",
        "hf_split": "train",
        "hf_text_column": "question",
        "hf_subset": "main",
        "hf_streaming": True,
        "text_key": "question",
        "solution_key": "answer",
    },
    "openmath": {
        "hf_dataset": "TIGER-Lab/OpenMathInstruct-2",
        "hf_split": "train",
        "hf_text_column": "problem",
        "hf_streaming": True,
        "text_key": "problem",
        "solution_key": "solution",
    },
    "stack": {
        "hf_dataset": "bigcode/the-stack",
        "hf_split": "train",
        "hf_text_column": "content",
        "hf_subset": "Python",
        "hf_streaming": True,
        "text_key": "content",
        "filter_language": "en",
    },
}

PRESET_GROUPS = {
    "quickstart": ["openwebtext"],
    "math": ["gsm8k", "openmath"],
    "all": ["openwebtext", "wikipedia", "arxiv", "gsm8k", "openmath", "stack"],
}

MIN_CHARS = int(os.environ.get("WING_MIN_CHARS", "120"))
SHARD_SIZE = int(os.environ.get("WING_SHARD_SIZE", "2000"))


def _hash_line(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _record_text(row: dict, text_key: str, solution_key: str | None) -> str:
    return row.get(text_key, "")


def _record_solution(row: dict, solution_key: str | None) -> str | None:
    if solution_key and solution_key in row:
        sol = row[solution_key]
        if isinstance(sol, dict) and "content" in sol:
            return sol["content"]
        return str(sol)
    return None


def _load_config_presets(path: str | Path = "configs/external_sources.yaml") -> dict:
    """Merge built-in presets with any configured presets in the YAML."""
    presets = dict(DEFAULT_PRESETS)
    try:
        import yaml

        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for name, block in (raw.get("external_sources", {}).get("presets", {}) or {}).items():
                sources = block.get("sources", [])
                if sources and name not in presets:
                    s = sources[0]
                    presets[name] = {
                        "hf_dataset": s.get("hf_dataset"),
                        "hf_split": s.get("hf_split", "train"),
                        "hf_text_column": s.get("hf_text_column", "text"),
                        "hf_subset": s.get("hf_subset"),
                        "hf_streaming": s.get("hf_streaming", True),
                        "text_key": s.get("hf_text_column", "text"),
                    }
    except Exception as exc:
        log.warning("could not merge yaml presets: %s", exc)
    return presets


def ingest(
    source_name: str,
    hf_dataset: str,
    hf_split: str = "train",
    hf_text_column: str = "text",
    hf_subset: str | None = None,
    hf_streaming: bool = True,
    text_key: str | None = None,
    solution_key: str | None = None,
    max_samples: int = 10000,
    output_dir: str = "datasets/raw",
    min_chars: int = MIN_CHARS,
    shard_size: int = SHARD_SIZE,
) -> Path:
    from datasets import load_dataset

    kwargs: dict = {"split": hf_split, "streaming": hf_streaming}
    if hf_subset:
        kwargs["name"] = hf_subset
    if not hf_streaming:
        kwargs["max_samples"] = max_samples

    log.info("loading %s (streaming=%s, subset=%s)", hf_dataset, hf_streaming, hf_subset)
    ds = load_dataset(hf_dataset, **kwargs)

    if not text_key:
        text_key = hf_text_column

    out_root = Path(output_dir) / source_name
    out_root.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    count = 0
    shard_idx = 0
    shard_file = None
    shard_handle = None

    def _next_shard():
        nonlocal shard_idx, shard_file, shard_handle
        if shard_handle:
            shard_handle.close()
        shard_idx += 1
        shard_file = out_root / f"{source_name}_{shard_idx:04d}.jsonl"
        shard_handle = shard_file.open("w", encoding="utf-8")
        return shard_handle

    h = _next_shard()
    for row in ds:
        if count >= max_samples:
            break
        text = _record_text(row, text_key, solution_key)
        if not text or len(text) < min_chars:
            continue
        if hf_dataset == "cc100" and "en" not in (hf_subset or "").lower():
            continue
        record = {"source": source_name, "id": _hash_line({"text": text}), "text": text}
        if solution_key:
            sol = _record_solution(row, solution_key)
            if sol:
                record["solution"] = sol
        line_hash = _hash_line(record)
        if line_hash in seen:
            continue
        seen.add(line_hash)
        record["content_hash"] = line_hash
        h.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % shard_size == 0:
            h = _next_shard()

    if shard_handle:
        shard_handle.close()
    report = {"source": source_name, "dataset": hf_dataset, "rows": count, "shards": shard_idx, "out": str(out_root)}
    (out_root / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("ingested %d rows into %s", count, out_root)
    return out_root


def resolve_presets(preset_arg: str, presets: dict) -> list[str]:
    if preset_arg in PRESET_GROUPS:
        return [s for s in PRESET_GROUPS[preset_arg] if s in presets]
    if preset_arg in presets:
        return [preset_arg]
    raise SystemExit(f"unknown preset '{preset_arg}'; available: {sorted(set(list(presets) + list(PRESET_GROUPS)))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest free HuggingFace datasets into datasets/raw")
    parser.add_argument("--preset", default="quickstart", help="preset name (openwebtext/wikipedia/arxiv/gsm8k/openmath/stack or group math/all)")
    parser.add_argument("--max-samples", type=int, default=5000, help="max rows per source")
    parser.add_argument("--output-dir", default="datasets/raw")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    presets = _load_config_presets()
    if args.list_presets:
        print("Built-in presets:")
        for k, v in presets.items():
            print(f"  {k}: {v['hf_dataset']} ({'streaming' if v.get('hf_streaming') else 'batch'})")
        print("Groups:", ", ".join(PRESET_GROUPS))
        return

    sources = resolve_presets(args.preset, presets)
    for name in sources:
        spec = presets[name]
        ingest(
            source_name=name,
            hf_dataset=spec["hf_dataset"],
            hf_split=spec.get("hf_split", "train"),
            hf_text_column=spec.get("hf_text_column", "text"),
            hf_subset=spec.get("hf_subset"),
            hf_streaming=spec.get("hf_streaming", True),
            text_key=spec.get("text_key") or spec.get("hf_text_column", "text"),
            solution_key=spec.get("solution_key"),
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            min_chars=args.min_chars,
            shard_size=args.shard_size,
        )


if __name__ == "__main__":
    main()

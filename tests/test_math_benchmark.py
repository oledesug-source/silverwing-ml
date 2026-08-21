"""Tests for the M09 independent mathematics benchmark release flow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks import BenchmarkRunner, DummyModel
from benchmarks.math import (
    MathBenchmarkConfig,
    flag_corpus_overlap,
    generate_math_benchmark,
    write_math_benchmark,
)
from benchmarks.registry import BenchmarkRegistry
from foundation.corpus.schema import DocumentRecord, Provenance, Split
from foundation.corpus.storage import ShardWriter

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, topics: tuple[str, ...] = ("arithmetic", "geometry")) -> MathBenchmarkConfig:
    return MathBenchmarkConfig(
        seed=123,
        items_per_topic=3,
        topics=topics,
        output_path=str(tmp_path / "math.jsonl"),
        corpus_dir=str(tmp_path / "corpus"),
    )


def test_generation_is_deterministic_unique_and_numeric(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = generate_math_benchmark(cfg)
    second = generate_math_benchmark(cfg)
    assert first == second
    assert len(first) == cfg.total_items
    assert len({item.item_id for item in first}) == cfg.total_items
    assert len({item.prompt for item in first}) == cfg.total_items
    assert {item.category for item in first} == set(cfg.topics)
    assert all(item.reference.lstrip("-").isdigit() for item in first)


def test_release_writes_hash_manifest_and_category_metrics(tmp_path: Path) -> None:
    cfg = _config(tmp_path, ("arithmetic", "number_theory", "trigonometry"))
    release = write_math_benchmark(cfg)
    data_path = release["path"]
    manifest = json.loads(release["manifest_path"].read_text(encoding="utf-8"))
    assert data_path.exists()
    assert manifest["items"] == cfg.total_items
    assert len(manifest["benchmark_hash"]) == 64
    assert manifest["config_digest"] == cfg.digest()
    assert manifest["items_by_category"] == dict.fromkeys(cfg.topics, 3)

    registry = BenchmarkRegistry()
    registry.register("math-test", data_path, "numeric")
    result = BenchmarkRunner(DummyModel(answer="0"), registry).run("math-test")
    assert set(result.category_metrics) == set(cfg.topics)
    assert sum(metric["n"] for metric in result.category_metrics.values()) == cfg.total_items


def test_corpus_overlap_is_detected(tmp_path: Path) -> None:
    record = generate_math_benchmark(_config(tmp_path, ("geometry",)))[0]
    corpus_record = DocumentRecord.build(
        document_id="overlap",
        text=f"{record.prompt} The expected response is {record.reference}.",
        provenance=Provenance(source_id="test", source_type="manual", domain="math", language="en"),
    )
    corpus_dir = tmp_path / "corpus"
    ShardWriter(corpus_dir).write({Split.TRAIN.value: [corpus_record]})
    flagged = flag_corpus_overlap([record], corpus_dir, n=8, threshold=0.6)
    assert record.item_id in flagged


def test_generation_cli_allows_explicit_development_skip(tmp_path: Path) -> None:
    output_path = tmp_path / "math.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_math_benchmark.py",
            "--output-path",
            str(output_path),
            "--items-per-topic",
            "1",
            "--skip-contamination-check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert output_path.with_suffix(".manifest.json").exists()
    assert "not a publishable benchmark" in result.stderr


def test_generation_cli_releases_against_verified_clean_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_record = DocumentRecord.build(
        document_id="unrelated",
        text=(
            "Botanical field notes describe seasonal rainfall, root structures, and pollinator behavior "
            "across a protected forest ecosystem. "
        )
        * 4,
        provenance=Provenance(source_id="test", source_type="manual", domain="science", language="en"),
    )
    ShardWriter(corpus_dir).write({Split.TRAIN.value: [corpus_record]})
    output_path = tmp_path / "math.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_math_benchmark.py",
            "--corpus-dir",
            str(corpus_dir),
            "--output-path",
            str(output_path),
            "--items-per-topic",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

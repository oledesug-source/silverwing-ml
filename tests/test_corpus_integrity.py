"""Tests for M03: dataset integrity, contamination hardening, config loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundation.corpus.config import (
    build_pipeline_from_config,
    load_corpus_config,
    pipeline_config_digest,
    source_configs_from_env,
)
from foundation.corpus.contamination import ContaminationDetector, build_default_detector
from foundation.corpus.deduplication import MinHash
from foundation.corpus.pipeline import CorpusPipeline
from foundation.corpus.schema import DocumentRecord, Provenance, Split
from foundation.corpus.storage import ShardWriter
from foundation.corpus.verify import verify_dataset

TEXT_A = (
    "The scientific method is a systematic process of inquiry that begins with observation "
    "and hypothesis formation, then proceeds through controlled experimentation and "
    "careful analysis of results, with published evidence reviewed by peers across the "
    "entire community of researchers."
)
TEXT_B = (
    "The industrial revolution transformed agriculture through mechanization and crop "
    "rotation, which increased yields and reduced the labor required to feed growing "
    "urban populations throughout Europe and beyond."
)


def make_record(doc_id: str, text: str) -> DocumentRecord:
    return DocumentRecord.build(
        document_id=doc_id,
        text=text,
        provenance=Provenance(source_id="t", source_type="manual", domain="general", language="en"),
    )


@pytest.fixture
def written_dataset(tmp_path: Path) -> tuple[Path, dict]:
    records = [make_record(f"doc{i}", TEXT_A if i % 2 == 0 else TEXT_B) for i in range(3000)]
    writer = ShardWriter(tmp_path / "ds")
    manifest = writer.write({Split.TRAIN.value: records})
    return tmp_path / "ds", manifest


def test_manifest_has_dataset_hash(written_dataset):
    _, manifest = written_dataset
    assert len(manifest["dataset_hash"]) == 64


def test_dataset_hash_is_deterministic(tmp_path: Path):
    records = [make_record(f"doc{i}", TEXT_A) for i in range(500)]
    first = ShardWriter(tmp_path / "a").write({Split.TRAIN.value: records})
    second = ShardWriter(tmp_path / "b").write({Split.TRAIN.value: records})
    assert first["dataset_hash"] == second["dataset_hash"]


def test_verify_dataset_ok(written_dataset):
    ds_dir, manifest = written_dataset
    result = verify_dataset(ds_dir, expected_dataset_hash=manifest["dataset_hash"])
    assert result.ok
    assert result.computed_dataset_hash == manifest["dataset_hash"]


def test_verify_detects_tampering(written_dataset):
    ds_dir, manifest = written_dataset
    shard = ds_dir / "train.0.jsonl"
    shard.write_text(shard.read_text(encoding="utf-8") + "\n{\"tampered\": true}\n", encoding="utf-8")
    result = verify_dataset(ds_dir)
    assert not result.ok
    assert any("mismatch" in err for err in result.split_errors)


def test_verify_detects_missing_shard(written_dataset):
    ds_dir, manifest = written_dataset
    (ds_dir / "train.0.jsonl").unlink()
    result = verify_dataset(ds_dir)
    assert not result.ok
    assert "train.0.jsonl" in result.missing_shards


def test_verify_pinned_hash_mismatch(written_dataset):
    ds_dir, manifest = written_dataset
    result = verify_dataset(ds_dir, expected_dataset_hash="0" * 64)
    assert not result.ok


def test_contamination_fuzzy_normalization():
    detector = ContaminationDetector(n=3, threshold=0.5)
    detector.add_benchmark("Alpha BETA, Gamma: delta")
    doc = make_record("d", "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta")
    assert detector.is_contaminated(doc)
    clean = make_record("c", " ".join(f"word{i}" for i in range(200)))
    assert not detector.is_contaminated(clean)


def test_contamination_per_benchmark_and_report(tmp_path: Path):
    detector = ContaminationDetector(n=3, threshold=0.5)
    detector.add_benchmark("one two three four", name="math")
    detector.add_benchmark("alpha beta gamma delta", name="code")
    doc = make_record("d", "one two three four repeated one two three four one two three four")
    ratios = detector.per_benchmark_ratios(doc.text)
    assert ratios["math"] >= 0.5
    assert ratios["code"] == 0.0
    path = detector.write_report([doc], tmp_path / "contamination.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["benchmarks"]["math"]["over_threshold"] == 1


def test_default_detector_has_named_benchmark():
    detector = build_default_detector()
    assert "heldout-suite" in detector.benchmark_names


def test_minhash_normalize_catches_case_variants():
    text = TEXT_A + " " + TEXT_A
    upper = text.upper()
    sig_a = MinHash(num_hashes=128).signature(text)
    sig_b = MinHash(num_hashes=128).signature(upper)
    assert MinHash.similarity(sig_a, sig_b) > 0.9


def test_load_corpus_config_from_repo():
    config = load_corpus_config()
    assert config["seed"] == 42
    assert config["chunking"]["max_tokens"] == 1024
    assert config["contamination"]["ngram_n"] == 8


def test_config_digest_deterministic():
    config = {"seed": 42, "chunking": {"max_tokens": 1024}}
    assert pipeline_config_digest(config) == pipeline_config_digest(config)
    assert pipeline_config_digest(config) != pipeline_config_digest({"seed": 43, "chunking": {"max_tokens": 1024}})


def test_build_pipeline_from_config_end_to_end(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "sources"
    corpus_dir.mkdir()
    for i in range(60):
        text = TEXT_A if i % 2 == 0 else TEXT_B
        (corpus_dir / f"doc{i}.txt").write_text(text + f" uniquely identified by the number {i} in this test", encoding="utf-8")
    monkeypatch.setenv("CORPUS_SOURCE_0", str(corpus_dir))
    config = load_corpus_config()
    config["sources"] = [
        {
            "source_id": "test-corpus",
            "path": str(corpus_dir),
            "source_type": "manual",
            "domain": "general",
            "kind": "directory",
        }
    ]
    sources = source_configs_from_env(config)
    pipeline = build_pipeline_from_config(config, sources=sources, output_dir=str(tmp_path / "out"))
    assert isinstance(pipeline, CorpusPipeline)
    assert pipeline.chunker.max_tokens == 1024
    report = pipeline.run()
    assert report.manifest is not None
    result = verify_dataset(tmp_path / "out")
    assert result.ok


def test_quality_config_fields_present():
    config = load_corpus_config()
    quality = config["filtering"]["quality"]
    assert quality["min_chars"] == 200
    assert "max_duplicate_line_ratio" in quality

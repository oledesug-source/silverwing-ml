"""End-to-end and unit tests for the corpus platform."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundation.corpus.chunking import Chunker, estimate_tokens
from foundation.corpus.contamination import ContaminationDetector
from foundation.corpus.deduplication import Deduplicator, MinHash
from foundation.corpus.filtering import QualityFilter, detect_domain, detect_language, detect_script
from foundation.corpus.ingestion import Ingestor, SourceConfig
from foundation.corpus.normalization import normalize_document_id, normalize_text
from foundation.corpus.pipeline import CorpusPipeline
from foundation.corpus.schema import DocumentRecord, Provenance, Split, SplitOptions
from foundation.corpus.split import Splitter
from foundation.corpus.storage import ShardWriter

LONG_TEXT = (
    "The scientific method is a systematic process of inquiry that begins with observation "
    "and hypothesis formation, then proceeds through controlled experimentation and "
    "careful analysis of results. Researchers design experiments to test their predictions "
    "while controlling for confounding variables that might otherwise distort the outcome. "
    "The resulting evidence is published and reviewed by peers, building a cumulative body "
    "of knowledge that is revised whenever new data contradicts previous conclusions. "
    "This cycle of theory, prediction, experimentation, and revision has proven remarkably "
    "effective across every branch of modern science, from physics to biology to economics."
)


def make_record(doc_id: str, text: str, language: str = "en") -> DocumentRecord:
    return DocumentRecord.build(
        document_id=doc_id,
        text=text,
        provenance=Provenance(source_id="test", source_type="manual", domain="science", language=language),
    )


def test_normalize_text_is_deterministic():
    dirty = "\ufeffHello\u00a0  world\r\n\n\n   \x00There\x07"
    clean = normalize_text(dirty)
    assert clean == normalize_text(dirty)
    assert "\ufeff" not in clean
    assert "\x00" not in clean
    assert "\n\n\n" not in clean


def test_normalize_document_id_sanitizes():
    assert normalize_document_id("A B!C/../D") == "A-B-C-D"
    assert normalize_document_id("  ") == "document"


def test_quality_filter_rejects_short_spammy():
    filt = QualityFilter(min_chars=200, min_words=40)
    assert filt.keep(make_record("ok", LONG_TEXT))
    assert not filt.keep(make_record("short", "hello world"))
    spam = " ".join(["https://spam.example/buy-now"] * 50)
    assert not filt.keep(make_record("spam", spam))


def test_language_and_script_detection():
    assert detect_script("the quick brown fox jumps") == "latin"
    assert detect_script("这是一个测试句子这是") == "cjk"
    assert detect_language("The quick brown fox jumps over the lazy dog and then continues walking") == "en"
    assert detect_language("Das ist ein deutscher Satz mit genug Worten fuer die Erkennung") == "de"
    assert detect_domain("def foo(): return 42  import math  class Bar:") == "code"


def test_minhash_similarity_and_lsh():
    near = LONG_TEXT + " revised whenever new data contradicts previous conclusions."
    sig_a = MinHash().signature(LONG_TEXT)
    sig_b = MinHash().signature(near)
    assert MinHash.similarity(sig_a, sig_b) > 0.9
    unrelated = " ".join(f"word{i}" for i in range(200))
    assert MinHash.similarity(sig_a, MinHash().signature(unrelated)) < 0.3


def test_deduplicator_removes_exact_and_near():
    dedup = Deduplicator()
    records = [
        make_record("a", LONG_TEXT),
        make_record("b", LONG_TEXT),  # exact duplicate
        make_record("c", LONG_TEXT + " revised whenever new data contradicts previous conclusions."),  # near duplicate
        make_record("d", " ".join(f"token{i}" for i in range(300))),
    ]
    kept, dropped = dedup.apply(records)
    assert dropped == 2
    assert {r.document_id for r in kept} == {"a", "d"}


def test_chunker_keeps_provenance_and_tokens():
    chunker = Chunker(max_tokens=40, overlap_tokens=8)
    chunks = chunker.chunk(make_record("doc", LONG_TEXT * 3))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.provenance.parent_document == "doc"
        assert chunk.token_count <= 40
    assert estimate_tokens(LONG_TEXT) > 0


def test_contamination_detector_flags_overlap():
    detector = ContaminationDetector(n=4, threshold=0.5)
    detector.add_benchmark("hypothesis formation then proceeds through controlled experimentation")
    contaminated = make_record("c1", "hypothesis formation then proceeds through controlled experimentation " * 10)
    clean = make_record("c2", " ".join(f"token{i}" for i in range(200)))
    assert detector.is_contaminated(contaminated)
    assert not detector.is_contaminated(clean)


def test_splitter_is_deterministic_and_partitioned():
    records = [make_record(f"doc{i}", LONG_TEXT) for i in range(2000)]
    splitter = Splitter(SplitOptions(train=0.96, validation=0.02, test=0.02))
    first = splitter.split(records)
    second = splitter.split([make_record(f"doc{i}", LONG_TEXT) for i in range(2000)])
    for name in (Split.TRAIN.value, Split.VALIDATION.value, Split.TEST.value):
        assert {r.document_id for r in first[name]} == {r.document_id for r in second[name]}
    total = sum(len(v) for v in first.values())
    assert abs(len(first[Split.TRAIN.value]) / total - 0.96) < 0.05
    assert abs(len(first[Split.VALIDATION.value]) / total - 0.02) < 0.03
    assert abs(len(first[Split.TEST.value]) / total - 0.02) < 0.03


def test_split_options_validate():
    with pytest.raises(ValueError):
        SplitOptions(train=0.5, validation=0.5, test=0.5)


def test_shard_writer_writes_manifest(tmp_path: Path):
    records = [make_record(f"doc{i}", LONG_TEXT) for i in range(25000)]
    writer = ShardWriter(tmp_path)
    manifest = writer.write({Split.TRAIN.value: records})
    assert manifest["splits"]["train"]["records"] == 25000
    shards = [p for p in tmp_path.iterdir() if p.suffix == ".jsonl"]
    assert len(shards) == 3
    assert (tmp_path / "manifest.json").exists()


def test_ingestor_reads_text_and_jsonl(tmp_path: Path):
    (tmp_path / "one.txt").write_text(LONG_TEXT, encoding="utf-8")
    lines = [json.dumps({"id": 1, "text": LONG_TEXT}), json.dumps({"id": 2, "text": "short"})]
    (tmp_path / "two.jsonl").write_text("\n".join(lines), encoding="utf-8")
    ingest = Ingestor(
        [
            SourceConfig(source_id="txt-src", path=str(tmp_path / "one.txt"), source_type="manual", domain="science"),
            SourceConfig(source_id="jsonl-src", path=str(tmp_path / "two.jsonl"), source_type="manual", kind="jsonl"),
        ]
    )
    records = list(ingest.ingest())
    assert len(records) == 3
    assert all(r.provenance.source_id in ("txt-src", "jsonl-src") for r in records)


def test_pipeline_end_to_end(tmp_path: Path):
    docs = [
        make_record(
            f"doc{i}",
            LONG_TEXT
            + f" This document is uniquely numbered {i} and contains additional discussion about topic number "
            f"{i} covering distinct concepts, historical background, and terminology that differ from every other "
            f"document in this corpus to avoid near-duplicate detection across samples.",
        )
        for i in range(200)
    ]
    corpus_dir = tmp_path / "sources"
    corpus_dir.mkdir(parents=True)
    for doc in docs:
        (corpus_dir / f"{doc.document_id}.txt").write_text(doc.text, encoding="utf-8")
    ingest = Ingestor(
        [SourceConfig(source_id="corpus", path=str(corpus_dir), source_type="manual", domain="science", kind="directory")]
    )
    pipeline = CorpusPipeline(ingest=ingest, output_dir=str(tmp_path / "out"), chunker=Chunker(max_tokens=64, overlap_tokens=8))
    report = pipeline.run()
    assert report.stages["ingested"] == 200
    assert report.completed_at is not None
    assert report.manifest is not None
    assert report.manifest["splits"]["train"]["records"] > 0
    assert report.manifest["splits"]["validation"]["records"] > 0
    assert report.manifest["splits"]["test"]["records"] > 0
    assert (tmp_path / "out" / "manifest.json").exists()

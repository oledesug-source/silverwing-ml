"""Tests for M05: Tokenizer V2 (byte-level BPE)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundation.tokenizer import SPECIAL_TOKENS, TokenizerV2, train_tokenizer_from_corpus
from foundation.tokenizer.bpe import MERGE_BASE, NUM_SPECIALS, train_bpe
from foundation.tokenizer.tokenizer import TOKENIZER_VERSION

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump!"
)


def sample_corpus() -> list[str]:
    return [
        TEXT * 3,
        "the quick brown fox the quick brown fox" * 5,
        "lazy dog lazy dog lazy dog" * 4,
        "Ünïcödé tèxt wïth ëmöjis 🚀 and mixed symbols §≠Ω" * 3,
        "".join(chr(0x4E00 + i) for i in range(20)) * 4,
    ]


def test_round_trip_ascii() -> None:
    tok = TokenizerV2(train_bpe(sample_corpus(), vocab_size=1024)[0])
    text = "Pack my box with five dozen liquor jugs."
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_unicode() -> None:
    tok = TokenizerV2(train_bpe(sample_corpus(), vocab_size=1024)[0])
    text = "Ünïcödé tèxt wïth ëmöjis 🚀 and mixed symbols §≠Ω 中文文本"
    assert tok.decode(tok.encode(text)) == text


def test_no_out_of_vocab() -> None:
    tok = TokenizerV2(train_bpe(sample_corpus(), vocab_size=2048)[0])
    text = "".join(chr(0x1000 + i) for i in range(500))  # bytes not necessarily in vocab
    for token_id in tok.encode(text):
        assert 0 <= token_id < tok.vocab_size


def test_vocab_size_layout() -> None:
    merges, stats = train_bpe(sample_corpus(), vocab_size=4096, min_frequency=1)
    tok = TokenizerV2(merges)
    assert tok.vocab_size == MERGE_BASE + len(merges)
    assert stats["requested_merges"] == 4096 - 256 - NUM_SPECIALS
    assert tok.vocab_size <= 4096


def test_min_frequency_early_stop() -> None:
    small = ["a" * 20, "b" * 20]  # (a,a) and (b,b) occur 19 times < threshold
    merges, stats = train_bpe(small, vocab_size=4096, min_frequency=50)
    assert stats["early_stopped"] is True
    assert stats["produced_merges"] == 0
    tok = TokenizerV2(merges)
    assert tok.decode(tok.encode("aaaaaaaaaaaaaaaaaaaa")) == "a" * 20


def test_deterministic_training() -> None:
    merges_a, _ = train_bpe(sample_corpus(), vocab_size=2048)
    merges_b, _ = train_bpe(sample_corpus(), vocab_size=2048)
    assert TokenizerV2(merges_a).digest() == TokenizerV2(merges_b).digest()


def test_special_tokens() -> None:
    merges, _ = train_bpe(sample_corpus(), vocab_size=512)
    tok = TokenizerV2(merges)
    for idx, special in enumerate(SPECIAL_TOKENS):
        assert tok.encode(special) == [idx]
    text = "hello world"
    ids = tok.encode(f"{SPECIAL_TOKENS[0]}{text}{SPECIAL_TOKENS[2]}")
    assert ids[0] == 0 and ids[-1] == 2
    decoded = tok.decode(ids)
    assert SPECIAL_TOKENS[0] in decoded and SPECIAL_TOKENS[2] in decoded
    assert text in decoded


def test_save_load_round_trip(tmp_path: Path) -> None:
    tok = TokenizerV2(train_bpe(sample_corpus(), vocab_size=1024)[0])
    out = tok.save(tmp_path / "tok")
    loaded = TokenizerV2.load(out)
    assert loaded.digest() == tok.digest()
    assert loaded.version == TOKENIZER_VERSION
    text = "Sphinx of black quartz, judge my vow 🚀"
    assert loaded.decode(loaded.encode(text)) == text
    assert (out / "config.json").exists()
    assert (out / "merges.json").exists()
    assert (out / "vocab.json").exists()
    assert (out / "tokenizer_hash").read_text(encoding="utf-8") == tok.digest()


def test_load_rejects_wrong_version(tmp_path: Path) -> None:
    tok = TokenizerV2(train_bpe(sample_corpus(), vocab_size=512)[0])
    out = tok.save(tmp_path / "tok")
    with pytest.raises(ValueError):
        TokenizerV2.load(out, version="tokenizer-v999")


def test_train_from_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shard = corpus_dir / "train.000.jsonl"
    docs = [
        {"document_id": "d1", "text": TEXT, "split": "train"},
        {"document_id": "d2", "text": "the quick brown fox 🦊 " * 10, "split": "train"},
        {"document_id": "d3", "text": "lazy dog lazy dog " * 8, "split": "train"},
    ]
    shard.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    manifest = corpus_dir / "manifest.json"
    manifest.write_text(json.dumps({"dataset_hash": "abc123"}), encoding="utf-8")

    out = tmp_path / "tok"
    report = train_tokenizer_from_corpus(corpus_dir=corpus_dir, vocab_size=1024, output_dir=out)
    assert report["corpus_dataset_hash"] == "abc123"
    assert report["vocab_size"] <= 1024
    assert report["tokenizer_hash"]
    assert (out / "tokenizer_training_report.json").exists()

    loaded = TokenizerV2.load(out)
    assert loaded.decode(loaded.encode("the quick brown fox")) == "the quick brown fox"

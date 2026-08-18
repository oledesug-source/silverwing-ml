"""Tests for M04: benchmark engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import (
    BenchmarkRunner,
    DummyModel,
    SilverwingModel,
    default_registry,
    load_items,
    write_evaluation_report,
)
from benchmarks.dataset import BenchmarkItem
from benchmarks.guard import build_corpus_fingerprint, flag_contaminated, item_contamination_ratio
from benchmarks.metrics import (
    accuracy,
    compute_metrics,
    exact_match,
    parse_number,
    perplexity_from_log_probs,
    relative_error,
    rmse,
)
from benchmarks.registry import BenchmarkRegistry
from benchmarks.runner import EvalResult, current_git_commit
from foundation.corpus.schema import DocumentRecord, Provenance, Split
from foundation.corpus.storage import ShardWriter

ARITHMETIC = """{"id": "a1", "prompt": "7 + 5 = ?", "reference": "12", "task_type": "numeric"}
{"id": "a2", "prompt": "100 - 37 = ?", "reference": "63", "task_type": "numeric"}
{"id": "a3", "prompt": "6 * 9 = ?", "reference": "54", "task_type": "numeric"}
"""

EXACT = """{"id": "e1", "prompt": "Capital of France?", "reference": "Paris"}
{"id": "e2", "prompt": "Capital of Japan?", "reference": "Tokyo"}
"""


def make_corpus_record(doc_id: str, text: str) -> DocumentRecord:
    return DocumentRecord.build(
        document_id=doc_id,
        text=text,
        provenance=Provenance(source_id="t", source_type="manual", domain="general", language="en"),
    )


def test_normalize_answer_and_exact_match():
    assert exact_match("Paris!", " paris .") is True
    assert exact_match("12.0", "12") is True
    assert exact_match("$1,234.50", "1234.5") is True
    assert exact_match("Tokyo", "Paris") is False


def test_parse_number_variants():
    assert parse_number("$1,234.50") == 1234.5
    assert parse_number("25%") == 25.0
    assert parse_number("-3.5") == -3.5
    assert parse_number("no number") is None
    assert parse_number("approx 2e3") == 2000.0


def test_numeric_metrics():
    metrics = compute_metrics(["10", "20", "30"], ["12", "18", "30"], "numeric")
    assert metrics["parsed"] == 3
    assert metrics["mae"] == pytest.approx((2 + 2 + 0) / 3)
    assert metrics["rmse"] == pytest.approx(rmse([2, 2, 0]))
    assert metrics["n"] == 3


def test_exact_match_metrics():
    metrics = compute_metrics(["paris.", "tokyo"], ["Paris", "Tokyo"], "exact_match")
    assert metrics["accuracy"] == 1.0
    assert metrics["correct"] == 2


def test_accuracy_and_perplexity():
    assert accuracy([True, True, False]) == pytest.approx(2 / 3)
    assert perplexity_from_log_probs([-math.log(2), -math.log(2)]) == pytest.approx(2.0)
    assert perplexity_from_log_probs([]) == float("inf")
    assert relative_error(10, 20) == 0.5


def test_load_items_jsonl(tmp_path: Path):
    path = tmp_path / "bench.jsonl"
    path.write_text(ARITHMETIC, encoding="utf-8")
    items = load_items(path, default_task_type="numeric")
    assert len(items) == 3
    assert items[0].item_id == "a1"
    assert items[0].task_type == "numeric"


def test_load_items_json_list(tmp_path: Path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps([{"id": "x", "prompt": "p", "reference": "r"}]), encoding="utf-8")
    items = load_items(path)
    assert items[0].category == "general"


def test_registry_roundtrip(tmp_path: Path):
    path = tmp_path / "b.jsonl"
    path.write_text(ARITHMETIC, encoding="utf-8")
    registry = BenchmarkRegistry()
    registry.register("arithmetic-v1", path, "numeric")
    assert registry.names() == ["arithmetic-v1"]
    spec = registry.get("arithmetic-v1")
    assert spec.task_type == "numeric"
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(FileNotFoundError):
        registry.register("missing", tmp_path / "nope.jsonl", "numeric")


def test_default_registry_has_sample():
    registry = default_registry()
    assert "sample_arithmetic" in registry.names()


def test_runner_numeric_with_dummy(tmp_path: Path):
    path = tmp_path / "arith.jsonl"
    path.write_text(ARITHMETIC, encoding="utf-8")
    registry = BenchmarkRegistry()
    registry.register("arith", path, "numeric")
    runner = BenchmarkRunner(DummyModel(answer="12"), registry)
    result = runner.run("arith", limit=2)
    assert isinstance(result, EvalResult)
    assert result.n_items == 2
    assert result.metrics["task_type"] == "numeric"
    assert set(result.item_scores) == {"a1", "a2"}
    assert result.category_metrics["general"]["n"] == 2
    assert result.model_id == "dummy"
    assert result.git_commit is not None
    assert result.git_commit == current_git_commit()


def test_runner_exact_match_with_dummy(tmp_path: Path):
    path = tmp_path / "exact.jsonl"
    path.write_text(EXACT, encoding="utf-8")
    registry = BenchmarkRegistry()
    registry.register("exact", path, "exact_match")
    runner = BenchmarkRunner(DummyModel(answer="Paris"), registry)
    result = runner.run("exact")
    assert result.metrics["accuracy"] == 0.5


def test_runner_rejects_non_adapter():
    with pytest.raises(TypeError):
        BenchmarkRunner(object())  # type: ignore[arg-type]


def test_guard_detects_contamination(tmp_path: Path):
    corpus_text = (
        "The capital of the small nation is Paris and the river Seine flows through it, "
        "which makes the city the center of culture and finance for the entire region, "
        "observed by historians across centuries."
    )
    records = [make_corpus_record(f"doc{i}", corpus_text) for i in range(20)]
    ShardWriter(tmp_path / "corpus").write({Split.TRAIN.value: records})
    fingerprint = build_corpus_fingerprint(tmp_path / "corpus", n=8)
    assert fingerprint

    contaminated = BenchmarkItem(item_id="c1", prompt=corpus_text, reference="x")
    clean = BenchmarkItem(item_id="c2", prompt=" ".join(f"word{i}" for i in range(300)), reference="y")
    assert item_contamination_ratio(contaminated, fingerprint, n=8) >= 0.6
    assert item_contamination_ratio(clean, fingerprint, n=8) < 0.6
    flagged = flag_contaminated([contaminated, clean], tmp_path / "corpus", n=8, threshold=0.6)
    assert set(flagged) == {"c1"}


def test_write_evaluation_report(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = EvalResult(
        benchmark="arith",
        model_id="dummy",
        task_type="numeric",
        metrics={"n": 1, "mae": 0.0},
        item_scores={"a1": 1.0},
        n_items=1,
        git_commit="abc123",
    )
    path = write_evaluation_report(result, tmp_path / "eval")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"]["mae"] == 0.0
    assert data["git_commit"] == "abc123"
    assert "environment" in data
    assert "python" in data["environment"]


def _tiny_model_and_checkpoint(tmp_path: Path) -> Path:
    import torch

    from foundation.model import ModelConfig, build_model
    from foundation.tokenizer import TokenizerV2
    from foundation.training.checkpoint import save_checkpoint

    tokenizer = TokenizerV2(merges=[])
    cfg = ModelConfig.from_dict(
        {
            "model_name": "tiny-eval",
            "vocab_size": tokenizer.vocab_size,
            "block_size": 64,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "mlp_hidden_size": 32,
            "dropout": 0.0,
            "tie_embeddings": True,
            "bias": False,
        }
    )
    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    path = save_checkpoint(
        ckpt_dir,
        step=1,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        run_id="test",
        config_digest="deadbeef",
        tokenizer_hash=tokenizer.digest(),
        dataset_hash="cafe",
        git_commit="test",
    )
    tokenizer.save(tmp_path / "tokenizer")
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("model:\n  vocab_size: %d\n  block_size: 64\n  n_layer: 1\n  n_head: 2\n  n_kv_head: 2\n  n_embd: 16\n  mlp_hidden_size: 32\n  tie_embeddings: true\n  bias: false\n" % tokenizer.vocab_size, encoding="utf-8")
    return path


def test_silverwing_model_completes_and_scores(tmp_path: Path):
    import torch

    checkpoint = _tiny_model_and_checkpoint(tmp_path)
    model = SilverwingModel(
        str(checkpoint),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        model_config=str(tmp_path / "model.yaml"),
    )
    assert model.model_id.startswith("silverwing:")
    text = model.complete("hello", max_new_tokens=16)
    assert isinstance(text, str)
    assert "<|" not in text
    logp = model.log_prob("hello", "world")
    assert isinstance(logp, float)
    assert torch.isfinite(torch.tensor(logp))

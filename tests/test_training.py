"""Tests for the Training Engine V2 (M07)."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import pytest
import torch
import yaml

from foundation.model import ModelConfig, build_model
from foundation.corpus.schema import DocumentRecord, Provenance, Split
from foundation.corpus.storage import ShardWriter
from foundation.tokenizer import TokenizerV2
from foundation.tokenizer.bpe import train_bpe
from foundation.training import (
    BEST_FILENAME,
    FINAL_FILENAME,
    PretrainingData,
    TrainConfig,
    build_optimizer,
    git_commit,
    git_is_clean,
    load_checkpoint,
    preflight_train,
    require_clean_repo,
    save_checkpoint,
    schedule_lr,
    train,
)

TINY_TEXT = "the quick brown fox jumps over the lazy dog. "


def _write_corpus(root: Path, n_train: int = 24, n_val: int = 4) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    train_lines = [
        json.dumps({"document_id": f"doc{i}", "text": TINY_TEXT * ((i % 3) + 1)})
        for i in range(n_train)
    ]
    (corpus / "train.0.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_lines = [
        json.dumps({"document_id": f"val{i}", "text": TINY_TEXT * 2}) for i in range(n_val)
    ]
    (corpus / "validation.0.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    return corpus


def _make_tokenizer(corpus: Path) -> TokenizerV2:
    lines = (corpus / "train.0.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines]
    merges, _ = train_bpe(texts, vocab_size=300, min_frequency=2)
    return TokenizerV2(merges=merges)


def _write_released_corpus(root: Path, n_train: int = 24, n_val: int = 4) -> Path:
    corpus = root / "released-corpus"
    train = [
        DocumentRecord.build(
            document_id=f"train-{i}",
            text=TINY_TEXT * ((i % 3) + 2),
            provenance=Provenance(source_id="test", source_type="manual", domain="general", language="en"),
        )
        for i in range(n_train)
    ]
    validation = [
        DocumentRecord.build(
            document_id=f"validation-{i}",
            text=TINY_TEXT * 2,
            provenance=Provenance(source_id="test", source_type="manual", domain="general", language="en"),
        )
        for i in range(n_val)
    ]
    ShardWriter(corpus).write({Split.TRAIN.value: train, Split.VALIDATION.value: validation})
    return corpus


def _write_tokenizer_dir(root: Path, tokenizer: TokenizerV2) -> Path:
    tokenizer_dir = root / "tokenizer"
    tokenizer.save(tokenizer_dir)
    return tokenizer_dir


def _write_model_config(root: Path, vocab_size: int, block_size: int = 64) -> str:
    path = root / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_layer": 2,
                    "n_head": 2,
                    "n_kv_head": 1,
                    "n_embd": 32,
                    "mlp_hidden_size": 64,
                    "mlp_activation": "swiglu",
                    "norm_eps": 1.0e-5,
                    "tie_embeddings": True,
                    "bias": False,
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _train_config(root: Path, **overrides) -> TrainConfig:
    corpus = _write_corpus(root)
    tokenizer = _make_tokenizer(corpus)
    tokenizer_dir = _write_tokenizer_dir(root, tokenizer)
    model_config_path = _write_model_config(root, tokenizer.vocab_size)
    defaults = {
        "model_config_path": model_config_path,
        "corpus_dir": str(corpus),
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint_dir": str(root / "checkpoints"),
        "batch_size": 2,
        "block_size": 8,
        "max_steps": 10,
        "warmup_steps": 2,
        "lr": 1.0e-3,
        "verify_dataset": False,
        "require_validation": True,
        "require_clean_repo": False,
        "device": "cpu",
    }
    defaults.update(overrides)
    return TrainConfig.from_dict(defaults)


class TestSchedule:
    def test_warmup_then_cosine(self) -> None:
        lr = 1.0
        values = [schedule_lr(s, lr, warmup_steps=20, max_steps=100, min_lr_ratio=0.1) for s in range(120)]
        assert values[0] == pytest.approx(lr / 20)
        assert values[19] == pytest.approx(lr)
        assert values[-1] == pytest.approx(0.1)
        assert values[21] < values[20]
        for i in range(20, len(values) - 1):
            assert values[i + 1] <= values[i] + 1e-12
        assert min(values[20:]) >= 0.1 - 1e-9

    def test_no_warmup(self) -> None:
        assert schedule_lr(0, 0.1, warmup_steps=0, max_steps=50) == pytest.approx(0.1)


class TestOptimizer:
    def test_grouping_decays_2d_weights_only(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        cfg = ModelConfig.from_yaml(_write_model_config(tmp_path, tokenizer.vocab_size))
        model = build_model(cfg)
        optimizer, report = build_optimizer(model, lr=1e-3, weight_decay=0.1)
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]
        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0
        assert report["decay_params"] + report["no_decay_params"] == sum(
            1 for p in model.parameters() if p.requires_grad
        )
        no_decay = set(id(p) for p in no_decay_group["params"])
        for name, param in model.named_parameters():
            if param.ndim == 1:
                assert id(param) in no_decay
            else:
                assert id(param) not in no_decay


class TestData:
    def test_shapes_shift_and_eos(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        data = PretrainingData(corpus, tokenizer, split="train", block_size=8)
        assert len(data) == (len(data.tokens) - 1) // 8
        x, y = data.batch([0, 1])
        assert x.shape == (2, 8) and y.shape == (2, 8)
        assert torch.equal(y[:, :-1], x[:, 1:])
        assert int(x.min()) >= 0 and int(x.max()) < tokenizer.vocab_size
        assert data.tokens[-1] == tokenizer.special_ids["<|endoftext|>"]
        assert data.n_documents == 24

    def test_batches_are_pure_permutations(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        data = PretrainingData(corpus, tokenizer, split="train", block_size=8)
        first = data.shuffled_indices(random.Random(7))
        second = data.shuffled_indices(random.Random(7))
        assert first == second
        assert sorted(first) == list(range(len(data)))
        other = data.shuffled_indices(random.Random(12345))
        assert other != first

    def test_ordered_batch_is_stable(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        data = PretrainingData(corpus, tokenizer, split="validation", block_size=8)
        x1, y1 = data.ordered_batch(2)
        x2, y2 = data.ordered_batch(2)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)

    def test_batch_stream_resume_uses_exact_next_batch(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        data = PretrainingData(corpus, tokenizer, split="train", block_size=8)
        stream = data.batch_stream(batch_size=2, seed=99)
        for _ in range(3):
            next(stream)
        state = stream.state_dict()
        expected_x, expected_y = next(stream)
        resumed = data.batch_stream(batch_size=2, seed=99)
        resumed.load_state_dict(state)
        actual_x, actual_y = next(resumed)
        assert torch.equal(expected_x, actual_x)
        assert torch.equal(expected_y, actual_y)


class TestCheckpoint:
    def test_round_trip(self, tmp_path: Path) -> None:
        corpus = _write_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        cfg = ModelConfig.from_yaml(_write_model_config(tmp_path, tokenizer.vocab_size))
        model = build_model(cfg)
        optimizer, _ = build_optimizer(model, lr=1e-3)
        path = save_checkpoint(
            tmp_path,
            step=42,
            model=model,
            optimizer=optimizer,
            run_id="test-run",
            config_digest="abc",
            tokenizer_hash="def",
            dataset_hash=None,
            git_commit=git_commit(),
        )
        model2 = build_model(cfg)
        optimizer2, _ = build_optimizer(model2, lr=1e-3)
        ckpt = load_checkpoint(path, model2, optimizer2)
        assert ckpt["step"] == 42
        assert ckpt["git_commit"] == git_commit()
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)
        for g1, g2 in zip(optimizer.param_groups, optimizer2.param_groups):
            assert g1["weight_decay"] == g2["weight_decay"]


class TestRepoGuard:
    def test_commit_hash_format(self) -> None:
        assert re.fullmatch(r"[0-9a-f]{40}", git_commit())

    def test_guard_detects_uncommitted_files(self) -> None:
        probe_rel = "tests/_repo_probe.txt"
        assert git_is_clean([probe_rel])
        probe = Path(__file__).parent / "_repo_probe.txt"
        try:
            probe.write_text("dirty", encoding="utf-8")
            assert not git_is_clean([probe_rel])
            with pytest.raises(RuntimeError, match="committed repository"):
                require_clean_repo([probe_rel])
        finally:
            probe.unlink(missing_ok=True)
        assert git_is_clean([probe_rel])


class TestTrain:
    def test_config_yaml_round_trip_and_coercion(self) -> None:
        cfg = TrainConfig.from_yaml("configs/training.yaml")
        assert isinstance(cfg.betas, tuple) and cfg.betas == (0.9, 0.95)
        assert isinstance(cfg.lr, float)
        assert isinstance(cfg.max_steps, int)
        again = TrainConfig.from_dict(cfg.to_dict())
        assert again.digest() == cfg.digest()
        assert again.resume_digest() == cfg.resume_digest()

    def test_preflight_verifies_released_dataset(self, tmp_path: Path) -> None:
        corpus = _write_released_corpus(tmp_path)
        tokenizer = _make_tokenizer(corpus)
        tokenizer_dir = _write_tokenizer_dir(tmp_path, tokenizer)
        model_path = _write_model_config(tmp_path, tokenizer.vocab_size)
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        cfg = TrainConfig.from_dict(
            {
                "model_config_path": model_path,
                "corpus_dir": str(corpus),
                "tokenizer_dir": str(tokenizer_dir),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "batch_size": 2,
                "block_size": 8,
                "max_steps": 2,
                "expected_dataset_hash": manifest["dataset_hash"],
                "require_clean_repo": False,
            }
        )
        inputs = preflight_train(cfg)
        assert inputs.dataset_hash == manifest["dataset_hash"]
        assert inputs.dataset_verification and inputs.dataset_verification["ok"]

    def test_preflight_rejects_unreleased_dataset(self, tmp_path: Path) -> None:
        cfg = _train_config(tmp_path, verify_dataset=True)
        with pytest.raises(ValueError, match="dataset integrity verification failed"):
            preflight_train(cfg)

    def test_overfits_small_corpus(self, tmp_path: Path) -> None:
        cfg = _train_config(
            tmp_path,
            max_steps=120,
            warmup_steps=10,
            lr=2.0e-3,
            eval_steps=40,
            eval_sequences=2,
            save_steps=40,
        )
        report = train(cfg)
        assert report["steps_done"] == 120
        assert report["final_train_loss"] < 0.8
        assert report["final_eval_loss"] is not None and math.isfinite(report["final_eval_loss"])
        assert report["best_eval_loss"] is not None
        assert re.fullmatch(r"[0-9a-f]{40}", report["git_commit"])
        assert re.fullmatch(r"[0-9a-f]{64}", report["model_config_digest"])
        assert re.fullmatch(r"[0-9a-f]{64}", report["train_config_digest"])
        assert re.fullmatch(r"[0-9a-f]{64}", report["tokenizer_hash"])
        assert report["num_parameters"] > 0
        assert report["final_perplexity"] > 1.0

        checkpoint_dir = Path(cfg.checkpoint_dir)
        assert (checkpoint_dir / "training_report.json").exists()
        assert (checkpoint_dir / BEST_FILENAME).exists()
        assert (checkpoint_dir / FINAL_FILENAME).exists()
        assert (checkpoint_dir / f"step-{40:08d}.pt").exists()
        manifest = json.loads((checkpoint_dir / "training_report.json").read_text(encoding="utf-8"))
        assert manifest["dataset_hash"] is None or isinstance(manifest["dataset_hash"], str)

        model = build_model(ModelConfig.from_yaml(cfg.model_config_path))
        load_checkpoint(checkpoint_dir / FINAL_FILENAME, model)
        logits = model(torch.randint(0, 8, (1, 8)))
        assert logits.shape[-1] == ModelConfig.from_yaml(cfg.model_config_path).vocab_size

    def test_dirty_repo_blocked(self, tmp_path: Path) -> None:
        cfg = _train_config(tmp_path, max_steps=4)
        cfg = TrainConfig.from_dict({**cfg.to_dict(), "require_clean_repo": True})
        probe = Path(__file__).parent / "_repo_probe.txt"
        try:
            probe.write_text("dirty", encoding="utf-8")
            with pytest.raises(RuntimeError, match="committed repository"):
                train(cfg)
        finally:
            probe.unlink(missing_ok=True)

    def test_empty_train_split_rejected(self, tmp_path: Path) -> None:
        corpus = tmp_path / "empty"
        corpus.mkdir()
        tokenizer = _make_tokenizer(_write_corpus(tmp_path))
        tokenizer_dir = _write_tokenizer_dir(tmp_path, tokenizer)
        model_config_path = _write_model_config(tmp_path, tokenizer.vocab_size)
        cfg = TrainConfig.from_dict(
            {
                "model_config_path": model_config_path,
                "corpus_dir": str(corpus),
                "tokenizer_dir": str(tokenizer_dir),
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "batch_size": 2,
                "block_size": 8,
                "max_steps": 4,
                "verify_dataset": False,
                "require_clean_repo": False,
            }
        )
        with pytest.raises(ValueError, match="empty"):
            train(cfg)

    def test_resume_restores_exact_training_state(self, tmp_path: Path) -> None:
        cfg = _train_config(
            tmp_path / "full",
            max_steps=4,
            warmup_steps=1,
            grad_accum_steps=2,
            eval_steps=2,
            save_steps=2,
        )
        train(cfg, log=lambda _: None)
        step_two = Path(cfg.checkpoint_dir) / "step-00000002.pt"
        resumed_cfg = TrainConfig.from_dict(
            {
                **cfg.to_dict(),
                "checkpoint_dir": str(tmp_path / "resumed"),
                "resume_from": str(step_two),
            }
        )
        resumed_report = train(resumed_cfg, log=lambda _: None)
        assert resumed_report["start_step"] == 3
        full_model = build_model(ModelConfig.from_yaml(cfg.model_config_path))
        resumed_model = build_model(ModelConfig.from_yaml(cfg.model_config_path))
        load_checkpoint(Path(cfg.checkpoint_dir) / FINAL_FILENAME, full_model)
        load_checkpoint(Path(resumed_cfg.checkpoint_dir) / FINAL_FILENAME, resumed_model)
        for first, second in zip(full_model.parameters(), resumed_model.parameters()):
            assert torch.equal(first, second)

    def test_resume_rejects_changed_training_dynamics(self, tmp_path: Path) -> None:
        cfg = _train_config(tmp_path, max_steps=2, save_steps=1, eval_steps=1)
        train(cfg, log=lambda _: None)
        changed = TrainConfig.from_dict(
            {
                **cfg.to_dict(),
                "checkpoint_dir": str(tmp_path / "changed"),
                "resume_from": str(Path(cfg.checkpoint_dir) / "step-00000001.pt"),
                "lr": 2.0e-3,
            }
        )
        with pytest.raises(ValueError, match="resume_config_digest"):
            train(changed, log=lambda _: None)

    def test_init_from_starts_fresh_run(self, tmp_path: Path) -> None:
        source = _train_config(tmp_path / "source", max_steps=2, save_steps=1, eval_steps=1)
        train(source, log=lambda _: None)
        init_cfg = _train_config(
            tmp_path / "init",
            max_steps=3,
            warmup_steps=1,
            eval_steps=1,
            save_steps=1,
        )
        init_cfg = TrainConfig.from_dict(
            {**init_cfg.to_dict(), "init_from": str(Path(source.checkpoint_dir) / FINAL_FILENAME)}
        )
        report = train(init_cfg, log=lambda _: None)
        assert report["init_from"] == str(Path(source.checkpoint_dir) / FINAL_FILENAME)
        assert report["start_step"] == 1
        assert report["steps_done"] == 3
        assert report["final_checkpoint"] is not None
        assert Path(report["final_checkpoint"]).exists()

    def test_init_from_missing_checkpoint_rejected(self, tmp_path: Path) -> None:
        cfg = _train_config(tmp_path, max_steps=2)
        cfg = TrainConfig.from_dict({**cfg.to_dict(), "init_from": str(tmp_path / "nope.pt")})
        with pytest.raises(ValueError, match="init_from checkpoint does not exist"):
            train(cfg, log=lambda _: None)

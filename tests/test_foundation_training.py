"""Tests for foundation/training/ — config, scheduler, checkpoint, optimizer, data."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from foundation.tokenizer import TokenizerV2
from foundation.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from foundation.training.config import TRAINING_VERSION, TrainConfig, _coerce
from foundation.training.data import PretrainingData
from foundation.training.optimizer import build_optimizer
from foundation.training.scheduler import schedule_lr

# ── Helpers ───────────────────────────────────────────────────────────────


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def _tiny_model_config(tokenizer: TokenizerV2) -> object:
    from foundation.model import ModelConfig

    return ModelConfig.from_dict(
        {
            "model_name": "tiny-train",
            "vocab_size": tokenizer.vocab_size,
            "block_size": 32,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "mlp_hidden_size": 32,
            "tie_embeddings": True,
            "bias": False,
        }
    )


def _tiny_model_yaml(tokenizer: TokenizerV2, tmp_path: Path) -> Path:

    cfg = _tiny_model_config(tokenizer)
    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(
        "model:\n"
        f"  vocab_size: {cfg.vocab_size}\n"
        f"  block_size: {cfg.block_size}\n"
        f"  n_layer: {cfg.n_layer}\n"
        f"  n_head: {cfg.n_head}\n"
        f"  n_kv_head: {cfg.n_kv_head}\n"
        f"  n_embd: {cfg.n_embd}\n"
        f"  mlp_hidden_size: {cfg.mlp_hidden_size}\n"
        f"  tie_embeddings: {str(cfg.tie_embeddings).lower()}\n"
        f"  bias: {str(cfg.bias).lower()}\n",
        encoding="utf-8",
    )
    return model_cfg_path


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path, TokenizerV2]:
    from foundation.model import build_model

    tokenizer = make_tokenizer()
    model_cfg = _tiny_model_config(tokenizer)
    model = build_model(model_cfg)
    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir()
    path = save_checkpoint(
        ckpt_dir,
        step=10,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        run_id="pretrain-test",
        config_digest="c0ffee",
        tokenizer_hash=tokenizer.digest(),
        dataset_hash="b0b",
        git_commit="abc123",
    )
    tokenizer.save(tmp_path / "tokenizer")
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    return path, model_cfg_path, tokenizer


def _write_corpus(tmp_path: Path, tokenizer: TokenizerV2, n_docs: int = 5) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shard = corpus_dir / "train.00000.jsonl"
    lines = []
    for i in range(n_docs):
        text = f"Document number {i} with some words for tokenization. " * 10
        lines.append(json.dumps({"text": text}, ensure_ascii=False))
    shard.write_text("\n".join(lines), encoding="utf-8")
    val_shard = corpus_dir / "validation.00000.jsonl"
    val_lines = []
    for i in range(3):
        text = f"Validation document {i} with different words. " * 10
        val_lines.append(json.dumps({"text": text}, ensure_ascii=False))
    val_shard.write_text("\n".join(val_lines), encoding="utf-8")
    return corpus_dir


# ── Config Tests ──────────────────────────────────────────────────────────


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.version == TRAINING_VERSION
        assert cfg.batch_size == 8
        assert cfg.lr == 3e-4
        assert cfg.device == "cpu"

    def test_to_dict(self):
        cfg = TrainConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["batch_size"] == 8
        assert isinstance(d["betas"], list)
        assert len(d["betas"]) == 2

    def test_digest_deterministic(self):
        cfg = TrainConfig()
        assert cfg.digest() == cfg.digest()

    def test_digest_differs_on_change(self):
        cfg1 = TrainConfig(lr=1e-3)
        cfg2 = TrainConfig(lr=2e-3)
        assert cfg1.digest() != cfg2.digest()

    def test_digest_is_sha256(self):
        cfg = TrainConfig()
        d = cfg.digest()
        assert len(d) == 64
        int(d, 16)

    def test_resume_digest_excludes_operational_fields(self):
        cfg1 = TrainConfig(checkpoint_dir="a", resume_from="b")
        cfg2 = TrainConfig(checkpoint_dir="x", resume_from="y")
        assert cfg1.resume_digest() == cfg2.resume_digest()

    def test_resume_digest_differs_on_training_change(self):
        cfg1 = TrainConfig(lr=1e-3)
        cfg2 = TrainConfig(lr=2e-3)
        assert cfg1.resume_digest() != cfg2.resume_digest()

    def test_from_dict(self):
        d = {"lr": 0.005, "batch_size": 16, "max_steps": 100}
        cfg = TrainConfig.from_dict(d)
        assert cfg.lr == 0.005
        assert cfg.batch_size == 16
        assert cfg.max_steps == 100

    def test_from_dict_defaults_fill_missing(self):
        cfg = TrainConfig.from_dict({"lr": 0.01})
        assert cfg.batch_size == 8
        assert cfg.block_size == 512

    def test_from_yaml(self, tmp_path: Path):
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(
            "training:\n  lr: 0.005\n  batch_size: 4\n  max_steps: 50\n",
            encoding="utf-8",
        )
        cfg = TrainConfig.from_yaml(yaml_path)
        assert cfg.lr == 0.005
        assert cfg.batch_size == 4
        assert cfg.max_steps == 50

    def test_validation_batch_size_positive(self):
        try:
            TrainConfig(batch_size=0)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_max_steps(self):
        try:
            TrainConfig(max_steps=0)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_lr_positive(self):
        try:
            TrainConfig(lr=-1.0)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_min_lr_ratio(self):
        try:
            TrainConfig(min_lr_ratio=1.5)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_weight_decay(self):
        try:
            TrainConfig(weight_decay=-0.1)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_grad_clip(self):
        try:
            TrainConfig(grad_clip=-1.0)
            assert False, "should raise"
        except ValueError:
            pass

    def test_validation_eval_steps(self):
        try:
            TrainConfig(eval_steps=-1)
            assert False, "should raise"
        except ValueError:
            pass

    def test_coerce_bool(self):
        assert _coerce("true", "bool") is True
        assert _coerce("false", "bool") is False
        assert _coerce(1, "bool") is True

    def test_coerce_optional(self):
        assert _coerce(None, "Optional[str]") is None
        assert _coerce(42, "Optional[int]") == 42

    def test_coerce_tuple(self):
        result = _coerce([0.9, 0.95], "tuple[float, float]")
        assert result == (0.9, 0.95)


# ── Scheduler Tests ───────────────────────────────────────────────────────


class TestScheduler:
    def test_warmup_linear(self):
        lr = 1e-3
        assert abs(schedule_lr(0, lr, warmup_steps=10, max_steps=100) - lr / 10) < 1e-8
        assert abs(schedule_lr(9, lr, warmup_steps=10, max_steps=100) - lr) < 1e-8

    def test_no_warmup(self):
        lr = 1e-3
        val = schedule_lr(0, lr, warmup_steps=0, max_steps=100)
        assert val > 0

    def test_cosine_decay(self):
        lr = 1e-3
        mid = schedule_lr(50, lr, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
        end = schedule_lr(100, lr, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
        assert mid < lr
        assert abs(end - lr * 0.1) < 1e-8

    def test_after_max_steps(self):
        lr = 1e-3
        val = schedule_lr(200, lr, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
        assert abs(val - lr * 0.1) < 1e-8

    def test_warmup_step_zero(self):
        lr = 1e-3
        val = schedule_lr(0, lr, warmup_steps=5, max_steps=100)
        assert abs(val - lr / 5) < 1e-8

    def test_min_lr_ratio_one(self):
        lr = 1e-3
        val = schedule_lr(100, lr, warmup_steps=0, max_steps=100, min_lr_ratio=1.0)
        assert abs(val - lr) < 1e-8


# ── Checkpoint Tests ──────────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        ckpt_dir = tmp_path / "ckpts"
        path = save_checkpoint(
            ckpt_dir,
            step=42,
            model=model,
            optimizer=optimizer,
            run_id="test-run",
            config_digest="abc",
            tokenizer_hash="def",
            dataset_hash="ghi",
            git_commit="commit1",
        )
        assert path.exists()
        assert path.name == "step-00000042.pt"

        model2 = build_model(model_cfg)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        loaded = load_checkpoint(path, model2, optimizer2, "cpu")
        assert loaded["step"] == 42
        assert loaded["run_id"] == "test-run"
        assert loaded["config_digest"] == "abc"

    def test_custom_filename(self, tmp_path: Path):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)

        path = save_checkpoint(
            tmp_path, step=1, model=model, optimizer=None,
            run_id="r", config_digest="", tokenizer_hash="",
            dataset_hash=None, git_commit="", filename="best.pt",
        )
        assert path.name == "best.pt"

    def test_metadata_fields(self, tmp_path: Path):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)

        path = save_checkpoint(
            tmp_path, step=5, model=model, optimizer=None,
            run_id="meta-test", config_digest="digest1",
            tokenizer_hash="tok1", dataset_hash="data1",
            git_commit="commit2",
            model_config_digest="mcd1",
            resume_config_digest="rcd1",
            best_eval_loss=2.5,
            eval_loss=3.0,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["model_config_digest"] == "mcd1"
        assert ckpt["resume_config_digest"] == "rcd1"
        assert ckpt["best_eval_loss"] == 2.5
        assert ckpt["eval_loss"] == 3.0
        assert "saved_at" in ckpt

    def test_rng_capture_restore(self):
        torch.manual_seed(42)
        state = capture_rng_state()
        assert "torch" in state
        torch.manual_seed(99)
        restore_rng_state(state)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert abs(val1 - val2) < 1e-6

    def test_rng_restore_missing_key(self):
        try:
            restore_rng_state({})
            assert False, "should raise"
        except ValueError:
            pass

    def test_load_without_optimizer(self, tmp_path: Path):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)

        path = save_checkpoint(
            tmp_path, step=1, model=model, optimizer=None,
            run_id="r", config_digest="", tokenizer_hash="",
            dataset_hash=None, git_commit="",
        )
        model2 = build_model(model_cfg)
        loaded = load_checkpoint(path, model2, None, "cpu")
        assert loaded["optimizer_state"] is None


# ── Optimizer Tests ───────────────────────────────────────────────────────


class TestOptimizer:
    def test_build_optimizer(self):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)
        optimizer, report = build_optimizer(model, lr=1e-3)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert "decay_params" in report
        assert "no_decay_params" in report
        assert report["decay_params"] + report["no_decay_params"] > 0

    def test_optimizer_report_counts(self):
        from foundation.model import build_model

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)
        _, report = build_optimizer(model, lr=1e-3, weight_decay=0.1)
        total = report["decay_parameters"] + report["no_decay_parameters"]
        assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Data Tests ────────────────────────────────────────────────────────────


class TestPretrainingData:
    def test_loads_corpus(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=5)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        assert data.n_documents == 5
        assert data.n_blocks > 0
        assert len(data) == data.n_blocks

    def test_num_tokens(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=3)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        assert data.num_tokens() > 0

    def test_batch_shape(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=5)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        xs, ys = data.batch([0, 1])
        assert xs.shape == (2, 32)
        assert ys.shape == (2, 32)

    def test_batch_targets_shifted(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=5)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        xs, ys = data.batch([0])
        assert xs[0, 1].item() == ys[0, 0].item()

    def test_ordered_batch(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=5)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        result = data.ordered_batch(2)
        assert result is not None
        xs, ys = result
        assert xs.shape[0] == 2

    def test_ordered_batch_empty(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=0)
        # Write an empty shard
        shard = corpus_dir / "train.00000.jsonl"
        shard.write_text("", encoding="utf-8")
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        assert data.n_blocks == 0
        assert data.ordered_batch(5) is None

    def test_max_tokens_limit(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=10)
        data_limited = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32, max_tokens=100)
        data_full = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        assert data_limited.num_tokens() <= data_full.num_tokens()
        assert data_limited.n_documents <= data_full.n_documents

    def test_shuffled_indices(self, tmp_path: Path):
        import random

        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=5)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        rng = random.Random(42)
        indices = data.shuffled_indices(rng)
        assert len(indices) == data.n_blocks
        assert sorted(indices) == list(range(data.n_blocks))


class TestShuffledBatchStream:
    def test_yields_batches(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=10)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        stream = data.batch_stream(batch_size=2, seed=42)
        xs, ys = next(stream)
        assert xs.shape == (2, 32)
        assert ys.shape == (2, 32)

    def test_state_dict_round_trip(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=10)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        stream = data.batch_stream(batch_size=2, seed=42)
        next(stream)
        next(stream)
        state = stream.state_dict()
        assert "rng_state" in state
        assert "indices" in state
        assert state["position"] > 0

        stream2 = data.batch_stream(batch_size=2, seed=42)
        stream2.load_state_dict(state)
        xs1, _ = next(stream)
        xs2, _ = next(stream2)
        assert torch.equal(xs1, xs2)

    def test_load_state_dict_batch_size_mismatch(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = _write_corpus(tmp_path, tokenizer, n_docs=10)
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        stream = data.batch_stream(batch_size=2, seed=42)
        state = stream.state_dict()
        state["batch_size"] = 99
        try:
            stream.load_state_dict(state)
            assert False, "should raise"
        except ValueError:
            pass

    def test_empty_data_raises(self, tmp_path: Path):
        tokenizer = make_tokenizer()
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        shard = corpus_dir / "train.00000.jsonl"
        shard.write_text("", encoding="utf-8")
        data = PretrainingData(corpus_dir, tokenizer, split="train", block_size=32)
        stream = data.batch_stream(batch_size=2, seed=42)
        try:
            next(stream)
            assert False, "should raise"
        except ValueError:
            pass


# ── Integration: Config YAML round-trip ───────────────────────────────────


class TestConfigIntegration:
    def test_yaml_round_trip(self, tmp_path: Path):
        cfg = TrainConfig(lr=0.002, batch_size=4, max_steps=100)
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(
            f"training:\n  lr: {cfg.lr}\n  batch_size: {cfg.batch_size}\n  max_steps: {cfg.max_steps}\n",
            encoding="utf-8",
        )
        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.lr == cfg.lr
        assert loaded.batch_size == cfg.batch_size
        assert loaded.digest() == cfg.digest()

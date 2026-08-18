"""Tests for M12: preference alignment (DPO dataset, config, trainer smoke)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from foundation.alignment import (
    IGNORE_INDEX,
    AlignmentConfig,
    PreferenceDataset,
    PreferenceExample,
    compute_dpo_loss,
    dataset_hash,
    load_preferences,
    train_alignment,
)
from foundation.tokenizer import TokenizerV2


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def make_preferences(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"pref-{i}",
            "instruction": f"Compute x + {i}.",
            "chosen": str(40 + i),
            "rejected": str(10 + i),
        }
        for i in range(n)
    ]


def write_dataset(path: Path, preferences: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preferences),
        encoding="utf-8",
    )
    return path


def test_load_preferences_and_hash(tmp_path: Path):
    dataset_path = write_dataset(tmp_path / "dpo.jsonl", make_preferences())
    examples = load_preferences(dataset_path)
    assert len(examples) == 4
    assert examples[0].chosen == "40"
    assert examples[0].rejected == "10"
    assert isinstance(dataset_hash(dataset_path), str)
    assert len(dataset_hash(dataset_path)) == 64


def test_load_preferences_missing_field(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "instruction": "q"}), encoding="utf-8")
    try:
        load_preferences(path)
        assert False, "should have raised"
    except ValueError:
        pass


def test_load_preferences_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    try:
        load_preferences(path)
        assert False, "should have raised"
    except ValueError:
        pass


def _make_examples(preferences: list[dict]) -> list[PreferenceExample]:
    return [
        PreferenceExample(
            example_id=e["id"],
            instruction=e["instruction"],
            chosen=e["chosen"],
            rejected=e["rejected"],
        )
        for e in preferences
    ]


def test_dataset_train_eval_disjoint():
    tokenizer = make_tokenizer()
    examples = _make_examples(make_preferences(8))
    train = PreferenceDataset(
        examples, tokenizer, block_size=64, seed=7, eval_fraction=0.25, split="train"
    )
    evl = PreferenceDataset(
        examples, tokenizer, block_size=64, seed=7, eval_fraction=0.25, split="eval"
    )
    assert train.n_examples + evl.n_examples == 8
    assert 0 < evl.n_examples < 8


def test_dataset_block_shapes():
    tokenizer = make_tokenizer()
    examples = _make_examples(make_preferences(2))
    data = PreferenceDataset(
        examples, tokenizer, block_size=64, seed=1, eval_fraction=0.0, split="train"
    )
    assert data.n_blocks >= 1
    w_ids, w_lbl, l_ids, l_lbl = data.block(0)
    assert w_ids.shape == (64,)
    assert w_lbl.shape == (64,)
    assert l_ids.shape == (64,)
    assert l_lbl.shape == (64,)
    # Prompt tokens should be IGNORE_INDEX in labels
    assert w_lbl[0].item() == IGNORE_INDEX
    # At least some response tokens are supervised
    supervised = w_lbl != IGNORE_INDEX
    assert supervised.sum() > 0


def test_dpo_config_yaml(tmp_path: Path):
    config_path = tmp_path / "alignment.yaml"
    config_path.write_text(
        "alignment:\n  init_from: experiments/checkpoints/best.pt\n  max_steps: 7\n  lr: 5.0e-5\n  dpo_beta: 0.2\n  device: cpu\n",
        encoding="utf-8",
    )
    cfg = AlignmentConfig.from_yaml(config_path)
    assert cfg.max_steps == 7
    assert cfg.lr == 5.0e-5
    assert cfg.dpo_beta == 0.2
    assert cfg.version == "alignment-v1"
    assert isinstance(cfg.digest(), str)
    assert len(cfg.digest()) == 64


def test_dpo_config_invalid_rejected():
    for bad_kwargs in (
        {"max_steps": 0},
        {"lr": -1.0},
        {"dpo_beta": 0.0},
        {"block_size": -1},
        {"eval_fraction": 1.0},
    ):
        try:
            AlignmentConfig(**bad_kwargs)
            assert False, f"should have raised for {bad_kwargs}"
        except ValueError:
            pass


def _tiny_model_config(tokenizer: TokenizerV2) -> object:
    from foundation.model import ModelConfig

    return ModelConfig.from_dict(
        {
            "model_name": "tiny-dpo",
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
    model_cfg_path = tmp_path / "model.yaml"
    model_cfg_path.write_text(
        "model:\n"
        f"  vocab_size: {tokenizer.vocab_size}\n"
        "  block_size: 32\n"
        "  n_layer: 1\n"
        "  n_head: 2\n"
        "  n_kv_head: 2\n"
        "  n_embd: 16\n"
        "  mlp_hidden_size: 32\n"
        "  tie_embeddings: true\n"
        "  bias: false\n",
        encoding="utf-8",
    )
    return model_cfg_path


def _tiny_pretrained_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    from foundation.model import build_model
    from foundation.training.checkpoint import save_checkpoint

    tokenizer = make_tokenizer()
    cfg = _tiny_model_config(tokenizer)
    model = build_model(cfg)
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
        git_commit="test",
    )
    tokenizer.save(tmp_path / "tokenizer")
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    return path, model_cfg_path


def test_compute_dpo_loss_returns_scalar(tmp_path: Path):
    from foundation.model import ModelConfig, build_model
    from foundation.training.checkpoint import load_checkpoint

    checkpoint, model_cfg_path = _tiny_pretrained_checkpoint(tmp_path)
    model_cfg = ModelConfig.from_yaml(model_cfg_path)
    model = build_model(model_cfg)
    ref_model = copy.deepcopy(model)
    load_checkpoint(str(checkpoint), model, None, "cpu")
    load_checkpoint(str(checkpoint), ref_model, None, "cpu")

    bs, seq = 2, 16
    vocab = model_cfg.vocab_size
    w_ids = torch.randint(0, vocab, (bs, seq))
    l_ids = torch.randint(0, vocab, (bs, seq))
    w_lbl = w_ids.clone()
    l_lbl = l_ids.clone()

    loss = compute_dpo_loss(model, ref_model, w_ids, w_lbl, l_ids, l_lbl, beta=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dpo_loss_when_responses_identical():
    """When chosen == rejected, the preference logit is 0 so loss = -log σ(0) = log 2."""
    from foundation.model import build_model

    tokenizer = make_tokenizer()
    model_cfg = _tiny_model_config(tokenizer)
    model = build_model(model_cfg)
    ref_model = copy.deepcopy(model)

    ids = torch.randint(0, tokenizer.vocab_size, (1, 8))
    lbl = ids.clone()
    loss = compute_dpo_loss(model, ref_model, ids, lbl, ids, lbl, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-4


def test_dpo_loss_differs_when_policy_neq_ref():
    """When the policy model is perturbed away from the reference, the DPO
    loss should differ from log(2) (the model==ref baseline)."""
    from foundation.model import build_model

    tokenizer = make_tokenizer()
    model_cfg = _tiny_model_config(tokenizer)
    model = build_model(model_cfg)
    ref_model = copy.deepcopy(model)
    # Perturb the policy so it differs from the frozen reference
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    ids_w = torch.tensor([[5, 5, 5, 5, 5, 5, 5, 5]])
    ids_l = torch.tensor([[10, 10, 10, 10, 10, 10, 10, 10]])
    loss_w = compute_dpo_loss(
        model, ref_model, ids_w, ids_w.clone(), ids_l, ids_l.clone(), beta=0.1
    )
    loss_l = compute_dpo_loss(
        model, ref_model, ids_l, ids_l.clone(), ids_w, ids_w.clone(), beta=0.1
    )
    # Swapping chosen/rejected negates the preference logit; with policy != ref
    # the two losses must differ
    assert torch.isfinite(loss_w)
    assert torch.isfinite(loss_l)
    assert loss_w.item() != loss_l.item()


def test_train_alignment_smoke(tmp_path: Path):
    dataset_path = write_dataset(tmp_path / "dpo.jsonl", make_preferences(6))
    checkpoint, model_cfg_path = _tiny_pretrained_checkpoint(tmp_path)
    cfg = AlignmentConfig(
        model_config_path=str(model_cfg_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        init_from=str(checkpoint),
        dataset_path=str(dataset_path),
        checkpoint_dir=str(tmp_path / "out"),
        block_size=32,
        max_steps=3,
        warmup_steps=1,
        lr=1e-4,
        log_steps=1,
        eval_steps=2,
        eval_examples=2,
        save_steps=0,
        eval_fraction=0.5,
        require_clean_repo=False,
        device="cpu",
    )
    report = train_alignment(cfg, log=lambda _: None)
    assert report["stage"] == "alignment"
    assert report["final_train_loss"] is not None
    assert Path(report["final_checkpoint"]).exists()
    assert (
        Path(report["final_checkpoint"])
        .parent.joinpath("alignment_report.json")
        .exists()
    )
    assert report["dataset_hash"] == dataset_hash(dataset_path)
    assert report["init_from"] == str(checkpoint)
    assert report["dpo_beta"] == cfg.dpo_beta


def test_train_alignment_requires_checkpoint(tmp_path: Path):
    tokenizer = make_tokenizer()
    _tiny_model_config(tokenizer)
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    tokenizer.save(tmp_path / "tokenizer")

    cfg = AlignmentConfig(
        model_config_path=str(model_cfg_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        init_from=str(tmp_path / "nonexistent.pt"),
        dataset_path=str(write_dataset(tmp_path / "dpo.jsonl", make_preferences(2))),
        checkpoint_dir=str(tmp_path / "out"),
        max_steps=1,
        block_size=32,
        eval_fraction=0.5,
        eval_examples=1,
        require_clean_repo=False,
        device="cpu",
    )
    try:
        train_alignment(cfg, log=lambda _: None)
        assert False, "should have raised"
    except ValueError as e:
        assert "does not exist" in str(e)

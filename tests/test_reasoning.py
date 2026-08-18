"""Tests for M13: reasoning-chain training (dataset, config, trainer smoke)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from foundation.reasoning import (
    EXAMPLE_SEPARATOR,
    FINAL_ANSWER_HEADER,
    IGNORE_INDEX,
    PROBLEM_HEADER,
    REASONING_HEADER,
    ReasoningConfig,
    ReasoningDataset,
    ReasoningExample,
    dataset_hash,
    load_reasoning_examples,
    split_into_steps,
    train_reasoning,
)
from foundation.tokenizer import TokenizerV2


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def make_reasoning_records(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"rl-{i}",
            "reasoning_type": "multi_step",
            "domain": "arithmetic",
            "problem": f"Compute {i} + {i + 1}.",
            "reasoning_steps": [
                f"Add {i} and {i + 1}.",
                f"The result is {2 * i + 1}.",
            ],
            "final_answer": str(2 * i + 1),
            "difficulty": 0.3,
            "quality_score": 1.0,
        }
        for i in range(n)
    ]


def write_dataset(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def test_load_reasoning_examples_and_hash(tmp_path: Path):
    dataset_path = write_dataset(tmp_path / "reasoning.jsonl", make_reasoning_records())
    examples = load_reasoning_examples(dataset_path)
    assert len(examples) == 4
    assert examples[0].reasoning_steps[0] == "Add 0 and 1."
    assert examples[0].final_answer == "1"
    assert isinstance(dataset_hash(dataset_path), str)
    assert len(dataset_hash(dataset_path)) == 64


def test_load_reasoning_examples_missing_field(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "problem": "q"}), encoding="utf-8")
    try:
        load_reasoning_examples(path)
        assert False, "should have raised"
    except ValueError:
        pass


def test_load_reasoning_examples_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    try:
        load_reasoning_examples(path)
        assert False, "should have raised"
    except ValueError:
        pass


def test_split_into_steps():
    assert split_into_steps("One sentence. Two sentences.") == [
        "One sentence.",
        "Two sentences.",
    ]
    assert split_into_steps("") == []
    assert split_into_steps("  ") == []
    steps = split_into_steps("Step one. Step two; step three.")
    assert len(steps) == 3


def test_example_format_text():
    example = ReasoningExample(
        example_id="rl-0",
        problem="What is 2 + 3?",
        reasoning_steps=["Add 2 and 3.", "Result is 5."],
        final_answer="5",
    )
    text = example.format_text()
    assert text.startswith(PROBLEM_HEADER)
    assert "What is 2 + 3?" in text
    assert REASONING_HEADER in text
    assert "1. Add 2 and 3." in text
    assert "2. Result is 5." in text
    assert FINAL_ANSWER_HEADER in text
    assert "5" in text
    assert text.endswith(EXAMPLE_SEPARATOR)


def test_dataset_train_eval_disjoint():
    tokenizer = make_tokenizer()
    records = make_reasoning_records(8)
    examples = [
        ReasoningExample(
            example_id=r["id"],
            problem=r["problem"],
            reasoning_steps=r["reasoning_steps"],
            final_answer=r["final_answer"],
            reasoning_type=r["reasoning_type"],
            domain=r["domain"],
        )
        for r in records
    ]
    train = ReasoningDataset(
        examples, tokenizer, block_size=128, seed=7, eval_fraction=0.25, split="train"
    )
    evl = ReasoningDataset(
        examples, tokenizer, block_size=128, seed=7, eval_fraction=0.25, split="eval"
    )
    assert train.n_examples + evl.n_examples == 8
    assert 0 < evl.n_examples < 8
    total_blocks = train.n_blocks + evl.n_blocks
    assert total_blocks > 0


def test_dataset_block_shapes():
    tokenizer = make_tokenizer()
    records = make_reasoning_records(2)
    examples = [
        ReasoningExample(
            example_id=r["id"],
            problem=r["problem"],
            reasoning_steps=r["reasoning_steps"],
            final_answer=r["final_answer"],
            reasoning_type=r["reasoning_type"],
            domain=r["domain"],
        )
        for r in records
    ]
    data = ReasoningDataset(
        examples, tokenizer, block_size=64, seed=1, eval_fraction=0.0, split="train"
    )
    assert data.n_blocks >= 1
    x, y = data.block(0)
    assert x.shape == (64,)
    assert y.shape == (64,)
    # Problem tokens should be IGNORE_INDEX in labels
    assert y[0].item() == IGNORE_INDEX
    # At least some response tokens are supervised (reasoning steps + answer)
    supervised = y != IGNORE_INDEX
    assert supervised.sum() > 0
    assert data.supervised_tokens > 0


def test_reasoning_config_yaml(tmp_path: Path):
    config_path = tmp_path / "reasoning.yaml"
    config_path.write_text(
        "reasoning:\n  init_from: experiments/checkpoints/best.pt\n  max_steps: 7\n"
        "  lr: 5.0e-5\n  block_size: 256\n  device: cpu\n",
        encoding="utf-8",
    )
    cfg = ReasoningConfig.from_yaml(config_path)
    assert cfg.max_steps == 7
    assert cfg.lr == 5.0e-5
    assert cfg.block_size == 256
    assert cfg.version == "reasoning-v1"
    assert isinstance(cfg.digest(), str)
    assert len(cfg.digest()) == 64


def test_reasoning_config_invalid_rejected():
    for bad_kwargs in (
        {"max_steps": 0},
        {"lr": -1.0},
        {"block_size": 0},
        {"eval_fraction": 1.0},
        {"eval_examples": 0},
    ):
        try:
            ReasoningConfig(**bad_kwargs)
            assert False, f"should have raised for {bad_kwargs}"
        except ValueError:
            pass


def _tiny_model_config(tokenizer: TokenizerV2) -> object:
    from foundation.model import ModelConfig

    return ModelConfig.from_dict(
        {
            "model_name": "tiny-reasoning",
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


def test_train_reasoning_smoke(tmp_path: Path):
    records = make_reasoning_records(6)
    dataset_path = write_dataset(tmp_path / "reasoning.jsonl", records)
    checkpoint, model_cfg_path = _tiny_pretrained_checkpoint(tmp_path)
    cfg = ReasoningConfig(
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
    report = train_reasoning(cfg, log=lambda _: None)
    assert report["stage"] == "reasoning"
    assert report["final_train_loss"] is not None
    assert report["final_eval_loss"] is not None
    assert math.isfinite(report["final_eval_loss"])
    assert Path(report["final_checkpoint"]).exists()
    assert (
        Path(report["final_checkpoint"])
        .parent.joinpath("reasoning_report.json")
        .exists()
    )
    assert report["dataset_hash"] == dataset_hash(dataset_path)
    assert report["init_from"] == str(checkpoint)
    assert report["supervised_tokens_train"] > 0


def test_train_reasoning_requires_checkpoint(tmp_path: Path):
    tokenizer = make_tokenizer()
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    tokenizer.save(tmp_path / "tokenizer")

    cfg = ReasoningConfig(
        model_config_path=str(model_cfg_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        init_from=str(tmp_path / "nonexistent.pt"),
        dataset_path=str(
            write_dataset(tmp_path / "reasoning.jsonl", make_reasoning_records(2))
        ),
        checkpoint_dir=str(tmp_path / "out"),
        max_steps=1,
        block_size=32,
        eval_fraction=0.5,
        eval_examples=1,
        require_clean_repo=False,
        device="cpu",
    )
    try:
        train_reasoning(cfg, log=lambda _: None)
        assert False, "should have raised"
    except ValueError as e:
        assert "does not exist" in str(e)


def test_train_reasoning_empty_train_split(tmp_path: Path):
    """A dataset with only 1 example and eval_fraction=0.99 leaves 0 train blocks."""
    tokenizer = make_tokenizer()
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    tokenizer.save(tmp_path / "tokenizer")
    checkpoint, _ = _tiny_pretrained_checkpoint(tmp_path)

    cfg = ReasoningConfig(
        model_config_path=str(model_cfg_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        init_from=str(checkpoint),
        dataset_path=str(
            write_dataset(tmp_path / "reasoning.jsonl", make_reasoning_records(1))
        ),
        checkpoint_dir=str(tmp_path / "out"),
        max_steps=1,
        block_size=32,
        eval_fraction=0.99,
        eval_examples=1,
        require_clean_repo=False,
        device="cpu",
    )
    try:
        train_reasoning(cfg, log=lambda _: None)
        assert False, "should have raised"
    except ValueError as e:
        assert "empty" in str(e).lower()

"""Tests for M11: supervised fine-tuning (dataset masking, config, trainer smoke)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from foundation.sft import IGNORE_INDEX, SftConfig, SftDataset, SftExample, load_examples, train_sft
from foundation.sft.dataset import dataset_hash
from foundation.tokenizer import TokenizerV2


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def make_examples(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"ex-{i}",
            "instruction": f"Compute x + {i}.",
            "response": str(40 + i),
        }
        for i in range(n)
    ]


def write_dataset(path: Path, examples: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples),
        encoding="utf-8",
    )
    return path


def test_load_examples_and_hash(tmp_path: Path):
    dataset_path = write_dataset(tmp_path / "sft.jsonl", make_examples())
    examples = load_examples(dataset_path)
    assert len(examples) == 4
    assert examples[0].response == "40"
    assert isinstance(dataset_hash(dataset_path), str)
    assert len(dataset_hash(dataset_path)) == 64


def test_dataset_masks_only_response(tmp_path: Path):
    tokenizer = make_tokenizer()
    examples = [SftExample(example_id=e["id"], instruction=e["instruction"], response=e["response"]) for e in make_examples(2)]
    data = SftDataset(examples, tokenizer, block_size=64, seed=1, eval_fraction=0.0, split="train")
    assert data.n_blocks >= 1
    xs, ys = data.block(0)
    assert xs.shape == (64,)
    supervised = (ys != IGNORE_INDEX)
    assert supervised.sum() > 0
    assert xs[0].item() != IGNORE_INDEX
    answer_prefix = tokenizer.encode("Answer: ")
    assert len(answer_prefix) > 0


def test_dataset_train_eval_disjoint():
    tokenizer = make_tokenizer()
    examples = [SftExample(example_id=e["id"], instruction=e["instruction"], response=e["response"]) for e in make_examples(8)]
    train = SftDataset(examples, tokenizer, block_size=64, seed=7, eval_fraction=0.25, split="train")
    evl = SftDataset(examples, tokenizer, block_size=64, seed=7, eval_fraction=0.25, split="eval")
    assert train.n_examples + evl.n_examples == 8
    assert 0 < evl.n_examples < 8


def test_shifted_targets():
    tokenizer = make_tokenizer()
    examples = [SftExample(example_id="a", instruction="Add 1 and 2.", response="3")]
    data = SftDataset(examples, tokenizer, block_size=32, seed=0, eval_fraction=0.0, split="train")
    xs, ys = data.block(0)
    supervised_flags = [ys[i] != IGNORE_INDEX for i in range(len(ys))]
    assert not supervised_flags[0]
    for i in range(1, len(xs)):
        if ys[i] != IGNORE_INDEX:
            assert xs[i - 1] != IGNORE_INDEX


def test_sft_config_yaml(tmp_path: Path):
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        "sft:\n  init_from: experiments/checkpoints/best.pt\n  max_steps: 7\n  lr: 5.0e-5\n  device: cpu\n",
        encoding="utf-8",
    )
    cfg = SftConfig.from_yaml(config_path)
    assert cfg.max_steps == 7
    assert cfg.lr == 5.0e-5
    assert cfg.version == "sft-v1"
    assert isinstance(cfg.digest(), str)
    assert len(cfg.digest()) == 64


def _tiny_pretrained_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer = make_tokenizer()
    from foundation.model import ModelConfig, build_model
    from foundation.training.checkpoint import save_checkpoint

    cfg = ModelConfig.from_dict(
        {
            "model_name": "tiny-sft",
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
    return path, model_cfg_path


def test_train_sft_smoke(tmp_path: Path):
    dataset_path = write_dataset(tmp_path / "sft.jsonl", make_examples(6))
    checkpoint, model_cfg_path = _tiny_pretrained_checkpoint(tmp_path)
    cfg = SftConfig(
        model_config_path=str(model_cfg_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        init_from=str(checkpoint),
        dataset_path=str(dataset_path),
        checkpoint_dir=str(tmp_path / "out"),
        batch_size=1,
        block_size=32,
        max_steps=3,
        warmup_steps=1,
        log_steps=1,
        eval_steps=2,
        eval_examples=2,
        save_steps=0,
        eval_fraction=0.5,
        require_clean_repo=False,
        device="cpu",
    )
    report = train_sft(cfg, log=lambda _: None)
    assert report["stage"] == "sft"
    assert report["final_eval_loss"] is not None
    assert Path(report["final_checkpoint"]).exists()
    assert Path(report["final_checkpoint"]).parent.joinpath("sft_report.json").exists()
    assert report["dataset_hash"] == dataset_hash(dataset_path)
    assert report["init_from"] == str(checkpoint)


def test_build_sft_dataset_script(tmp_path: Path):
    output = tmp_path / "sft" / "sft.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--output",
            str(output),
            "--per-topic",
            "2",
            "--seed",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    from foundation.math_corpus import PROBLEM_GENERATORS

    assert len(lines) == 2 * len(PROBLEM_GENERATORS)
    first = json.loads(lines[0])
    assert "instruction" in first and "response" in first
    rerun = output.read_bytes()
    result2 = subprocess.run(
        [
            sys.executable,
            "scripts/build_sft_dataset.py",
            "--output",
            str(output),
            "--per-topic",
            "2",
            "--seed",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result2.returncode == 0
    assert output.read_bytes() == rerun

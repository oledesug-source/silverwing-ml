"""Tests for foundation/curriculum/ — config and smoke training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from foundation.curriculum.config import CurriculumConfig, StageConfig
from foundation.tokenizer import TokenizerV2

# ── Helpers ───────────────────────────────────────────────────────────────


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def _tiny_model_config(tokenizer: TokenizerV2) -> object:
    from foundation.model import ModelConfig

    return ModelConfig.from_dict(
        {
            "model_name": "tiny-curriculum",
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


def _tiny_pretrained_checkpoint(tmp_path: Path) -> tuple[Path, Path, TokenizerV2]:
    from foundation.model import build_model
    from foundation.training.checkpoint import save_checkpoint

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
        git_commit="test",
    )
    tokenizer.save(tmp_path / "tokenizer")
    model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
    return path, model_cfg_path, tokenizer


def _write_sft_dataset(path: Path, n: int = 4) -> Path:
    examples = []
    for i in range(n):
        examples.append({
            "id": f"ex-{i}",
            "instruction": f"Compute {i} + {i}.",
            "response": str(i + i),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples),
        encoding="utf-8",
    )
    return path


# ── StageConfig Tests ─────────────────────────────────────────────────────


class TestStageConfig:
    def test_defaults(self):
        stage = StageConfig(name="stage1", dataset_path="/data/train.jsonl")
        assert stage.name == "stage1"
        assert stage.max_steps == 500
        assert stage.lr == 1e-4
        assert stage.warmup_steps == 50
        assert stage.grad_clip == 1.0

    def test_to_dict(self):
        stage = StageConfig(name="s1", dataset_path="/d.jsonl", max_steps=100)
        d = stage.to_dict()
        assert d["name"] == "s1"
        assert d["max_steps"] == 100
        assert d["dataset_path"] == "/d.jsonl"

    def test_custom_values(self):
        stage = StageConfig(
            name="advanced",
            dataset_path="/data/advanced.jsonl",
            max_steps=2000,
            lr=5e-5,
            warmup_steps=200,
            grad_clip=0.5,
            description="Hard stage",
        )
        assert stage.lr == 5e-5
        assert stage.description == "Hard stage"

    def test_frozen(self):
        stage = StageConfig(name="s", dataset_path="/d")
        try:
            stage.name = "changed"
            assert False, "should raise"
        except AttributeError:
            pass


# ── CurriculumConfig Tests ────────────────────────────────────────────────


class TestCurriculumConfig:
    def test_defaults(self):
        cfg = CurriculumConfig()
        assert cfg.version == "curriculum-v1"
        assert cfg.batch_size == 1
        assert cfg.device == "cpu"
        assert cfg.stages == []

    def test_to_dict(self):
        cfg = CurriculumConfig(stages=[
            StageConfig(name="s1", dataset_path="/d1"),
            StageConfig(name="s2", dataset_path="/d2"),
        ])
        d = cfg.to_dict()
        assert isinstance(d["stages"], list)
        assert len(d["stages"]) == 2
        assert d["stages"][0]["name"] == "s1"
        assert isinstance(d["betas"], list)

    def test_digest_deterministic(self):
        cfg = CurriculumConfig()
        assert cfg.digest() == cfg.digest()

    def test_digest_differs(self):
        cfg1 = CurriculumConfig(batch_size=1)
        cfg2 = CurriculumConfig(batch_size=4)
        assert cfg1.digest() != cfg2.digest()

    def test_digest_is_sha256(self):
        cfg = CurriculumConfig()
        d = cfg.digest()
        assert len(d) == 64
        int(d, 16)

    def test_from_yaml(self, tmp_path: Path):
        yaml_path = tmp_path / "curriculum.yaml"
        yaml_path.write_text(
            "curriculum:\n"
            "  batch_size: 2\n"
            "  lr: 0.001\n"
            "  stages:\n"
            "    - name: easy\n"
            "      dataset_path: /data/easy.jsonl\n"
            "      max_steps: 100\n"
            "    - name: hard\n"
            "      dataset_path: /data/hard.jsonl\n"
            "      max_steps: 200\n"
            "      lr: 0.0005\n",
            encoding="utf-8",
        )
        cfg = CurriculumConfig.from_yaml(yaml_path)
        assert cfg.batch_size == 2
        assert len(cfg.stages) == 2
        assert cfg.stages[0].name == "easy"
        assert cfg.stages[1].name == "hard"
        assert cfg.stages[1].lr == 0.0005

    def test_from_yaml_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "curriculum.yaml"
        yaml_path.write_text(
            "curriculum:\n"
            "  stages:\n"
            "    - name: s1\n"
            "      dataset_path: /d\n",
            encoding="utf-8",
        )
        cfg = CurriculumConfig.from_yaml(yaml_path)
        assert cfg.batch_size == 1
        assert cfg.stages[0].max_steps == 500

    def test_empty_stages(self):
        cfg = CurriculumConfig()
        d = cfg.to_dict()
        assert d["stages"] == []


# ── Smoke Training Test ───────────────────────────────────────────────────


class TestCurriculumTrainer:
    def test_train_curriculum_smoke(self, tmp_path: Path):
        from foundation.curriculum.trainer import train_curriculum
        from foundation.model import build_model
        from foundation.training.checkpoint import save_checkpoint

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)

        ckpt_dir = tmp_path / "pretrained"
        ckpt_dir.mkdir()
        init_ckpt = save_checkpoint(
            ckpt_dir, step=1, model=model, optimizer=None,
            run_id="init", config_digest="", tokenizer_hash=tokenizer.digest(),
            dataset_hash=None, git_commit="",
        )

        tokenizer.save(tmp_path / "tokenizer")
        model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)

        dataset1 = _write_sft_dataset(tmp_path / "data" / "stage1.jsonl", n=8)
        dataset2 = _write_sft_dataset(tmp_path / "data" / "stage2.jsonl", n=8)

        stages = [
            StageConfig(
                name="easy",
                dataset_path=str(dataset1),
                max_steps=2,
                lr=1e-4,
                warmup_steps=1,
                log_steps=1,
                eval_steps=1,
                save_steps=0,
                checkpoint_dir=str(tmp_path / "ckpts" / "stage1"),
            ),
            StageConfig(
                name="hard",
                dataset_path=str(dataset2),
                max_steps=2,
                lr=5e-5,
                warmup_steps=1,
                log_steps=1,
                eval_steps=1,
                save_steps=0,
                checkpoint_dir=str(tmp_path / "ckpts" / "stage2"),
            ),
        ]

        cfg = CurriculumConfig(
            model_config_path=str(model_cfg_path),
            tokenizer_dir=str(tmp_path / "tokenizer"),
            init_from=str(init_ckpt),
            block_size=32,
            batch_size=1,
            stages=stages,
            eval_fraction=0.3,
            eval_examples=2,
            require_clean_repo=False,
            device="cpu",
        )

        report = train_curriculum(cfg, log=lambda _: None)
        assert report["num_stages"] == 2
        assert report["total_steps"] == 4
        assert len(report["stages"]) == 2
        assert report["stages"][0]["name"] == "easy"
        assert report["stages"][1]["name"] == "hard"
        assert Path(report["final_checkpoint"]).exists()
        assert report["device"] == "cpu"
        assert report["elapsed_seconds"] > 0

    def test_curriculum_report_written(self, tmp_path: Path):
        from foundation.curriculum.trainer import train_curriculum
        from foundation.model import build_model
        from foundation.training.checkpoint import save_checkpoint

        tokenizer = make_tokenizer()
        model_cfg = _tiny_model_config(tokenizer)
        model = build_model(model_cfg)

        ckpt_dir = tmp_path / "pretrained"
        ckpt_dir.mkdir()
        init_ckpt = save_checkpoint(
            ckpt_dir, step=1, model=model, optimizer=None,
            run_id="init", config_digest="", tokenizer_hash=tokenizer.digest(),
            dataset_hash=None, git_commit="",
        )
        tokenizer.save(tmp_path / "tokenizer")
        model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)

        dataset = _write_sft_dataset(tmp_path / "data.jsonl", n=8)
        stages = [
            StageConfig(
                name="s1", dataset_path=str(dataset),
                max_steps=2, lr=1e-4, warmup_steps=1,
                log_steps=0, eval_steps=1, save_steps=0,
                checkpoint_dir=str(tmp_path / "ckpts"),
            ),
        ]
        cfg = CurriculumConfig(
            model_config_path=str(model_cfg_path),
            tokenizer_dir=str(tmp_path / "tokenizer"),
            init_from=str(init_ckpt),
            block_size=32, batch_size=1, stages=stages,
            eval_fraction=0.3, eval_examples=2,
            require_clean_repo=False, device="cpu",
        )
        train_curriculum(cfg, log=lambda _: None)
        report_path = tmp_path / "ckpts" / "curriculum_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "run_id" in report
        assert "stages" in report

    def test_curriculum_requires_checkpoint(self, tmp_path: Path):
        from foundation.curriculum.trainer import train_curriculum

        tokenizer = make_tokenizer()
        model_cfg_path = _tiny_model_yaml(tokenizer, tmp_path)
        tokenizer.save(tmp_path / "tokenizer")

        cfg = CurriculumConfig(
            model_config_path=str(model_cfg_path),
            tokenizer_dir=str(tmp_path / "tokenizer"),
            init_from=str(tmp_path / "nonexistent.pt"),
            block_size=32,
            stages=[StageConfig(name="s", dataset_path="/nonexistent")],
            require_clean_repo=False,
            device="cpu",
        )
        try:
            train_curriculum(cfg, log=lambda _: None)
            assert False, "should raise"
        except ValueError as e:
            assert "does not exist" in str(e)

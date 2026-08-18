"""Validate every immutable input before a real pretraining run starts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..corpus import verify_dataset
from ..model import ModelConfig
from ..tokenizer import TokenizerV2
from .config import TrainConfig
from .data import PretrainingData


@dataclass
class TrainingInputs:
    model_config: ModelConfig
    tokenizer: TokenizerV2
    train_data: PretrainingData
    validation_data: PretrainingData
    dataset_hash: str | None
    dataset_verification: dict | None

    def report(self) -> dict:
        return {
            "model_config_digest": self.model_config.digest(),
            "tokenizer_hash": self.tokenizer.digest(),
            "dataset_hash": self.dataset_hash,
            "dataset_verification": self.dataset_verification,
            "train_documents": self.train_data.n_documents,
            "train_tokens": self.train_data.num_tokens(),
            "train_blocks": self.train_data.n_blocks,
            "validation_documents": self.validation_data.n_documents,
            "validation_tokens": self.validation_data.num_tokens(),
            "validation_blocks": self.validation_data.n_blocks,
        }


def _manifest_dataset_hash(corpus_dir: str | Path) -> str | None:
    manifest = Path(corpus_dir) / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("dataset_hash")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid corpus manifest: {manifest}") from exc


def _validate_device(device_name: str) -> None:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ({device_name}) but CUDA is unavailable")


def preflight_train(cfg: TrainConfig) -> TrainingInputs:
    """Resolve and validate the exact assets that a training run will consume."""
    _validate_device(cfg.device)
    model_config = ModelConfig.from_yaml(cfg.model_config_path)
    if cfg.block_size > model_config.block_size:
        raise ValueError(
            f"training block_size {cfg.block_size} exceeds model block_size {model_config.block_size}"
        )
    tokenizer = TokenizerV2.load(cfg.tokenizer_dir, version=cfg.tokenizer_version)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            "tokenizer and model vocabularies must match exactly for pretraining: "
            f"tokenizer={tokenizer.vocab_size}, model={model_config.vocab_size}"
        )

    verification = None
    if cfg.verify_dataset:
        result = verify_dataset(cfg.corpus_dir, expected_dataset_hash=cfg.expected_dataset_hash)
        verification = result.to_dict()
        if not result.ok:
            details = "; ".join(result.split_errors + result.missing_shards)
            raise ValueError(f"dataset integrity verification failed: {details or 'unknown error'}")
        dataset_hash = result.recorded_dataset_hash
    else:
        dataset_hash = _manifest_dataset_hash(cfg.corpus_dir)
        if cfg.expected_dataset_hash is not None and dataset_hash != cfg.expected_dataset_hash:
            raise ValueError("dataset_hash does not match expected_dataset_hash")

    train_data = PretrainingData(
        cfg.corpus_dir,
        tokenizer,
        split="train",
        block_size=cfg.block_size,
        max_tokens=cfg.max_tokens,
    )
    validation_data = PretrainingData(
        cfg.corpus_dir,
        tokenizer,
        split="validation",
        block_size=cfg.block_size,
        max_tokens=cfg.max_tokens,
    )
    if train_data.n_blocks == 0:
        raise ValueError(f"train split is empty in {cfg.corpus_dir}")
    if cfg.require_validation and validation_data.n_blocks == 0:
        raise ValueError(f"validation split is empty in {cfg.corpus_dir}")
    return TrainingInputs(
        model_config=model_config,
        tokenizer=tokenizer,
        train_data=train_data,
        validation_data=validation_data,
        dataset_hash=dataset_hash,
        dataset_verification=verification,
    )

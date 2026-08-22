"""CLI entry point for pretraining Silverwing Decoder V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Silverwing Decoder V2")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-clean-repo-check", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    section = raw.get("training", raw)
    overrides = {
        "model_config_path": args.model_config,
        "corpus_dir": args.corpus_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "resume_from": args.resume_from,
        "init_from": args.init_from,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "block_size": args.block_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
    }
    merged = {**section, **{k: v for k, v in overrides.items() if v is not None}}
    cfg = TrainConfig.from_dict(merged)
    if args.no_clean_repo_check:
        cfg = TrainConfig.from_dict({**cfg.to_dict(), "require_clean_repo": False})
    if args.amp:
        cfg = TrainConfig.from_dict({**cfg.to_dict(), "amp": True})

    report = train(cfg)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

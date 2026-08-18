"""Supervised fine-tuning CLI (M11).

Run the SFT stage over a pretrained checkpoint:
    python scripts/train_sft.py --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.sft import SftConfig, train_sft  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning (M11)")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-clean-repo-check", action="store_true")
    args = parser.parse_args()

    cfg = SftConfig.from_yaml(args.config)
    overrides: dict = {}
    if args.init_from:
        overrides["init_from"] = args.init_from
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)
    if args.no_clean_repo_check:
        from dataclasses import replace
        cfg = replace(cfg, require_clean_repo=False)

    report = train_sft(cfg)
    print(report["final_checkpoint"])
    print(report["final_eval_loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

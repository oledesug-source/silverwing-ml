"""Supervised fine-tuning CLI (M11).

Run the SFT stage over a pretrained checkpoint:
    python scripts/train_sft.py --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-clean-repo-check", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    section = raw.get("sft", raw)
    overrides: dict = {}
    if args.init_from:
        overrides["init_from"] = args.init_from
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device
    if args.no_clean_repo_check:
        overrides["require_clean_repo"] = False
    if args.amp:
        overrides["amp"] = True
    merged = {**section, **{k: v for k, v in overrides.items() if v is not None}}
    cfg = SftConfig.from_dict(merged)

    report = train_sft(cfg)
    print(report["final_checkpoint"])
    print(report["final_eval_loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

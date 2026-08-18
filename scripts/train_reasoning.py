"""Reasoning chain-of-thought training CLI (M13).

Run the reasoning training stage over a pretrained or SFT/DPO checkpoint:
    python scripts/train_reasoning.py --config configs/reasoning.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.reasoning import ReasoningConfig, train_reasoning


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reasoning chain-of-thought training (M13)"
    )
    parser.add_argument("--config", default="configs/reasoning.yaml")
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = ReasoningConfig.from_yaml(args.config)
    overrides: dict = {}
    if args.init_from:
        overrides["init_from"] = args.init_from
    if args.dataset_path:
        overrides["dataset_path"] = args.dataset_path
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

    report = train_reasoning(cfg)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

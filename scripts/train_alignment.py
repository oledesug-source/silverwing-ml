"""DPO alignment CLI (M12).

Run the DPO alignment stage over a pretrained checkpoint:
    python scripts/train_alignment.py --config configs/alignment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.alignment import AlignmentConfig, train_alignment


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO preference alignment (M12)")
    parser.add_argument("--config", default="configs/alignment.yaml")
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dpo-beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-clean-repo-check", action="store_true")
    args = parser.parse_args()

    cfg = AlignmentConfig.from_yaml(args.config)
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
    if args.dpo_beta is not None:
        overrides["dpo_beta"] = args.dpo_beta
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

    report = train_alignment(cfg)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

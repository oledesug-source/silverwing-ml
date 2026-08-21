"""CLI for curriculum learning (M14).

Runs progressive multi-stage SFT training:
    python scripts/train_curriculum.py --config configs/curriculum.yaml --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.curriculum import CurriculumConfig, train_curriculum  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Curriculum learning (M14)")
    parser.add_argument("--config", default="configs/curriculum.yaml")
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-clean-repo-check", action="store_true")
    parser.add_argument("--max-steps-per-stage", type=int, default=None)
    args = parser.parse_args()

    cfg = CurriculumConfig.from_yaml(args.config)
    overrides = {}
    if args.init_from:
        overrides["init_from"] = args.init_from
    if args.device:
        overrides["device"] = args.device
    if args.no_clean_repo_check:
        overrides["require_clean_repo"] = False
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)

    if args.checkpoint_dir:
        cfg = replace(
            cfg,
            stages=[replace(s, checkpoint_dir=args.checkpoint_dir) for s in cfg.stages],
        )

    if args.max_steps_per_stage:
        cfg = replace(
            cfg,
            stages=[replace(s, max_steps=args.max_steps_per_stage) for s in cfg.stages],
        )

    report = train_curriculum(cfg)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

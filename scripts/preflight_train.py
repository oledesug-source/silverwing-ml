"""Validate immutable M10 training inputs without allocating or training a model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.training import TrainConfig, preflight_train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight a Silverwing real-pretraining run")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-dataset-verify", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    cfg = replace(
        cfg,
        corpus_dir=args.corpus_dir or cfg.corpus_dir,
        tokenizer_dir=args.tokenizer_dir or cfg.tokenizer_dir,
        model_config_path=args.model_config or cfg.model_config_path,
        device=args.device or cfg.device,
        verify_dataset=False if args.no_dataset_verify else cfg.verify_dataset,
    )
    inputs = preflight_train(cfg)
    print(json.dumps(inputs.report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

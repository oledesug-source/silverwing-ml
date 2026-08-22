#!/usr/bin/env python3
"""Validate all YAML files in ``configs/`` against Pydantic schemas.

Files with a registered schema (training*, model*, tokenizer*) are validated
strictly; unregistered files are reported as unchecked.  Exits non-zero if
any registered config fails validation — wire this into CI or preflight to
catch bad configs before a training run starts.

Usage::

    python scripts/validate_configs.py [--configs-dir configs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from foundation.config.schemas import load_config, schema_for  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Directory containing YAML configs (default: configs)",
    )
    args = parser.parse_args(argv)

    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_dir():
        print(f"error: configs dir not found: {configs_dir}", file=sys.stderr)
        return 2

    yaml_files = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))
    if not yaml_files:
        print(f"error: no YAML files found in {configs_dir}", file=sys.stderr)
        return 2

    failures = 0
    checked = 0
    for path in yaml_files:
        if schema_for(path) is None:
            print(f"SKIP  {path} (no schema registered)")
            continue
        try:
            # Parse first so pure YAML syntax errors get a clear message even
            # before schema validation runs.
            yaml.safe_load(path.read_text(encoding="utf-8"))
            load_config(path)
            print(f"OK    {path}")
            checked += 1
        except (ValueError, OSError) as exc:
            print(f"FAIL  {path}: {exc}", file=sys.stderr)
            failures += 1

    summary = f"{checked} validated, {failures} failed"
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

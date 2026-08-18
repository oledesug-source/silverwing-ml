"""Package Silverwing-ML for Google Colab upload.

Creates silverwing-colab.zip with source code + data, excluding large
checkpoints and local venv. Run from the project root:

    python scripts/package_for_colab.py

Output: silverwing-colab.zip in the project root.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "silverwing-colab.zip"

# Directories/files to include
INCLUDE_DIRS = [
    "foundation",
    "intelligence",
    "benchmarks",
    "serving",
    "scripts",
    "configs",
    "tests",
]

INCLUDE_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "README.md",
]

# Data files (needed for training) — include small ones, exclude checkpoints
INCLUDE_DATA = [
    "experiments/corpus/manifest.json",
    "experiments/corpus/pipeline_report.json",
    "experiments/tokenizer",
    "experiments/sft/sft-v1.jsonl",
    "experiments/sft/sft-v1-combined.jsonl",
    "experiments/reasoning/reasoning-v1.jsonl",
    "experiments/reasoning/reasoning-v1-sft.jsonl",
    "experiments/reasoning/reasoning-v1.manifest.json",
    "experiments/math_benchmarks",
    "experiments/eval",
]

# Excludes
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "*.egg-info",
    ".ruff_cache",
    ".pytest_cache",
    ".idea",
]


def main() -> int:
    if OUT.exists():
        OUT.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "Silverwing-ML"
        tmp.mkdir()

        # Copy source directories
        for d in INCLUDE_DIRS:
            src = ROOT / d
            if src.exists():
                dst = tmp / d
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS))
                print(f"  copied {d}/")

        # Copy individual files
        for f in INCLUDE_FILES:
            src = ROOT / f
            if src.exists():
                shutil.copy2(src, tmp / f)
                print(f"  copied {f}")

        # Copy data files
        for item in INCLUDE_DATA:
            src = ROOT / item
            if src.is_dir():
                dst = tmp / item
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS))
                print(f"  copied {item}/")
            elif src.is_file():
                dst = tmp / item
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  copied {item}")

        # Copy corpus data (train/val/test jsonl files)
        corpus_dir = ROOT / "experiments" / "corpus"
        for f in corpus_dir.glob("*.jsonl"):
            dst = tmp / "experiments" / "corpus" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            print(f"  copied experiments/corpus/{f.name}")

        # Copy colab notebook
        nb = ROOT / "colab_train.ipynb"
        if nb.exists():
            shutil.copy2(nb, tmp / "colab_train.ipynb")
            print(f"  copied colab_train.ipynb")

        # Create zip
        shutil.make_archive(str(OUT.with_suffix("")), "zip", ROOT.parent, "Silverwing-ML")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nCreated {OUT.name} ({size_mb:.1f} MB)")
    print(f"Upload to Google Colab: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

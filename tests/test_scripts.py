"""Black-box coverage for the foundation operational scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_and_verify_corpus_scripts(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "Silverwing trains a language model from transparent, traceable data. " * 12,
        encoding="utf-8",
    )
    output_dir = tmp_path / "corpus"

    build = _run(
        "scripts/build_corpus.py",
        "--source",
        f"local={source}",
        "--output-dir",
        str(output_dir),
    )
    assert build.returncode == 0, build.stderr
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "pipeline_report.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    verify = _run(
        "scripts/verify_corpus.py",
        "--output-dir",
        str(output_dir),
        "--expected-hash",
        manifest["dataset_hash"],
        "--json",
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["ok"] is True


def test_build_corpus_refuses_empty_release(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("too short", encoding="utf-8")
    output_dir = tmp_path / "corpus"

    result = _run(
        "scripts/build_corpus.py",
        "--source",
        f"local={source}",
        "--output-dir",
        str(output_dir),
    )
    assert result.returncode == 1
    assert "refusing to release an empty corpus" in result.stderr

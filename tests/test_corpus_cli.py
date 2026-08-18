"""Tests for the corpus build and verify CLI scripts (M02/M03 entry points)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    "The history of the industrial revolution is a story of machines, steam power, and the "
    "transformation of agriculture into factory production across Europe and North America.",
    "Neural networks learn patterns from data by adjusting many small parameters during training, "
    "and this process of learning from examples is called supervised learning in the research literature.",
    "The solar system contains eight major planets that orbit the sun in nearly circular paths, and the "
    "study of their motion is one of the oldest branches of the physical sciences.",
    "Economic growth depends on investment in capital, education, and technology, which together raise "
    "the productive capacity of a nation over long periods of history.",
]

EMPTY_CORPUS_CONFIG = """
corpus:
  seed: 42
  chunking:
    max_tokens: 1024
    overlap_tokens: 128
  splits:
    train: 0.96
    validation: 0.02
    test: 0.02
  filtering:
    allowed_languages: [en]
    quality:
      min_chars: 1
      min_words: 1
      min_alpha_ratio: 0.0
      max_punct_ratio: 1.0
      max_url_ratio: 1.0
      max_email_count: 100
      max_duplicate_line_ratio: 1.0
  deduplication:
    num_hashes: 128
    bands: 16
    similarity_threshold: 0.85
    normalize: true
  contamination:
    ngram_n: 8
    threshold: 0.6
  integrity:
    algorithm: sha256
    dataset_hash: recomputed
  output_dir: experiments/corpus
  sources: []
"""


def _run_script(script: str, *args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=180,
    )


def _write_relaxed_config(root: Path) -> Path:
    path = root / "corpus.yaml"
    path.write_text(EMPTY_CORPUS_CONFIG, encoding="utf-8")
    return path


def _write_sources(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"id": i, "text": text}) for i, text in enumerate(DOCS)
    ]
    (raw / "sources.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw


class TestBuildCorpus:
    def test_build_produces_verified_dataset(self, tmp_path: Path) -> None:
        config = _write_relaxed_config(tmp_path)
        raw = _write_sources(tmp_path)
        output = tmp_path / "corpus"
        result = _run_script(
            "build_corpus.py",
            "--config", str(config),
            "--output-dir", str(output),
            "--source", f"raw={raw / 'sources.jsonl'}",
        )
        assert result.returncode == 0, result.stderr
        assert re.search(r"dataset_hash=[0-9a-f]{64}", result.stdout)
        assert (output / "manifest.json").exists()
        assert (output / "pipeline_report.json").exists()
        assert any((output).glob("train.*.jsonl"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        total = sum(info["records"] for info in manifest["splits"].values())
        assert total > 0
        report = json.loads((output / "pipeline_report.json").read_text(encoding="utf-8"))
        assert re.fullmatch(r"[0-9a-f]{40}", report["git_commit"] or "")
        assert re.fullmatch(r"[0-9a-f]{64}", report["config_digest"])
        assert report["sources"][0]["source_id"] == "raw"

    def test_empty_output_is_refused(self, tmp_path: Path) -> None:
        config = _write_relaxed_config(tmp_path)
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = _run_script(
            "build_corpus.py",
            "--config", str(config),
            "--output-dir", str(tmp_path / "out"),
            "--source", f"empty={empty}",
        )
        assert result.returncode != 0
        assert "empty corpus" in result.stderr

    def test_allow_empty_flag_bypasses_guard(self, tmp_path: Path) -> None:
        config = _write_relaxed_config(tmp_path)
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = _run_script(
            "build_corpus.py",
            "--config", str(config),
            "--output-dir", str(tmp_path / "out"),
            "--source", f"empty={empty}",
            "--allow-empty",
        )
        assert result.returncode == 0, result.stderr
        assert "records: total=0" in result.stdout

    def test_missing_source_is_rejected(self, tmp_path: Path) -> None:
        config = _write_relaxed_config(tmp_path)
        result = _run_script("build_corpus.py", "--config", str(config))
        assert result.returncode != 0
        assert "no sources" in result.stderr


class TestVerifyCorpus:
    def _build(self, tmp_path: Path) -> Path:
        config = _write_relaxed_config(tmp_path)
        raw = _write_sources(tmp_path)
        output = tmp_path / "corpus"
        result = _run_script(
            "build_corpus.py",
            "--config", str(config),
            "--output-dir", str(output),
            "--source", f"raw={raw / 'sources.jsonl'}",
        )
        assert result.returncode == 0, result.stderr
        return output

    def test_verify_ok(self, tmp_path: Path) -> None:
        output = self._build(tmp_path)
        result = _run_script("verify_corpus.py", "--output-dir", str(output))
        assert result.returncode == 0, result.stderr
        assert "ok: True" in result.stdout

    def test_verify_detects_corruption(self, tmp_path: Path) -> None:
        output = self._build(tmp_path)
        shard = next(output.glob("train.*.jsonl"))
        shard.write_bytes(shard.read_bytes() + b"CORRUPT")
        result = _run_script("verify_corpus.py", "--output-dir", str(output))
        assert result.returncode == 1
        assert "dataset_hash mismatch" in result.stdout

    def test_verify_rejects_wrong_pinned_hash(self, tmp_path: Path) -> None:
        output = self._build(tmp_path)
        result = _run_script(
            "verify_corpus.py",
            "--output-dir", str(output),
            "--expected-hash", "0" * 64,
        )
        assert result.returncode == 1
        assert "pinned expected hash" in result.stdout

    def test_verify_json_output(self, tmp_path: Path) -> None:
        output = self._build(tmp_path)
        result = _run_script("verify_corpus.py", "--output-dir", str(output), "--json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["recorded_dataset_hash"]

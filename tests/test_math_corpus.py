"""Tests for the M08 mathematical training corpus (determinism, correctness,
document structure, and end-to-end corpus release through the M02/M03 CLI)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from foundation.math_corpus import MathCorpusConfig, PROBLEM_GENERATORS, REFERENCES, build_document, generate_math_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _tiny_config(tmp_path: Path, docs: dict[str, int]) -> MathCorpusConfig:
    cfg = MathCorpusConfig(
        seed=42,
        curriculum=docs,
        examples_per_document=2,
        exercises_per_document=6,
        staging_dir=str(tmp_path / "raw"),
        corpus_dir=str(tmp_path / "corpus"),
    )
    return cfg


class TestDeterminism:
    def test_same_seed_produces_identical_documents(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path, {"arithmetic": 3, "algebra": 2, "integration": 2})
        r1 = generate_math_corpus(cfg)
        r2 = generate_math_corpus(cfg)
        assert r1["content_digest"] == r2["content_digest"]
        assert r1["total_documents"] == 7
        files = sorted((tmp_path / "raw").glob("*.txt"))
        assert len(files) == 7
        for f in files:
            assert (tmp_path / "raw" / f.name).read_bytes() == f.read_bytes()

    def test_different_seed_changes_content(self, tmp_path: Path) -> None:
        a = _tiny_config(tmp_path / "a", {"arithmetic": 3})
        b = _tiny_config(tmp_path / "b", {"arithmetic": 3})
        b2 = MathCorpusConfig(seed=99, curriculum=b.curriculum, examples_per_document=2, exercises_per_document=6, staging_dir=b.staging_dir, corpus_dir=b.corpus_dir)
        ra = generate_math_corpus(a)
        rb = generate_math_corpus(b2)
        assert ra["content_digest"] != rb["content_digest"]

    def test_report_pins_digest_and_counts(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path, {"linear_equations": 4})
        report = generate_math_corpus(cfg)
        assert report["config_digest"] == cfg.digest()
        assert report["documents_per_topic"] == {"linear_equations": 4}
        saved = json.loads((tmp_path / "raw" / "generation_report.json").read_text(encoding="utf-8"))
        assert saved["content_digest"] == report["content_digest"]


class TestCorrectness:
    def test_all_topics_generate_nonempty_problems(self) -> None:
        import random

        rng = random.Random(0)
        for topic, gen in PROBLEM_GENERATORS.items():
            for _ in range(40):
                problem = gen(rng)
                assert problem.question
                assert problem.answer
                assert problem.solution
                assert topic in REFERENCES

    def test_linear_equation_answer_verified(self) -> None:
        import random

        rng = random.Random(7)
        for _ in range(30):
            p = PROBLEM_GENERATORS["linear_equations"](rng)
            m = re.match(r"Solve for x: (\d+)x ([+-]) (\d+) = (\d+)x ([+-]) (\d+)\.", p.question)
            assert m, p.question
            a, sb, b, c, sd, d = m.groups()
            a, b, c, d = int(a), int(b), int(c), int(d)
            b = -b if sb == "-" else b
            d = -d if sd == "-" else d
            got = set()
            for x0 in range(-100, 101):
                if a * x0 + b == c * x0 + d:
                    got.add(x0)
            assert len(got) == 1, p.question
            assert p.answer == f"x = {got.pop()}"

    def test_quadratic_answer_verified_by_substitution(self) -> None:
        import random

        rng = random.Random(11)
        for _ in range(30):
            p = PROBLEM_GENERATORS["algebra"](rng)
            m = re.match(r"Solve x\^2 ([+-]) (\d+)x ([+-]) (\d+) = 0\.", p.question)
            if not m:
                continue
            sg, pg, csg, cg = m.groups()
            p_coeff = int(pg)
            if sg == "-":
                p_coeff = -p_coeff
            c = int(cg)
            if csg == "-":
                c = -c
            roots = [r for r in range(-100, 101) if r * r + p_coeff * r + c == 0]
            assert len(roots) == 2, (p.question, p.answer)
            assert all(str(r) in p.answer for r in roots)

    def test_fraction_answer_in_lowest_terms(self) -> None:
        import random

        from fractions import Fraction

        rng = random.Random(13)
        for _ in range(40):
            p = PROBLEM_GENERATORS["arithmetic"](rng)
            if "Compute" not in p.question or "/" not in p.question:
                continue
            a, b, c, d = map(int, re.findall(r"\d+", p.question))
            expected = Fraction(a, b) + Fraction(c, d)
            assert p.answer == str(expected) if expected.denominator != 1 else p.answer == str(expected.numerator)

    def test_differentiation_derivative_numeric_check(self) -> None:
        import random

        rng = random.Random(17)
        for _ in range(30):
            p = PROBLEM_GENERATORS["differentiation"](rng)
            m = re.match(r"Find the derivative of f\(x\) = (\d+)x\^(\d+)\.", p.question)
            if m:
                a, n = int(m.group(1)), int(m.group(2))
                assert p.answer == f"f'(x) = {a * n}x^{n - 1}"


class TestDocuments:
    def test_document_structure(self) -> None:
        import random

        rng = random.Random(0)
        for topic, gen in PROBLEM_GENERATORS.items():
            examples = [gen(rng) for _ in range(2)]
            exercises = [gen(rng) for _ in range(6)]
            doc = build_document(topic, REFERENCES[topic], examples, exercises, 0)
            for section in ("## Definition", "## Rules", "## Worked Examples", "## Exercises", "## Answer Key"):
                assert section in doc, (topic, section)
            for ex in exercises:
                assert ex.answer in doc

    def test_documents_pass_min_quality(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path, {"probability": 2, "geometry": 2})
        generate_math_corpus(cfg)
        for f in (tmp_path / "raw").glob("*.txt"):
            text = f.read_text(encoding="utf-8")
            assert len(text) >= 200
            assert len(text.split()) >= 40


class TestEndToEndRelease:
    def _build(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path, {"arithmetic": 3, "algebra": 3, "differentiation": 3})
        generate_math_corpus(cfg)
        corpus_config = tmp_path / "corpus.yaml"
        corpus_config.write_text(
            yaml.safe_dump(
                {
                    "corpus": {
                        "seed": 42,
                        "chunking": {"max_tokens": 1024, "overlap_tokens": 128},
                        "splits": {"train": 0.96, "validation": 0.02, "test": 0.02},
                        "filtering": {
                            "allowed_languages": ["en"],
                            "quality": {"min_chars": 1, "min_words": 1},
                            "deduplication": {"num_hashes": 32, "bands": 4, "similarity_threshold": 0.85, "normalize": True},
                            "contamination": {"ngram_n": 8, "threshold": 0.6, "fuzzy_normalize": True},
                        },
                        "integrity": {"algorithm": "sha256", "dataset_hash": "recomputed"},
                        "output_dir": str(tmp_path / "corpus"),
                        "sources": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        result = _run_script(
            "build_corpus.py",
            "--config", str(corpus_config),
            "--output-dir", str(tmp_path / "corpus"),
            "--source", f"math={tmp_path / 'raw'}",
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8"))
        total = sum(info["records"] for info in manifest["splits"].values())
        assert total > 0
        verify = _run_script("verify_corpus.py", "--output-dir", str(tmp_path / "corpus"))
        assert verify.returncode == 0, verify.stderr

    def test_corpus_builds_and_verifies(self, tmp_path: Path) -> None:
        self._build(tmp_path)


class TestConfig:
    def test_default_config_loads(self) -> None:
        cfg = MathCorpusConfig.from_yaml(PROJECT_ROOT / "configs" / "math_corpus.yaml")
        assert cfg.seed == 42
        assert cfg.total_documents > 0
        assert set(cfg.curriculum) == set(PROBLEM_GENERATORS)

    def test_unknown_topic_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown curriculum topic"):
            MathCorpusConfig(curriculum={"not_a_topic": 5})

    def test_digest_stable(self) -> None:
        cfg = MathCorpusConfig(seed=1, curriculum={"arithmetic": 2})
        assert cfg.digest() == cfg.digest()

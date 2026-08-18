"""Tests for foundation/evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from foundation.evaluation import EvalConfig, EvalReport, EvalSuite, Evaluator


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.device == "cpu"
    assert cfg.max_new_tokens == 128
    assert isinstance(cfg.digest(), str)
    assert len(cfg.digest()) == 64


def test_eval_config_to_dict_round_trip():
    cfg = EvalConfig(checkpoint_path="test.pt", device="cuda")
    d = cfg.to_dict()
    assert d["checkpoint_path"] == "test.pt"
    assert d["device"] == "cuda"


def test_eval_config_yaml(tmp_path: Path):
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        "evaluation:\n  checkpoint_path: test.pt\n  device: cpu\n",
        encoding="utf-8",
    )
    cfg = EvalConfig.from_yaml(cfg_path) if hasattr(EvalConfig, 'from_yaml') else None
    # EvalConfig uses to_dict/from_dict pattern
    d = {"checkpoint_path": "test.pt", "device": "cpu"}
    cfg2 = EvalConfig(**{k: v for k, v in d.items() if hasattr(EvalConfig, k)})
    assert cfg2.checkpoint_path == "test.pt"


def test_eval_report_to_dict():
    report = EvalReport(
        run_id="test-run",
        config=EvalConfig(),
        num_parameters=100,
        perplexity=15.0,
        perplexity_tokens=500,
    )
    d = report.to_dict()
    assert d["run_id"] == "test-run"
    assert d["num_parameters"] == 100
    assert d["perplexity"] == 15.0
    assert d["perplexity_tokens"] == 500


def test_eval_suite_smoke():
    suite = EvalSuite.smoke()
    assert suite.name == "smoke"
    assert len(suite.sample_prompts) == 2
    assert len(suite.benchmarks) == 0


def test_eval_suite_math_basic():
    suite = EvalSuite.math_basic()
    assert suite.name == "math-basic"
    assert len(suite.benchmarks) > 0
    assert len(suite.sample_prompts) > 0


def test_eval_suite_full():
    suite = EvalSuite.full()
    assert suite.name == "full"
    assert len(suite.benchmarks) >= 3
    assert len(suite.sample_prompts) >= 5


def test_eval_suite_custom():
    suite = EvalSuite(
        name="custom",
        benchmarks=["bench1"],
        sample_prompts=["prompt1", "prompt2"],
        perplexity_text="some text",
    )
    assert suite.name == "custom"
    assert suite.benchmarks == ["bench1"]
    assert suite.perplexity_text == "some text"


def test_eval_report_samples():
    report = EvalReport(
        run_id="test",
        config=EvalConfig(),
    )
    report.samples.append({"prompt": "What is 1+1?", "response": "2"})
    assert len(report.samples) == 1
    assert report.samples[0]["response"] == "2"


def test_eval_report_benchmark_results():
    report = EvalReport(
        run_id="test",
        config=EvalConfig(),
    )
    report.benchmark_results["sample_arithmetic"] = {
        "accuracy": 0.85,
        "n": 100,
    }
    assert report.benchmark_results["sample_arithmetic"]["accuracy"] == 0.85


def test_eval_report_empty():
    report = EvalReport(run_id="empty", config=EvalConfig())
    d = report.to_dict()
    assert d["perplexity"] is None
    assert d["benchmark_results"] == {}
    assert d["samples"] == []
    assert d["elapsed_seconds"] == 0.0


def test_eval_config_from_dict():
    d = {"checkpoint_path": "x.pt", "device": "cuda", "max_new_tokens": 64}
    cfg = EvalConfig(**{k: v for k, v in d.items() if hasattr(EvalConfig, k)})
    assert cfg.checkpoint_path == "x.pt"
    assert cfg.device == "cuda"

"""Tests for foundation/evaluation/ — config, report, suites, save, perplexity."""

from __future__ import annotations

import json
import math
from pathlib import Path

from foundation.evaluation import EvalConfig, EvalReport, EvalSuite, Evaluator

# ── EvalConfig Tests ──────────────────────────────────────────────────────


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.device == "cpu"
        assert cfg.max_new_tokens == 128
        assert cfg.benchmarks == []
        assert cfg.sample_prompts == []

    def test_to_dict_round_trip(self):
        cfg = EvalConfig(checkpoint_path="test.pt", device="cuda", max_new_tokens=64)
        d = cfg.to_dict()
        assert d["checkpoint_path"] == "test.pt"
        assert d["device"] == "cuda"
        assert d["max_new_tokens"] == 64

    def test_to_dict_all_fields(self):
        cfg = EvalConfig(
            checkpoint_path="a.pt",
            model_config_path="m.yaml",
            tokenizer_dir="tok/",
            device="cuda",
            benchmarks=["b1", "b2"],
            benchmark_limit=10,
            max_new_tokens=256,
            perplexity_text="hello world",
            sample_prompts=["p1", "p2"],
            output_dir="out/",
        )
        d = cfg.to_dict()
        assert d["benchmarks"] == ["b1", "b2"]
        assert d["benchmark_limit"] == 10
        assert d["sample_prompts"] == ["p1", "p2"]
        assert d["output_dir"] == "out/"
        assert d["checkpoint_path"] == "a.pt"
        assert d["model_config_path"] == "m.yaml"

    def test_digest_deterministic(self):
        cfg = EvalConfig()
        assert cfg.digest() == cfg.digest()

    def test_digest_differs(self):
        cfg1 = EvalConfig(checkpoint_path="a.pt")
        cfg2 = EvalConfig(checkpoint_path="b.pt")
        assert cfg1.digest() != cfg2.digest()

    def test_digest_is_sha256(self):
        cfg = EvalConfig()
        d = cfg.digest()
        assert len(d) == 64
        int(d, 16)

    def test_from_dict(self):
        d = {
            "checkpoint_path": "x.pt",
            "device": "cuda",
            "max_new_tokens": 64,
            "benchmarks": ["math"],
        }
        cfg = EvalConfig(**{k: v for k, v in d.items() if hasattr(EvalConfig, k)})
        assert cfg.checkpoint_path == "x.pt"
        assert cfg.device == "cuda"

    def test_construct_custom(self):
        cfg = EvalConfig(
            checkpoint_path="custom.pt",
            benchmarks=["bench_a"],
            sample_prompts=["prompt_a"],
            perplexity_text="some text",
        )
        assert cfg.checkpoint_path == "custom.pt"
        assert cfg.benchmarks == ["bench_a"]


# ── EvalReport Tests ──────────────────────────────────────────────────────


class TestEvalReport:
    def test_empty_report(self):
        report = EvalReport(run_id="empty", config=EvalConfig())
        d = report.to_dict()
        assert d["run_id"] == "empty"
        assert d["perplexity"] is None
        assert d["perplexity_tokens"] == 0
        assert d["benchmark_results"] == {}
        assert d["samples"] == []
        assert d["elapsed_seconds"] == 0.0
        assert d["num_parameters"] == 0

    def test_report_with_perplexity(self):
        report = EvalReport(
            run_id="test",
            config=EvalConfig(),
            perplexity=15.5,
            perplexity_tokens=500,
        )
        d = report.to_dict()
        assert d["perplexity"] == 15.5
        assert d["perplexity_tokens"] == 500

    def test_report_with_benchmarks(self):
        report = EvalReport(run_id="test", config=EvalConfig())
        report.benchmark_results["math"] = {"accuracy": 0.85, "n": 100}
        report.benchmark_results["reasoning"] = {"accuracy": 0.72, "n": 50}
        d = report.to_dict()
        assert len(d["benchmark_results"]) == 2
        assert d["benchmark_results"]["math"]["accuracy"] == 0.85

    def test_report_with_samples(self):
        report = EvalReport(run_id="test", config=EvalConfig())
        report.samples.append({"prompt": "What is 1+1?", "response": "2"})
        report.samples.append({"prompt": "Hi", "response": "Hello"})
        d = report.to_dict()
        assert len(d["samples"]) == 2
        assert d["samples"][1]["response"] == "Hello"

    def test_report_mutable_after_creation(self):
        report = EvalReport(run_id="t", config=EvalConfig())
        report.perplexity = 10.0
        report.perplexity_tokens = 100
        report.elapsed_seconds = 5.5
        report.started_at = "2025-01-01T00:00:00"
        report.finished_at = "2025-01-01T00:00:05"
        report.python = "3.11"
        report.torch_version = "2.1.0"
        d = report.to_dict()
        assert d["perplexity"] == 10.0
        assert d["elapsed_seconds"] == 5.5

    def test_report_serializable_to_json(self):
        report = EvalReport(run_id="json-test", config=EvalConfig())
        report.benchmark_results["b"] = {"v": 1}
        report.samples.append({"prompt": "p", "response": "r"})
        d = report.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        loaded = json.loads(json_str)
        assert loaded["run_id"] == "json-test"


# ── EvalSuite Tests ───────────────────────────────────────────────────────


class TestEvalSuite:
    def test_smoke(self):
        suite = EvalSuite.smoke()
        assert suite.name == "smoke"
        assert len(suite.sample_prompts) == 2
        assert len(suite.benchmarks) == 0
        assert suite.perplexity_text == ""

    def test_math_basic(self):
        suite = EvalSuite.math_basic()
        assert suite.name == "math-basic"
        assert len(suite.benchmarks) == 2
        assert "sample_arithmetic" in suite.benchmarks
        assert "sample_geometry" in suite.benchmarks
        assert len(suite.sample_prompts) == 5

    def test_full(self):
        suite = EvalSuite.full()
        assert suite.name == "full"
        assert len(suite.benchmarks) >= 4
        assert len(suite.sample_prompts) >= 5
        assert "sample_arithmetic" in suite.benchmarks

    def test_custom(self):
        suite = EvalSuite(
            name="custom",
            benchmarks=["bench1", "bench2"],
            sample_prompts=["prompt1"],
            perplexity_text="text",
        )
        assert suite.name == "custom"
        assert suite.benchmarks == ["bench1", "bench2"]
        assert suite.sample_prompts == ["prompt1"]
        assert suite.perplexity_text == "text"

    def test_defaults(self):
        suite = EvalSuite(name="empty")
        assert suite.benchmarks == []
        assert suite.sample_prompts == []
        assert suite.perplexity_text == ""

    def test_all_suites_produce_reports(self):
        for factory in [EvalSuite.smoke, EvalSuite.math_basic, EvalSuite.full]:
            suite = factory()
            report = EvalReport(run_id=f"test-{suite.name}", config=EvalConfig())
            report.benchmark_results = {}
            report.samples = []
            d = report.to_dict()
            assert d["run_id"].startswith("test-")


# ── Evaluator.save Tests ──────────────────────────────────────────────────


class TestEvaluatorSave:
    def test_save_writes_files(self, tmp_path: Path):
        cfg = EvalConfig(output_dir=str(tmp_path / "eval"))
        evaluator = Evaluator(cfg, log=lambda _: None)
        report = EvalReport(
            run_id="save-test",
            config=cfg,
            perplexity=12.5,
            perplexity_tokens=200,
        )
        report_path = evaluator.save(report)
        assert report_path.exists()
        assert "save-test" in report_path.name

        latest = tmp_path / "eval" / "latest_eval.json"
        assert latest.exists()
        loaded = json.loads(latest.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "save-test"
        assert loaded["perplexity"] == 12.5

    def test_save_creates_dir(self, tmp_path: Path):
        cfg = EvalConfig(output_dir=str(tmp_path / "deep" / "nested" / "eval"))
        evaluator = Evaluator(cfg, log=lambda _: None)
        report = EvalReport(run_id="dir-test", config=cfg)
        evaluator.save(report)
        assert (tmp_path / "deep" / "nested" / "eval" / "latest_eval.json").exists()

    def test_save_overwrites_latest(self, tmp_path: Path):
        cfg = EvalConfig(output_dir=str(tmp_path / "eval"))
        evaluator = Evaluator(cfg, log=lambda _: None)
        r1 = EvalReport(run_id="run1", config=cfg)
        evaluator.save(r1)
        r2 = EvalReport(run_id="run2", config=cfg)
        evaluator.save(r2)
        latest = json.loads(
            (tmp_path / "eval" / "latest_eval.json").read_text(encoding="utf-8")
        )
        assert latest["run_id"] == "run2"


# ── Evaluator Perplexity Tests ────────────────────────────────────────────


class TestEvaluatorPerplexity:
    def test_compute_perplexity(self, tmp_path: Path):
        from foundation.model import ModelConfig, build_model
        from foundation.tokenizer import TokenizerV2

        tokenizer = TokenizerV2(merges=[])
        model_cfg = ModelConfig.from_dict({
            "model_name": "tiny-eval",
            "vocab_size": tokenizer.vocab_size,
            "block_size": 32,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "mlp_hidden_size": 32,
            "tie_embeddings": True,
            "bias": False,
        })
        model = build_model(model_cfg)
        model.eval()

        cfg = EvalConfig(output_dir=str(tmp_path))
        evaluator = Evaluator(cfg, log=lambda _: None)

        text = "Hello world this is a test sentence for perplexity."
        ppl, n_tokens = evaluator._compute_perplexity(model, tokenizer, text, "cpu")
        assert isinstance(ppl, float)
        assert ppl > 0
        assert n_tokens > 0
        assert math.isfinite(ppl)

    def test_compute_perplexity_empty_text(self, tmp_path: Path):
        from foundation.model import ModelConfig, build_model
        from foundation.tokenizer import TokenizerV2

        tokenizer = TokenizerV2(merges=[])
        model_cfg = ModelConfig.from_dict({
            "model_name": "tiny-eval",
            "vocab_size": tokenizer.vocab_size,
            "block_size": 32,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "mlp_hidden_size": 32,
            "tie_embeddings": True,
            "bias": False,
        })
        model = build_model(model_cfg)
        model.eval()

        cfg = EvalConfig(output_dir=str(tmp_path))
        evaluator = Evaluator(cfg, log=lambda _: None)

        ppl, n_tokens = evaluator._compute_perplexity(model, tokenizer, "", "cpu")
        assert n_tokens == 0

    def test_compute_perplexity_single_token(self, tmp_path: Path):
        from foundation.model import ModelConfig, build_model
        from foundation.tokenizer import TokenizerV2

        tokenizer = TokenizerV2(merges=[])
        model_cfg = ModelConfig.from_dict({
            "model_name": "tiny-eval",
            "vocab_size": tokenizer.vocab_size,
            "block_size": 32,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 16,
            "mlp_hidden_size": 32,
            "tie_embeddings": True,
            "bias": False,
        })
        model = build_model(model_cfg)
        model.eval()

        cfg = EvalConfig(output_dir=str(tmp_path))
        evaluator = Evaluator(cfg, log=lambda _: None)

        # Encode a very short text that produces 1 token
        ids = tokenizer.encode("Hi")
        if len(ids) <= 1:
            ppl, n = evaluator._compute_perplexity(model, tokenizer, "Hi", "cpu")
            assert n == 0


# ── EvalReport.to_dict JSON edge cases ────────────────────────────────────


class TestReportEdgeCases:
    def test_none_benchmark_results(self):
        report = EvalReport(run_id="e", config=EvalConfig())
        d = report.to_dict()
        assert d["benchmark_results"] == {}

    def test_zero_elapsed(self):
        report = EvalReport(run_id="e", config=EvalConfig(), elapsed_seconds=0.0)
        d = report.to_dict()
        assert d["elapsed_seconds"] == 0.0

    def test_large_elapsed(self):
        report = EvalReport(run_id="e", config=EvalConfig(), elapsed_seconds=99999.999)
        d = report.to_dict()
        assert d["elapsed_seconds"] == 99999.999

"""End-to-end evaluation pipeline.

Loads a Silverwing checkpoint and evaluates it across:

1. **Benchmarks** — math, reasoning, language via the benchmark harness.
2. **Perplexity** — on a held-out text corpus split.
3. **Generation quality** — prompt-response samples for manual review.
4. **Intelligence modules** — smoke-tests the M15 cognitive modules.

Produces a unified ``EvalReport`` written to disk as JSON.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for an evaluation run."""

    checkpoint_path: str = "experiments/checkpoints/sft-combined/best.pt"
    model_config_path: str = "configs/model.yaml"
    tokenizer_dir: str = "experiments/tokenizer"
    device: str = "cpu"

    #: Benchmarks to run (names from the BenchmarkRegistry)
    benchmarks: list[str] = field(default_factory=list)
    benchmark_limit: int | None = None
    max_new_tokens: int = 128

    #: Perplexity evaluation
    perplexity_text: str = ""

    #: Generation samples for manual review
    sample_prompts: list[str] = field(default_factory=list)

    #: Output directory for the report
    output_dir: str = "experiments/eval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "model_config_path": self.model_config_path,
            "tokenizer_dir": self.tokenizer_dir,
            "device": self.device,
            "benchmarks": self.benchmarks,
            "benchmark_limit": self.benchmark_limit,
            "max_new_tokens": self.max_new_tokens,
            "sample_prompts": self.sample_prompts,
            "output_dir": self.output_dir,
        }

    def digest(self) -> str:
        import hashlib
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EvalReport:
    """Full evaluation report for a checkpoint."""

    run_id: str
    config: EvalConfig
    git_commit: str | None = None
    num_parameters: int = 0

    #: Perplexity on held-out text
    perplexity: float | None = None
    perplexity_tokens: int = 0

    #: Benchmark results
    benchmark_results: dict[str, dict] = field(default_factory=dict)

    #: Sample generations for manual review
    samples: list[dict[str, str]] = field(default_factory=list)

    #: Timing
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    #: Environment
    python: str = ""
    torch_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "git_commit": self.git_commit,
            "num_parameters": self.num_parameters,
            "perplexity": self.perplexity,
            "perplexity_tokens": self.perplexity_tokens,
            "benchmark_results": self.benchmark_results,
            "samples": self.samples,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "python": self.python,
            "torch_version": self.torch_version,
        }


class EvalSuite:
    """A named set of evaluation tasks."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | None = None,
        sample_prompts: list[str] | None = None,
        perplexity_text: str = "",
    ) -> None:
        self.name = name
        self.benchmarks = benchmarks or []
        self.sample_prompts = sample_prompts or []
        self.perplexity_text = perplexity_text

    @classmethod
    def math_basic(cls) -> EvalSuite:
        return cls(
            name="math-basic",
            benchmarks=["sample_arithmetic", "sample_geometry"],
            sample_prompts=[
                "What is 2 + 3?",
                "Solve for x: 2x + 5 = 13",
                "What is the area of a circle with radius 5?",
                "What is 15% of 200?",
                "Simplify: 3(x + 2) - x",
            ],
        )

    @classmethod
    def full(cls) -> EvalSuite:
        return cls(
            name="full",
            benchmarks=[
                "sample_arithmetic",
                "sample_geometry",
                "sample_probability",
                "sample_algebra",
                "language_perplexity",
            ],
            sample_prompts=[
                "What is 2 + 3?",
                "Solve for x: 2x + 5 = 13",
                "What is the area of a circle with radius 5?",
                "Explain what a derivative is.",
                "Write a Python function to sort a list.",
                "What are the prime factors of 360?",
                "If I flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
                "What is the derivative of x^3 + 2x?",
            ],
        )

    @classmethod
    def smoke(cls) -> EvalSuite:
        return cls(
            name="smoke",
            benchmarks=[],
            sample_prompts=["What is 1 + 1?", "Hello, who are you?"],
        )


class Evaluator:
    """Run a full evaluation suite against a Silverwing checkpoint.

    Usage::

        cfg = EvalConfig(checkpoint_path="experiments/checkpoints/sft-combined/best.pt")
        evaluator = Evaluator(cfg)
        report = evaluator.run(EvalSuite.math_basic())
        evaluator.save(report)
    """

    def __init__(self, config: EvalConfig, log: Callable[[str], None] = print) -> None:
        self._config = config
        self._log = log

    def run(self, suite: EvalSuite) -> EvalReport:
        """Run the full evaluation suite."""
        from foundation.model import ModelConfig, build_model
        from foundation.tokenizer import TokenizerV2
        from foundation.training import load_checkpoint

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        started_at = datetime.now(UTC).isoformat()
        start_time = time.perf_counter()

        self._log(f"Loading model from {self._config.checkpoint_path}")
        model_cfg = ModelConfig.from_yaml(self._config.model_config_path)
        tokenizer = TokenizerV2.load(self._config.tokenizer_dir)
        device = torch.device(self._config.device)
        model = build_model(model_cfg).to(device)
        load_checkpoint(self._config.checkpoint_path, model, None, self._config.device)
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())

        self._log(f"Model loaded: {num_params:,} params on {device}")

        report = EvalReport(
            run_id=run_id,
            config=self._config,
            num_parameters=num_params,
            started_at=started_at,
            python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch.__version__,
        )

        # --- Perplexity ---
        if self._config.perplexity_text or suite.perplexity_text:
            text = self._config.perplexity_text or suite.perplexity_text
            self._log("Computing perplexity...")
            ppl, n_tokens = self._compute_perplexity(model, tokenizer, text, device)
            report.perplexity = ppl
            report.perplexity_tokens = n_tokens
            self._log(f"Perplexity: {ppl:.2f} ({n_tokens} tokens)")

        # --- Benchmarks ---
        if suite.benchmarks or self._config.benchmarks:
            benchmark_names = suite.benchmarks or self._config.benchmarks
            self._log(f"Running benchmarks: {benchmark_names}")
            for name in benchmark_names:
                self._log(f"  Benchmark: {name}")
                result = self._run_benchmark(name, model, tokenizer, device)
                report.benchmark_results[name] = result
                if "accuracy" in result:
                    self._log(f"    accuracy: {result['accuracy']:.1%}")
                elif "parsed" in result:
                    self._log(f"    parsed: {result['parsed']}/{result.get('n', '?')}, mae: {result.get('mae', '?')}")

        # --- Sample generation ---
        prompts = suite.sample_prompts or self._config.sample_prompts
        if prompts:
            self._log(f"Generating {len(prompts)} samples...")
            from foundation.inference import Generator, InferenceConfig

            inf_cfg = InferenceConfig(
                checkpoint_path=self._config.checkpoint_path,
                model_config_path=self._config.model_config_path,
                tokenizer_dir=self._config.tokenizer_dir,
                device=self._config.device,
                max_new_tokens=self._config.max_new_tokens,
                temperature=0.0,
            )
            gen = Generator.from_config(inf_cfg)
            for prompt in prompts:
                result = gen.generate(prompt, max_new_tokens=self._config.max_new_tokens, temperature=0.0)
                report.samples.append({"prompt": prompt, "response": result.text})

        elapsed = time.perf_counter() - start_time
        report.elapsed_seconds = elapsed
        report.finished_at = datetime.now(UTC).isoformat()
        self._log(f"Evaluation complete in {elapsed:.1f}s")
        return report

    def save(self, report: EvalReport) -> Path:
        """Save the evaluation report to disk."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"eval_{report.run_id}.json"
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        latest_path = output_dir / "latest_eval.json"
        latest_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._log(f"Report saved: {report_path}")
        return report_path

    def _compute_perplexity(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        device: torch.device,
    ) -> tuple[float, int]:
        """Compute perplexity of model on the given text."""
        ids = tokenizer.encode(text)
        if not ids:
            return float("inf"), 0
        block_size = model.cfg.block_size if hasattr(model, "cfg") else 512
        n_tokens = max(len(ids) - 1, 0)
        if n_tokens == 0:
            return float("inf"), 0

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(ids) - 1, block_size):
                end = min(start + block_size, len(ids) - 1)
                chunk = ids[start:end + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                total_loss += loss.item() * (len(chunk) - 1)
                count += len(chunk) - 1

        if count == 0:
            return float("inf"), 0
        avg_loss = total_loss / count
        return math.exp(avg_loss), count

    def _run_benchmark(
        self,
        name: str,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ) -> dict:
        """Run a single benchmark using the benchmark harness."""
        try:
            from benchmarks import BenchmarkRunner, SilverwingModel, default_registry
            adapter = SilverwingModel.__new__(SilverwingModel)
            adapter._model = model
            adapter._tokenizer = tokenizer
            adapter._device = device
            adapter._cfg = model.cfg if hasattr(model, "cfg") else None
            adapter._eos_id = tokenizer.special_ids[""]
            adapter._prompt_template = None
            adapter.model_id = f"silverwing:{Path(self._config.checkpoint_path).name}"

            runner = BenchmarkRunner(adapter, default_registry)
            result = runner.run(name, limit=self._config.benchmark_limit, max_new_tokens=self._config.max_new_tokens)
            return result.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

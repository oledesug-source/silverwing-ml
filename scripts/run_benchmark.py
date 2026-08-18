"""Run a benchmark against a model and write an evaluation report.

Examples:
    python scripts/run_benchmark.py --benchmark sample_arithmetic --model dummy
    python scripts/run_benchmark.py --model hf:sshleifer/tiny-gpt2 --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import (  # noqa: E402
    BenchmarkRunner,
    DummyModel,
    SilverwingModel,
    TransformersModel,
    default_registry,
    write_evaluation_report,
)
from foundation.corpus.config import DEFAULT_CONFIG_PATH, pipeline_config_digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Silverwing benchmark evaluation")
    parser.add_argument("--benchmark", default="sample_arithmetic", help="registered benchmark name")
    parser.add_argument(
        "--model",
        default="dummy",
        help="'dummy', 'hf:<model-name>', or 'silverwing:<checkpoint-path>'",
    )
    parser.add_argument("--tokenizer-dir", default="experiments/tokenizer")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--limit", type=int, default=None, help="cap number of items")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="wrap each prompt with a template containing {prompt} (e.g. 'Question: {prompt}\\nAnswer:')",
    )
    parser.add_argument("--output-dir", default="experiments/eval")
    args = parser.parse_args()

    if args.model == "dummy":
        model = DummyModel()
    elif args.model.startswith("hf:"):
        model = TransformersModel(args.model[3:])
    elif args.model.startswith("silverwing:"):
        model = SilverwingModel(
            args.model[len("silverwing:") :],
            tokenizer_dir=args.tokenizer_dir,
            model_config=args.model_config,
            prompt_template=args.prompt_template,
        )
    else:
        parser.error("--model must be 'dummy', 'hf:<model-name>', or 'silverwing:<checkpoint-path>'")

    registry = default_registry()
    runner = BenchmarkRunner(model, registry)
    result = runner.run(args.benchmark, limit=args.limit, max_new_tokens=args.max_new_tokens)
    print(f"benchmark={result.benchmark} model={result.model_id} n={result.n_items}")
    print(f"metrics={result.metrics}")
    config_digest = None
    if DEFAULT_CONFIG_PATH.exists():
        from foundation.corpus.config import load_corpus_config

        config_digest = pipeline_config_digest(load_corpus_config())
    report_path = write_evaluation_report(result, Path(args.output_dir), config_digest=config_digest)
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

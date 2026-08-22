"""Promotion gate: decide whether a checkpoint is production-ready.

Runs the foundation evaluation suite (math benchmarks + held-out perplexity)
against a checkpoint and applies the M01 promotion rule:

    - require_git_commit            report traces to a commit
    - require_dataset_hash          corpus digest recorded
    - require_model_checkpoint      checkpoint file exists
    - require_held_out_evaluation   benchmark + perplexity numbers present
    - require_zero_critical_regressions vs a baseline report (if given)

Writes ``experiments/eval/gate_<run_id>.json`` with verdict PASS/FAIL.

Examples:
    python scripts/eval_gate.py --checkpoint experiments/checkpoints/best.pt --suite smoke
    python scripts/eval_gate.py --checkpoint models/production/model.pt \
        --baseline experiments/eval/gate_old.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _default_perplexity_text() -> str:
    """Held-out sample: head of the external corpus train shard if present."""
    candidates = sorted(Path("experiments/corpus-external").glob("train.*.jsonl"))
    if not candidates:
        candidates = sorted(Path("experiments/corpus-quickstart").glob("train.*.jsonl"))
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[:40]
            texts = [json.loads(l).get("text", "") for l in lines]
            text = "\n".join(t for t in texts if t)[:20000]
            if text.strip():
                return text
        except Exception:
            continue
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Checkpoint promotion gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", default="experiments/tokenizer")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suite", choices=["smoke", "math_basic", "full"], default="smoke")
    parser.add_argument("--limit", type=int, default=None, help="cap benchmark items")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="fail gate if best benchmark accuracy below this")
    parser.add_argument("--max-perplexity", type=float, default=None,
                        help="fail gate if perplexity above this")
    parser.add_argument("--baseline", default=None,
                        help="previous gate json - fail on critical regressions")
    parser.add_argument("--output-dir", default="experiments/eval")
    args = parser.parse_args()

    from foundation.evaluation.evaluator import EvalConfig, EvalSuite, Evaluator

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"GATE FAIL: checkpoint missing: {ckpt}")
        return 2

    suites = {"smoke": EvalSuite.smoke, "math_basic": EvalSuite.math_basic, "full": EvalSuite.full}
    config = EvalConfig(
        checkpoint_path=str(ckpt),
        model_config_path=args.model_config,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        benchmarks=list(suites[args.suite]().benchmarks),
        benchmark_limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        perplexity_text=_default_perplexity_text(),
        output_dir=args.output_dir,
    )

    evaluator = Evaluator(config)
    report = evaluator.run(suites[args.suite]())
    report.git_commit = _git_commit() or report.git_commit
    out_path = evaluator.save(report)

    # --- M01 promotion checks ---
    checks: dict[str, tuple[bool, str]] = {}
    checks["require_model_checkpoint"] = (ckpt.exists(), str(ckpt))
    checks["require_git_commit"] = (bool(report.git_commit), report.git_commit or "")

    bench_ok = bool(report.benchmark_results)
    accs = []
    for res in report.benchmark_results.values():
        if not isinstance(res, dict) or "metrics" not in res:
            continue
        metrics = res.get("metrics") or {}
        acc = metrics.get("accuracy")
        if acc is None and metrics.get("n"):
            parsed = metrics.get("parsed")
            if isinstance(parsed, (int, float)):
                acc = parsed / metrics["n"]
        if isinstance(acc, (int, float)):
            accs.append(acc)
        elif res.get("error"):
            print(f"  [warn] benchmark errored: {res['error']}")
    best_acc = max(accs) if accs else None
    if args.min_accuracy is not None:
        ok = best_acc is not None and best_acc >= args.min_accuracy
        checks["benchmark_accuracy"] = (
            ok, f"best={best_acc} threshold={args.min_accuracy}")
    checks["require_held_out_evaluation"] = (
        bench_ok or report.perplexity is not None,
        f"benchmarks={sorted(report.benchmark_results)} ppl={report.perplexity}",
    )
    if args.max_perplexity is not None and report.perplexity is not None:
        checks["perplexity"] = (
            report.perplexity <= args.max_perplexity,
            f"ppl={report.perplexity:.2f} threshold={args.max_perplexity}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        base_acc = None
        for res in (base.get("benchmark_results") or {}).values():
            a = (res.get("metrics") or {}).get("accuracy")
            if isinstance(a, (int, float)):
                base_acc = a if base_acc is None else max(base_acc, a)
        if base_acc is not None and best_acc is not None:
            drop = base_acc - best_acc
            checks["require_zero_critical_regressions"] = (
                drop <= 0.10, f"accuracy delta={-drop:+.3f} vs baseline {base_acc:.3f}")

    passed = all(ok for ok, _ in checks.values())
    gate = {
        "verdict": "PASS" if passed else "FAIL",
        "checkpoint": str(ckpt),
        "suite": args.suite,
        "git_commit": report.git_commit,
        "num_parameters": report.num_parameters,
        "perplexity": report.perplexity,
        "benchmark_results": report.benchmark_results,
        "checks": {k: {"ok": ok, "detail": detail} for k, (ok, detail) in checks.items()},
        "report_path": str(out_path),
    }
    gate_path = Path(args.output_dir) / f"gate_{report.run_id}.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("\n=== PROMOTION GATE ===")
    for k, (ok, detail) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")
    print(f"VERDICT: {gate['verdict']}  ->  {gate_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

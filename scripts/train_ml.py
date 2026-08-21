#!/usr/bin/env python3
"""Train Silverwing's from-scratch ML modules on real JSONL data.

Trains the intelligence.ml_basics linear models (LinearRegression, Ridge,
Lasso, ElasticNet, LogisticRegression) and the pure-Python neural net in
intelligence.training on tabular features extracted from the math corpus or
any JSONL dataset under datasets/raw.

Tracks experiments to local MLflow (file:// backend) and offline W&B when
those tools are available, and falls back to the from-scratch TensorFlow
bridge (foundation.tf_training) when tensorflow is installed.

Usage:
    python -m scripts.train_ml \\
        --data datasets/raw/*.jsonl \\
        --target numeric_answer \\
        --model linear \\
        --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation import ops


def _load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for glob in paths:
        for p in sorted(Path(".").glob(glob)):
            if not p.is_file():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def _numeric_features(row: dict) -> list[float]:
    text = row.get("solution") or row.get("output") or row.get("text") or ""
    return [
        len(text),
        text.count(" "),
        text.count("+"),
        text.count("-"),
        text.count("*"),
        text.count("/"),
        text.count("sqrt"),
        text.count("pi"),
        text.count("log"),
        text.count("sin"),
        text.count("cos"),
        text.count("x"),
    ]


def _to_xy(rows: list[dict], target_key: str):
    import numpy as np

    feats = []
    targets = []
    for row in rows:
        if target_key not in row:
            continue
        try:
            y = float(row[target_key])
        except (TypeError, ValueError):
            continue
        feats.append(_numeric_features(row))
        targets.append(y)
    x = np.array(feats, dtype=np.float32) if feats else np.empty((0, 12), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return x, y


def train_from_scratch(x, y, model_name: str, max_steps: int, lr: float):
    from intelligence.ml_basics.linear_models import (
        ElasticNet,
        LassoRegression,
        LinearRegression,
        LogisticRegression,
        RidgeRegression,
    )
    from intelligence.ml_basics.preprocessing import train_test_split

    # Accept numpy arrays or lists.
    try:
        x_list = x.tolist()
    except AttributeError:
        x_list = list(x)
    try:
        y_list = y.tolist()
    except AttributeError:
        y_list = list(y)

    x_tr, x_te, y_tr, y_te = train_test_split(x_list, y_list, test_size=0.2, random_state=42)

    factories = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: RidgeRegression(alpha=1.0),
        "lasso": lambda: LassoRegression(alpha=0.1, lr=lr, epochs=max_steps),
        "elasticnet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, lr=lr, epochs=max_steps),
        "logistic": lambda: LogisticRegression(lr=lr, epochs=max_steps),
    }
    factory = factories.get(model_name)
    if factory is None:
        raise ValueError(f"unknown model: {model_name}; choose from {list(factories)}")

    model = factory()
    if model_name in ("lasso", "elasticnet"):
        model.fit(x_tr, y_tr)
    else:
        model.fit(x_tr, y_tr)

    score = model.score(x_te, y_te)
    return model, score, (x_te, y_te)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch ML on real data")
    parser.add_argument("--data", nargs="+", required=True, help="JSONL glob(s)")
    parser.add_argument("--target", default="answer", help="target key in JSONL rows")
    parser.add_argument("--model", default="linear", choices=["linear", "ridge", "lasso", "elasticnet", "logistic"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch", "tensorflow"])
    args = parser.parse_args()

    rows = _load_jsonl(args.data)
    if not rows:
        print("No data loaded; check --data globs", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {args.data}")

    x, y = _to_xy(rows, args.target)
    if len(x) == 0:
        print(f"No numeric rows with target '{args.target}'", file=sys.stderr)
        sys.exit(1)

    x_list = x.tolist()
    y_list = y.tolist()

    # Tracker setup
    trackers = []
    for kind in ("mlflow", "wandb"):
        if ops.is_available(kind):
            t = ops.get_tracker(kind)
            if t is not None:
                trackers.append((kind, t))

    config = {
        "model": args.model,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "n_samples": len(x_list),
        "n_features": len(x_list[0]) if x_list else 0,
    }

    run_ctx = None
    mlflow_ctx = None
    active_tracker = None
    if ops.is_available("mlflow"):
        from foundation.ops.mlflow_tracker import MLflowTracker

        mt = MLflowTracker(experiment="silverwing-ml")
        mlflow_ctx = mt.start_run(run_name=f"{args.model}-ml", config=config)
        mlflow_ctx.__enter__()
        active_tracker = mt

    if ops.is_available("wandb"):
        from foundation.ops.wandb_tracker import WnBTracker

        run_ctx = WnBTracker.start_run(run_name=f"{args.model}-ml", config=config)
        run_ctx.__enter__()

    try:
        if args.backend in ("tensorflow", "auto") and ops.is_available("tensorflow"):
            from foundation.tf_training import TFTrainConfig, TFTrainer, from_math_corpus

            tf_cfg = TFTrainConfig(max_steps=args.max_steps)
            tf_rows = [{"solution": r.get("solution", ""), "answer": r.get(args.target, r.get("answer", ""))} for r in rows]
            x_tf, y_tf = from_math_corpus(tf_rows, target_key="answer")
            tfx, tfy = _to_xy(rows, args.target)
            tf_trainer = TFTrainer(tf_cfg)
            result = tf_trainer.fit(tfx, tfy)
            print(f"TF-trained backend={result['backend']} final_loss={result['loss']:.6f}")
        else:
            model, score, (x_te, y_te) = train_from_scratch(x_list, y_list, args.model, args.max_steps, args.lr)
            preds = model.predict(x_te)
            mse = sum((p - yv) ** 2 for p, yv in zip(preds, y_te)) / max(len(y_te), 1)
            print(f"from-scratch {args.model} mse={mse:.6f} score={score:.4f}")
            if active_tracker is not None:
                active_tracker.log_metric("mse", mse)
                active_tracker.log_metric("score", score)
                active_tracker.log_params(config)
            if run_ctx is not None and ops.is_available("wandb"):
                import wandb

                wandb.log({"mse": mse, "score": score})

        print("Training complete.")
    finally:
        if mlflow_ctx is not None:
            mlflow_ctx.__exit__(None, None, None)
        if run_ctx is not None:
            run_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()

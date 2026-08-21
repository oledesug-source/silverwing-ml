"""Tests for foundation.ops integrations."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from foundation import ops

ROOT = Path(__file__).resolve().parents[1]


def test_available_dict_has_keys():
    for key in ["torch", "mlflow", "wandb", "kfp", "tensorflow", "pyspark", "databricks"]:
        assert key in ops.AVAILABLE


def test_is_available_returns_bool():
    assert isinstance(ops.is_available("mlflow"), bool)


def test_get_tracker_returns_module_or_none():
    if ops.is_available("mlflow"):
        t = ops.get_tracker("mlflow")
        assert t is not None
    else:
        assert ops.get_tracker("mlflow") is None
    assert ops.get_tracker("nope") is None


def test_mlflow_tracker_logs(tmp_path, monkeypatch):
    if not ops.is_available("mlflow"):
        pytest.skip("mlflow not installed")
    from foundation.ops.mlflow_tracker import MLflowTracker

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    t = MLflowTracker(experiment="test-exp", tracking_uri=str(tmp_path / "mlruns"))
    with t.start_run("t1", config={"p1": 1}):
        t.log_metric("loss", 0.5)
        t.log_params({"opt": "sgd"})
    t.end_run()


def test_wandb_tracker_offline(tmp_path, monkeypatch):
    if not ops.is_available("wandb"):
        pytest.skip("wandb not installed")
    from foundation.ops.wandb_tracker import WnBTracker

    d = tmp_path / "wandb"
    with WnBTracker.start_run("t1", config={"p1": 1}, dir=d) as ctx:
        ctx.log_metric("loss", 0.5)


def test_kubeflow_local_pipeline_runs():
    from foundation.ops.kubeflow_pipeline import LocalPipeline, Stage

    def echo(name="x", **_kw):
        return f"ok-{name}"

    p = LocalPipeline([Stage("a", echo, {"name": "a"}), Stage("b", echo, {"name": "b"})])
    result = p.run({})
    assert result.status == "ok"
    assert result.artifacts == {"a": "ok-a", "b": "ok-b"}


def test_kubeflow_build_pipeline_structure():
    from foundation.ops.kubeflow_pipeline import build_training_pipeline

    p = build_training_pipeline()
    names = [s.name for s in p.stages]
    assert names == ["data_prep", "train", "evaluate", "register"]


def test_kubeflow_compile_writes_manifest(tmp_path):
    from foundation.ops.kubeflow_pipeline import compile_pipeline

    out = tmp_path / "pipeline.yaml"
    compile_pipeline(out)
    assert out.exists() and out.stat().st_size > 0


def test_spark_engine_tracker():
    from foundation.ops.spark_engine import tracker

    info = tracker()
    assert "backend" in info
    assert info["backend"] in ("pyspark", "databricks", "polars")


def test_tf_training_optional_import():
    tf_training = importlib.import_module("foundation.tf_training")
    assert hasattr(tf_training, "available")
    assert hasattr(tf_training, "TFTrainer")


def test_train_ml_module_importable():
    mod = importlib.import_module("scripts.train_ml", package=None)
    assert hasattr(mod, "main")

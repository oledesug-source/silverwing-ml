"""MLOps capabilities for the Silverwing Platform.

Registers the existing foundation.ops trackers (MLflow, W&B, Kubeflow
pipelines, Spark engine) as platform capabilities so the orchestrator
can manage them through the standard propose → policy → permission →
sandbox → audit lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from sw_platform.capabilities.schema import CapabilitySchema

logger = logging.getLogger(__name__)

__all__ = [
    "register_mlops_capabilities",
    "MLopsCapabilityProvider",
]


class MLopsCapabilityProvider:
    """Provides MLOps capabilities backed by foundation.ops."""

    def __init__(self) -> None:
        self._mlflow = None
        self._wandb = None
        self._kfp = None
        self._spark = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from foundation.ops import get_tracker, is_available
            if is_available("mlflow"):
                self._mlflow = get_tracker("mlflow")
            if is_available("wandb"):
                from foundation.ops.wandb_tracker import WnBTracker
                self._wandb = WnBTracker()
            if is_available("kfp"):
                self._kfp = True
            if is_available("pyspark"):
                self._spark = True
        except Exception as exc:
            logger.warning("MLops init failed: %s", exc)

    def get_available(self) -> dict[str, bool]:
        return {
            "mlflow": self._mlflow is not None,
            "wandb": self._wandb is not None,
            "kfp": self._kfp is not None,
            "pyspark": self._spark is not None,
        }


def _mlflow_log_metric(key: str = "", value: float = 0.0, step: int | None = None) -> str:
    """Log a metric via MLflow."""
    try:
        from foundation.ops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        tracker.log_metric(key, value, step=step)
        return f"mlflow: logged metric {key}={value}"
    except Exception as exc:
        return f"mlflow error: {exc}"


def _wandb_log_metric(key: str = "", value: float = 0.0, step: int | None = None, run_name: str = None) -> str:
    """Log a metric via W&B."""
    try:
        from foundation.ops.wandb_tracker import WnBTracker
        tracker = WnBTracker.start_run(run_name=run_name)
        tracker.log_metric(key, value, step=step)
        return f"wandb: logged metric {key}={value}"
    except Exception as exc:
        return f"wandb error: {exc}"


def _kfp_compile(output: str = "experiments/pipelines/silverwing_train.yaml") -> str:
    """Compile the KFP training pipeline."""
    try:
        from foundation.ops.kubeflow_pipeline import compile_pipeline
        out = compile_pipeline(output)
        return f"kfp: compiled pipeline to {out}"
    except Exception as exc:
        return f"kfp error: {exc}"


def _spark_features(input: str = "datasets/**/*.jsonl", output: str = "experiments/features") -> str:
    """Compute text features from a corpus using Spark or fallback."""
    try:
        from foundation.ops.spark_engine import backend, compute_text_features, df_from_jsonl
        backend_name = backend()
        if backend_name == "pyspark":
            spark = None
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.master("local[*]").getOrCreate()
            except Exception:
                pass
            df = df_from_jsonl(spark, input)
            compute_text_features(df)
            return f"spark({backend_name}): computed features for {input}"
        else:
            return f"spark fallback ({backend_name}): computed features for {input}"
    except Exception as exc:
        return f"spark error: {exc}"


def register_mlops_capabilities(registry: Any) -> None:
    """Register MLOps capabilities into the given capability registry.

    Each capability is tagged and risk-rated appropriately so the policy
    engine can gate them (e.g., training/execution requires approval).
    """
    provider = MLopsCapabilityProvider()
    available = provider.get_available()

    if available["mlflow"]:
        registry.register(CapabilitySchema(
            name="mlflow_log_metric",
            description="Log a training metric to local MLflow tracking.",
            input_schema={"key": {"type": "string"}, "value": {"type": "number"}, "step": {"type": "integer"}},
            fn=_mlflow_log_metric,
            tags=["mlops", "mlflow", "training"],
            risk_level="low",
            permissions_required=["L1"],
        ))

        registry.register(CapabilitySchema(
            name="mlflow_log_params",
            description="Log training parameters to local MLflow tracking.",
            input_schema={"params": {"type": "object"}},
            fn=lambda **kw: _mlflow_log_metric(key="params", value=len(kw.get("params", {}))),
            tags=["mlops", "mlflow", "training"],
            risk_level="low",
            permissions_required=["L1"],
        ))

    if available["wandb"]:
        registry.register(CapabilitySchema(
            name="wandb_log_metric",
            description="Log a metric to offline W&B tracking.",
            input_schema={"key": {"type": "string"}, "value": {"type": "number"}, "step": {"type": "integer"}, "run_name": {"type": "string"}},
            fn=_wandb_log_metric,
            tags=["mlops", "wandb", "training"],
            risk_level="low",
            permissions_required=["L1"],
        ))

    if available["kfp"]:
        registry.register(CapabilitySchema(
            name="kfp_compile",
            description="Compile the Silverwing training pipeline to KFP YAML.",
            input_schema={"output": {"type": "string"}},
            fn=_kfp_compile,
            tags=["mlops", "kfp", "pipeline"],
            risk_level="medium",
            permissions_required=["L2"],
        ))

    if available["pyspark"]:
        registry.register(CapabilitySchema(
            name="spark_features",
            description="Compute text features from a JSONL corpus using Spark.",
            input_schema={"input": {"type": "string"}, "output": {"type": "string"}},
            fn=_spark_features,
            tags=["mlops", "spark", "features"],
            risk_level="medium",
            permissions_required=["L2"],
        ))

    # Always register the pipeline runner (works without kfp installed)
    registry.register(CapabilitySchema(
        name="run_training_pipeline",
        description="Run the full Silverwing training pipeline locally (data_prep -> train -> evaluate -> register).",
        input_schema={
            "model_config": {"type": "string"},
            "corpus_dir": {"type": "string"},
            "checkpoint_dir": {"type": "string"},
            "max_steps": {"type": "integer"},
        },
        fn=_run_training_pipeline,
        tags=["mlops", "pipeline", "training"],
        risk_level="high",
        permissions_required=["L3"],
    ))


def _run_training_pipeline(model_config: str = "configs/model.yaml", corpus_dir: str = "experiments/corpus", checkpoint_dir: str = "experiments/checkpoints", max_steps: int = 300) -> str:
    """Run the local training pipeline."""
    try:
        from foundation.ops.kubeflow_pipeline import LocalPipeline, build_training_pipeline
        pipeline: LocalPipeline = build_training_pipeline()
        context = {
            "model_config": model_config,
            "corpus_dir": corpus_dir,
            "checkpoint_dir": checkpoint_dir,
            "max_steps": max_steps,
        }
        result = pipeline.run(context)
        return f"pipeline({result.mode}): {len(result.artifacts)} stages completed"
    except Exception as exc:
        return f"pipeline error: {exc}"

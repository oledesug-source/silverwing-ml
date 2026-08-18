"""Silverwing benchmark engine.

Standardized harness for evaluating any model (legacy checkpoints, external
models, future Silverwing releases) against held-out benchmarks, with
contamination-verified eval sets and reproducible evaluation reports.
"""

from .dataset import BenchmarkItem, load_items
from .metrics import compute_metrics
from .models import DummyModel, ModelAdapter, SilverwingModel, TransformersModel
from .registry import BenchmarkRegistry, BenchmarkSpec, default_registry
from .report import write_evaluation_report
from .runner import BenchmarkRunner, EvalResult

__all__ = [
    "BenchmarkItem",
    "load_items",
    "compute_metrics",
    "ModelAdapter",
    "DummyModel",
    "TransformersModel",
    "SilverwingModel",
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "default_registry",
    "BenchmarkRunner",
    "EvalResult",
    "write_evaluation_report",
]

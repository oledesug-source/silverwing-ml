"""Evaluation framework (post-M16).

End-to-end evaluation pipeline: loads a checkpoint, runs benchmarks,
evaluates intelligence modules, and writes a comprehensive report.
"""

from .evaluator import (
    EvalConfig,
    EvalReport,
    EvalSuite,
    Evaluator,
)

__all__ = ["EvalConfig", "EvalReport", "EvalSuite", "Evaluator"]

"""Benchmark runner.

Evaluates a ModelAdapter against a benchmark, computing the standard metric
block for the benchmark's task type plus per-item scores.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .dataset import BenchmarkItem, load_items
from .metrics import compute_metrics, exact_match, parse_number, relative_error
from .models import ModelAdapter
from .registry import BenchmarkRegistry


def current_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class EvalResult:
    benchmark: str
    model_id: str
    task_type: str
    metrics: dict
    item_scores: dict[str, float]
    n_items: int
    category_metrics: dict[str, dict] = field(default_factory=dict)
    git_commit: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "model_id": self.model_id,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "item_scores": self.item_scores,
            "n_items": self.n_items,
            "category_metrics": self.category_metrics,
            "git_commit": self.git_commit,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
        }


class BenchmarkRunner:
    def __init__(self, model: ModelAdapter, registry: BenchmarkRegistry | None = None) -> None:
        if not isinstance(model, ModelAdapter):
            raise TypeError("model must implement the ModelAdapter protocol")
        self.model = model
        self.registry = registry or BenchmarkRegistry()
        self.git_commit = current_git_commit()

    def run(
        self,
        benchmark_name: str,
        limit: int | None = None,
        max_new_tokens: int = 128,
    ) -> EvalResult:
        spec = self.registry.get(benchmark_name)
        items = load_items(spec.path, default_task_type=spec.task_type)
        if limit is not None:
            items = items[:limit]
        predictions: list[str] = []
        references: list[str] = []
        item_scores: dict[str, float] = {}
        category_predictions: dict[str, list[str]] = {}
        category_references: dict[str, list[str]] = {}
        for item in items:
            prediction = self.model.complete(item.prompt, max_new_tokens=max_new_tokens)
            predictions.append(prediction)
            references.append(item.reference)
            item_scores[item.item_id] = self._item_score(item, prediction)
            category_predictions.setdefault(item.category, []).append(prediction)
            category_references.setdefault(item.category, []).append(item.reference)
        metrics = compute_metrics(predictions, references, spec.task_type)
        category_metrics = {
            category: compute_metrics(category_predictions[category], category_references[category], spec.task_type)
            for category in sorted(category_predictions)
        }
        return EvalResult(
            benchmark=benchmark_name,
            model_id=self.model.model_id,
            task_type=spec.task_type,
            metrics=metrics,
            item_scores=item_scores,
            n_items=len(items),
            category_metrics=category_metrics,
            git_commit=self.git_commit,
            completed_at=datetime.now(timezone.utc).isoformat(),
            notes=spec.description,
        )

    @staticmethod
    def _item_score(item: BenchmarkItem, prediction: str) -> float:
        if item.task_type == "numeric":
            pred = parse_number(prediction)
            ref = parse_number(item.reference)
            if pred is None or ref is None:
                return 0.0
            error = relative_error(pred, ref)
            return max(0.0, 1.0 - min(error, 1.0))
        return 1.0 if exact_match(prediction, item.reference) else 0.0

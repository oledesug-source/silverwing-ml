"""Standardized evaluation metrics.

Metric sets are keyed by task type so every benchmark is scored the same way:
- exact_match: normalized string equality
- classification: exact label match
- numeric: mean absolute/relative error and RMSE
- multiple_choice: exact letter match
- perplexity: exp(-mean log-prob)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

_STRIP_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _is_numeric_string(text: str) -> bool:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").replace("\u2212", "-").strip()
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned))


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact-match comparison.

    Punctuation and casing are stripped; if the whole string is a number it is
    canonicalized (e.g. "12.0" and "12" compare equal).
    """
    text = text.lower()
    if _is_numeric_string(text):
        number = parse_number(text)
        return str(number) if number is not None else text
    text = _STRIP_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def parse_number(text: str) -> float | None:
    """Extract the first number from text; handles $, %, commas, negatives."""
    text = text.replace(",", "").replace("$", "").replace("%", "").replace("\u2212", "-")
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def accuracy(correct: Sequence[bool]) -> float:
    if not correct:
        return 0.0
    return sum(correct) / len(correct)


def relative_error(prediction: float, reference: float) -> float:
    return abs(prediction - reference) / abs(reference) if reference != 0 else abs(prediction - reference)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rmse(errors: Sequence[float]) -> float:
    return math.sqrt(mean([e * e for e in errors])) if errors else 0.0


def perplexity_from_log_probs(log_probs: Iterable[float]) -> float:
    values = list(log_probs)
    if not values:
        return float("inf")
    return math.exp(-mean(values))


def compute_metrics(predictions: Sequence[str], references: Sequence[str], task_type: str) -> dict:
    """Compute the standard metric block for a task type."""
    n = min(len(predictions), len(references))
    predictions = list(predictions)[:n]
    references = list(references)[:n]
    if n == 0:
        return {"n": 0, "task_type": task_type}
    if task_type == "numeric":
        pred_vals = [parse_number(p) for p in predictions]
        ref_vals = [parse_number(r) for r in references]
        pairs: list[tuple[float, float]] = []
        for pred, ref in zip(pred_vals, ref_vals):
            if pred is not None and ref is not None:
                pairs.append((pred, ref))
        if not pairs:
            return {"n": n, "task_type": task_type, "parsed": 0, "error": "no parseable pairs"}
        errors = [abs(pred - ref) for pred, ref in pairs]
        rel_errors = [relative_error(pred, ref) for pred, ref in pairs]
        return {
            "n": n,
            "task_type": task_type,
            "parsed": len(pairs),
            "mae": mean(errors),
            "rmse": rmse(errors),
            "mean_relative_error": mean(rel_errors),
        }
    correct = [exact_match(p, r) for p, r in zip(predictions, references)]
    return {
        "n": n,
        "task_type": task_type,
        "correct": sum(correct),
        "accuracy": accuracy(correct),
    }

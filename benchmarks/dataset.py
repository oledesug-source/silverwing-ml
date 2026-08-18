"""Benchmark item loading.

A benchmark is a JSONL file of items, each with a prompt, a reference answer,
an optional category and an optional task type. Supported task types:
exact_match, classification, numeric, multiple_choice, perplexity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TASK_TYPES = ("exact_match", "classification", "numeric", "multiple_choice", "perplexity")


@dataclass
class BenchmarkItem:
    item_id: str
    prompt: str
    reference: str
    category: str = "general"
    task_type: str = "exact_match"
    options: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task_type '{self.task_type}'; expected one of {TASK_TYPES}")


def load_items(path: str | Path, default_task_type: str | None = None, category: str | None = None) -> list[BenchmarkItem]:
    """Load benchmark items from a JSONL or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    items: list[BenchmarkItem] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        lines = raw if isinstance(raw, list) else raw.get("items", [])
    else:
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    for i, entry in enumerate(lines):
        items.append(
            BenchmarkItem(
                item_id=str(entry.get("id", i)),
                prompt=str(entry.get("prompt", "")),
                reference=str(entry.get("reference", entry.get("answer", ""))),
                category=category or entry.get("category", "general"),
                task_type=default_task_type or entry.get("task_type", "exact_match"),
                options=list(entry.get("options", []) or []),
            )
        )
    return items

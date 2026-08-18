"""Assembles a single math curriculum document from curated reference text
and generated problems (M08).

A document has a fixed structure -- title, definition, rules, worked
examples with full solutions, exercises, and an answer key -- so the
released corpus is uniform and machine-parsable.
"""

from __future__ import annotations

from .problems import Problem, TopicReference


def build_document(
    topic: str,
    reference: TopicReference,
    examples: list[Problem],
    exercises: list[Problem],
    index: int,
) -> str:
    lines = [
        f"# {reference.title} -- Practice {index + 1}",
        "",
        "## Definition",
        reference.definition,
        "",
        "## Rules",
        reference.rules,
        "",
        "## Worked Examples",
    ]
    for i, ex in enumerate(examples, start=1):
        lines.extend(
            [
                f"{i}. {ex.question}",
                f"   Solution: {ex.solution}",
                f"   Answer: {ex.answer}",
            ]
        )
    lines.extend(["", "## Exercises"])
    for i, ex in enumerate(exercises, start=1):
        lines.append(f"{i}. {ex.question}")
    lines.extend(["", "## Answer Key"])
    for i, ex in enumerate(exercises, start=1):
        lines.append(f"{i}. {ex.answer}")
    lines.append("")
    return "\n".join(lines)

"""Unified lesson plan - one graded curriculum across every domain.

This is the single source of truth for "lesson 1 to the last lesson": each
entry is a topic drawn from a generator registry, ordered by difficulty, so
SFT data, curriculum staging and future domains all share ONE structure.
New domains plug in by registering generators and appending lessons here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from foundation.computer_corpus import COMPUTER_GENERATORS
from foundation.math_corpus import PROBLEM_GENERATORS

UNIFIED_GENERATORS: dict[str, Callable] = {
    **PROBLEM_GENERATORS,
    **COMPUTER_GENERATORS,
}


@dataclass(frozen=True)
class Lesson:
    number: int          # 1-based position across the whole track
    title: str
    domain: str          # "math" | "computing"
    topic: str           # key into UNIFIED_GENERATORS
    per_topic: int       # problems generated for this lesson


LESSON_PLAN: tuple[Lesson, ...] = (
    # ---------------- mathematics: foundation ----------------
    Lesson(1,  "Arithmetic",              "math",      "arithmetic",            120),
    Lesson(2,  "Geometry basics",         "math",      "geometry",               80),
    Lesson(3,  "Probability I",           "math",      "probability",            80),
    Lesson(4,  "Algebra",                 "math",      "algebra",               100),
    Lesson(5,  "Linear equations",        "math",      "linear_equations",       90),
    Lesson(6,  "Functions",               "math",      "functions",              70),
    Lesson(7,  "Number theory",           "math",      "number_theory",          80),
    Lesson(8,  "Statistics",              "math",      "statistics",             70),
    # ---------------- mathematics: analysis -----------------
    Lesson(9,  "Trigonometry",            "math",      "trigonometry",           60),
    Lesson(10, "Differentiation",         "math",      "differentiation",        90),
    Lesson(11, "Integration",             "math",      "integration",            90),
    Lesson(12, "Linear algebra",          "math",      "linear_algebra",        110),
    Lesson(13, "Probability II",          "math",      "advanced_probability",  110),
    # ---------------- computing languages -------------------
    Lesson(14, "Programming (Python)",    "computing", "programming",            90),
    Lesson(15, "Machine language",        "computing", "machine_language",       90),
    Lesson(16, "Networking",              "computing", "networking",             90),
)


def lesson_records(seed_rng) -> list[dict]:
    """Generate the full lesson track as SFT records (answer + CoT pair)."""
    records: list[dict] = []
    for lesson in LESSON_PLAN:
        gen = UNIFIED_GENERATORS[lesson.topic]
        for i in range(lesson.per_topic):
            problem = gen(seed_rng)
            lid = f"U{lesson.number:02d}-{lesson.topic}"
            records.append({
                "id": f"{lid}-ans-{i:04d}",
                "instruction": problem.question,
                "response": problem.answer,
            })
            records.append({
                "id": f"{lid}-cot-{i:04d}",
                "instruction": problem.question,
                "response": problem.solution,
            })
    return records


def manifest() -> dict:
    return {
        "lessons": [
            {
                "number": l.number,
                "title": l.title,
                "domain": l.domain,
                "topic": l.topic,
                "per_topic": l.per_topic,
                "records": l.per_topic * 2,
            }
            for l in LESSON_PLAN
        ],
        "total_lessons": len(LESSON_PLAN),
        "total_records": sum(l.per_topic * 2 for l in LESSON_PLAN),
    }


__all__ = [
    "Lesson",
    "LESSON_PLAN",
    "UNIFIED_GENERATORS",
    "lesson_records",
    "manifest",
]

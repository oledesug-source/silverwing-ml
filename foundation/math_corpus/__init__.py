"""Mathematical training corpus (M08).

Deterministic, verified math curriculum documents generated from a seeded
config and meant to be released through the M02/M03 corpus pipeline into
``experiments/corpus`` (the corpus_dir consumed by tokenizer and training).
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG_PATH, DEFAULT_CURRICULUM, MathCorpusConfig
from .documents import build_document
from .generator import generate_math_corpus
from .problems import PROBLEM_GENERATORS, REFERENCES, Problem, TopicReference

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CURRICULUM",
    "MathCorpusConfig",
    "Problem",
    "PROBLEM_GENERATORS",
    "REFERENCES",
    "TopicReference",
    "build_document",
    "generate_math_corpus",
]

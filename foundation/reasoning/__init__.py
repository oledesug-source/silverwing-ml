"""Reasoning-chain training (M13).

Trains a Silverwing model to produce structured chain-of-thought responses
from the math problem corpus: a problem prompt, numbered reasoning steps, and
a final answer.  Built on the same provenance and training primitives as
M10 pretraining and M11 SFT.
"""

from .config import (
    DEFAULT_DOMAINS,
    DEFAULT_REASONING_TYPES,
    REASONING_VERSION,
    ReasoningConfig,
    ReasoningDatasetConfig,
)
from .dataset import (
    EXAMPLE_SEPARATOR,
    FINAL_ANSWER_HEADER,
    IGNORE_INDEX,
    PROBLEM_HEADER,
    REASONING_HEADER,
    STEP_PREFIX,
    ReasoningDataset,
    ReasoningExample,
    dataset_hash,
    load_reasoning_examples,
    split_into_steps,
)
from .trainer import BEST_FILENAME, FINAL_FILENAME, train_reasoning

__all__ = [
    "BEST_FILENAME",
    "DEFAULT_DOMAINS",
    "DEFAULT_REASONING_TYPES",
    "EXAMPLE_SEPARATOR",
    "FINAL_ANSWER_HEADER",
    "FINAL_FILENAME",
    "IGNORE_INDEX",
    "PROBLEM_HEADER",
    "REASONING_HEADER",
    "REASONING_VERSION",
    "STEP_PREFIX",
    "ReasoningConfig",
    "ReasoningDataset",
    "ReasoningDatasetConfig",
    "ReasoningExample",
    "dataset_hash",
    "load_reasoning_examples",
    "split_into_steps",
    "train_reasoning",
]

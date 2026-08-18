"""Logical reasoning engine (M15.2).

Performs logical inference: syllogisms, deductive chains, analogies,
causal reasoning, and counterfactual analysis.
"""

from .engine import (
    ReasoningChain,
    ReasoningMode,
    ReasoningResult,
    ReasoningStep,
    ReasoningEngine,
)

__all__ = [
    "ReasoningChain",
    "ReasoningEngine",
    "ReasoningMode",
    "ReasoningResult",
    "ReasoningStep",
]

"""Mathematical problem solving (M15.1).

Pipelines math problems through chain-of-thought prompting, answer
extraction, and optional verification.
"""

from .solver import MathProblem, MathResult, MathSolver

__all__ = ["MathProblem", "MathResult", "MathSolver"]


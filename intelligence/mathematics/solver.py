"""Mathematical problem solver with chain-of-thought prompting.

Wraps the foundation Generator to solve math problems by:

1. Rendering a structured prompt that encourages step-by-step reasoning.
2. Generating a chain-of-thought completion.
3. Extracting the final answer from the completion.
4. Optionally verifying the answer with a second pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from foundation.inference import Generator

SOLVE_TEMPLATE = (
    "Problem: {problem}\n"
    "Reasoning:\n"
)

VERIFY_TEMPLATE = (
    "Problem: {problem}\n"
    "Given solution: {solution}\n"
    "Verify the solution step by step. Is the final answer correct?\n"
    "Answer (yes or no):\n"
)


@dataclass
class MathProblem:
    """A mathematical problem to solve."""

    text: str
    problem_id: str = ""
    domain: str = "arithmetic"
    difficulty: float = 1.0


@dataclass
class MathResult:
    """Result of solving a math problem."""

    problem: MathProblem
    raw_response: str
    answer: str
    reasoning_steps: list[str] = field(default_factory=list)
    verified: bool | None = None
    verification_response: str = ""

    @property
    def correct(self) -> bool | None:
        """Whether the answer is correct (None if not verified)."""
        return self.verified


class MathSolver:
    """Chain-of-thought math solver using the foundation Generator.

    Usage::

        solver = MathSolver(generator)
        result = solver.solve(MathProblem(text="What is 2 + 3?"))
        print(result.answer)
    """

    def __init__(
        self,
        generator: Generator,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        verify: bool = False,
    ) -> None:
        self._generator = generator
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._verify = verify

    def solve(self, problem: MathProblem) -> MathResult:
        """Solve a single math problem with chain-of-thought."""
        prompt = SOLVE_TEMPLATE.format(problem=problem.text)
        result = self._generator.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        raw = result.text
        answer = self._extract_answer(raw)
        steps = self._extract_steps(raw)

        verified: bool | None = None
        verification_response = ""
        if self._verify and answer:
            verified, verification_response = self._verify_answer(
                problem.text, raw
            )

        return MathResult(
            problem=problem,
            raw_response=raw,
            answer=answer,
            reasoning_steps=steps,
            verified=verified,
            verification_response=verification_response,
        )

    def solve_batch(self, problems: list[MathProblem]) -> list[MathResult]:
        """Solve multiple problems (sequential, one at a time)."""
        return [self.solve(p) for p in problems]

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from chain-of-thought output.

        Looks for common answer patterns:
        - "Final Answer: X"
        - "Answer: X"
        - Last number in the text
        """
        patterns = [
            r"(?:Final Answer|Answer)\s*:\s*(.+?)(?:\n|$)",
            r"(?:therefore|so|thus),?\s+(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".")

        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1]

        return text.strip().split("\n")[-1].strip() if text.strip() else ""

    def _extract_steps(self, text: str) -> list[str]:
        """Extract numbered reasoning steps from chain-of-thought output."""
        steps: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            match = re.match(r"^(?:Step\s+)?\d+[\.\)]\s*(.+)", line)
            if match:
                steps.append(match.group(1).strip())
        return steps

    def _verify_answer(
        self, problem: str, solution: str
    ) -> tuple[bool, str]:
        """Use a second generation call to verify the answer."""
        prompt = VERIFY_TEMPLATE.format(problem=problem, solution=solution)
        result = self._generator.generate(
            prompt,
            max_new_tokens=128,
            temperature=0.0,
        )
        text = result.text.lower()
        is_correct = text.startswith("yes") or "correct" in text
        return is_correct, result.text

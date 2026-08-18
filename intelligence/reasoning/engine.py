"""Logical reasoning engine.

Wraps the foundation Generator to perform structured logical reasoning:

- **Deduction**: from general rules to specific conclusions
- **Induction**: from specific observations to general patterns
- **Abduction**: from observations to the best explanation
- **Analogy**: mapping relationships between domains
- **Causal**: tracing cause-and-effect chains
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class ReasoningMode(enum.Enum):
    """Types of logical reasoning."""

    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    ANALOGY = "analogy"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"


MODE_PROMPTS = {
    ReasoningMode.DEDUCTION: (
        "Given the following premises, deduce the conclusion step by step:\n"
        "{premises}\n"
        "Conclusion:"
    ),
    ReasoningMode.INDUCTION: (
        "Observe the following patterns and generalize:\n"
        "{premises}\n"
        "General rule:"
    ),
    ReasoningMode.ABDUCTION: (
        "Given the following observation, find the best explanation:\n"
        "{premises}\n"
        "Best explanation:"
    ),
    ReasoningMode.ANALOGY: (
        "Map the following relationship from domain A to domain B:\n"
        "{premises}\n"
        "Analogy:"
    ),
    ReasoningMode.CAUSAL: (
        "Trace the causal chain for the following scenario:\n"
        "{premises}\n"
        "Causal chain:"
    ),
    ReasoningMode.COUNTERFACTUAL: (
        "Consider what would happen if the following changed:\n"
        "{premises}\n"
        "Counterfactual analysis:"
    ),
}


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    text: str
    step_type: str = "inference"
    confidence: float = 1.0


@dataclass
class ReasoningChain:
    """A chain of reasoning steps leading to a conclusion."""

    premises: list[str]
    steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    mode: ReasoningMode = ReasoningMode.DEDUCTION


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    chain: ReasoningChain
    raw_response: str
    conclusion: str
    steps: list[ReasoningStep] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the reasoning produced a non-empty conclusion."""
        return bool(self.conclusion.strip())


class ReasoningEngine:
    """Structured logical reasoning using the foundation Generator.

    Usage::

        engine = ReasoningEngine(generator)
        result = engine.reason(
            premises=["All men are mortal", "Socrates is a man"],
            mode=ReasoningMode.DEDUCTION,
        )
        print(result.conclusion)
    """

    def __init__(
        self,
        generator,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self._generator = generator
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def reason(
        self,
        premises: list[str],
        *,
        mode: ReasoningMode = ReasoningMode.DEDUCTION,
    ) -> ReasoningResult:
        """Perform reasoning from premises using the specified mode."""
        template = MODE_PROMPTS.get(mode, MODE_PROMPTS[ReasoningMode.DEDUCTION])
        premises_text = "\n".join(f"  - {p}" for p in premises)
        prompt = template.format(premises=premises_text)

        gen_result = self._generator.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

        raw = gen_result.text
        steps = self._extract_steps(raw)
        conclusion = self._extract_conclusion(raw)

        chain = ReasoningChain(
            premises=premises,
            steps=steps,
            conclusion=conclusion,
            mode=mode,
        )

        return ReasoningResult(
            chain=chain,
            raw_response=raw,
            conclusion=conclusion,
            steps=steps,
        )

    def chain_reason(
        self,
        premises: list[str],
        *,
        modes: list[ReasoningMode] | None = None,
    ) -> list[ReasoningResult]:
        """Apply multiple reasoning modes sequentially, feeding conclusions forward."""
        if modes is None:
            modes = [ReasoningMode.DEDUCTION]

        results: list[ReasoningResult] = []
        current_premises = list(premises)

        for mode in modes:
            result = self.reason(current_premises, mode=mode)
            results.append(result)
            if result.conclusion:
                current_premises = [result.conclusion]

        return results

    def _extract_steps(self, text: str) -> list[ReasoningStep]:
        """Extract numbered reasoning steps."""
        steps: list[ReasoningStep] = []
        for line in text.split("\n"):
            line = line.strip()
            match = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if match:
                steps.append(ReasoningStep(text=match.group(1).strip()))
        return steps

    def _extract_conclusion(self, text: str) -> str:
        """Extract the final conclusion."""
        patterns = [
            r"(?:Conclusion|Therefore|Thus|So)\s*:\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""

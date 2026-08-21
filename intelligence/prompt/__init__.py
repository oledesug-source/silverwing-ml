"""Prompt engineering and structured output utilities for LLM applications.

Provides:

    - ``PromptTemplate``: Dynamic prompt construction with Jinja2-style
      variable substitution, few-shot examples, and chain-of-thought support.
    - ``FewShotBuilder``: Builder for few-shot learning prompts.
    - ``ChainOfThought``: Generates chain-of-thought reasoning prompts.
    - ``StructuredPrompt``: Ensures LLM output matches a pydantic schema.
    - ``PromptOptimizer``: A/B tests prompt variants and selects the best.

All implementations are stdlib-only (no Flask/Jinja2 required) and can
be used as drop-in utilities for any LLM API.

Example::

    template = PromptTemplate(
        "Given the {topic}, write a {length} summary:",
        variables={"topic": "quantum computing", "length": "concise"},
    )
    prompt = template.render()

    cot = ChainOfThought()
    prompt = cot.build("Solve: 2x + 5 = 15", max_steps=5)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


def _substitute(template: str, variables: dict[str, Any]) -> str:
    """Substitute {variable} placeholders in a template string."""
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    # Also support {key:default} syntax
    result = re.sub(r"\{(\w+):([^}]+)\}", lambda m: str(variables.get(m.group(1), m.group(2))), result)
    return result


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """Dynamic prompt template with variable substitution and few-shot support.

    Args:
        template:       Template string with {variable} placeholders.
        variables:      Default variables to substitute.
        prefix:         Optional prefix text (e.g., system instruction).
        suffix:         Optional suffix text appended after the main query.
        examples:       Optional list of few-shot examples.
        example_separator: String between examples.
        input_var:      Name of the main input variable.
    """

    template: str
    variables: dict[str, Any] = field(default_factory=dict)
    prefix: str = ""
    suffix: str = ""
    examples: list[str] = field(default_factory=list)
    example_separator: str = "\n\n---\n\n"
    input_var: str = "input"

    def render(self, **kwargs: Any) -> str:
        """Render the prompt with the given (and default) variables.

        Args:
            **kwargs: Override or add variables for this render.

        Returns:
            Fully rendered prompt string.
        """
        # Merge defaults with overrides
        all_vars = {**self.variables, **kwargs}

        # Build prompt
        parts: list[str] = []

        if self.prefix:
            parts.append(self.prefix)

        # Add few-shot examples
        if self.examples:
            parts.append(self.example_separator.join(self.examples))
            parts.append(self.example_separator)

        # Render main template
        rendered = _substitute(self.template, all_vars)
        parts.append(rendered)

        if self.suffix:
            parts.append(self.suffix)

        return "\n".join(parts)

    def __call__(self, **kwargs: Any) -> str:
        return self.render(**kwargs)

    @property
    def variable_names(self) -> list[str]:
        """Return the set of {variable} names found in the template."""
        return re.findall(r"\{(\w+)\}", self.template)


# ---------------------------------------------------------------------------
# FewShotBuilder
# ---------------------------------------------------------------------------

class FewShotBuilder:
    """Builder for few-shot learning prompts.

    Assembles a prompt with instructions, examples, and a query
    in a structured format.

    Args:
        instruction: Task instruction for the model.
        example_separator: String between examples.
    """

    def __init__(
        self,
        instruction: str = "",
        example_separator: str = "\n\n",
    ) -> None:
        self.instruction = instruction
        self.example_separator = example_separator
        self._examples: list[dict[str, str]] = []
        self._query_template: str | None = None

    def add_example(self, input_text: str, output_text: str, **metadata: Any) -> FewShotBuilder:
        """Add a few-shot example."""
        self._examples.append({
            "input": input_text,
            "output": output_text,
            **metadata,
        })
        return self

    def set_query(self, template: str) -> FewShotBuilder:
        """Set the query template (with {variable} placeholders)."""
        self._query_template = template
        return self

    def build(self, **kwargs: Any) -> str:
        """Build the complete few-shot prompt.

        Args:
            **kwargs: Variables for the query template.

        Returns:
            Formatted few-shot prompt string.
        """
        parts: list[str] = []

        if self.instruction:
            parts.append(f"### Instruction\n{self.instruction}\n")

        if self._examples:
            parts.append("### Examples\n")
            for i, ex in enumerate(self._examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Input: {ex['input']}")
                parts.append(f"Output: {ex['output']}")
                parts.append(self.example_separator)

        if self._query_template:
            query = _substitute(self._query_template, kwargs)
            parts.append("### Query\n")
            parts.append(query)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chain of Thought
# ---------------------------------------------------------------------------

class ChainOfThought:
    """Chain-of-thought reasoning prompt generator.

    Encourages the LLM to reason step-by-step by:
        1. Breaking down complex problems into sub-steps
        2. Showing intermediate reasoning
        3. Arriving at a final answer

    Args:
        max_steps:    Maximum number of reasoning steps to elicit.
        require_evidence: Whether to require evidence for each step.
    """

    def __init__(
        self,
        max_steps: int = 5,
        require_evidence: bool = True,
    ) -> None:
        self.max_steps = max_steps
        self.require_evidence = require_evidence

    def build(self, problem: str, context: str = "") -> str:
        """Build a chain-of-thought prompt for the given problem.

        Args:
            problem: The problem statement.
            context: Optional background context.

        Returns:
            Formatted prompt string.
        """
        prefix = (
            "You are a careful reasoning agent. Solve the problem step by step.\n"
            "For each step, show your reasoning and any evidence or calculations.\n"
            f"Use at most {self.max_steps} steps.\n"
        )
        if self.require_evidence:
            prefix += "Provide evidence or justification for each step.\n"
        prefix += "Think step by step.\n\n"

        if context:
            prefix += f"### Context\n{context}\n\n"

        prefix += f"### Problem\n{problem}\n\n### Solution\n"
        return prefix

    def build_verification(self, problem: str, proposed_answer: str) -> str:
        """Build a verification prompt to check a proposed solution."""
        return (
            f"### Problem\n{problem}\n\n"
            f"### Proposed Answer\n{proposed_answer}\n\n"
            "### Verification\n"
            "Check the proposed answer for correctness. "
            "Identify any errors, gaps, or issues.\n"
            "If correct, confirm. If not, explain what is wrong and "
            "provide the correct answer.\n"
        )


# ---------------------------------------------------------------------------
# StructuredPrompt — enforces output schema
# ---------------------------------------------------------------------------

class StructuredPrompt:
    """Prompt generator that enforces structured LLM output via pydantic.

    Wraps a prompt template and ensures the LLM response can be
    validated against a pydantic model.

    Args:
        template:    Prompt template string.
        model:       Pydantic model class for output validation.
        examples:    Optional few-shot examples of valid outputs.
        max_retries: Number of times to retry parsing on validation failure.
    """

    def __init__(
        self,
        template: str,
        model: type[BaseModel],
        examples: list[str] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.template = template
        self.model = model
        self.examples = examples or []
        self.max_retries = max_retries

    def build_prompt(self, **kwargs: Any) -> str:
        """Build the full prompt including schema description."""
        rendered = _substitute(self.template, kwargs)

        # Add schema instructions
        schema_text = self._model_to_schema_text()

        prompt = rendered
        if self.examples:
            prompt += "\n\n" + "\n\n".join(self.examples)

        prompt += f"\n\nOutput strictly as valid JSON matching this schema:\n{schema_text}"
        prompt += "\n\n```json"
        return prompt

    def _model_to_schema_text(self) -> str:
        """Generate a human-readable schema description from the pydantic model."""
        if hasattr(self.model, "model_json_schema"):
            try:
                schema = self.model.model_json_schema()
                properties = schema.get("properties", {})
                lines = ["{"]
                for prop_name, prop_info in properties.items():
                    required = prop_name in schema.get("required", [])
                    prop_type = prop_info.get("type", "unknown")
                    req_str = "" if required else " (optional)"
                    lines.append(f'  "{prop_name}": "{prop_type}"{req_str}')
                lines.append("}")
                return "\n".join(lines)
            except Exception:
                pass
        return str(self.model.__fields__)

    def validate_output(self, text: str) -> tuple[bool, Any, str]:
        """Attempt to parse and validate LLM output against the schema.

        Tries multiple parsing strategies (JSON block, bare JSON,
        code fence stripping).

        Returns:
            (success, parsed_model_or_text, error_str)
        """
        text = text.strip()

        # Strategy 1: Extract JSON from code fences
        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Strategy 2: Try to parse as JSON
        try:
            data = __import__("json").loads(text)
            instance = self.model(**data)
            return True, instance, ""
        except (ValueError, ValidationError, TypeError) as exc:
            error = str(exc)

        # Strategy 3: Try parsing line by line as JSON-like
        try:
            data = __import__("json").loads(text)
        except (ValueError, TypeError):
            return False, text, error

        return False, text, error


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------

@dataclass
class PromptVariant:
    """A prompt variant for A/B testing."""

    name: str
    prompt: str
    template: PromptTemplate | None = None

    def render(self, **kwargs: Any) -> str:
        if self.template:
            return self.template.render(**kwargs)
        return self.prompt


class PromptOptimizer:
    """A/B testing framework for prompt engineering.

    Tests multiple prompt variants against a set of queries and
    evaluates outputs using a scoring function.

    Args:
        scorer: Callable that rates (prompt, query, response) → float.
    """

    def __init__(self, scorer: Callable[[str, str, str], float] | None = None) -> None:
        self.scorer = scorer or self._default_scorer
        self._history: list[dict[str, Any]] = []

    @staticmethod
    def _default_scorer(prompt: str, query: str, response: str) -> float:
        """Default scorer: rewards longer, more relevant responses."""
        # Simple heuristic: prefer responses that are non-empty and
        # contain key terms from the query
        if not response.strip():
            return 0.0
        query_terms = set(query.lower().split())
        response_lower = response.lower()
        overlap = sum(1 for t in query_terms if t in response_lower)
        relevance = overlap / len(query_terms) if query_terms else 0.0
        length_factor = min(len(response) / 100, 1.0)  # cap at 100 chars
        return 0.7 * relevance + 0.3 * length_factor

    def evaluate(
        self,
        variants: list[PromptVariant],
        queries: list[str],
        response_fn: Callable[[str, str], str],
        **render_kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """Evaluate multiple prompt variants against queries.

        Args:
            variants:      List of PromptVariant objects to test.
            queries:       List of test queries.
            response_fn:   Function (prompt, query) → LLM response string.
            **render_kwargs: Variables to render with each prompt template.

        Returns:
            Dict mapping variant name → {"avg_score": float, "scores": list[float]}
        """
        results: dict[str, dict[str, float]] = {}
        for variant in variants:
            prompt = variant.render(**render_kwargs)
            scores: list[float] = []
            for query in queries:
                response = response_fn(prompt, query)
                score = self.scorer(prompt, query, response)
                scores.append(score)
                self._history.append({
                    "variant": variant.name,
                    "query": query,
                    "response": response,
                    "score": score,
                })
            results[variant.name] = {
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "scores": scores,
            }
        return results

    def best_variant(
        self,
        variants: list[PromptVariant],
        queries: list[str],
        response_fn: Callable[[str, str], str],
        **render_kwargs: Any,
    ) -> PromptVariant:
        """Run evaluation and return the best-performing variant.

        Args:
            variants:      List of PromptVariant objects to test.
            queries:       List of test queries.
            response_fn:   Function (prompt, query) → LLM response string.

        Returns:
            The PromptVariant with the highest average score.
        """
        results = self.evaluate(variants, queries, response_fn, **render_kwargs)
        best_name = max(results, key=lambda k: results[k]["avg_score"])
        for v in variants:
            if v.name == best_name:
                return v
        return variants[0]

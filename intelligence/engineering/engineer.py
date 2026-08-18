"""Code engineering engine.

Wraps the foundation Generator for code-related tasks:

- **generate**: write code from a natural language description
- **explain**: explain what a piece of code does
- **review**: review code for issues and improvements
- **transform**: refactor or translate code between languages
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class CodeTaskType(enum.Enum):
    """Types of code engineering tasks."""

    GENERATE = "generate"
    EXPLAIN = "explain"
    REVIEW = "review"
    TRANSFORM = "transform"
    DEBUG = "debug"
    DOCUMENT = "document"


TASK_PROMPTS = {
    CodeTaskType.GENERATE: (
        "Write {language} code for the following:\n{description}\n\n"
        "Code:\n```{language}\n"
    ),
    CodeTaskType.EXPLAIN: (
        "Explain the following {language} code step by step:\n"
        "```{language}\n{code}\n```\n\n"
        "Explanation:\n"
    ),
    CodeTaskType.REVIEW: (
        "Review the following {language} code for bugs, performance, "
        "and style issues:\n```{language}\n{code}\n```\n\n"
        "Review:\n"
    ),
    CodeTaskType.TRANSFORM: (
        "Convert the following {source_language} code to {target_language}:\n"
        "```{source_language}\n{code}\n```\n\n"
        "```{target_language}\n"
    ),
    CodeTaskType.DEBUG: (
        "Find and fix the bug in this {language} code:\n"
        "```{language}\n{code}\n```\n"
        "Error: {error}\n\n"
        "Fixed code:\n```{language}\n"
    ),
    CodeTaskType.DOCUMENT: (
        "Add docstrings and comments to this {language} code:\n"
        "```{language}\n{code}\n```\n\n"
        "Documented code:\n```{language}\n"
    ),
}


@dataclass
class CodeTask:
    """A code engineering task."""

    task_type: CodeTaskType
    code: str = ""
    description: str = ""
    language: str = "python"
    source_language: str = ""
    target_language: str = ""
    error: str = ""


@dataclass
class CodeResult:
    """Result of a code engineering task."""

    task: CodeTask
    raw_response: str
    extracted_code: str = ""
    explanation: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether a code block was extracted from the response."""
        return bool(self.extracted_code.strip())


class CodeEngineer:
    """Code engineering using the foundation Generator.

    Usage::

        engineer = CodeEngineer(generator)
        result = engineer.solve(CodeTask(
            task_type=CodeTaskType.GENERATE,
            description="sort a list of numbers",
            language="python",
        ))
        print(result.extracted_code)
    """

    def __init__(
        self,
        generator,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._generator = generator
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def solve(self, task: CodeTask) -> CodeResult:
        """Execute a code engineering task."""
        prompt = self._build_prompt(task)
        gen_result = self._generator.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

        raw = gen_result.text
        code = self._extract_code_block(raw)
        issues = self._extract_issues(raw) if task.task_type == CodeTaskType.REVIEW else []

        return CodeResult(
            task=task,
            raw_response=raw,
            extracted_code=code,
            explanation=raw if task.task_type in (CodeTaskType.EXPLAIN, CodeTaskType.REVIEW) else "",
            issues=issues,
        )

    def generate(self, description: str, language: str = "python") -> CodeResult:
        """Shorthand: generate code from a description."""
        return self.solve(CodeTask(
            task_type=CodeTaskType.GENERATE,
            description=description,
            language=language,
        ))

    def explain(self, code: str, language: str = "python") -> CodeResult:
        """Shorthand: explain a piece of code."""
        return self.solve(CodeTask(
            task_type=CodeTaskType.EXPLAIN,
            code=code,
            language=language,
        ))

    def review(self, code: str, language: str = "python") -> CodeResult:
        """Shorthand: review code for issues."""
        return self.solve(CodeTask(
            task_type=CodeTaskType.REVIEW,
            code=code,
            language=language,
        ))

    def _build_prompt(self, task: CodeTask) -> str:
        template = TASK_PROMPTS.get(task.task_type, TASK_PROMPTS[CodeTaskType.GENERATE])
        return template.format(
            code=task.code,
            description=task.description,
            language=task.language,
            source_language=task.source_language or task.language,
            target_language=task.target_language or "python",
            error=task.error,
        )

    def _extract_code_block(self, text: str) -> str:
        """Extract the first fenced code block."""
        match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_issues(self, text: str) -> list[str]:
        """Extract identified issues from a code review."""
        issues: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^[-*]\s+.+", line):
                issues.append(re.sub(r"^[-*]\s+", "", line))
        return issues

"""Task decomposition and planning engine.

Wraps the foundation Generator to:

1. Decompose a high-level goal into concrete steps.
2. Track execution state per step (pending / running / done / failed).
3. Re-plan if a step fails.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class PlanStatus(enum.Enum):
    """Execution status of a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""

    description: str
    status: PlanStatus = PlanStatus.PENDING
    result: str = ""
    step_id: int = 0

    @property
    def completed(self) -> bool:
        return self.status in (PlanStatus.DONE, PlanStatus.SKIPPED)


@dataclass
class Plan:
    """A decomposed plan with steps and execution state."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def all_done(self) -> bool:
        return bool(self.steps) and all(s.completed for s in self.steps)

    @property
    def any_failed(self) -> bool:
        return any(s.status == PlanStatus.FAILED for s in self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.completed)
        return done / len(self.steps)

    def next_step(self) -> PlanStep | None:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                return step
        return None

    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.completed]


DECOMPOSE_TEMPLATE = (
    "Break down the following goal into numbered steps:\n"
    "Goal: {goal}\n"
    "Provide up to {max_steps} concrete steps.\n"
    "Steps:\n"
)

REPLAN_TEMPLATE = (
    "The following plan failed at step {failed_step}:\n"
    "Goal: {goal}\n"
    "Steps attempted:\n{steps}\n"
    "Failure: {failure}\n"
    "Provide a revised plan (numbered steps) to achieve the goal:\n"
)


class Planner:
    """Task decomposition and planning using the foundation Generator.

    Usage::

        planner = Planner(generator)
        plan = planner.decompose("Train a math model on arithmetic data")
        for step in plan.steps:
            print(f"Step {step.step_id}: {step.description}")
    """

    def __init__(
        self,
        generator,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_steps: int = 10,
    ) -> None:
        self._generator = generator
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._max_steps = max_steps

    def decompose(self, goal: str) -> Plan:
        """Decompose a goal into a plan with numbered steps."""
        prompt = DECOMPOSE_TEMPLATE.format(goal=goal, max_steps=self._max_steps)
        result = self._generator.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        steps = self._parse_steps(result.text)
        return Plan(goal=goal, steps=steps)

    def replan(self, plan: Plan, failed_step: PlanStep, failure_reason: str) -> Plan:
        """Re-plan after a step failure."""
        steps_text = "\n".join(
            f"  {s.step_id}. [{s.status.value}] {s.description}"
            for s in plan.steps
        )
        prompt = REPLAN_TEMPLATE.format(
            failed_step=failed_step.step_id,
            goal=plan.goal,
            steps=steps_text,
            failure=failure_reason,
        )
        result = self._generator.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        new_steps = self._parse_steps(result.text)
        return Plan(goal=plan.goal, steps=new_steps)

    def summarize_plan(self, plan: Plan) -> str:
        """Generate a natural language summary of the plan."""
        lines = [f"Goal: {plan.goal}"]
        lines.append(f"Progress: {plan.progress:.0%}")
        for step in plan.steps:
            marker = {"pending": "[ ]", "running": "[~]", "done": "[x]", "failed": "[!]", "skipped": "[-]"}
            lines.append(f"  {marker.get(step.status.value, '[?]')} Step {step.step_id}: {step.description}")
        return "\n".join(lines)

    def _parse_steps(self, text: str) -> list[PlanStep]:
        """Parse numbered steps from generated text."""
        steps: list[PlanStep] = []
        for line in text.split("\n"):
            line = line.strip()
            match = re.match(r"^(\d+)[\.\)]\s*(.+)", line)
            if match:
                step_id = int(match.group(1))
                description = match.group(2).strip()
                steps.append(PlanStep(description=description, step_id=step_id))
        return steps

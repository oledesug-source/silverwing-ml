"""Simple sequential workflow engine.

A ``Workflow`` is a named sequence of ``WorkflowStep`` objects, each
invoking a registered capability.  Workflows provide higher-level
composition on top of the ``Orchestrator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .orchestration import ChatRequest, Orchestrator


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Attributes:
        name: Human-readable step name.
        capability_name: Name of the capability to invoke.
        input_template: Template string with ``{prev_output}`` and ``{user_input}`` placeholders.
        description: Optional description.
    """

    name: str
    capability_name: str
    input_template: str = "{prev_output}"
    description: str = ""

    def render(self, user_input: str, prev_output: str = "") -> str:
        """Render the input template with concrete values."""
        return self.input_template.format(
            user_input=user_input,
            prev_output=prev_output,
        )


@dataclass
class Workflow:
    """A named sequence of workflow steps.

    Usage::

        workflow = Workflow(
            name="math_chain",
            steps=[
                WorkflowStep("parse", "calculator", "{user_input}"),
            ],
        )
        result = workflow.execute(orchestrator, "What is 2+2?")
    """

    name: str
    steps: list[WorkflowStep] = field(default_factory=list)
    description: str = ""

    def execute(
        self,
        orchestrator: Orchestrator,
        initial_input: str,
    ) -> WorkflowResult:
        """Execute the workflow step by step.

        Each step's rendered input is sent through the Orchestrator as a
        chat request, and its output becomes the next step's context.
        """
        step_results: list[StepResult] = []
        prev_output = ""

        for step in self.steps:
            rendered = step.render(initial_input, prev_output)
            request = ChatRequest(
                message=rendered,
                metadata={"workflow": self.name, "step": step.name},
            )
            response = orchestrator.handle_request(request)
            prev_output = response.text
            step_results.append(StepResult(
                step=step,
                input_text=rendered,
                output_text=response.text,
                success=response.success,
                rounds=response.rounds,
            ))

        final_text = prev_output
        return WorkflowResult(
            workflow=self,
            input_text=initial_input,
            output_text=final_text,
            step_results=step_results,
        )


@dataclass
class StepResult:
    """Result of a single workflow step."""

    step: WorkflowStep
    input_text: str
    output_text: str
    success: bool = True
    rounds: int = 0


@dataclass
class WorkflowResult:
    """Result of executing a complete workflow."""

    workflow: Workflow
    input_text: str
    output_text: str
    step_results: list[StepResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(sr.success for sr in self.step_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.name,
            "input": self.input_text,
            "output": self.output_text,
            "steps": [
                {
                    "name": sr.step.name,
                    "success": sr.success,
                    "rounds": sr.rounds,
                    "output": sr.output_text,
                }
                for sr in self.step_results
            ],
            "all_succeeded": self.all_succeeded,
        }

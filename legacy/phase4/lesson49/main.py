# Silverwing ML
# Phase 4 - Lesson 49
# Agent Planning and Multi-Step Execution


import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 49")
print("Agent Planning and Multi-Step Execution")
print()


# ==================================================
# 1. BASIC TOOL IMPLEMENTATIONS
# ==================================================

def calculator(expression: str) -> float:

    allowed_characters = (
        "0123456789+-*/(). "
    )

    if any(
            character not in allowed_characters
            for character in expression
    ):
        raise ValueError(
            "Unsupported characters in expression."
        )

    result = eval(
        expression,
        {"__builtins__": {}},
        {}
    )

    if not isinstance(
            result,
            (int, float)
    ):
        raise ValueError(
            "Expression must return a number."
        )

    return float(result)


def analyze_machine(
        temperature: float,
        pressure: float,
        rpm: float
) -> Dict[str, Any]:

    score = 0

    if temperature >= 100:
        score += 40

    elif temperature >= 80:
        score += 20

    if rpm > 3000:
        score += 40

    elif rpm > 2500:
        score += 15

    if pressure >= 160:
        score += 20

    elif pressure >= 130:
        score += 10

    if score >= 70:
        level = "CRITICAL"

    elif score >= 40:
        level = "HIGH"

    elif score >= 20:
        level = "MEDIUM"

    else:
        level = "LOW"

    return {
        "risk_score": score,
        "risk_level": level
    }


def summarize_results(
        results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    successful = [
        result
        for result in results
        if result["status"] == "success"
    ]

    failed = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed)
    }


def current_time():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. TOOL REGISTRY
# ==================================================

class ToolRegistry:

    def __init__(self):

        self.tools = {
            "calculator": calculator,
            "machine_analyzer": analyze_machine,
            "summarize_results": summarize_results,
            "current_time": current_time
        }


    def execute(
            self,
            tool_name: str,
            arguments: Dict[str, Any]
    ):

        if tool_name not in self.tools:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        return self.tools[
            tool_name
        ](
            **arguments
        )


registry = ToolRegistry()


print("TEST 1: Tool Registry")
print()

print(
    "Available tools:"
)

for name in registry.tools:

    print(
        "-",
        name
    )

print()


# ==================================================
# 3. PLAN DATA STRUCTURE
# ==================================================

@dataclass
class PlanStep:

    step_id: str

    description: str

    tool_name: str

    arguments: Dict[str, Any]

    depends_on: List[str] = field(
        default_factory=list
    )

    status: str = "pending"

    result: Any = None

    error: str | None = None


print("TEST 2: Plan Step")
print()


example_step = PlanStep(
    step_id="step-1",
    description="Calculate a value.",
    tool_name="calculator",
    arguments={
        "expression": "25 * 8"
    }
)


print(
    example_step
)

print()


# ==================================================
# 4. EXECUTION PLAN
# ==================================================

@dataclass
class ExecutionPlan:

    goal: str

    steps: List[PlanStep]

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    def get_step(
            self,
            step_id: str
    ):

        for step in self.steps:

            if step.step_id == step_id:

                return step

        return None


    def completed_steps(self):

        return [
            step
            for step in self.steps
            if step.status == "completed"
        ]


    def failed_steps(self):

        return [
            step
            for step in self.steps
            if step.status == "failed"
        ]


print("TEST 3: Execution Plan")
print()


plan = ExecutionPlan(
    goal=(
        "Analyze machine condition and calculate "
        "a supporting value."
    ),
    steps=[
        PlanStep(
            step_id="step-1",
            description=(
                "Calculate baseline value."
            ),
            tool_name="calculator",
            arguments={
                "expression": "25 * 8"
            }
        ),
        PlanStep(
            step_id="step-2",
            description=(
                "Analyze machine."
            ),
            tool_name="machine_analyzer",
            arguments={
                "temperature": 97,
                "pressure": 130,
                "rpm": 2600
            }
        )
    ]
)


print(
    "Goal:",
    plan.goal
)

print()

for step in plan.steps:

    print(
        step.step_id,
        "->",
        step.description
    )

print()


# ==================================================
# 5. PLANNER
# ==================================================

class Planner:
    """
    Converts a high-level goal into executable steps.

    This lesson uses deterministic planning so
    the architecture is easy to understand.
    Later, an LLM can generate these plans.
    """

    def create_machine_analysis_plan(
            self,
            temperature,
            pressure,
            rpm
    ):

        return ExecutionPlan(
            goal=(
                "Analyze machine risk and prepare "
                "a final assessment."
            ),
            steps=[
                PlanStep(
                    step_id="analysis",
                    description=(
                        "Analyze machine operating conditions."
                    ),
                    tool_name="machine_analyzer",
                    arguments={
                        "temperature":
                            temperature,
                        "pressure":
                            pressure,
                        "rpm":
                            rpm
                    }
                ),
                PlanStep(
                    step_id="timestamp",
                    description=(
                        "Record analysis time."
                    ),
                    tool_name="current_time",
                    arguments={}
                ),
                PlanStep(
                    step_id="summary",
                    description=(
                        "Summarize completed operations."
                    ),
                    tool_name="summarize_results",
                    arguments={},
                    depends_on=[
                        "analysis",
                        "timestamp"
                    ]
                )
            ]
        )


planner = Planner()


print("TEST 4: Create Machine Analysis Plan")
print()


machine_plan = (
    planner.create_machine_analysis_plan(
        temperature=97,
        pressure=130,
        rpm=2600
    )
)


print(
    "Goal:",
    machine_plan.goal
)

print()

for step in machine_plan.steps:

    print(
        step.step_id,
        "|",
        step.tool_name,
        "| depends on:",
        step.depends_on
    )

print()


# ==================================================
# 6. DEPENDENCY CHECK
# ==================================================

def dependencies_completed(
        plan: ExecutionPlan,
        step: PlanStep
):

    for dependency_id in (
            step.depends_on
    ):

        dependency = plan.get_step(
            dependency_id
        )

        if dependency is None:

            return False

        if dependency.status != "completed":

            return False

    return True


# ==================================================
# 7. PLANNING CONTEXT
# ==================================================

def build_execution_context(
        plan: ExecutionPlan
):

    context = []

    for step in plan.steps:

        if step.status == "completed":

            context.append(
                {
                    "step_id": step.step_id,
                    "result": step.result
                }
            )

    return context


# ==================================================
# 8. RESULT-AWARE ARGUMENT RESOLUTION
# ==================================================

def resolve_arguments(
        step: PlanStep,
        plan: ExecutionPlan
):

    arguments = dict(
        step.arguments
    )

    # Example:
    # The summary tool can consume the outputs
    # of previously completed steps.

    if (
            step.tool_name
            ==
            "summarize_results"
    ):

        completed_results = []

        for dependency_id in (
                step.depends_on
        ):

            dependency = plan.get_step(
                dependency_id
            )

            if dependency is not None:

                completed_results.append(
                    {
                        "step_id":
                            dependency.step_id,

                        "status":
                            dependency.status,

                        "result":
                            dependency.result
                    }
                )


        arguments["results"] = (
            completed_results
        )


    return arguments


# ==================================================
# 9. AGENT EXECUTOR
# ==================================================

class AgentExecutor:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def execute_plan(
            self,
            plan: ExecutionPlan
    ):

        print(
            "PLAN EXECUTION STARTED"
        )

        print()

        while True:

            pending_steps = [
                step
                for step in plan.steps
                if step.status == "pending"
            ]


            if not pending_steps:

                break


            progress = False


            for step in pending_steps:

                if not dependencies_completed(
                        plan,
                        step
                ):

                    continue


                print(
                    "Executing:",
                    step.step_id
                )

                print(
                    "Tool:",
                    step.tool_name
                )

                print(
                    "Description:",
                    step.description
                )


                step.status = "running"


                try:

                    arguments = resolve_arguments(
                        step,
                        plan
                    )


                    result = (
                        self.registry.execute(
                            step.tool_name,
                            arguments
                        )
                    )


                    step.result = result

                    step.status = "completed"

                    progress = True


                    print(
                        "Result:",
                        result
                    )

                    print()


                except Exception as error:

                    step.status = "failed"

                    step.error = str(
                        error
                    )

                    progress = True


                    print(
                        "Error:",
                        error
                    )

                    print()


            if not progress:

                unresolved = [
                    step.step_id
                    for step in plan.steps
                    if step.status
                       == "pending"
                ]

                raise RuntimeError(
                    "Plan cannot make progress. "
                    f"Unresolved steps: {unresolved}"
                )


        return plan


executor = AgentExecutor(
    registry
)


# ==================================================
# 10. EXECUTE PLAN
# ==================================================

print("TEST 5: Execute Plan")
print()


executed_plan = (
    executor.execute_plan(
        machine_plan
    )
)


# ==================================================
# 11. PLAN STATUS
# ==================================================

print("TEST 6: Plan Status")
print()


for step in executed_plan.steps:

    print(
        step.step_id,
        "->",
        step.status
    )

    print(
        "Result:",
        step.result
    )

    if step.error:

        print(
            "Error:",
            step.error
        )

    print()


# ==================================================
# 12. COMPLETION CHECK
# ==================================================

print("TEST 7: Completion Check")
print()


completed = (
    len(
        executed_plan.completed_steps()
    )
)


failed = (
    len(
        executed_plan.failed_steps()
    )
)


print(
    "Completed:",
    completed
)

print(
    "Failed:",
    failed
)

print()


# ==================================================
# 13. PLAN SERIALIZATION
# ==================================================

print("TEST 8: Serialize Plan")
print()


def serialize_plan(
        plan: ExecutionPlan
):

    return {
        "goal": plan.goal,
        "created_at": plan.created_at,
        "steps": [
            {
                "step_id":
                    step.step_id,

                "description":
                    step.description,

                "tool_name":
                    step.tool_name,

                "arguments":
                    step.arguments,

                "depends_on":
                    step.depends_on,

                "status":
                    step.status,

                "result":
                    step.result,

                "error":
                    step.error
            }
            for step in plan.steps
        ]
    }


serialized_plan = (
    serialize_plan(
        executed_plan
    )
)


print(
    json.dumps(
        serialized_plan,
        indent=4
    )
)

print()


# ==================================================
# 14. MULTI-TASKING PLAN
# ==================================================

print("TEST 9: Multi-Tasking Plan")
print()


multitask_plan = ExecutionPlan(
    goal=(
        "Perform several independent operations."
    ),
    steps=[
        PlanStep(
            step_id="task-a",
            description="Calculate A.",
            tool_name="calculator",
            arguments={
                "expression": "20 * 5"
            }
        ),
        PlanStep(
            step_id="task-b",
            description="Calculate B.",
            tool_name="calculator",
            arguments={
                "expression": "100 / 4"
            }
        ),
        PlanStep(
            step_id="task-c",
            description="Get current time.",
            tool_name="current_time",
            arguments={}
        ),
        PlanStep(
            step_id="task-d",
            description="Analyze machine.",
            tool_name="machine_analyzer",
            arguments={
                "temperature": 105,
                "pressure": 140,
                "rpm": 3200
            }
        )
    ]
)


multitask_result = (
    executor.execute_plan(
        multitask_plan
    )
)


print(
    "Completed tasks:"
)

for step in (
        multitask_result.steps
):

    print(
        step.step_id,
        "->",
        step.result
    )

print()


# ==================================================
# 15. DEPENDENT TASK EXAMPLE
# ==================================================

print("TEST 10: Dependent Tasks")
print()


dependent_plan = ExecutionPlan(
    goal=(
        "Perform analysis and then summarize it."
    ),
    steps=[
        PlanStep(
            step_id="measurement",
            description=(
                "Calculate measurement."
            ),
            tool_name="calculator",
            arguments={
                "expression": "12 * 15"
            }
        ),
        PlanStep(
            step_id="analysis",
            description=(
                "Analyze machine."
            ),
            tool_name="machine_analyzer",
            arguments={
                "temperature": 101,
                "pressure": 135,
                "rpm": 3100
            }
        ),
        PlanStep(
            step_id="summary",
            description=(
                "Summarize earlier results."
            ),
            tool_name="summarize_results",
            arguments={},
            depends_on=[
                "measurement",
                "analysis"
            ]
        )
    ]
)


dependent_result = (
    executor.execute_plan(
        dependent_plan
    )
)


for step in (
        dependent_result.steps
):

    print(
        step.step_id,
        "->",
        step.status
    )

    print(
        "Result:",
        step.result
    )

    print()


# ==================================================
# 16. FAILURE HANDLING
# ==================================================

print("TEST 11: Failure Handling")
print()


failure_plan = ExecutionPlan(
    goal=(
        "Demonstrate a failed tool call."
    ),
    steps=[
        PlanStep(
            step_id="valid",
            description="Valid operation.",
            tool_name="calculator",
            arguments={
                "expression": "10 * 10"
            }
        ),
        PlanStep(
            step_id="invalid",
            description="Invalid operation.",
            tool_name="unknown_tool",
            arguments={}
        )
    ]
)


failure_result = (
    executor.execute_plan(
        failure_plan
    )
)


for step in (
        failure_result.steps
):

    print(
        step.step_id,
        "->",
        step.status
    )

    print(
        "Error:",
        step.error
    )

    print()


# ==================================================
# 17. VERIFICATION
# ==================================================

print("TEST 12: Verification")
print()


def verify_plan(
        plan: ExecutionPlan
):

    verification = {
        "complete":
            True,

        "failed_steps":
            [],

        "missing_results":
            []
    }


    for step in plan.steps:

        if step.status == "failed":

            verification[
                "complete"
            ] = False

            verification[
                "failed_steps"
            ].append(
                step.step_id
            )


        if (
                step.status == "completed"
                and
                step.result is None
        ):

            verification[
                "complete"
            ] = False

            verification[
                "missing_results"
            ].append(
                step.step_id
            )


    return verification


verification = verify_plan(
    executed_plan
)


print(
    json.dumps(
        verification,
        indent=4
    )
)

print()


# ==================================================
# 18. AGENT LOOP
# ==================================================

print("AGENT LOOP")
print()

print("User Goal")
print("   ↓")
print("Understand Goal")
print("   ↓")
print("Create Plan")
print("   ↓")
print("Resolve Dependencies")
print("   ↓")
print("Execute Tasks")
print("   ↓")
print("Observe Results")
print("   ↓")
print("Verify")
print("   ↓")
print("Re-plan if necessary")
print("   ↓")
print("Final Response")

print()


# ==================================================
# 19. MULTITASKING ARCHITECTURE
# ==================================================

print("MULTITASKING ARCHITECTURE")
print()

print("                    Agent")
print("                      │")
print("                    Planner")
print("                      │")
print("              Execution Plan")
print("                      │")
print("       ┌──────────────┼──────────────┐")
print("       ↓              ↓              ↓")
print("    Task A          Task B          Task C")
print("       ↓              ↓              ↓")
print("     Tool A         Tool B         Tool C")
print("       └──────────────┼──────────────┘")
print("                      ↓")
print("                    Results")
print("                      ↓")
print("                  Verifier")
print("                      ↓")
print("                 Final Answer")

print()


# ==================================================
# 20. SEQUENTIAL VS PARALLEL
# ==================================================

print("SEQUENTIAL VS PARALLEL EXECUTION")
print()

print(
    "Independent tasks can potentially run "
    "in parallel."
)

print()

print(
    "Dependent tasks must wait for their "
    "required results."
)

print()

print(
    "A future scheduler can decide which "
    "operations are safe to execute concurrently."
)

print()


# ==================================================
# 21. FUTURE TASK SCHEDULER
# ==================================================

print("FUTURE TASK SCHEDULER")
print()

print("Task Queue")
print("    ↓")
print("Dependency Graph")
print("    ↓")
print("Scheduler")
print("    ↓")
print("Worker Pool")
print("    ↓")
print("Tool / Service Execution")
print("    ↓")
print("Results")
print("    ↓")
print("Verifier")

print()


# ==================================================
# 22. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This lesson uses deterministic plans."
)

print()

print(
    "A modern agent can use an LLM to generate "
    "plans dynamically, but production systems "
    "still need validation, permissions, "
    "timeouts, retries, observability, and "
    "resource limits."
)

print()


# ==================================================
# 23. SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("LLM")
print(" ↓")
print("Conversation")
print(" ↓")
print("Memory")
print(" ↓")
print("Semantic Retrieval")
print(" ↓")
print("Tool Registry")
print(" ↓")
print("Structured Tool Calls")
print(" ↓")
print("Planning")
print(" ↓")
print("Multi-Step Execution")
print(" ↓")
print("Verification")
print(" ↓")
print("Multitasking Agent")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 49 COMPLETE ===")

# Silverwing ML
# Phase 4 - Lesson 51
# Verification, Retries and Self-Correction


import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 51")
print("Verification, Retries and Self-Correction")
print()


# ==================================================
# 1. UTILITIES
# ==================================================

def timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. TOOL RESULT
# ==================================================

@dataclass
class ToolResult:

    success: bool

    result: Any = None

    error: Optional[str] = None

    attempts: int = 0

    timestamp: str = field(
        default_factory=timestamp
    )


# ==================================================
# 3. SIMULATED TOOLS
# ==================================================

async def reliable_calculator(
        expression: str
):

    await asyncio.sleep(0.3)

    allowed = "0123456789+-*/(). "

    if any(
            character not in allowed
            for character in expression
    ):

        raise ValueError(
            "Unsupported characters."
        )

    result = eval(
        expression,
        {"__builtins__": {}},
        {}
    )

    return result


async def unreliable_service(
        failure_probability: float = 0.5
):

    await asyncio.sleep(0.5)

    if random.random() < failure_probability:

        raise RuntimeError(
            "Temporary service failure."
        )

    return {
        "status": "healthy",
        "message": "Service responded successfully."
    }


async def machine_analyzer(
        temperature: float,
        pressure: float,
        rpm: float
):

    await asyncio.sleep(0.4)

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


async def intentionally_inconsistent_tool():

    await asyncio.sleep(0.2)

    # Simulates a result that may fail validation.

    value = random.choice(
        [
            {"status": "ok", "value": 42},
            {"status": "ok"},
            {"status": "error", "value": None}
        ]
    )

    return value


# ==================================================
# 4. VERIFIER
# ==================================================

class Verifier:
    """
    Checks whether tool results satisfy
    expected conditions.
    """

    def verify(
            self,
            tool_name: str,
            result: Any
    ) -> Dict[str, Any]:

        if result is None:

            return {
                "valid": False,
                "reason": "Result is None."
            }


        if tool_name == "calculator":

            if isinstance(
                    result,
                    (int, float)
            ):

                return {
                    "valid": True,
                    "reason": "Numeric result verified."
                }


            return {
                "valid": False,
                "reason":
                    "Calculator result is not numeric."
            }


        if tool_name == "health_check":

            if (
                    isinstance(result, dict)
                    and
                    result.get("status")
                    == "healthy"
            ):

                return {
                    "valid": True,
                    "reason":
                        "Health status verified."
                }


            return {
                "valid": False,
                "reason":
                    "Service health check failed."
            }


        if tool_name == "machine_analyzer":

            required_keys = {
                "risk_score",
                "risk_level"
            }


            if not required_keys.issubset(
                    result.keys()
            ):

                return {
                    "valid": False,
                    "reason":
                        "Machine result is missing fields."
                }


            if result["risk_level"] not in {
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            }:

                return {
                    "valid": False,
                    "reason":
                        "Invalid risk level."
                }


            return {
                "valid": True,
                "reason":
                    "Machine analysis verified."
            }


        if tool_name == "inconsistent_tool":

            if (
                    result.get("status")
                    == "ok"
                    and
                    "value"
                    in result
                    and
                    result["value"]
                    is not None
            ):

                return {
                    "valid": True,
                    "reason":
                        "Required fields verified."
                }


            return {
                "valid": False,
                "reason":
                    "Incomplete or invalid result."
            }


        return {
            "valid": True,
            "reason": "No special validation rule."
        }


verifier = Verifier()


# ==================================================
# 5. RETRY POLICY
# ==================================================

@dataclass
class RetryPolicy:

    max_attempts: int = 3

    initial_delay: float = 0.5

    backoff: float = 2.0


# ==================================================
# 6. RETRY EXECUTOR
# ==================================================

class ReliableExecutor:

    def __init__(
            self,
            retry_policy: RetryPolicy,
            verifier: Verifier
    ):

        self.retry_policy = retry_policy

        self.verifier = verifier


    async def execute(
            self,
            tool_name: str,
            function,
            arguments: Dict[str, Any]
    ) -> ToolResult:

        delay = self.retry_policy.initial_delay


        for attempt in range(
                1,
                self.retry_policy.max_attempts + 1
        ):

            print(
                f"Attempt {attempt}: {tool_name}"
            )


            try:

                result = await function(
                    **arguments
                )


                verification = (
                    self.verifier.verify(
                        tool_name,
                        result
                    )
                )


                if verification["valid"]:

                    print(
                        "Verification:",
                        verification["reason"]
                    )


                    return ToolResult(
                        success=True,
                        result=result,
                        attempts=attempt
                    )


                print(
                    "Verification failed:",
                    verification["reason"]
                )


                if (
                        attempt
                        >= self.retry_policy.max_attempts
                ):

                    return ToolResult(
                        success=False,
                        result=result,
                        error=(
                            verification[
                                "reason"
                            ]
                        ),
                        attempts=attempt
                    )


            except Exception as error:

                print(
                    "Execution failed:",
                    error
                )


                if (
                        attempt
                        >= self.retry_policy.max_attempts
                ):

                    return ToolResult(
                        success=False,
                        error=str(error),
                        attempts=attempt
                    )


            print(
                f"Retrying in {delay:.2f}s..."
            )


            await asyncio.sleep(
                delay
            )


            delay *= self.retry_policy.backoff


        return ToolResult(
            success=False,
            error="Maximum attempts reached."
        )


# ==================================================
# 7. CREATE EXECUTOR
# ==================================================

retry_policy = RetryPolicy(
    max_attempts=4,
    initial_delay=0.3,
    backoff=2.0
)


executor = ReliableExecutor(
    retry_policy,
    verifier
)


# ==================================================
# 8. RELIABLE CALCULATOR
# ==================================================

print("TEST 1: Reliable Calculator")
print()


async def test_calculator():

    result = await executor.execute(
        "calculator",
        reliable_calculator,
        {
            "expression": "125 * 8 / 4"
        }
    )


    print(
        "Final result:",
        result
    )


asyncio.run(
    test_calculator()
)


print()


# ==================================================
# 9. UNRELIABLE SERVICE
# ==================================================

print("TEST 2: Retry Unreliable Service")
print()


async def test_unreliable_service():

    result = await executor.execute(
        "health_check",
        unreliable_service,
        {
            "failure_probability": 0.6
        }
    )


    print(
        "Final result:",
        result
    )


asyncio.run(
    test_unreliable_service()
)


print()


# ==================================================
# 10. MACHINE VERIFICATION
# ==================================================

print("TEST 3: Machine Result Verification")
print()


async def test_machine():

    result = await executor.execute(
        "machine_analyzer",
        machine_analyzer,
        {
            "temperature": 105,
            "pressure": 140,
            "rpm": 3200
        }
    )


    print(
        "Final result:",
        result
    )


asyncio.run(
    test_machine()
)


print()


# ==================================================
# 11. INCONSISTENT TOOL
# ==================================================

print("TEST 4: Invalid Result Recovery")
print()


async def test_inconsistent_tool():

    result = await executor.execute(
        "inconsistent_tool",
        intentionally_inconsistent_tool,
        {}
    )


    print(
        "Final result:",
        result
    )


asyncio.run(
    test_inconsistent_tool()
)


print()


# ==================================================
# 12. REASONING STATE
# ==================================================

@dataclass
class AgentState:

    goal: str

    current_step: str = ""

    observations: List[Dict[str, Any]] = field(
        default_factory=list
    )

    corrections: List[str] = field(
        default_factory=list
    )

    completed: bool = False


# ==================================================
# 13. SELF-CORRECTING AGENT
# ==================================================

class SelfCorrectingAgent:

    def __init__(
            self,
            executor
    ):

        self.executor = executor


    async def run(
            self,
            state: AgentState
    ):

        print(
            "AGENT GOAL:"
        )

        print(
            state.goal
        )

        print()


        state.current_step = (
            "Analyze machine."
        )


        result = await self.executor.execute(
            "machine_analyzer",
            machine_analyzer,
            {
                "temperature": 102,
                "pressure": 135,
                "rpm": 3100
            }
        )


        state.observations.append(
            {
                "step":
                    state.current_step,

                "result":
                    result.result,

                "success":
                    result.success,

                "attempts":
                    result.attempts
            }
        )


        if not result.success:

            state.corrections.append(
                (
                    "Machine analysis failed; "
                    "retry required."
                )
            )


            return state


        machine_result = result.result


        state.current_step = (
            "Check machine risk."
        )


        risk_level = (
            machine_result["risk_level"]
        )


        if risk_level in {
            "HIGH",
            "CRITICAL"
        }:

            state.corrections.append(
                (
                    "Elevated machine risk detected; "
                    "additional verification required."
                )
            )


            verification_result = (
                await self.executor.execute(
                    "machine_analyzer",
                    machine_analyzer,
                    {
                        "temperature": 102,
                        "pressure": 135,
                        "rpm": 3100
                    }
                )
            )


            state.observations.append(
                {
                    "step":
                        "Verification analysis",

                    "result":
                        verification_result.result,

                    "success":
                        verification_result.success,

                    "attempts":
                        verification_result.attempts
                }
            )


            if not verification_result.success:

                state.completed = False

                return state


            if (
                    verification_result.result
                    !=
                    machine_result
            ):

                state.corrections.append(
                    (
                        "Verification disagreed with "
                        "the original result."
                    )
                )


                state.completed = False

                return state


        state.completed = True

        return state


# ==================================================
# 14. RUN SELF-CORRECTING AGENT
# ==================================================

print("TEST 5: Self-Correcting Agent")
print()


agent = SelfCorrectingAgent(
    executor
)


agent_state = AgentState(
    goal=(
        "Analyze the machine and verify that "
        "the resulting risk assessment is reliable."
    )
)


async def run_agent():

    return await agent.run(
        agent_state
    )


final_state = asyncio.run(
    run_agent()
)


print(
    "Completed:",
    final_state.completed
)

print()

print(
    "Current step:",
    final_state.current_step
)

print()

print(
    "Observations:"
)

for observation in (
        final_state.observations
):

    print(
        observation
    )

print()

print(
    "Corrections:"
)

for correction in (
        final_state.corrections
):

    print(
        "-",
        correction
    )

print()


# ==================================================
# 15. RESULT CONFIDENCE
# ==================================================

print("TEST 6: Result Confidence")
print()


def confidence_from_verification(
        success,
        attempts,
        verification_count
):

    if not success:

        return 0.0


    base = 1.0


    # More attempts slightly reduce confidence.

    attempt_penalty = (
            0.05
            *
            max(
                attempts - 1,
                0
            )
    )


    verification_bonus = min(
        0.05
        *
        verification_count,
        0.1
    )


    confidence = (
            base
            -
            attempt_penalty
            +
            verification_bonus
    )


    return max(
        0.0,
        min(
            1.0,
            confidence
        )
    )


confidence = (
    confidence_from_verification(
        success=True,
        attempts=1,
        verification_count=1
    )
)


print(
    "Example confidence:",
    confidence
)

print()


# ==================================================
# 16. RETRY VS REPLAN
# ==================================================

print("TEST 7: Retry vs Re-Plan")
print()


print(
    "Retry is appropriate when:"
)

print(
    "- the operation may succeed if repeated."
)

print(
    "- the input is still valid."
)

print(
    "- the failure appears temporary."
)

print()


print(
    "Re-planning is appropriate when:"
)

print(
    "- the strategy itself is wrong."
)

print(
    "- a required dependency changed."
)

print(
    "- repeated attempts fail."
)

print(
    "- another tool or approach is needed."
)

print()


# ==================================================
# 17. FAILURE CLASSIFICATION
# ==================================================

print("TEST 8: Failure Classification")
print()


def classify_failure(
        error_text
):

    text = error_text.lower()


    if any(
            word in text
            for word in [
                "timeout",
                "temporarily",
                "connection"
            ]
    ):

        return "transient"


    if any(
            word in text
            for word in [
                "permission",
                "unauthorized",
                "forbidden"
            ]
    ):

        return "permission"


    if any(
            word in text
            for word in [
                "invalid",
                "missing",
                "unsupported"
            ]
    ):

        return "input"


    return "unknown"


for error in [
    "Temporary service failure.",
    "Permission denied.",
    "Invalid argument.",
    "Unknown execution error."
]:

    print(
        error,
        "->",
        classify_failure(error)
    )

print()


# ==================================================
# 18. AGENT SELF-CORRECTION LOOP
# ==================================================

print("SELF-CORRECTION LOOP")
print()

print("Goal")
print(" ↓")
print("Plan")
print(" ↓")
print("Execute")
print(" ↓")
print("Observe")
print(" ↓")
print("Verify")
print(" ↓")
print(" ┌──────────────────────┐")
print(" │ Valid?               │")
print(" └───────┬──────────────┘")
print("         │")
print("    ┌────┴────┐")
print("    ↓         ↓")
print("   YES        NO")
print("    ↓         ↓")
print(" Continue   Classify Failure")
print("              ↓")
print("        ┌─────┴──────┐")
print("        ↓            ↓")
print("      Retry       Re-plan")
print("        │            │")
print("        └─────┬──────┘")
print("              ↓")
print("          Execute")
print("              ↓")
print("           Verify")

print()


# ==================================================
# 19. ADVANCED AGENT ARCHITECTURE
# ==================================================

print("ADVANCED SILVERWING AGENT")
print()

print("User Goal")
print("   ↓")
print("LLM Reasoning")
print("   ↓")
print("Planner")
print("   ↓")
print("Task Scheduler")
print("   ↓")
print("Tool Execution")
print("   ↓")
print("Observation")
print("   ↓")
print("Verifier")
print("   ↓")
print("Failure Classifier")
print("   ↓")
print("Retry / Re-plan")
print("   ↓")
print("Verification")
print("   ↓")
print("Final Response")

print()


# ==================================================
# 20. PRODUCTION PRINCIPLES
# ==================================================

print("PRODUCTION PRINCIPLES")
print()

print(
    "Retries should have bounded limits."
)

print()

print(
    "Failures should be observable and logged."
)

print()

print(
    "Repeated failure should not create infinite loops."
)

print()

print(
    "High-impact operations should require "
    "appropriate authorization."
)

print()

print(
    "Verification should test actual outcomes, "
    "not merely assume that execution succeeded."
)

print()


# ==================================================
# 21. SILVERWING PROGRESS
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
print("Tools")
print(" ↓")
print("Structured Tool Calling")
print(" ↓")
print("Planning")
print(" ↓")
print("Async Multitasking")
print(" ↓")
print("Verification")
print(" ↓")
print("Retries")
print(" ↓")
print("Self-Correction")
print(" ↓")
print("Advanced Agent")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 51 COMPLETE ===")

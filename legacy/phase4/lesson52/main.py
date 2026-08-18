# Silverwing ML
# Phase 4 - Lesson 52
# Observability, Logging and Agent Traces

import json
import logging
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 52")
print("Observability, Logging and Agent Traces")
print()


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LOG_FILE = (
        BASE_DIR / "silverwing.log"
)

TRACE_FILE = (
        BASE_DIR / "agent_traces.jsonl"
)


print("TEST 1: Observability Configuration")
print()

print("Log file:", LOG_FILE)
print("Trace file:", TRACE_FILE)

print()


# ==================================================
# 2. LOGGING CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger(
    "silverwing"
)


# ==================================================
# 3. TIMESTAMP
# ==================================================

def utc_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 4. TRACE EVENT
# ==================================================

@dataclass
class TraceEvent:

    trace_id: str

    event_type: str

    component: str

    message: str

    timestamp: str = field(
        default_factory=utc_timestamp
    )

    duration_ms: Optional[float] = None

    status: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 5. TRACE MANAGER
# ==================================================

class TraceManager:
    """
    Records structured execution traces.
    """

    def __init__(
            self,
            trace_file: Path
    ):

        self.trace_file = Path(
            trace_file
        )


    def record(
            self,
            event: TraceEvent
    ):

        data = {
            "trace_id":
                event.trace_id,

            "event_type":
                event.event_type,

            "component":
                event.component,

            "message":
                event.message,

            "timestamp":
                event.timestamp,

            "duration_ms":
                event.duration_ms,

            "status":
                event.status,

            "metadata":
                event.metadata
        }


        with open(
                self.trace_file,
                "a",
                encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False
                )
                +
                "\n"
            )


        logger.info(
            "%s | %s | %s",
            event.component,
            event.event_type,
            event.message
        )


    def read_trace(
            self,
            trace_id: Optional[str] = None
    ):

        if not self.trace_file.exists():

            return []


        events = []


        with open(
                self.trace_file,
                "r",
                encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()


                if not line:

                    continue


                event = json.loads(
                    line
                )


                if (
                        trace_id is None
                        or
                        event["trace_id"]
                        ==
                        trace_id
                ):

                    events.append(
                        event
                    )


        return events


trace_manager = TraceManager(
    TRACE_FILE
)


# ==================================================
# 6. TRACE CONTEXT
# ==================================================

class TraceContext:

    def __init__(
            self,
            trace_manager: TraceManager
    ):

        self.trace_manager = (
            trace_manager
        )


    def start(
            self,
            message: str,
            component="agent",
            metadata=None
    ):

        trace_id = str(
            uuid.uuid4()
        )


        event = TraceEvent(
            trace_id=trace_id,
            event_type="trace_started",
            component=component,
            message=message,
            status="started",
            metadata=(
                metadata
                if metadata
                else {}
            )
        )


        self.trace_manager.record(
            event
        )


        return trace_id


    def event(
            self,
            trace_id,
            event_type,
            component,
            message,
            status=None,
            duration_ms=None,
            metadata=None
    ):

        event = TraceEvent(
            trace_id=trace_id,
            event_type=event_type,
            component=component,
            message=message,
            status=status,
            duration_ms=duration_ms,
            metadata=(
                metadata
                if metadata
                else {}
            )
        )


        self.trace_manager.record(
            event
        )


trace_context = TraceContext(
    trace_manager
)


print("TEST 2: Trace Manager")
print()

print(
    "Trace manager initialized."
)

print()


# ==================================================
# 7. METRICS
# ==================================================

@dataclass
class Metrics:

    tool_calls: int = 0

    successful_tools: int = 0

    failed_tools: int = 0

    total_execution_time_ms: float = 0.0

    retries: int = 0

    plans_created: int = 0

    plans_completed: int = 0

    plans_failed: int = 0

    verification_successes: int = 0

    verification_failures: int = 0


    def summary(self):

        average_time = 0.0


        if self.tool_calls > 0:

            average_time = (
                    self.total_execution_time_ms
                    /
                    self.tool_calls
            )


        return {
            "tool_calls":
                self.tool_calls,

            "successful_tools":
                self.successful_tools,

            "failed_tools":
                self.failed_tools,

            "retries":
                self.retries,

            "plans_created":
                self.plans_created,

            "plans_completed":
                self.plans_completed,

            "plans_failed":
                self.plans_failed,

            "verification_successes":
                self.verification_successes,

            "verification_failures":
                self.verification_failures,

            "total_execution_time_ms":
                round(
                    self.total_execution_time_ms,
                    3
                ),

            "average_tool_time_ms":
                round(
                    average_time,
                    3
                )
        }


metrics = Metrics()


# ==================================================
# 8. OBSERVABLE TOOL
# ==================================================

class ObservableToolExecutor:
    """
    Wraps tool execution with traces,
    metrics, timing and errors.
    """

    def __init__(
            self,
            trace_context: TraceContext,
            metrics: Metrics
    ):

        self.trace_context = (
            trace_context
        )

        self.metrics = metrics


    def execute(
            self,
            trace_id,
            tool_name,
            function,
            arguments
    ):

        self.metrics.tool_calls += 1


        start = time.perf_counter()


        self.trace_context.event(
            trace_id=trace_id,
            event_type="tool_started",
            component="tool_executor",
            message=(
                f"Starting tool: {tool_name}"
            ),
            status="running",
            metadata={
                "tool":
                    tool_name,

                "arguments":
                    arguments
            }
        )


        try:

            result = function(
                **arguments
            )


            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            self.metrics.successful_tools += 1

            self.metrics.total_execution_time_ms += (
                duration
            )


            self.trace_context.event(
                trace_id=trace_id,
                event_type="tool_completed",
                component="tool_executor",
                message=(
                    f"Tool completed: {tool_name}"
                ),
                status="success",
                duration_ms=duration,
                metadata={
                    "tool":
                        tool_name
                }
            )


            return result


        except Exception as error:

            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            self.metrics.failed_tools += 1

            self.metrics.total_execution_time_ms += (
                duration
            )


            self.trace_context.event(
                trace_id=trace_id,
                event_type="tool_failed",
                component="tool_executor",
                message=(
                    f"Tool failed: {tool_name}"
                ),
                status="failed",
                duration_ms=duration,
                metadata={
                    "tool":
                        tool_name,

                    "error":
                        str(error)
                }
            )


            raise


# ==================================================
# 9. DEMONSTRATION TOOLS
# ==================================================

def calculator(
        expression
):

    return eval(
        expression,
        {"__builtins__": {}},
        {}
    )


def machine_analyzer(
        temperature,
        pressure,
        rpm
):

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
        "risk_score":
            score,

        "risk_level":
            level
    }


def failing_tool():

    raise RuntimeError(
        "Demonstration failure."
    )


# ==================================================
# 10. BASIC TOOL EXECUTION
# ==================================================

print("TEST 3: Observable Tool Execution")
print()


trace_id = trace_context.start(
    message="Test observable calculator.",
    component="lesson52"
)


observable_executor = (
    ObservableToolExecutor(
        trace_context,
        metrics
    )
)


result = observable_executor.execute(
    trace_id=trace_id,
    tool_name="calculator",
    function=calculator,
    arguments={
        "expression":
            "25 * 8"
    }
)


print(
    "Calculator result:",
    result
)

print()


# ==================================================
# 11. MACHINE ANALYSIS TRACE
# ==================================================

print("TEST 4: Machine Analysis Trace")
print()


machine_result = (
    observable_executor.execute(
        trace_id=trace_id,
        tool_name="machine_analyzer",
        function=machine_analyzer,
        arguments={
            "temperature":
                105,

            "pressure":
                140,

            "rpm":
                3200
        }
    )
)


print(
    "Machine result:",
    machine_result
)

print()


# ==================================================
# 12. FAILURE TRACE
# ==================================================

print("TEST 5: Failure Trace")
print()


try:

    observable_executor.execute(
        trace_id=trace_id,
        tool_name="failing_tool",
        function=failing_tool,
        arguments={}
    )

except Exception as error:

    print(
        "Caught expected failure:"
    )

    print(
        error
    )

print()


# ==================================================
# 13. VERIFY RESULT
# ==================================================

print("TEST 6: Verification Trace")
print()


verification_valid = (
        machine_result["risk_level"]
        in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        }
)


if verification_valid:

    metrics.verification_successes += 1

    trace_context.event(
        trace_id=trace_id,
        event_type="verification",
        component="verifier",
        message=(
            "Machine result verified."
        ),
        status="success",
        metadata={
            "risk_level":
                machine_result[
                    "risk_level"
                ]
        }
    )

else:

    metrics.verification_failures += 1

    trace_context.event(
        trace_id=trace_id,
        event_type="verification",
        component="verifier",
        message=(
            "Machine result failed verification."
        ),
        status="failed"
    )


print(
    "Verification:",
    verification_valid
)

print()


# ==================================================
# 14. AGENT PLAN TRACE
# ==================================================

print("TEST 7: Agent Plan Trace")
print()


plan_trace_id = trace_context.start(
    message=(
        "Start multi-step machine analysis."
    ),
    component="planner"
)


metrics.plans_created += 1


trace_context.event(
    trace_id=plan_trace_id,
    event_type="plan_created",
    component="planner",
    message="Three-step analysis plan created.",
    metadata={
        "steps": [
            "machine_analysis",
            "verification",
            "response"
        ]
    }
)


# Step 1

trace_context.event(
    trace_id=plan_trace_id,
    event_type="plan_step",
    component="planner",
    message="Executing machine analysis.",
    status="running"
)


machine_result_2 = (
    observable_executor.execute(
        trace_id=plan_trace_id,
        tool_name="machine_analyzer",
        function=machine_analyzer,
        arguments={
            "temperature":
                97,

            "pressure":
                130,

            "rpm":
                2600
        }
    )
)


# Step 2

trace_context.event(
    trace_id=plan_trace_id,
    event_type="plan_step",
    component="verifier",
    message="Verifying machine analysis.",
    status="running"
)


verification_2 = (
        machine_result_2["risk_level"]
        in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        }
)


if verification_2:

    metrics.verification_successes += 1

    trace_context.event(
        trace_id=plan_trace_id,
        event_type="verification",
        component="verifier",
        message="Verification succeeded.",
        status="success"
    )

else:

    metrics.verification_failures += 1

    trace_context.event(
        trace_id=plan_trace_id,
        event_type="verification",
        component="verifier",
        message="Verification failed.",
        status="failed"
    )


# Step 3

trace_context.event(
    trace_id=plan_trace_id,
    event_type="response_ready",
    component="agent",
    message="Final result ready.",
    status="success",
    metadata={
        "result":
            machine_result_2
    }
)


metrics.plans_completed += 1


trace_context.event(
    trace_id=plan_trace_id,
    event_type="plan_completed",
    component="planner",
    message="Plan completed successfully.",
    status="success"
)


print(
    "Plan result:",
    machine_result_2
)

print()


# ==================================================
# 15. TRACE RETRIEVAL
# ==================================================

print("TEST 8: Retrieve Trace")
print()


trace_events = trace_manager.read_trace(
    trace_id
)


print(
    "Trace events:",
    len(trace_events)
)

print()


for event in trace_events:

    print(
        event["event_type"],
        "|",
        event["component"],
        "|",
        event["status"]
    )

print()


# ==================================================
# 16. DETAILED TRACE
# ==================================================

print("TEST 9: Detailed Trace")
print()


for event in trace_events:

    print(
        json.dumps(
            event,
            indent=2
        )
    )

    print()


# ==================================================
# 17. METRICS
# ==================================================

print("TEST 10: Metrics")
print()


print(
    json.dumps(
        metrics.summary(),
        indent=4
    )
)

print()


# ==================================================
# 18. ERROR RATE
# ==================================================

print("TEST 11: Error Rate")
print()


summary = metrics.summary()


total_calls = (
    summary["tool_calls"]
)


failed_calls = (
    summary["failed_tools"]
)


if total_calls > 0:

    error_rate = (
            failed_calls
            /
            total_calls
    )

else:

    error_rate = 0.0


print(
    "Tool calls:",
    total_calls
)

print(
    "Failed:",
    failed_calls
)

print(
    "Error rate:",
    round(
        error_rate,
        4
    )
)

print()


# ==================================================
# 19. RETRY OBSERVABILITY
# ==================================================

print("TEST 12: Retry Tracking")
print()


metrics.retries += 1


trace_context.event(
    trace_id=trace_id,
    event_type="retry",
    component="retry_manager",
    message="Retry event recorded.",
    metadata={
        "attempt":
            2,

        "reason":
            "Transient failure"
    }
)


print(
    "Retries:",
    metrics.retries
)

print()


# ==================================================
# 20. AGENT AUDIT TRAIL
# ==================================================

print("TEST 13: Agent Audit Trail")
print()


audit = {
    "trace_id":
        plan_trace_id,

    "created_at":
        utc_timestamp(),

    "actions": [
        {
            "component":
                "planner",

            "action":
                "created_plan"
        },

        {
            "component":
                "tool_executor",

            "action":
                "machine_analysis"
        },

        {
            "component":
                "verifier",

            "action":
                "verified_result"
        },

        {
            "component":
                "agent",

            "action":
                "prepared_response"
        }
    ]
}


print(
    json.dumps(
        audit,
        indent=4
    )
)

print()


# ==================================================
# 21. OBSERVABILITY CATEGORIES
# ==================================================

print("TEST 14: Observability Categories")
print()

categories = [
    "logs",
    "traces",
    "metrics",
    "errors",
    "latency",
    "task state",
    "tool calls",
    "model calls",
    "resource usage",
    "audit events"
]


for category in categories:

    print(
        "-",
        category
    )

print()


# ==================================================
# 22. WHAT A PRODUCTION SYSTEM NEEDS
# ==================================================

print("TEST 15: Production Observability")
print()

print(
    "A production AI system should be able "
    "to answer:"
)

print()

questions = [
    "Which request created this trace?",
    "Which model was called?",
    "Which tools were used?",
    "How long did each operation take?",
    "How many retries occurred?",
    "Where did execution fail?",
    "What was the final result?",
    "How much compute was consumed?"
]


for question in questions:

    print(
        "-",
        question
    )

print()


# ==================================================
# 23. AGENT TRACE FLOW
# ==================================================

print("AGENT TRACE FLOW")
print()

print("User Request")
print("     ↓")
print("Trace Created")
print("     ↓")
print("Planner")
print("     ↓")
print("Tool Calls")
print("     ↓")
print("Tool Results")
print("     ↓")
print("Verification")
print("     ↓")
print("Retries / Re-plan")
print("     ↓")
print("Final Response")
print("     ↓")
print("Trace Completed")

print()


# ==================================================
# 24. SILVERWING OBSERVABILITY ARCHITECTURE
# ==================================================

print("SILVERWING OBSERVABILITY")
print()

print("                    SILVERWING")
print("                        │")
print("                 Agent / Services")
print("                        │")
print("             ┌──────────┼──────────┐")
print("             ↓          ↓          ↓")
print("           Logs       Traces     Metrics")
print("             │          │          │")
print("             └──────────┼──────────┘")
print("                        ↓")
print("                Observability Layer")
print("                        ↓")
print("             Dashboard / Diagnostics")
print("                        ↓")
print("                  Human Operator")

print()


# ==================================================
# 25. FUTURE OBSERVABILITY STACK
# ==================================================

print("FUTURE OBSERVABILITY STACK")
print()

print(
    "Application logs"
)

print(
    "       ↓"
)

print(
    "Structured traces"
)

print(
    "       ↓"
)

print(
    "Metrics"
)

print(
    "       ↓"
)

print(
    "Centralized observability service"
)

print(
    "       ↓"
)

print(
    "Dashboards + alerts"
)

print()


# ==================================================
# 26. PERSONAL AI IMPORTANCE
# ==================================================

print("PERSONAL AI IMPORTANCE")
print()

print(
    "A highly capable personal AI should be "
    "observable because it may perform many "
    "operations across different services."
)

print()

print(
    "Traceability lets the owner understand "
    "what the system did and where something "
    "went wrong."
)

print()


# ==================================================
# 27. CURRENT SILVERWING PROGRESS
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
print("Self-Correction")
print(" ↓")
print("Observability")
print(" ↓")
print("Advanced Agent Runtime")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 52 COMPLETE ===")

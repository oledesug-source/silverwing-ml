# Silverwing ML
# Phase 4 - Lesson 50
# Async Task Scheduler and Real Multitasking


import asyncio
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 50")
print("Async Task Scheduler and Real Multitasking")
print()


# ==================================================
# 1. TIMESTAMP
# ==================================================

def timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. TOOL FUNCTIONS
# ==================================================

async def calculate(
        expression: str
):

    # Simulated async work.
    await asyncio.sleep(1)

    allowed_characters = (
        "0123456789+-*/(). "
    )

    if any(
            character not in allowed_characters
            for character in expression
    ):

        raise ValueError(
            "Unsupported characters."
        )

    result = eval(
        expression,
        {
            "__builtins__": {}
        },
        {}
    )

    return result


async def machine_analysis(
        temperature: float,
        pressure: float,
        rpm: float
):

    # Simulated sensor-analysis latency.
    await asyncio.sleep(2)

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


async def retrieve_information(
        topic: str
):

    # Simulated external information retrieval.
    await asyncio.sleep(3)

    return {
        "topic": topic,
        "information": (
            f"Retrieved information about {topic}."
        )
    }


async def system_check():

    # Simulated system operation.
    await asyncio.sleep(1.5)

    return {
        "cpu_status": "available",
        "memory_status": "available",
        "service_status": "healthy"
    }


async def generate_report(
        results: List[Dict[str, Any]]
):

    await asyncio.sleep(0.5)

    successful = [
        result
        for result in results
        if result["status"] == "completed"
    ]

    failed = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    return {
        "total_tasks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "generated_at": timestamp()
    }


# ==================================================
# 3. TOOL REGISTRY
# ==================================================

class AsyncToolRegistry:

    def __init__(self):

        self.tools = {
            "calculator": calculate,
            "machine_analysis": machine_analysis,
            "retrieve_information":
                retrieve_information,
            "system_check": system_check,
            "generate_report": generate_report
        }


    def get(
            self,
            name: str
    ):

        return self.tools.get(
            name
        )


registry = AsyncToolRegistry()


print("TEST 1: Async Tool Registry")
print()

for name in registry.tools:

    print(
        "-",
        name
    )

print()


# ==================================================
# 4. TASK DATA STRUCTURE
# ==================================================

@dataclass
class AsyncTask:

    task_id: str

    name: str

    tool_name: str

    arguments: Dict[str, Any]

    depends_on: List[str] = field(
        default_factory=list
    )

    status: str = "pending"

    result: Any = None

    error: Optional[str] = None

    started_at: Optional[str] = None

    completed_at: Optional[str] = None

    duration: Optional[float] = None


# ==================================================
# 5. TASK SCHEDULER
# ==================================================

class TaskScheduler:

    def __init__(
            self,
            registry: AsyncToolRegistry,
            max_concurrency: int = 4
    ):

        self.registry = registry

        self.max_concurrency = (
            max_concurrency
        )

        self.running_tasks = {}

        self.semaphore = asyncio.Semaphore(
            max_concurrency
        )


    # ----------------------------------------------
    # Check dependencies
    # ----------------------------------------------

    def dependencies_ready(
            self,
            task: AsyncTask,
            tasks: Dict[str, AsyncTask]
    ):

        for dependency_id in (
                task.depends_on
        ):

            dependency = tasks.get(
                dependency_id
            )


            if dependency is None:

                return False


            if dependency.status != (
                    "completed"
            ):

                return False


        return True


    # ----------------------------------------------
    # Resolve dependent arguments
    # ----------------------------------------------

    def resolve_arguments(
            self,
            task: AsyncTask,
            tasks: Dict[str, AsyncTask]
    ):

        arguments = dict(
            task.arguments
        )


        # A report can receive results from
        # earlier completed tasks.

        if task.tool_name == (
                "generate_report"
        ):

            dependency_results = []


            for dependency_id in (
                    task.depends_on
            ):

                dependency = tasks.get(
                    dependency_id
                )


                if dependency:

                    dependency_results.append(
                        {
                            "task_id":
                                dependency.task_id,

                            "name":
                                dependency.name,

                            "status":
                                dependency.status,

                            "result":
                                dependency.result
                        }
                    )


            arguments["results"] = (
                dependency_results
            )


        return arguments


    # ----------------------------------------------
    # Execute one task
    # ----------------------------------------------

    async def execute_task(
            self,
            task: AsyncTask,
            tasks: Dict[str, AsyncTask]
    ):

        async with self.semaphore:

            function = self.registry.get(
                task.tool_name
            )


            if function is None:

                task.status = "failed"

                task.error = (
                    f"Unknown tool: "
                    f"{task.tool_name}"
                )

                return task


            task.status = "running"

            task.started_at = (
                timestamp()
            )


            start_time = time.perf_counter()


            print(
                f"[START] {task.name}"
            )


            try:

                arguments = (
                    self.resolve_arguments(
                        task,
                        tasks
                    )
                )


                task.result = await function(
                    **arguments
                )


                task.status = "completed"


            except Exception as error:

                task.status = "failed"

                task.error = str(
                    error
                )


            finally:

                task.completed_at = (
                    timestamp()
                )


                task.duration = (
                        time.perf_counter()
                        -
                        start_time
                )


                print(
                    f"[END] {task.name} "
                    f"({task.duration:.2f}s)"
                )


            return task


    # ----------------------------------------------
    # Execute complete plan
    # ----------------------------------------------

    async def execute(
            self,
            tasks: List[AsyncTask]
    ):

        task_map = {
            task.task_id: task
            for task in tasks
        }


        while True:

            pending = [
                task
                for task in tasks
                if task.status == "pending"
            ]


            if not pending:

                break


            ready = [
                task
                for task in pending
                if self.dependencies_ready(
                    task,
                    task_map
                )
            ]


            if not ready:

                unresolved = [
                    task.task_id
                    for task in pending
                ]


                raise RuntimeError(
                    "Scheduler cannot make progress. "
                    f"Unresolved tasks: {unresolved}"
                )


            # Launch all currently independent tasks.

            futures = [
                asyncio.create_task(
                    self.execute_task(
                        task,
                        task_map
                    )
                )
                for task in ready
            ]


            await asyncio.gather(
                *futures
            )


        return tasks


# ==================================================
# 6. SEQUENTIAL BENCHMARK
# ==================================================

async def sequential_execution():

    print(
        "TEST 2: Sequential Benchmark"
    )

    print()


    start = time.perf_counter()


    await calculate(
        "25 * 8"
    )

    await machine_analysis(
        temperature=97,
        pressure=130,
        rpm=2600
    )

    await retrieve_information(
        "machine learning"
    )

    await system_check()


    duration = (
            time.perf_counter()
            -
            start
    )


    print(
        "Sequential duration:",
        round(
            duration,
            2
        ),
        "seconds"
    )

    print()


    return duration


# ==================================================
# 7. PARALLEL BENCHMARK
# ==================================================

async def parallel_execution():

    print(
        "TEST 3: Parallel Benchmark"
    )

    print()


    start = time.perf_counter()


    await asyncio.gather(

        calculate(
            "25 * 8"
        ),

        machine_analysis(
            temperature=97,
            pressure=130,
            rpm=2600
        ),

        retrieve_information(
            "machine learning"
        ),

        system_check()

    )


    duration = (
            time.perf_counter()
            -
            start
    )


    print(
        "Parallel duration:",
        round(
            duration,
            2
        ),
        "seconds"
    )

    print()


    return duration


# ==================================================
# 8. BASIC CONCURRENCY DEMONSTRATION
# ==================================================

async def run_benchmark():

    sequential = (
        await sequential_execution()
    )


    parallel = (
        await parallel_execution()
    )


    speedup = (
        sequential / parallel
        if parallel > 0
        else 0
    )


    print(
        "Observed speedup:",
        round(
            speedup,
            2
        ),
        "x"
    )

    print()


# ==================================================
# 9. RUN BENCHMARK
# ==================================================

print("TEST 4: Concurrency Benchmark")
print()

asyncio.run(
    run_benchmark()
)


# ==================================================
# 10. CREATE MULTI-TASK PLAN
# ==================================================

print("TEST 5: Create Multitasking Plan")
print()


tasks = [

    AsyncTask(
        task_id="task-1",
        name="Machine analysis",
        tool_name="machine_analysis",
        arguments={
            "temperature": 97,
            "pressure": 130,
            "rpm": 2600
        }
    ),

    AsyncTask(
        task_id="task-2",
        name="Calculate operating metric",
        tool_name="calculator",
        arguments={
            "expression": "2500 * 0.8"
        }
    ),

    AsyncTask(
        task_id="task-3",
        name="Retrieve machine-learning information",
        tool_name="retrieve_information",
        arguments={
            "topic": "machine learning"
        }
    ),

    AsyncTask(
        task_id="task-4",
        name="System health check",
        tool_name="system_check",
        arguments={}
    )
]


for task in tasks:

    print(
        task.task_id,
        "->",
        task.name
    )

print()


# ==================================================
# 11. RUN MULTITASKING PLAN
# ==================================================

print("TEST 6: Execute Multitasking Plan")
print()


scheduler = TaskScheduler(
    registry,
    max_concurrency=4
)


async def execute_multitasking_plan():

    return await scheduler.execute(
        tasks
    )


completed_tasks = asyncio.run(
    execute_multitasking_plan()
)


print()


# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

print("TEST 7: Task Results")
print()


for task in completed_tasks:

    print(
        "Task:",
        task.name
    )

    print(
        "Status:",
        task.status
    )

    print(
        "Duration:",
        round(
            task.duration or 0,
            3
        ),
        "seconds"
    )

    print(
        "Result:",
        task.result
    )

    print()


# ==================================================
# 13. DEPENDENT TASK
# ==================================================

print("TEST 8: Dependent Task")
print()


dependent_tasks = [

    AsyncTask(
        task_id="analysis",
        name="Analyze machine",
        tool_name="machine_analysis",
        arguments={
            "temperature": 105,
            "pressure": 140,
            "rpm": 3200
        }
    ),

    AsyncTask(
        task_id="report",
        name="Generate final report",
        tool_name="generate_report",
        arguments={},
        depends_on=[
            "analysis"
        ]
    )
]


async def execute_dependent_plan():

    return await scheduler.execute(
        dependent_tasks
    )


dependent_results = asyncio.run(
    execute_dependent_plan()
)


for task in dependent_results:

    print(
        task.task_id,
        "->",
        task.status
    )

    print(
        "Result:",
        task.result
    )

    print()


# ==================================================
# 14. MIXED DEPENDENCY GRAPH
# ==================================================

print("TEST 9: Dependency Graph")
print()


mixed_tasks = [

    AsyncTask(
        task_id="sensor",
        name="Analyze sensor data",
        tool_name="machine_analysis",
        arguments={
            "temperature": 102,
            "pressure": 135,
            "rpm": 3100
        }
    ),

    AsyncTask(
        task_id="calculation",
        name="Calculate maintenance metric",
        tool_name="calculator",
        arguments={
            "expression": "3200 / 8"
        }
    ),

    AsyncTask(
        task_id="research",
        name="Retrieve maintenance information",
        tool_name="retrieve_information",
        arguments={
            "topic": "predictive maintenance"
        }
    ),

    AsyncTask(
        task_id="report",
        name="Combine task results",
        tool_name="generate_report",
        arguments={},
        depends_on=[
            "sensor",
            "calculation",
            "research"
        ]
    )
]


async def execute_mixed_plan():

    return await scheduler.execute(
        mixed_tasks
    )


mixed_results = asyncio.run(
    execute_mixed_plan()
)


for task in mixed_results:

    print(
        task.name,
        "->",
        task.status
    )

print()


# ==================================================
# 15. TASK SUMMARY
# ==================================================

print("TEST 10: Task Summary")
print()


def summarize_tasks(
        tasks
):

    completed = [
        task
        for task in tasks
        if task.status == "completed"
    ]


    failed = [
        task
        for task in tasks
        if task.status == "failed"
    ]


    return {
        "total":
            len(tasks),

        "completed":
            len(completed),

        "failed":
            len(failed),

        "total_execution_time":
            round(
                sum(
                    task.duration or 0
                    for task in tasks
                ),
                3
            )
    }


summary = summarize_tasks(
    mixed_results
)


for key, value in summary.items():

    print(
        key,
        ":",
        value
    )

print()


# ==================================================
# 16. TIMEOUT DEMONSTRATION
# ==================================================

print("TEST 11: Task Timeout")
print()


async def run_with_timeout():

    try:

        result = await asyncio.wait_for(
            retrieve_information(
                "advanced AI systems"
            ),
            timeout=5
        )


        print(
            "Task completed:"
        )

        print(
            result
        )


    except asyncio.TimeoutError:

        print(
            "Task exceeded timeout."
        )


asyncio.run(
    run_with_timeout()
)


print()


# ==================================================
# 17. CANCELLATION CONCEPT
# ==================================================

print("TEST 12: Task Cancellation Concept")
print()


async def cancellable_task():

    try:

        print(
            "Long-running task started."
        )


        await asyncio.sleep(
            10
        )


        print(
            "Long-running task completed."
        )


    except asyncio.CancelledError:

        print(
            "Long-running task cancelled."
        )

        raise


async def cancellation_demo():

    task = asyncio.create_task(
        cancellable_task()
    )


    await asyncio.sleep(
        0.5
    )


    task.cancel()


    try:

        await task

    except asyncio.CancelledError:

        pass


asyncio.run(
    cancellation_demo()
)


print()


# ==================================================
# 18. RESOURCE LIMIT
# ==================================================

print("TEST 13: Concurrency Limit")
print()


limited_scheduler = TaskScheduler(
    registry,
    max_concurrency=2
)


print(
    "Maximum concurrent tasks:",
    limited_scheduler.max_concurrency
)

print()


# ==================================================
# 19. MULTITASKING ARCHITECTURE
# ==================================================

print("MULTITASKING ARCHITECTURE")
print()

print("User Goal")
print("   ↓")
print("Agent Planner")
print("   ↓")
print("Dependency Graph")
print("   ↓")
print("Async Scheduler")
print("   ↓")
print("┌────────┬────────┬────────┐")
print("↓        ↓        ↓")
print("Task A  Task B  Task C")
print("↓        ↓        ↓")
print("Tool A  Tool B  Tool C")
print("└────────┴────────┴────────┘")
print("             ↓")
print("        Task Results")
print("             ↓")
print("          Verifier")
print("             ↓")
print("        Final Response")

print()


# ==================================================
# 20. WHY CONCURRENCY MATTERS
# ==================================================

print("WHY CONCURRENCY MATTERS")
print()

print(
    "Some AI operations spend time waiting for "
    "network responses, services, files, or other "
    "I/O operations."
)

print()

print(
    "Independent I/O-bound operations can often "
    "make progress concurrently."
)

print()

print(
    "CPU-heavy or GPU-heavy work may require "
    "different concurrency strategies."
)

print()


# ==================================================
# 21. IMPORTANT DISTINCTION
# ==================================================

print("IMPORTANT DISTINCTION")
print()

print(
    "Asynchronous concurrency is not the same "
    "as making every computation run faster."
)

print()

print(
    "The scheduler decides which independent "
    "operations can safely overlap."
)

print()

print(
    "Dependencies still enforce execution order."
)

print()


# ==================================================
# 22. FUTURE SILVERWING ORCHESTRATOR
# ==================================================

print("FUTURE SILVERWING ORCHESTRATOR")
print()

print("User")
print(" ↓")
print("LLM / Agent")
print(" ↓")
print("Plan")
print(" ↓")
print("Dependency Graph")
print(" ↓")
print("Task Scheduler")
print(" ↓")
print("Worker Pool")
print(" ↓")
print("Tools / Services / Models")
print(" ↓")
print("Results")
print(" ↓")
print("Verification")
print(" ↓")
print("Agent")
print(" ↓")
print("User")

print()


# ==================================================
# 23. CURRENT SILVERWING PROGRESS
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
print("Advanced Agent")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 50 COMPLETE ===")

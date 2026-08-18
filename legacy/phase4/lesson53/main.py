# Silverwing ML
# Phase 4 - Lesson 53
# Persistent Agent State and Resumable Jobs
#
# Corrected version:
# - Properly switches JobStore after application restart
# - No closed-database access during resume
# - Handles stale "running" steps
# - Uses a clean lesson database


import json
import sqlite3
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 53")
print("Persistent Agent State and Resumable Jobs")
print()


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = (
        BASE_DIR / "silverwing_agent_state_v2.db"
)


print("TEST 1: Configuration")
print()

print(
    "Database:",
    DATABASE_FILE
)

print()


# ==================================================
# 2. TIME
# ==================================================

def now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 3. JSON HELPERS
# ==================================================

def to_json(value):

    return json.dumps(
        value,
        ensure_ascii=False
    )


def from_json(
        value,
        default=None
):

    if value is None:

        return default

    try:

        return json.loads(
            value
        )

    except (
            json.JSONDecodeError,
            TypeError
    ):

        return default


# ==================================================
# 4. DATABASE CONNECTION
# ==================================================

def create_connection():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


connection = create_connection()


print("TEST 2: Database Connection")
print()

print(
    "Persistent state database connected."
)

print()


# ==================================================
# 5. DATABASE SCHEMA
# ==================================================

print("TEST 3: Create State Tables")
print()


cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS jobs (
                                        job_id TEXT PRIMARY KEY,
                                        goal TEXT NOT NULL,
                                        status TEXT NOT NULL,
                                        priority INTEGER DEFAULT 5,
                                        created_at TEXT NOT NULL,
                                        updated_at TEXT NOT NULL,
                                        completed_at TEXT
    )
    """
)


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS job_steps (
                                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                                             job_id TEXT NOT NULL,
                                             step_id TEXT NOT NULL,
                                             description TEXT NOT NULL,
                                             tool_name TEXT NOT NULL,
                                             arguments TEXT NOT NULL,
                                             depends_on TEXT NOT NULL,
                                             status TEXT NOT NULL,
                                             result TEXT,
                                             error TEXT,
                                             attempts INTEGER DEFAULT 0,
                                             started_at TEXT,
                                             completed_at TEXT,
                                             FOREIGN KEY(job_id) REFERENCES jobs(job_id)
        )
    """
)


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS job_events (
                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                              job_id TEXT NOT NULL,
                                              event_type TEXT NOT NULL,
                                              message TEXT NOT NULL,
                                              payload TEXT,
                                              created_at TEXT NOT NULL,
                                              FOREIGN KEY(job_id) REFERENCES jobs(job_id)
        )
    """
)


connection.commit()


print(
    "State tables ready."
)

print()


# ==================================================
# 6. JOB STORE
# ==================================================

class JobStore:
    """
    Persistent storage for jobs, steps and events.
    """

    def __init__(
            self,
            connection
    ):

        self.connection = connection


    # ----------------------------------------------
    # Create job
    # ----------------------------------------------

    def create_job(
            self,
            goal,
            priority=5
    ):

        job_id = str(
            uuid.uuid4()
        )

        timestamp = now()


        self.connection.execute(
            """
            INSERT INTO jobs (
                job_id,
                goal,
                status,
                priority,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                goal,
                "pending",
                priority,
                timestamp,
                timestamp
            )
        )


        self.connection.commit()

        return job_id


    # ----------------------------------------------
    # Add step
    # ----------------------------------------------

    def add_step(
            self,
            job_id,
            step_id,
            description,
            tool_name,
            arguments,
            depends_on=None
    ):

        if depends_on is None:

            depends_on = []


        self.connection.execute(
            """
            INSERT INTO job_steps (
                job_id,
                step_id,
                description,
                tool_name,
                arguments,
                depends_on,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                step_id,
                description,
                tool_name,
                to_json(arguments),
                to_json(depends_on),
                "pending"
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Update job
    # ----------------------------------------------

    def update_job_status(
            self,
            job_id,
            status
    ):

        completed_at = None


        if status == "completed":

            completed_at = now()


        self.connection.execute(
            """
            UPDATE jobs
            SET status = ?,
                updated_at = ?,
                completed_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                now(),
                completed_at,
                job_id
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Update step
    # ----------------------------------------------

    def update_step(
            self,
            job_id,
            step_id,
            status,
            result=None,
            error=None,
            attempts=0
    ):

        if status == "running":

            self.connection.execute(
                """
                UPDATE job_steps
                SET status = ?,
                    started_at = ?,
                    attempts = ?,
                    error = NULL
                WHERE job_id = ?
                  AND step_id = ?
                """,
                (
                    status,
                    now(),
                    attempts,
                    job_id,
                    step_id
                )
            )


        elif status == "completed":

            self.connection.execute(
                """
                UPDATE job_steps
                SET status = ?,
                    result = ?,
                    error = NULL,
                    completed_at = ?,
                    attempts = ?
                WHERE job_id = ?
                  AND step_id = ?
                """,
                (
                    status,
                    to_json(result),
                    now(),
                    attempts,
                    job_id,
                    step_id
                )
            )


        elif status == "failed":

            self.connection.execute(
                """
                UPDATE job_steps
                SET status = ?,
                    error = ?,
                    completed_at = ?,
                    attempts = ?
                WHERE job_id = ?
                  AND step_id = ?
                """,
                (
                    status,
                    error,
                    now(),
                    attempts,
                    job_id,
                    step_id
                )
            )


        elif status == "pending":

            self.connection.execute(
                """
                UPDATE job_steps
                SET status = ?,
                    error = NULL,
                    completed_at = NULL,
                    attempts = ?
                WHERE job_id = ?
                  AND step_id = ?
                """,
                (
                    status,
                    attempts,
                    job_id,
                    step_id
                )
            )


        self.connection.commit()


    # ----------------------------------------------
    # Record event
    # ----------------------------------------------

    def record_event(
            self,
            job_id,
            event_type,
            message,
            payload=None
    ):

        self.connection.execute(
            """
            INSERT INTO job_events (
                job_id,
                event_type,
                message,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                job_id,
                event_type,
                message,
                to_json(payload)
                if payload is not None
                else None,
                now()
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Get job
    # ----------------------------------------------

    def get_job(
            self,
            job_id
    ):

        row = self.connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,)
        ).fetchone()


        if row is None:

            return None


        job = dict(row)


        step_rows = self.connection.execute(
            """
            SELECT *
            FROM job_steps
            WHERE job_id = ?
            ORDER BY id ASC
            """,
            (job_id,)
        ).fetchall()


        steps = []


        for step_row in step_rows:

            step = dict(
                step_row
            )


            step["arguments"] = from_json(
                step["arguments"],
                {}
            )


            step["depends_on"] = from_json(
                step["depends_on"],
                []
            )


            step["result"] = from_json(
                step["result"]
            )


            steps.append(
                step
            )


        job["steps"] = steps

        return job


    # ----------------------------------------------
    # Events
    # ----------------------------------------------

    def get_events(
            self,
            job_id
    ):

        rows = self.connection.execute(
            """
            SELECT *
            FROM job_events
            WHERE job_id = ?
            ORDER BY id ASC
            """,
            (job_id,)
        ).fetchall()


        return [
            dict(row)
            for row in rows
        ]


    # ----------------------------------------------
    # Resumable jobs
    # ----------------------------------------------

    def get_resumable_jobs(self):

        rows = self.connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE status IN (
                             'pending',
                             'running',
                             'paused',
                             'failed'
                )
            ORDER BY priority DESC,
                     created_at ASC
            """
        ).fetchall()


        return [
            dict(row)
            for row in rows
        ]


# ==================================================
# 7. INITIAL STORE
# ==================================================

store = JobStore(
    connection
)


# ==================================================
# 8. CREATE JOB
# ==================================================

print("TEST 4: Create Persistent Job")
print()


job_id = store.create_job(
    goal=(
        "Analyze a machine, calculate a metric, "
        "and create a final summary."
    ),
    priority=9
)


print(
    "Job ID:",
    job_id
)

print()


# ==================================================
# 9. ADD JOB STEPS
# ==================================================

print("TEST 5: Create Job Steps")
print()


store.add_step(
    job_id,
    "machine-analysis",
    "Analyze machine condition.",
    "machine_analyzer",
    {
        "temperature": 105,
        "pressure": 140,
        "rpm": 3200
    }
)


store.add_step(
    job_id,
    "calculate-metric",
    "Calculate maintenance metric.",
    "calculator",
    {
        "expression": "3200 / 8"
    }
)


store.add_step(
    job_id,
    "final-summary",
    "Create final summary.",
    "summarize",
    {},
    [
        "machine-analysis",
        "calculate-metric"
    ]
)


print(
    "Three persistent steps created."
)

print()


# ==================================================
# 10. TOOL IMPLEMENTATIONS
# ==================================================

def machine_analyzer(
        temperature,
        pressure,
        rpm
):

    time.sleep(
        0.5
    )


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

        risk_level = "CRITICAL"

    elif score >= 40:

        risk_level = "HIGH"

    elif score >= 20:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    return {
        "risk_score":
            score,

        "risk_level":
            risk_level
    }


def calculator(
        expression
):

    time.sleep(
        0.3
    )


    allowed = (
        "0123456789+-*/(). "
    )


    if any(
            character not in allowed
            for character in expression
    ):

        raise ValueError(
            "Unsupported calculator expression."
        )


    return eval(
        expression,
        {
            "__builtins__": {}
        },
        {}
    )


def summarize(
        results
):

    return {
        "status":
            "summary_created",

        "results":
            results,

        "created_at":
            now()
    }


TOOLS = {
    "machine_analyzer":
        machine_analyzer,

    "calculator":
        calculator,

    "summarize":
        summarize
}


# ==================================================
# 11. DEPENDENCY CHECK
# ==================================================

def dependencies_completed(
        steps,
        step
):

    for dependency_id in (
            step["depends_on"]
    ):

        dependency = next(
            (
                candidate
                for candidate in steps
                if candidate["step_id"]
                   == dependency_id
            ),
            None
        )


        if dependency is None:

            return False


        if dependency["status"] != (
                "completed"
        ):

            return False


    return True


# ==================================================
# 12. STEP EXECUTION
# ==================================================

def execute_step(
        job_store,
        job_id,
        step
):
    """
    IMPORTANT:
    The active JobStore is passed explicitly.

    This is the fix for the previous bug.
    """

    tool_name = step[
        "tool_name"
    ]


    tool = TOOLS.get(
        tool_name
    )


    if tool is None:

        job_store.update_step(
            job_id,
            step["step_id"],
            "failed",
            error=(
                f"Unknown tool: {tool_name}"
            ),
            attempts=(
                             step.get("attempts") or 0
                     ) + 1
        )


        return False


    attempt = (
                      step.get("attempts") or 0
              ) + 1


    job_store.update_step(
        job_id,
        step["step_id"],
        "running",
        attempts=attempt
    )


    job_store.record_event(
        job_id,
        "step_started",
        f"Started {step['step_id']}",
        {
            "attempt":
                attempt
        }
    )


    try:

        arguments = dict(
            step["arguments"]
        )


        # ------------------------------------------
        # Build summary arguments from completed
        # dependency results.
        # ------------------------------------------

        if tool_name == "summarize":

            current_job = (
                job_store.get_job(
                    job_id
                )
            )


            dependency_results = {}


            for dependency_id in (
                    step["depends_on"]
            ):

                dependency = next(
                    (
                        candidate
                        for candidate
                        in current_job["steps"]
                        if candidate["step_id"]
                           ==
                           dependency_id
                    ),
                    None
                )


                if dependency is not None:

                    dependency_results[
                        dependency_id
                    ] = dependency[
                        "result"
                    ]


            arguments["results"] = (
                dependency_results
            )


        result = tool(
            **arguments
        )


        job_store.update_step(
            job_id,
            step["step_id"],
            "completed",
            result=result,
            attempts=attempt
        )


        job_store.record_event(
            job_id,
            "step_completed",
            f"Completed {step['step_id']}",
            {
                "result":
                    result,

                "attempt":
                    attempt
            }
        )


        return True


    except Exception as error:

        job_store.update_step(
            job_id,
            step["step_id"],
            "failed",
            error=str(error),
            attempts=attempt
        )


        job_store.record_event(
            job_id,
            "step_failed",
            f"Failed {step['step_id']}",
            {
                "error":
                    str(error),

                "attempt":
                    attempt
            }
        )


        return False


# ==================================================
# 13. RESUMABLE EXECUTOR
# ==================================================

class ResumableJobExecutor:

    def __init__(
            self,
            job_store
    ):

        self.job_store = job_store


    def run(
            self,
            job_id,
            simulate_shutdown=False
    ):

        job = self.job_store.get_job(
            job_id
        )


        if job is None:

            raise ValueError(
                "Job does not exist."
            )


        self.job_store.update_job_status(
            job_id,
            "running"
        )


        self.job_store.record_event(
            job_id,
            "job_started",
            "Job execution started."
        )


        print(
            "Starting job:",
            job_id
        )


        while True:

            job = self.job_store.get_job(
                job_id
            )


            steps = job["steps"]


            # --------------------------------------
            # Recover stale running steps
            # --------------------------------------

            for step in steps:

                if step["status"] == "running":

                    self.job_store.update_step(
                        job_id,
                        step["step_id"],
                        "pending",
                        attempts=(
                            step["attempts"]
                        )
                    )


            # Refresh state after normalization.

            job = self.job_store.get_job(
                job_id
            )

            steps = job["steps"]


            # --------------------------------------
            # Check completion
            # --------------------------------------

            if all(
                    step["status"]
                    ==
                    "completed"
                    for step in steps
            ):


                self.job_store.update_job_status(
                    job_id,
                    "completed"
                )


                self.job_store.record_event(
                    job_id,
                    "job_completed",
                    "All job steps completed."
                )


                print(
                    "Job completed."
                )


                return self.job_store.get_job(
                    job_id
                )


            # --------------------------------------
            # Find executable steps
            # --------------------------------------

            progress = False


            for step in steps:

                if step["status"] not in {
                    "pending",
                    "failed"
                }:

                    continue


                if not dependencies_completed(
                        steps,
                        step
                ):

                    continue


                if step["status"] == "failed":

                    self.job_store.record_event(
                        job_id,
                        "step_retry",
                        f"Retrying {step['step_id']}"
                    )


                print(
                    "Executing:",
                    step["step_id"]
                )


                success = execute_step(
                    self.job_store,
                    job_id,
                    step
                )


                progress = True


                print(
                    "Success:",
                    success
                )


                print()


                # ----------------------------------
                # Simulate crash after metric step.
                # ----------------------------------

                if (
                        simulate_shutdown
                        and
                        step["step_id"]
                        ==
                        "calculate-metric"
                ):

                    self.job_store.record_event(
                        job_id,
                        "simulated_shutdown",
                        (
                            "Application stopped after "
                            "this completed step."
                        )
                    )


                    print(
                        "SIMULATED APPLICATION SHUTDOWN"
                    )


                    return None


            if not progress:

                self.job_store.update_job_status(
                    job_id,
                    "paused"
                )


                self.job_store.record_event(
                    job_id,
                    "job_paused",
                    (
                        "No executable steps were "
                        "available."
                    )
                )


                return self.job_store.get_job(
                    job_id
                )


# ==================================================
# 14. FIRST EXECUTOR
# ==================================================

executor = ResumableJobExecutor(
    store
)


# ==================================================
# 15. FIRST EXECUTION
# ==================================================

print("TEST 6: First Execution")
print()


executor.run(
    job_id,
    simulate_shutdown=True
)


print()


# ==================================================
# 16. INSPECT STATE
# ==================================================

print("TEST 7: State After Simulated Shutdown")
print()


saved_job = store.get_job(
    job_id
)


print(
    "Job status:",
    saved_job["status"]
)

print()


for step in (
        saved_job["steps"]
):

    print(
        step["step_id"],
        "->",
        step["status"]
    )

print()


# ==================================================
# 17. RESUMABLE JOBS
# ==================================================

print("TEST 8: Resumable Jobs")
print()


resumable_jobs = (
    store.get_resumable_jobs()
)


print(
    "Resumable jobs:",
    len(resumable_jobs)
)

print()


for job in resumable_jobs:

    print(
        job["job_id"],
        "->",
        job["status"]
    )

print()


# ==================================================
# 18. SIMULATE APPLICATION RESTART
# ==================================================

print("TEST 9: Simulate Restart")
print()


connection.close()


print(
    "Original database connection closed."
)

print()


# ==================================================
# 19. CREATE NEW APPLICATION CONNECTION
# ==================================================

new_connection = create_connection()


new_store = JobStore(
    new_connection
)


print(
    "New application instance created."
)

print()


# ==================================================
# 20. RESTORE JOB
# ==================================================

print("TEST 10: Restore Job State")
print()


restored_job = new_store.get_job(
    job_id
)


print(
    "Restored goal:"
)

print(
    restored_job["goal"]
)

print()


print(
    "Restored status:",
    restored_job["status"]
)

print()


for step in (
        restored_job["steps"]
):

    print(
        step["step_id"],
        "->",
        step["status"],
        "| attempts:",
        step["attempts"]
    )

print()


# ==================================================
# 21. NEW EXECUTOR USING NEW STORE
# ==================================================

new_executor = ResumableJobExecutor(
    new_store
)


# ==================================================
# 22. RESUME JOB
# ==================================================

print("TEST 11: Resume Job")
print()


resumed_job = new_executor.run(
    job_id,
    simulate_shutdown=False
)


print()


# ==================================================
# 23. FINAL STATE
# ==================================================

print("TEST 12: Final Job State")
print()


final_job = new_store.get_job(
    job_id
)


print(
    "Job status:",
    final_job["status"]
)

print()


for step in (
        final_job["steps"]
):

    print(
        step["step_id"],
        "->",
        step["status"]
    )

    print(
        "Result:",
        step["result"]
    )

    print()


# ==================================================
# 24. EVENT HISTORY
# ==================================================

print("TEST 13: Event History")
print()


events = new_store.get_events(
    job_id
)


print(
    "Events:",
    len(events)
)

print()


for event in events:

    print(
        event["created_at"],
        "|",
        event["event_type"],
        "|",
        event["message"]
    )

print()


# ==================================================
# 25. VERIFY PERSISTENCE
# ==================================================

print("TEST 14: Persistence Verification")
print()


all_completed = all(
    step["status"]
    == "completed"
    for step in final_job[
        "steps"
    ]
)


job_completed = (
        final_job["status"]
        ==
        "completed"
)


print(
    "All steps completed:",
    all_completed
)

print(
    "Job completed:",
    job_completed
)

print()


if (
        all_completed
        and
        job_completed
):

    print(
        "PERSISTENT RESUME TEST PASSED."
    )

else:

    print(
        "PERSISTENT RESUME TEST FAILED."
    )

print()


# ==================================================
# 26. SERIALIZED FINAL STATE
# ==================================================

print("TEST 15: Serialized Final State")
print()


print(
    json.dumps(
        final_job,
        indent=4,
        ensure_ascii=False,
        default=str
    )
)

print()


# ==================================================
# 27. DURABLE AGENT WORKFLOW
# ==================================================

print("DURABLE AGENT WORKFLOW")
print()

print("User Goal")
print("   ↓")
print("Create Job")
print("   ↓")
print("Persist Plan")
print("   ↓")
print("Execute Step")
print("   ↓")
print("Persist Result")
print("   ↓")
print("Application Stops")
print("   ↓")
print("Application Restarts")
print("   ↓")
print("Restore Job")
print("   ↓")
print("Find Remaining Work")
print("   ↓")
print("Resume")
print("   ↓")
print("Verify")
print("   ↓")
print("Complete")

print()


# ==================================================
# 28. FUTURE JOB SYSTEM
# ==================================================

print("FUTURE SILVERWING JOB SYSTEM")
print()

print("Agent")
print(" ↓")
print("Job Manager")
print(" ↓")
print("Persistent State")
print(" ↓")
print("Scheduler")
print(" ↓")
print("Workers")
print(" ↓")
print("Tools / Services / Models")
print(" ↓")
print("Results")
print(" ↓")
print("Verifier")
print(" ↓")
print("Agent")

print()


# ==================================================
# 29. WHY THIS MATTERS
# ==================================================

print("WHY PERSISTENT AGENT STATE MATTERS")
print()

print(
    "Long-running work should not depend entirely "
    "on process memory."
)

print()

print(
    "A restart should not automatically destroy "
    "knowledge of what the agent already completed."
)

print()

print(
    "Intermediate results can be reused instead "
    "of performing the same work again."
)

print()


# ==================================================
# 30. CURRENT SILVERWING PROGRESS
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
print("Persistent Agent State")
print(" ↓")
print("Resumable Jobs")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 53 COMPLETE ===")


# ==================================================
# CLOSE FINAL CONNECTION
# ==================================================

new_connection.close()
# Silverwing ML
# Phase 4 - Lesson 44
# Persistent Memory with SQLite
#
# Goal:
# Build a persistent memory subsystem that can
# store conversations, facts, tasks, and events.


import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 44")
print("Persistent Memory with SQLite")
print()


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = (
        BASE_DIR / "silverwing_memory.db"
)


print("TEST 1: Database Configuration")
print()

print(
    "Database:",
    DATABASE_FILE
)

print()


# ==================================================
# 2. DATABASE CONNECTION
# ==================================================

connection = sqlite3.connect(
    DATABASE_FILE
)

connection.row_factory = (
    sqlite3.Row
)


cursor = connection.cursor()


print("TEST 2: Database Connection")
print()

print(
    "SQLite connection established."
)

print()


# ==================================================
# 3. CREATE MEMORY TABLES
# ==================================================

print("TEST 3: Create Memory Tables")
print()


cursor.execute("""
               CREATE TABLE IF NOT EXISTS memories (
                                                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                       memory_type TEXT NOT NULL,
                                                       subject TEXT,
                                                       content TEXT NOT NULL,
                                                       importance REAL DEFAULT 0.5,
                                                       source TEXT,
                                                       created_at TEXT NOT NULL,
                                                       updated_at TEXT NOT NULL
               )
               """)


cursor.execute("""
               CREATE TABLE IF NOT EXISTS conversations (
                                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                            session_id TEXT NOT NULL,
                                                            role TEXT NOT NULL,
                                                            content TEXT NOT NULL,
                                                            created_at TEXT NOT NULL
               )
               """)


cursor.execute("""
               CREATE TABLE IF NOT EXISTS tasks (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    task_id TEXT NOT NULL UNIQUE,
                                                    title TEXT NOT NULL,
                                                    status TEXT NOT NULL,
                                                    priority INTEGER DEFAULT 5,
                                                    details TEXT,
                                                    created_at TEXT NOT NULL,
                                                    updated_at TEXT NOT NULL
               )
               """)


cursor.execute("""
               CREATE TABLE IF NOT EXISTS events (
                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                     event_type TEXT NOT NULL,
                                                     source TEXT,
                                                     payload TEXT NOT NULL,
                                                     created_at TEXT NOT NULL
               )
               """)


connection.commit()


print(
    "Memory tables created."
)

print()


# ==================================================
# 4. TIMESTAMP FUNCTION
# ==================================================

def current_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 5. MEMORY MANAGER
# ==================================================

class MemoryManager:
    """
    Persistent memory service for Silverwing.
    """

    def __init__(
            self,
            connection
    ):

        self.connection = connection


    # ----------------------------------------------
    # Store general memory
    # ----------------------------------------------

    def store_memory(
            self,
            memory_type,
            content,
            subject=None,
            importance=0.5,
            source="system"
    ):

        timestamp = current_timestamp()


        cursor = self.connection.cursor()


        cursor.execute(
            """
            INSERT INTO memories (
                memory_type,
                subject,
                content,
                importance,
                source,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_type,
                subject,
                content,
                importance,
                source,
                timestamp,
                timestamp
            )
        )


        self.connection.commit()


        return cursor.lastrowid


    # ----------------------------------------------
    # Search memories
    # ----------------------------------------------

    def search_memories(
            self,
            query,
            limit=10
    ):

        pattern = (
                "%"
                + query
                + "%"
        )


        cursor = self.connection.cursor()


        cursor.execute(
            """
            SELECT *
            FROM memories
            WHERE content LIKE ?
               OR subject LIKE ?
            ORDER BY importance DESC,
                     updated_at DESC
                LIMIT ?
            """,
            (
                pattern,
                pattern,
                limit
            )
        )


        return [
            dict(row)
            for row in cursor.fetchall()
        ]


    # ----------------------------------------------
    # Store conversation
    # ----------------------------------------------

    def add_message(
            self,
            session_id,
            role,
            content
    ):

        cursor = self.connection.cursor()


        cursor.execute(
            """
            INSERT INTO conversations (
                session_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                current_timestamp()
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Get conversation
    # ----------------------------------------------

    def get_conversation(
            self,
            session_id,
            limit=20
    ):

        cursor = self.connection.cursor()


        cursor.execute(
            """
            SELECT *
            FROM conversations
            WHERE session_id = ?
            ORDER BY id DESC
                LIMIT ?
            """,
            (
                session_id,
                limit
            )
        )


        rows = cursor.fetchall()


        rows.reverse()


        return [
            dict(row)
            for row in rows
        ]


    # ----------------------------------------------
    # Create task
    # ----------------------------------------------

    def create_task(
            self,
            task_id,
            title,
            details="",
            priority=5
    ):

        timestamp = current_timestamp()


        cursor = self.connection.cursor()


        cursor.execute(
            """
            INSERT INTO tasks (
                task_id,
                title,
                status,
                priority,
                details,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                title,
                "pending",
                priority,
                details,
                timestamp,
                timestamp
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Update task
    # ----------------------------------------------

    def update_task(
            self,
            task_id,
            status
    ):

        cursor = self.connection.cursor()


        cursor.execute(
            """
            UPDATE tasks
            SET status = ?,
                updated_at = ?
            WHERE task_id = ?
            """,
            (
                status,
                current_timestamp(),
                task_id
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Get tasks
    # ----------------------------------------------

    def get_tasks(
            self,
            status=None
    ):

        cursor = self.connection.cursor()


        if status is None:

            cursor.execute(
                """
                SELECT *
                FROM tasks
                ORDER BY priority DESC,
                         updated_at DESC
                """
            )

        else:

            cursor.execute(
                """
                SELECT *
                FROM tasks
                WHERE status = ?
                ORDER BY priority DESC,
                         updated_at DESC
                """,
                (status,)
            )


        return [
            dict(row)
            for row in cursor.fetchall()
        ]


    # ----------------------------------------------
    # Store event
    # ----------------------------------------------

    def record_event(
            self,
            event_type,
            payload,
            source="system"
    ):

        cursor = self.connection.cursor()


        cursor.execute(
            """
            INSERT INTO events (
                event_type,
                source,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                event_type,
                source,
                json.dumps(payload),
                current_timestamp()
            )
        )


        self.connection.commit()


    # ----------------------------------------------
    # Database statistics
    # ----------------------------------------------

    def statistics(self):

        cursor = self.connection.cursor()


        tables = [
            "memories",
            "conversations",
            "tasks",
            "events"
        ]


        result = {}


        for table in tables:

            cursor.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {table}
                """
            )


            result[table] = (
                cursor.fetchone()["count"]
            )


        return result


# ==================================================
# 6. CREATE MEMORY MANAGER
# ==================================================

print("TEST 4: Memory Manager")
print()


memory = MemoryManager(
    connection
)


print(
    "Memory manager created."
)

print()


# ==================================================
# 7. STORE USER MEMORY
# ==================================================

print("TEST 5: Store Memory")
print()


memory_id = memory.store_memory(
    memory_type="project",
    subject="Silverwing",
    content=(
        "Silverwing is being developed as "
        "a personal, extensible AI system "
        "with communication, memory, tools, "
        "ML models, and multitasking."
    ),
    importance=1.0,
    source="user"
)


print(
    "Memory ID:",
    memory_id
)

print()


# ==================================================
# 8. STORE TECHNICAL MEMORY
# ==================================================

memory.store_memory(
    memory_type="technical",
    subject="ML",
    content=(
        "Silverwing-ML contains classical "
        "machine learning, deep learning, "
        "Transformer, and LLM engineering lessons."
    ),
    importance=0.9,
    source="lesson"
)


memory.store_memory(
    memory_type="architecture",
    subject="AI",
    content=(
        "Silverwing should separate the LLM "
        "from memory, tools, services, and "
        "the task orchestration layer."
    ),
    importance=0.95,
    source="architecture"
)


print(
    "Technical memories stored."
)

print()


# ==================================================
# 9. SEARCH MEMORY
# ==================================================

print("TEST 6: Search Memory")
print()


results = memory.search_memories(
    "Silverwing"
)


for result in results:

    print(
        "Type:",
        result["memory_type"]
    )

    print(
        "Subject:",
        result["subject"]
    )

    print(
        "Content:",
        result["content"]
    )

    print(
        "Importance:",
        result["importance"]
    )

    print()


# ==================================================
# 10. CREATE CONVERSATION
# ==================================================

print("TEST 7: Conversation Memory")
print()


session_id = (
    "lesson44-session"
)


memory.add_message(
    session_id,
    "user",
    "What are we building?"
)


memory.add_message(
    session_id,
    "assistant",
    (
        "We are building Silverwing as "
        "a broad personal AI system."
    )
)


memory.add_message(
    session_id,
    "user",
    "What should it have?"
)


memory.add_message(
    session_id,
    "assistant",
    (
        "It should have reasoning, memory, "
        "tools, services, ML models, and "
        "multitasking capabilities."
    )
)


conversation = memory.get_conversation(
    session_id
)


for message in conversation:

    print(
        message["role"].upper(),
        ":",
        message["content"]
    )

print()


# ==================================================
# 11. CREATE TASKS
# ==================================================

print("TEST 8: Task Memory")
print()


memory.create_task(
    task_id="ML-001",
    title="Train machine failure model",
    details=(
        "Prepare a larger representative dataset "
        "and evaluate candidate models."
    ),
    priority=8
)


memory.create_task(
    task_id="AI-001",
    title="Build memory subsystem",
    details=(
        "Create persistent and retrieval-ready "
        "memory architecture."
    ),
    priority=9
)


memory.create_task(
    task_id="AI-002",
    title="Build tool registry",
    details=(
        "Create a common interface for "
        "external tools and services."
    ),
    priority=9
)


print(
    "Tasks created."
)

print()


# ==================================================
# 12. READ TASKS
# ==================================================

print("TEST 9: Read Tasks")
print()


tasks = memory.get_tasks()


for task in tasks:

    print(
        "Task:",
        task["task_id"]
    )

    print(
        "Title:",
        task["title"]
    )

    print(
        "Status:",
        task["status"]
    )

    print(
        "Priority:",
        task["priority"]
    )

    print()


# ==================================================
# 13. UPDATE TASK
# ==================================================

print("TEST 10: Update Task")
print()


memory.update_task(
    "AI-001",
    "in_progress"
)


updated_tasks = memory.get_tasks(
    status="in_progress"
)


for task in updated_tasks:

    print(
        task["task_id"],
        "->",
        task["status"]
    )

print()


# ==================================================
# 14. RECORD SYSTEM EVENT
# ==================================================

print("TEST 11: Event Memory")
print()


memory.record_event(
    event_type="lesson_completed",
    source="silverwing-ml",
    payload={
        "lesson": 44,
        "phase": 4,
        "component": "memory",
        "status": "completed"
    }
)


print(
    "Event recorded."
)

print()


# ==================================================
# 15. MEMORY STATISTICS
# ==================================================

print("TEST 12: Memory Statistics")
print()


statistics = memory.statistics()


for name, count in statistics.items():

    print(
        name,
        ":",
        count
    )

print()


# ==================================================
# 16. BUILD CONTEXT FROM MEMORY
# ==================================================

print("TEST 13: Context Retrieval")
print()


memory_results = memory.search_memories(
    "AI"
)


context_items = []


for result in memory_results:

    context_items.append(
        {
            "type":
                result["memory_type"],

            "subject":
                result["subject"],

            "content":
                result["content"],

            "importance":
                result["importance"]
        }
    )


print(
    json.dumps(
        context_items,
        indent=4
    )
)

print()


# ==================================================
# 17. CREATE MEMORY CONTEXT
# ==================================================

print("TEST 14: Memory Context")
print()


def build_memory_context(
        memories
):

    lines = []


    for memory_item in memories:

        lines.append(
            (
                f"[{memory_item['type']}] "
                f"{memory_item['content']}"
            )
        )


    return "\n".join(
        lines
    )


memory_context = (
    build_memory_context(
        context_items
    )
)


print(
    memory_context
)

print()


# ==================================================
# 18. PRIORITIZE MEMORIES
# ==================================================

print("TEST 15: Memory Prioritization")
print()


def prioritize_memories(
        memories
):

    return sorted(
        memories,
        key=lambda item:
        item["importance"],
        reverse=True
    )


prioritized_memories = (
    prioritize_memories(
        context_items
    )
)


for item in prioritized_memories:

    print(
        item["importance"],
        "->",
        item["content"]
    )

print()


# ==================================================
# 19. PERSISTENCE TEST
# ==================================================

print("TEST 16: Persistence")
print()


connection.close()


print(
    "Original database connection closed."
)

print()


# Reopen database to prove persistence.

new_connection = sqlite3.connect(
    DATABASE_FILE
)

new_connection.row_factory = (
    sqlite3.Row
)


new_memory = MemoryManager(
    new_connection
)


restored_memories = (
    new_memory.search_memories(
        "Silverwing"
    )
)


print(
    "Restored memories:",
    len(restored_memories)
)

print()


for item in restored_memories:

    print(
        item["content"]
    )

print()


# ==================================================
# 20. RESTORED DATABASE STATISTICS
# ==================================================

print("TEST 17: Restored Statistics")
print()


restored_statistics = (
    new_memory.statistics()
)


for name, count in (
        restored_statistics.items()
):

    print(
        name,
        ":",
        count
    )

print()


# ==================================================
# 21. MEMORY ARCHITECTURE
# ==================================================

print("SILVERWING MEMORY ARCHITECTURE")
print()

print("User interaction")
print("      ↓")
print("Memory Manager")
print("      ↓")
print("┌────────────────────────────┐")
print("│ SQLite Memory Store        │")
print("├────────────────────────────┤")
print("│ Memories                   │")
print("│ Conversations              │")
print("│ Tasks                      │")
print("│ Events                     │")
print("└────────────────────────────┘")
print("      ↓")
print("Retrieval")
print("      ↓")
print("Context Builder")
print("      ↓")
print("LLM / Agent")

print()


# ==================================================
# 22. FUTURE MEMORY ARCHITECTURE
# ==================================================

print("FUTURE MEMORY ARCHITECTURE")
print()

print("Short-Term Memory")
print("       +")
print("Long-Term Memory")
print("       +")
print("Semantic Memory")
print("       +")
print("Episodic Memory")
print("       +")
print("Task Memory")
print("       +")
print("Knowledge Retrieval")
print("       ↓")
print("Memory Manager")
print("       ↓")
print("Context / Agent")

print()


# ==================================================
# 23. WHY SQLITE IS ONLY THE BEGINNING
# ==================================================

print("MEMORY EVOLUTION")
print()

print("JSON")
print(" ↓")
print("SQLite")
print(" ↓")
print("PostgreSQL")
print(" ↓")
print("Vector Database")
print(" ↓")
print("Hybrid Memory")
print(" ↓")
print("Semantic + Episodic + Task Memory")

print()


# ==================================================
# 24. PERSONAL AI MEMORY
# ==================================================

print("PERSONAL AI MEMORY")
print()

print(
    "A personal AI can use persistent memory "
    "to maintain continuity between sessions."
)

print()

print(
    "That memory can contain conversations, "
    "tasks, project knowledge, events, and "
    "other explicitly stored information."
)

print()


# ==================================================
# 25. IMPORTANT ARCHITECTURE PRINCIPLE
# ==================================================

print("ARCHITECTURE PRINCIPLE")
print()

print(
    "The LLM should not be the database."
)

print()

print(
    "The memory subsystem should store and "
    "retrieve information independently."
)

print()

print(
    "The context builder decides what memory "
    "is relevant to a particular reasoning task."
)

print()


# ==================================================
# 26. CURRENT SILVERWING ARCHITECTURE
# ==================================================

print("CURRENT SILVERWING ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation")
print(" ↓")
print("Memory Manager")
print(" ↓")
print("Context Builder")
print(" ↓")
print("LLM")
print(" ↓")
print("Response")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 44 COMPLETE ===")


# ==================================================
# CLOSE FINAL CONNECTION
# ==================================================

new_connection.close()

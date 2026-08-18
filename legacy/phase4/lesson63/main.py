# Silverwing ML
# Phase 4 - Lesson 63
# Agent + Tools Inside the Conversational API
#
# Corrected version 2
# - No shared long-lived SQLite connection
# - Fresh SQLite connection per operation
# - Safe with FastAPI/Uvicorn lifecycle
# - Persistent conversation memory
# - Semantic memory
# - Tool registry
# - Agent tool execution
# - Conversational API


import json
import sqlite3
import time
import uuid

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 63")
print("Agent + Tools Inside the Conversational API")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "sshleifer/tiny-gpt2"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DATABASE_FILE = (
        Path(__file__).resolve().parent
        / "silverwing_agent_api.db"
)

SEED = 42

set_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print("LLM:", MODEL_NAME)
print("Embedding model:", EMBEDDING_MODEL_NAME)
print("Database:", DATABASE_FILE)
print("Device:", DEVICE)

print()


# ==================================================
# 2. LOAD TOKENIZER
# ==================================================

print("TEST 2: Load Tokenizer")
print()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print("Tokenizer loaded.")
print()


# ==================================================
# 3. LOAD LOCAL LLM
# ==================================================

print("TEST 3: Load Local LLM")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print("LLM loaded.")
print()


# ==================================================
# 4. LOAD EMBEDDING MODEL
# ==================================================

print("TEST 4: Load Embedding Model")
print()


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


EMBEDDING_DIMENSION = (
    embedding_model.get_embedding_dimension()
)


print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION
)

print()


# ==================================================
# 5. SQLITE DATABASE MANAGER
# ==================================================

class DatabaseManager:
    """
    Opens a fresh SQLite connection for each
    database operation.

    This avoids using a connection that may have
    been closed by test code or invalidated by
    application lifecycle changes.
    """

    def __init__(
            self,
            database_file: Path
    ):

        self.database_file = (
            database_file
        )


    @contextmanager
    def connection(self):

        conn = sqlite3.connect(
            self.database_file,
            timeout=30.0
        )

        conn.row_factory = sqlite3.Row


        try:

            yield conn

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            conn.close()


database = DatabaseManager(
    DATABASE_FILE
)


# ==================================================
# 6. DATABASE INITIALIZATION
# ==================================================

print("TEST 5: Database")
print()


def initialize_database():

    with database.connection() as connection:

        cursor = connection.cursor()


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                                                    session_id TEXT PRIMARY KEY,
                                                    created_at TEXT NOT NULL,
                                                    updated_at TEXT NOT NULL
            )
            """
        )


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    session_id TEXT NOT NULL,
                                                    role TEXT NOT NULL,
                                                    content TEXT NOT NULL,
                                                    created_at TEXT NOT NULL
            )
            """
        )


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    session_id TEXT,
                                                    memory_type TEXT NOT NULL,
                                                    content TEXT NOT NULL,
                                                    importance REAL DEFAULT 0.5,
                                                    embedding BLOB NOT NULL,
                                                    created_at TEXT NOT NULL
            )
            """
        )


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_events (
                                                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                       session_id TEXT,
                                                       tool_name TEXT NOT NULL,
                                                       arguments TEXT NOT NULL,
                                                       result TEXT,
                                                       status TEXT NOT NULL,
                                                       created_at TEXT NOT NULL
            )
            """
        )


initialize_database()


print("Database ready.")
print()


# ==================================================
# 7. UTILITIES
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def to_json(
        value
):

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str
    )


def create_embedding(
        text
):

    vector = embedding_model.encode(
        text,
        normalize_embeddings=True
    )


    return np.asarray(
        vector,
        dtype=np.float32
    )


def serialize_embedding(
        vector
):

    return np.asarray(
        vector,
        dtype=np.float32
    ).tobytes()


def deserialize_embedding(
        blob
):

    return np.frombuffer(
        blob,
        dtype=np.float32
    )


def cosine_similarity(
        vector_a,
        vector_b
):

    denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        denominator
    )


# ==================================================
# 8. MEMORY MANAGER
# ==================================================

class MemoryManager:

    def create_session(self):

        session_id = str(
            uuid.uuid4()
        )


        with database.connection() as connection:

            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    utc_now(),
                    utc_now()
                )
            )


        return session_id


    def session_exists(
            self,
            session_id
    ):

        with database.connection() as connection:

            row = connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE session_id = ?
                """,
                (
                    session_id,
                )
            ).fetchone()


        return row is not None


    def touch_session(
            self,
            session_id
    ):

        with database.connection() as connection:

            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (
                    utc_now(),
                    session_id
                )
            )


    def add_message(
            self,
            session_id,
            role,
            content
    ):

        with database.connection() as connection:

            connection.execute(
                """
                INSERT INTO messages (
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
                    utc_now()
                )
            )


            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (
                    utc_now(),
                    session_id
                )
            )


    def get_messages(
            self,
            session_id,
            limit=12
    ):

        with database.connection() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                    LIMIT ?
                """,
                (
                    session_id,
                    limit
                )
            ).fetchall()


        return list(
            reversed(
                [
                    dict(row)
                    for row in rows
                ]
            )
        )


    def store_memory(
            self,
            session_id,
            content,
            memory_type="conversation",
            importance=0.5
    ):

        embedding = create_embedding(
            content
        )


        with database.connection() as connection:

            connection.execute(
                """
                INSERT INTO memories (
                    session_id,
                    memory_type,
                    content,
                    importance,
                    embedding,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    memory_type,
                    content,
                    importance,
                    serialize_embedding(
                        embedding
                    ),
                    utc_now()
                )
            )


    def search_memory(
            self,
            query,
            limit=5
    ):

        query_embedding = create_embedding(
            query
        )


        with database.connection() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM memories
                ORDER BY id ASC
                """
            ).fetchall()


        results = []


        for row in rows:

            memory_embedding = (
                deserialize_embedding(
                    row["embedding"]
                )
            )


            similarity = cosine_similarity(
                query_embedding,
                memory_embedding
            )


            results.append(
                {
                    "id":
                        row["id"],

                    "session_id":
                        row["session_id"],

                    "memory_type":
                        row["memory_type"],

                    "content":
                        row["content"],

                    "importance":
                        row["importance"],

                    "similarity":
                        similarity
                }
            )


        results.sort(
            key=lambda item: (
                item["similarity"],
                item["importance"]
            ),
            reverse=True
        )


        return results[:limit]


    def record_tool_event(
            self,
            session_id,
            tool_name,
            arguments,
            result,
            status
    ):

        with database.connection() as connection:

            connection.execute(
                """
                INSERT INTO tool_events (
                    session_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tool_name,
                    to_json(arguments),
                    to_json(result),
                    status,
                    utc_now()
                )
            )


    def statistics(self):

        tables = [
            "sessions",
            "messages",
            "memories",
            "tool_events"
        ]


        result = {}


        with database.connection() as connection:

            for table in tables:

                row = connection.execute(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM {table}
                    """
                ).fetchone()


                result[table] = row[
                    "count"
                ]


        return result


memory = MemoryManager()


# ==================================================
# 9. TOOL DEFINITIONS
# ==================================================

@dataclass
class ToolDefinition:

    name: str

    description: str

    parameters: Dict[str, Any]


    def schema(self):

        return {
            "name":
                self.name,

            "description":
                self.description,

            "parameters":
                self.parameters
        }


# ==================================================
# 10. TOOL IMPLEMENTATIONS
# ==================================================

def calculate(
        expression: str
):

    allowed = (
        "0123456789+-*/(). "
    )


    if any(
            character not in allowed
            for character in expression
    ):

        raise ValueError(
            "Unsupported characters."
        )


    result = eval(
        expression,
        {
            "__builtins__":
                {}
        },
        {}
    )


    if not isinstance(
            result,
            (int, float)
    ):

        raise ValueError(
            "Expression must return a number."
        )


    return {
        "result":
            result
    }


def machine_analyzer(
        temperature: float,
        pressure: float,
        rpm: float,
        operating_hours: float
):

    score = 0


    if temperature >= 100:

        score += 40

    elif temperature >= 80:

        score += 20


    if pressure >= 160:

        score += 20

    elif pressure >= 130:

        score += 10


    if rpm > 3000:

        score += 40

    elif rpm > 2500:

        score += 15


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
            level,

        "operating_hours":
            operating_hours
    }


def get_system_status():

    return {
        "status":
            "healthy",

        "service":
            "silverwing-agent-api",

        "timestamp":
            utc_now()
    }


# ==================================================
# 11. TOOL REGISTRY
# ==================================================

class ToolRegistry:

    def __init__(self):

        self.definitions = {

            "calculator":
                ToolDefinition(
                    name="calculator",
                    description=(
                        "Perform basic arithmetic calculations."
                    ),
                    parameters={
                        "type":
                            "object",

                        "properties": {
                            "expression": {
                                "type":
                                    "string",

                                "description":
                                    "Arithmetic expression."
                            }
                        },

                        "required": [
                            "expression"
                        ],

                        "additionalProperties":
                            False
                    }
                ),

            "machine_analyzer":
                ToolDefinition(
                    name="machine_analyzer",
                    description=(
                        "Analyze machine temperature, "
                        "pressure, RPM and operating hours."
                    ),
                    parameters={
                        "type":
                            "object",

                        "properties": {
                            "temperature": {
                                "type":
                                    "number"
                            },

                            "pressure": {
                                "type":
                                    "number"
                            },

                            "rpm": {
                                "type":
                                    "number"
                            },

                            "operating_hours": {
                                "type":
                                    "number"
                            }
                        },

                        "required": [
                            "temperature",
                            "pressure",
                            "rpm",
                            "operating_hours"
                        ],

                        "additionalProperties":
                            False
                    }
                ),

            "system_status":
                ToolDefinition(
                    name="system_status",
                    description=(
                        "Check Silverwing service status."
                    ),
                    parameters={
                        "type":
                            "object",

                        "properties": {},

                        "required": [],

                        "additionalProperties":
                            False
                    }
                )
        }


        self.functions = {

            "calculator":
                calculate,

            "machine_analyzer":
                machine_analyzer,

            "system_status":
                get_system_status
        }


    def get(
            self,
            tool_name
    ):

        return self.functions.get(
            tool_name
        )


    def definition(
            self,
            tool_name
    ):

        return self.definitions.get(
            tool_name
        )


    def schemas(self):

        return [
            definition.schema()
            for definition
            in self.definitions.values()
        ]


    def list_tools(self):

        return list(
            self.definitions.keys()
        )


tools = ToolRegistry()


# ==================================================
# 12. TOOL VALIDATION
# ==================================================

def validate_tool_arguments(
        definition,
        arguments
):

    if not isinstance(
            arguments,
            dict
    ):

        raise ValueError(
            "Tool arguments must be an object."
        )


    schema = definition.parameters


    properties = schema.get(
        "properties",
        {}
    )


    required = schema.get(
        "required",
        []
    )


    for name in required:

        if name not in arguments:

            raise ValueError(
                f"Missing required argument: {name}"
            )


    if (
            schema.get(
                "additionalProperties"
            )
            is False
    ):

        unknown = (
                set(arguments)
                -
                set(properties)
        )


        if unknown:

            raise ValueError(
                "Unknown arguments: "
                +
                ", ".join(
                    sorted(
                        unknown
                    )
                )
            )


    return True


# ==================================================
# 13. TOOL EXECUTOR
# ==================================================

class ToolExecutor:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def execute(
            self,
            session_id,
            tool_name,
            arguments
    ):

        definition = (
            self.registry.definition(
                tool_name
            )
        )


        function = (
            self.registry.get(
                tool_name
            )
        )


        if (
                definition is None
                or
                function is None
        ):

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )


        validate_tool_arguments(
            definition,
            arguments
        )


        start = time.perf_counter()


        try:

            result = function(
                **arguments
            )


            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            memory.record_tool_event(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result={
                    "result":
                        result,

                    "duration_ms":
                        round(
                            duration,
                            3
                        )
                },
                status="success"
            )


            return {
                "status":
                    "success",

                "tool":
                    tool_name,

                "result":
                    result,

                "duration_ms":
                    round(
                        duration,
                        3
                    )
            }


        except Exception as error:

            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            memory.record_tool_event(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result={
                    "error":
                        str(error),

                    "duration_ms":
                        round(
                            duration,
                            3
                        )
                },
                status="failed"
            )


            raise


executor = ToolExecutor(
    tools
)


# ==================================================
# 14. AGENT
# ==================================================

class Agent:

    def __init__(
            self,
            tool_registry,
            executor
    ):

        self.tool_registry = (
            tool_registry
        )

        self.executor = executor


    def choose_tool(
            self,
            message: str
    ):

        text = message.lower()


        if any(
                phrase in text
                for phrase in [
                    "calculate",
                    "compute",
                    "multiply",
                    "divide",
                    "add",
                    "subtract"
                ]
        ):

            return "calculator"


        if any(
                phrase in text
                for phrase in [
                    "machine",
                    "temperature",
                    "pressure",
                    "rpm",
                    "machine risk",
                    "machine condition"
                ]
        ):

            return "machine_analyzer"


        if any(
                phrase in text
                for phrase in [
                    "system status",
                    "system health",
                    "are you running",
                    "is silverwing running"
                ]
        ):

            return "system_status"


        return None


    def extract_arguments(
            self,
            message,
            tool_name
    ):

        if tool_name == "calculator":

            text = message.lower()

            expression = text


            prefixes = [
                "calculate ",
                "compute ",
                "what is "
            ]


            for prefix in prefixes:

                if expression.startswith(
                        prefix
                ):

                    expression = (
                        expression[
                            len(prefix):
                        ]
                    )

                    break


            return {
                "expression":
                    expression.strip()
            }


        if tool_name == "machine_analyzer":

            return {
                "temperature":
                    101,

                "pressure":
                    135,

                "rpm":
                    3100,

                "operating_hours":
                    4200
            }


        if tool_name == "system_status":

            return {}


        return {}


    def run(
            self,
            session_id,
            message
    ):

        tool_name = (
            self.choose_tool(
                message
            )
        )


        if tool_name is None:

            return {
                "used_tool":
                    False,

                "tool_name":
                    None,

                "arguments":
                    {},

                "tool_result":
                    None
            }


        arguments = (
            self.extract_arguments(
                message,
                tool_name
            )
        )


        result = self.executor.execute(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments
        )


        return {
            "used_tool":
                True,

            "tool_name":
                tool_name,

            "arguments":
                arguments,

            "tool_result":
                result
        }


agent = Agent(
    tools,
    executor
)


# ==================================================
# 15. CONTEXT BUILDER
# ==================================================

def build_context(
        session_id,
        user_message,
        agent_result
):

    messages = memory.get_messages(
        session_id,
        limit=12
    )


    semantic_memories = (
        memory.search_memory(
            user_message,
            limit=4
        )
    )


    lines = []


    lines.append(
        "SYSTEM:"
    )


    lines.append(
        (
            "You are Silverwing, a personal AI assistant."
        )
    )


    lines.append(
        "Use tool results when available."
    )


    lines.append(
        "Do not invent tool results."
    )


    lines.append(
        ""
    )


    lines.append(
        "RELEVANT LONG-TERM MEMORY:"
    )


    for memory_item in (
            semantic_memories
    ):

        lines.append(
            "- "
            +
            memory_item[
                "content"
            ]
        )


    lines.append(
        ""
    )


    lines.append(
        "RECENT CONVERSATION:"
    )


    for message in messages:

        lines.append(
            (
                f"{message['role'].upper()}: "
                f"{message['content']}"
            )
        )


    lines.append(
        ""
    )


    if (
            agent_result
            and
            agent_result.get(
                "used_tool"
            )
    ):

        lines.append(
            "TOOL RESULT:"
        )


        lines.append(
            to_json(
                agent_result[
                    "tool_result"
                ]
            )
        )


        lines.append(
            ""
        )


    lines.append(
        "LATEST USER REQUEST:"
    )


    lines.append(
        user_message
    )


    lines.append(
        ""
    )


    lines.append(
        "ASSISTANT:"
    )


    return "\n".join(
        lines
    )


# ==================================================
# 16. LLM GENERATION
# ==================================================

def generate_response(
        prompt,
        temperature,
        top_k,
        top_p,
        max_tokens
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(
            DEVICE
        )
        for key, value
        in inputs.items()
    }


    input_tokens = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=(
                tokenizer.pad_token_id
            )
        )


    total_tokens = (
        output_ids.shape[1]
    )


    generated_tokens = (
            total_tokens
            -
            input_tokens
    )


    full_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


    if full_text.startswith(
            prompt
    ):

        response = (
            full_text[
                len(prompt):
            ]
        ).strip()

    else:

        response = full_text


    return {
        "text":
            response,

        "input_tokens":
            input_tokens,

        "generated_tokens":
            generated_tokens
    }


# ==================================================
# 17. FASTAPI
# ==================================================

app = FastAPI(
    title="Silverwing Agent API",
    description=(
        "Conversational AI with persistent memory "
        "and tool execution."
    ),
    version="1.0.1"
)


# ==================================================
# 18. REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=4000
    )

    session_id: Optional[str] = None

    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0
    )

    top_k: int = Field(
        default=50,
        ge=1
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0
    )

    max_tokens: int = Field(
        default=50,
        ge=1,
        le=200
    )


# ==================================================
# 19. ROOT
# ==================================================

@app.get("/")
def root():

    return {
        "project":
            "Silverwing ML",

        "phase":
            4,

        "lesson":
            63,

        "service":
            "agent-api",

        "status":
            "running"
    }


# ==================================================
# 20. HEALTH
# ==================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "llm":
            MODEL_NAME,

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "device":
            str(
                DEVICE
            ),

        "database":
            str(
                DATABASE_FILE
            ),

        "tools":
            tools.list_tools(),

        "statistics":
            memory.statistics()
    }


# ==================================================
# 21. CAPABILITIES
# ==================================================

@app.get("/capabilities")
def capabilities():

    return {
        "service":
            "silverwing-agent-api",

        "capabilities": [
            "conversation",
            "persistent_memory",
            "semantic_memory",
            "tool_execution",
            "machine_analysis",
            "calculator",
            "system_status",
            "local_llm"
        ],

        "tools":
            tools.schemas()
    }


# ==================================================
# 22. CREATE SESSION
# ==================================================

@app.post("/sessions")
def create_session():

    session_id = (
        memory.create_session()
    )


    return {
        "session_id":
            session_id,

        "status":
            "created"
    }


# ==================================================
# 23. GET SESSION
# ==================================================

@app.get(
    "/sessions/{session_id}"
)
def get_session(
        session_id: str
):

    if not memory.session_exists(
            session_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )


    return {
        "session_id":
            session_id,

        "messages":
            memory.get_messages(
                session_id,
                limit=100
            )
    }


# ==================================================
# 24. TOOLS
# ==================================================

@app.get("/tools")
def list_tools():

    return {
        "tools":
            tools.schemas()
    }


@app.post(
    "/tools/{tool_name}"
)
def execute_tool_endpoint(
        tool_name: str,
        arguments: Dict[str, Any]
):

    session_id = (
        memory.create_session()
    )


    try:

        result = executor.execute(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments
        )


        return {
            "session_id":
                session_id,

            "result":
                result
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ==================================================
# 25. MEMORY SEARCH
# ==================================================

@app.post(
    "/memory/search"
)
def search_memory(
        query: str
):

    return {
        "query":
            query,

        "results":
            memory.search_memory(
                query,
                limit=10
            )
    }


# ==================================================
# 26. CHAT
# ==================================================

@app.post("/chat")
def chat(
        request: ChatRequest
):

    # ----------------------------------------------
    # Session
    # ----------------------------------------------

    session_id = (
        request.session_id
    )


    if session_id is None:

        session_id = (
            memory.create_session()
        )

    elif not memory.session_exists(
            session_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )


    # ----------------------------------------------
    # Store user message
    # ----------------------------------------------

    memory.add_message(
        session_id,
        "user",
        request.message
    )


    # ----------------------------------------------
    # Agent
    # ----------------------------------------------

    try:

        agent_result = agent.run(
            session_id,
            request.message
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Tool execution failed: {error}"
            )
        )


    # ----------------------------------------------
    # Context
    # ----------------------------------------------

    prompt = build_context(
        session_id,
        request.message,
        agent_result
    )


    # ----------------------------------------------
    # LLM
    # ----------------------------------------------

    start = time.perf_counter()


    result = generate_response(
        prompt=prompt,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )


    latency = (
                      time.perf_counter()
                      -
                      start
              ) * 1000


    assistant_response = result[
        "text"
    ]


    # ----------------------------------------------
    # Persist response
    # ----------------------------------------------

    memory.add_message(
        session_id,
        "assistant",
        assistant_response
    )


    # ----------------------------------------------
    # Persist tool history
    # ----------------------------------------------

    if agent_result.get(
            "used_tool"
    ):

        memory.store_memory(
            session_id,
            (
                f"Tool "
                f"{agent_result['tool_name']} "
                f"was used for "
                f"'{request.message}'."
            ),
            memory_type="tool_history",
            importance=0.75
        )


    return {
        "session_id":
            session_id,

        "request_id":
            str(
                uuid.uuid4()
            ),

        "message":
            assistant_response,

        "agent": {
            "used_tool":
                agent_result[
                    "used_tool"
                ],

            "tool":
                agent_result[
                    "tool_name"
                ],

            "arguments":
                agent_result[
                    "arguments"
                ],

            "tool_result":
                agent_result[
                    "tool_result"
                ]
        },

        "usage": {
            "input_tokens":
                result[
                    "input_tokens"
                ],

            "generated_tokens":
                result[
                    "generated_tokens"
                ]
        },

        "latency_ms":
            round(
                latency,
                3
            )
    }


# ==================================================
# 27. CONTEXT ENDPOINT
# ==================================================

@app.get(
    "/sessions/{session_id}/context"
)
def context_endpoint(
        session_id: str
):

    if not memory.session_exists(
            session_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )


    messages = memory.get_messages(
        session_id,
        limit=12
    )


    latest_message = ""


    if messages:

        latest_message = (
            messages[-1]["content"]
        )


    context = build_context(
        session_id,
        latest_message,
        {
            "used_tool":
                False
        }
    )


    tokens = tokenizer(
        context,
        return_tensors="pt"
    )


    return {
        "session_id":
            session_id,

        "context_tokens":
            tokens[
                "input_ids"
            ].shape[1],

        "prompt":
            context
    }


# ==================================================
# 28. MEMORY STATISTICS
# ==================================================

@app.get(
    "/memory/statistics"
)
def memory_statistics():

    return memory.statistics()


# ==================================================
# 29. LOCAL TESTS
# ==================================================

print("TEST 6: Tool Registry")
print()


for name in (
        tools.list_tools()
):

    print(
        "-",
        name
    )


print()


print("TEST 7: Tool Schemas")
print()


print(
    json.dumps(
        tools.schemas(),
        indent=4
    )
)

print()


# ==================================================
# 30. TOOL SELECTION TEST
# ==================================================

print("TEST 8: Agent Tool Selection")
print()


test_messages = [
    "Calculate 25 * 8",
    "Analyze this machine",
    "Is Silverwing running?",
    "Tell me about machine learning"
]


for message in test_messages:

    print(
        "Message:",
        message
    )

    print(
        "Selected tool:",
        agent.choose_tool(
            message
        )
    )

    print()


# ==================================================
# 31. TOOL EXECUTION TEST
# ==================================================

print("TEST 9: Tool Execution")
print()


test_session = (
    memory.create_session()
)


calculation_result = (
    executor.execute(
        session_id=test_session,
        tool_name="calculator",
        arguments={
            "expression":
                "25 * 8"
        }
    )
)


print(
    json.dumps(
        calculation_result,
        indent=4
    )
)

print()


machine_result = (
    executor.execute(
        session_id=test_session,
        tool_name="machine_analyzer",
        arguments={
            "temperature":
                101,

            "pressure":
                135,

            "rpm":
                3100,

            "operating_hours":
                4200
        }
    )
)


print(
    json.dumps(
        machine_result,
        indent=4
    )
)

print()


# ==================================================
# 32. AGENT LOOP TEST
# ==================================================

print("TEST 10: Agent Loop")
print()


agent_result = agent.run(
    test_session,
    "Analyze this machine."
)


print(
    json.dumps(
        agent_result,
        indent=4
    )
)

print()


# ==================================================
# 33. TOOL-AWARE CONTEXT TEST
# ==================================================

print("TEST 11: Tool-Aware Context")
print()


sample_context = build_context(
    test_session,
    "Analyze this machine.",
    agent_result
)


print(
    sample_context
)

print()


# ==================================================
# 34. MEMORY TEST
# ==================================================

print("TEST 12: Semantic Memory")
print()


memory.store_memory(
    test_session,
    (
        "Silverwing uses tools to extend "
        "its capabilities."
    ),
    memory_type="architecture",
    importance=0.9
)


memory.store_memory(
    test_session,
    (
        "Silverwing can analyze machine "
        "operating conditions."
    ),
    memory_type="capability",
    importance=0.85
)


memory_results = memory.search_memory(
    "What can Silverwing do with tools?",
    limit=3
)


for item in memory_results:

    print(
        "Similarity:",
        round(
            item["similarity"],
            4
        )
    )

    print(
        item["content"]
    )

    print()


# ==================================================
# 35. ARCHITECTURE
# ==================================================

print("AGENT ARCHITECTURE")
print()

print("User")
print(" ↓")
print("FastAPI")
print(" ↓")
print("Conversation / Memory")
print(" ↓")
print("Agent")
print(" ↓")
print("Tool Registry")
print(" ↓")
print("Tool Executor")
print(" ↓")
print("Tool Result")
print(" ↓")
print("Context Builder")
print(" ↓")
print("LLM")
print(" ↓")
print("Final Response")
print(" ↓")
print("Persistent Memory")

print()


# ==================================================
# 36. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "The tool-selection logic is deterministic."
)

print()

print(
    "The tiny GPT-2 model is not being trusted "
    "to produce structured tool calls."
)

print()

print(
    "The next architecture step is a structured "
    "tool-call protocol."
)

print()


# ==================================================
# 37. CURRENT PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("LLM")
print(" ↓")
print("Conversation")
print(" ↓")
print("Persistent Memory")
print(" ↓")
print("Semantic Retrieval")
print(" ↓")
print("Tool Registry")
print(" ↓")
print("Agent")
print(" ↓")
print("Tool Execution")
print(" ↓")
print("Tool-Augmented Conversation")
print(" ↓")
print("Conversational API")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 63 COMPLETE ===")
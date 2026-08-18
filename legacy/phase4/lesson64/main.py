# Silverwing ML
# Phase 4 - Lesson 64
# Structured Tool Calls and Reliable Response Generation

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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 64")
print("Structured Tool Calls and Reliable Response Generation")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DATABASE_FILE = (
        Path(__file__).resolve().parent
        / "silverwing_structured_agent.db"
)


print("TEST 1: Configuration")
print()

print(
    "Embedding model:",
    EMBEDDING_MODEL_NAME
)

print(
    "Database:",
    DATABASE_FILE
)

print()


# ==================================================
# 2. EMBEDDING MODEL
# ==================================================

print("TEST 2: Load Embedding Model")
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
# 3. DATABASE MANAGER
# ==================================================

class DatabaseManager:

    def __init__(
            self,
            database_file
    ):

        self.database_file = (
            database_file
        )


    @contextmanager
    def connection(self):

        connection = sqlite3.connect(
            self.database_file,
            timeout=30.0
        )

        connection.row_factory = sqlite3.Row


        try:

            yield connection

            connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()


database = DatabaseManager(
    DATABASE_FILE
)


def initialize_database():

    with database.connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                                                    session_id TEXT PRIMARY KEY,
                                                    created_at TEXT NOT NULL,
                                                    updated_at TEXT NOT NULL
            )
            """
        )


        connection.execute(
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


        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                                                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                      session_id TEXT NOT NULL,
                                                      call_id TEXT NOT NULL,
                                                      tool_name TEXT NOT NULL,
                                                      arguments TEXT NOT NULL,
                                                      status TEXT NOT NULL,
                                                      result TEXT,
                                                      error TEXT,
                                                      duration_ms REAL,
                                                      created_at TEXT NOT NULL
            )
            """
        )


initialize_database()


print("TEST 3: Database Ready")
print()


# ==================================================
# 4. UTILITIES
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


# ==================================================
# 5. SESSION STORE
# ==================================================

class SessionStore:

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


    def exists(
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
            limit=20
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


sessions = SessionStore()


# ==================================================
# 6. STRUCTURED TOOL CALL
# ==================================================

@dataclass
class ToolCall:

    call_id: str

    tool_name: str

    arguments: Dict[str, Any]


    def to_dict(self):

        return {
            "call_id":
                self.call_id,

            "tool_name":
                self.tool_name,

            "arguments":
                self.arguments
        }


# ==================================================
# 7. STRUCTURED TOOL RESULT
# ==================================================

@dataclass
class ToolResult:

    call_id: str

    tool_name: str

    status: str

    result: Any = None

    error: Optional[str] = None

    duration_ms: Optional[float] = None


    def to_dict(self):

        return {
            "call_id":
                self.call_id,

            "tool_name":
                self.tool_name,

            "status":
                self.status,

            "result":
                self.result,

            "error":
                self.error,

            "duration_ms":
                self.duration_ms
        }


# ==================================================
# 8. TOOL DEFINITION
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
# 9. TOOL IMPLEMENTATIONS
# ==================================================

def calculate(
        expression: str
):

    allowed = (
        "0123456789+-*/(). "
    )


    if not expression.strip():

        raise ValueError(
            "Expression cannot be empty."
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
            "Expression must produce a number."
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

        "temperature":
            temperature,

        "pressure":
            pressure,

        "rpm":
            rpm,

        "operating_hours":
            operating_hours
    }


def system_status():

    return {
        "status":
            "healthy",

        "service":
            "silverwing-agent",

        "timestamp":
            utc_now()
    }


# ==================================================
# 10. TOOL REGISTRY
# ==================================================

class ToolRegistry:

    def __init__(self):

        self.definitions = {

            "calculator":
                ToolDefinition(
                    name="calculator",

                    description=(
                        "Perform basic arithmetic "
                        "calculations."
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
                        "Analyze machine operating "
                        "conditions."
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
                        "Check Silverwing system status."
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
                system_status
        }


    def get(
            self,
            name
    ):

        return self.functions.get(
            name
        )


    def definition(
            self,
            name
    ):

        return self.definitions.get(
            name
        )


    def schemas(self):

        return [
            item.schema()
            for item
            in self.definitions.values()
        ]


    def list_tools(self):

        return list(
            self.definitions.keys()
        )


tools = ToolRegistry()


# ==================================================
# 11. ARGUMENT VALIDATION
# ==================================================

def validate_arguments(
        definition,
        arguments
):

    if not isinstance(
            arguments,
            dict
    ):

        raise ValueError(
            "Arguments must be an object."
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


    for required_name in required:

        if required_name not in arguments:

            raise ValueError(
                f"Missing required argument: "
                f"{required_name}"
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
# 12. TOOL EXECUTOR
# ==================================================

class ToolExecutor:

    def execute(
            self,
            session_id,
            call: ToolCall
    ):

        definition = tools.definition(
            call.tool_name
        )


        function = tools.get(
            call.tool_name
        )


        if (
                definition is None
                or
                function is None
        ):

            raise ValueError(
                f"Unknown tool: "
                f"{call.tool_name}"
            )


        validate_arguments(
            definition,
            call.arguments
        )


        started = time.perf_counter()


        try:

            result = function(
                **call.arguments
            )


            duration = (
                               time.perf_counter()
                               -
                               started
                       ) * 1000


            tool_result = ToolResult(
                call_id=call.call_id,

                tool_name=call.tool_name,

                status="success",

                result=result,

                duration_ms=round(
                    duration,
                    3
                )
            )


        except Exception as error:

            duration = (
                               time.perf_counter()
                               -
                               started
                       ) * 1000


            tool_result = ToolResult(
                call_id=call.call_id,

                tool_name=call.tool_name,

                status="error",

                error=str(
                    error
                ),

                duration_ms=round(
                    duration,
                    3
                )
            )


        with database.connection() as connection:

            connection.execute(
                """
                INSERT INTO tool_calls (
                    session_id,
                    call_id,
                    tool_name,
                    arguments,
                    status,
                    result,
                    error,
                    duration_ms,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,

                    tool_result.call_id,

                    tool_result.tool_name,

                    to_json(
                        call.arguments
                    ),

                    tool_result.status,

                    to_json(
                        tool_result.result
                    ),

                    tool_result.error,

                    tool_result.duration_ms,

                    utc_now()
                )
            )


        return tool_result


executor = ToolExecutor()


# ==================================================
# 13. AGENT INTENT DETECTION
# ==================================================

class Agent:

    def choose_tool(
            self,
            message
    ):

        text = message.lower()


        if any(
                phrase in text
                for phrase in [
                    "calculate",
                    "compute",
                    "multiply",
                    "divide",
                    "subtract",
                    "add"
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
                    "is silverwing running",
                    "are you running"
                ]
        ):

            return "system_status"


        return None


    def build_tool_call(
            self,
            message
    ):

        tool_name = self.choose_tool(
            message
        )


        if tool_name is None:

            return None


        call_id = (
                "call_"
                +
                uuid.uuid4().hex[:12]
        )


        if tool_name == "calculator":

            expression = (
                message.strip()
            )


            prefixes = [
                "calculate ",
                "compute ",
                "what is "
            ]


            lowered = expression.lower()


            for prefix in prefixes:

                if lowered.startswith(
                        prefix
                ):

                    expression = (
                        expression[
                            len(prefix):
                        ]
                    )

                    break


            arguments = {
                "expression":
                    expression.strip()
            }


        elif tool_name == "machine_analyzer":

            # Demonstration machine data.
            arguments = {
                "temperature":
                    101,

                "pressure":
                    135,

                "rpm":
                    3100,

                "operating_hours":
                    4200
            }


        else:

            arguments = {}


        return ToolCall(
            call_id=call_id,

            tool_name=tool_name,

            arguments=arguments
        )


agent = Agent()


# ==================================================
# 14. RESPONSE BUILDER
# ==================================================

class ResponseBuilder:
    """
    Reliable deterministic response layer.

    This prevents the weak tiny GPT-2 model from
    corrupting a correct tool result.

    A future instruction-tuned LLM can replace
    this component while keeping the same input
    and output contract.
    """


    def build(
            self,
            user_message: str,
            tool_call: Optional[ToolCall],
            tool_result: Optional[ToolResult]
    ):

        if (
                tool_call is None
                or
                tool_result is None
        ):

            return (
                "I received your request. "
                "The current lesson does not yet "
                "connect this type of request to "
                "a specialized capability."
            )


        if (
                tool_result.status
                !=
                "success"
        ):

            return (
                "I could not complete the requested "
                f"{tool_call.tool_name} operation. "
                "The tool reported an error: "
                f"{tool_result.error}"
            )


        result = tool_result.result


        if tool_call.tool_name == (
                "calculator"
        ):

            value = result[
                "result"
            ]


            return (
                f"The result is {value}."
            )


        if tool_call.tool_name == (
                "machine_analyzer"
        ):

            risk = result[
                "risk_level"
            ]

            score = result[
                "risk_score"
            ]

            temperature = result[
                "temperature"
            ]

            pressure = result[
                "pressure"
            ]

            rpm = result[
                "rpm"
            ]

            hours = result[
                "operating_hours"
            ]


            return (
                "Machine analysis complete. "
                f"Risk level: {risk}. "
                f"Risk score: {score}. "
                f"Temperature: {temperature}. "
                f"Pressure: {pressure}. "
                f"RPM: {rpm}. "
                f"Operating hours: {hours}."
            )


        if tool_call.tool_name == (
                "system_status"
        ):

            return (
                "Silverwing is running normally. "
                f"System status: "
                f"{result['status']}."
            )


        return (
            "The requested operation completed "
            "successfully."
        )


response_builder = ResponseBuilder()


# ==================================================
# 15. FASTAPI
# ==================================================

app = FastAPI(
    title="Silverwing Structured Agent API",
    description=(
        "Structured tool calling and reliable "
        "response generation."
    ),
    version="1.0.0"
)


# ==================================================
# 16. REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=4000
    )

    session_id: Optional[str] = None


# ==================================================
# 17. ROOT
# ==================================================

@app.get("/")
def root():

    return {
        "project":
            "Silverwing ML",

        "phase":
            4,

        "lesson":
            64,

        "service":
            "structured-agent-api",

        "status":
            "running"
    }


# ==================================================
# 18. HEALTH
# ==================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "tools":
            tools.list_tools(),

        "database":
            str(
                DATABASE_FILE
            )
    }


# ==================================================
# 19. CAPABILITIES
# ==================================================

@app.get("/capabilities")
def capabilities():

    return {
        "structured_tool_calls":
            True,

        "reliable_response_builder":
            True,

        "tools":
            tools.schemas(),

        "architecture": [
            "agent",
            "tool_registry",
            "tool_executor",
            "response_builder"
        ]
    }


# ==================================================
# 20. CREATE SESSION
# ==================================================

@app.post("/sessions")
def create_session():

    session_id = (
        sessions.create_session()
    )


    return {
        "session_id":
            session_id,

        "status":
            "created"
    }


# ==================================================
# 21. GET SESSION
# ==================================================

@app.get(
    "/sessions/{session_id}"
)
def get_session(
        session_id: str
):

    if not sessions.exists(
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
            sessions.get_messages(
                session_id
            )
    }


# ==================================================
# 22. TOOLS
# ==================================================

@app.get("/tools")
def list_tools():

    return {
        "tools":
            tools.schemas()
    }


# ==================================================
# 23. CHAT
# ==================================================

@app.post("/chat")
def chat(
        request: ChatRequest
):

    session_id = request.session_id


    if session_id is None:

        session_id = (
            sessions.create_session()
        )

    elif not sessions.exists(
            session_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )


    sessions.add_message(
        session_id,
        "user",
        request.message
    )


    # ----------------------------------------------
    # Step 1: create structured tool call
    # ----------------------------------------------

    tool_call = agent.build_tool_call(
        request.message
    )


    tool_result = None


    # ----------------------------------------------
    # Step 2: execute tool
    # ----------------------------------------------

    if tool_call is not None:

        tool_result = executor.execute(
            session_id,
            tool_call
        )


    # ----------------------------------------------
    # Step 3: build reliable response
    # ----------------------------------------------

    response_text = response_builder.build(
        user_message=request.message,
        tool_call=tool_call,
        tool_result=tool_result
    )


    sessions.add_message(
        session_id,
        "assistant",
        response_text
    )


    return {
        "session_id":
            session_id,

        "request_id":
            str(
                uuid.uuid4()
            ),

        "message":
            response_text,

        "tool_call":
            tool_call.to_dict()
            if tool_call
            else None,

        "tool_result":
            tool_result.to_dict()
            if tool_result
            else None
    }


# ==================================================
# 24. DIRECT STRUCTURED TOOL EXECUTION
# ==================================================

@app.post(
    "/tool-calls"
)
def structured_tool_call(
        call: Dict[str, Any]
):

    required_fields = {
        "tool_name",
        "arguments"
    }


    missing = (
            required_fields
            -
            set(call)
    )


    if missing:

        raise HTTPException(
            status_code=422,
            detail=(
                    "Missing fields: "
                    +
                    ", ".join(
                        sorted(
                            missing
                        )
                    )
            )
        )


    session_id = (
        sessions.create_session()
    )


    tool_call = ToolCall(
        call_id=str(
            uuid.uuid4()
        ),

        tool_name=call[
            "tool_name"
        ],

        arguments=call[
            "arguments"
        ]
    )


    try:

        result = executor.execute(
            session_id,
            tool_call
        )


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    return {
        "tool_call":
            tool_call.to_dict(),

        "tool_result":
            result.to_dict()
    }


# ==================================================
# 25. TEST 4: TOOL SCHEMAS
# ==================================================

print("TEST 4: Structured Tool Schemas")
print()


print(
    json.dumps(
        tools.schemas(),
        indent=4
    )
)

print()


# ==================================================
# 26. TEST 5: STRUCTURED CALL
# ==================================================

print("TEST 5: Structured Tool Call")
print()


test_call = agent.build_tool_call(
    "Calculate 25 * 8"
)


print(
    json.dumps(
        test_call.to_dict(),
        indent=4
    )
)

print()


# ==================================================
# 27. TEST 6: EXECUTION
# ==================================================

print("TEST 6: Structured Tool Execution")
print()


test_session = (
    sessions.create_session()
)


test_result = executor.execute(
    test_session,
    test_call
)


print(
    json.dumps(
        test_result.to_dict(),
        indent=4
    )
)

print()


# ==================================================
# 28. TEST 7: RESPONSE GENERATION
# ==================================================

print("TEST 7: Reliable Response Generation")
print()


reliable_response = (
    response_builder.build(
        "Calculate 25 * 8",
        test_call,
        test_result
    )
)


print(
    reliable_response
)

print()


# ==================================================
# 29. TEST 8: MACHINE RESPONSE
# ==================================================

print("TEST 8: Machine Tool Response")
print()


machine_call = agent.build_tool_call(
    "Analyze this machine"
)


machine_result = executor.execute(
    test_session,
    machine_call
)


machine_response = response_builder.build(
    "Analyze this machine",
    machine_call,
    machine_result
)


print(
    machine_response
)

print()


# ==================================================
# 30. TEST 9: SYSTEM RESPONSE
# ==================================================

print("TEST 9: System Tool Response")
print()


status_call = agent.build_tool_call(
    "Is Silverwing running?"
)


status_result = executor.execute(
    test_session,
    status_call
)


status_response = response_builder.build(
    "Is Silverwing running?",
    status_call,
    status_result
)


print(
    status_response
)

print()


# ==================================================
# 31. TEST 10: ERROR HANDLING
# ==================================================

print("TEST 10: Tool Error Handling")
print()


bad_call = ToolCall(
    call_id=str(
        uuid.uuid4()
    ),

    tool_name="calculator",

    arguments={
        "expression":
            "25 / 0"
    }
)


bad_result = executor.execute(
    test_session,
    bad_call
)


print(
    json.dumps(
        bad_result.to_dict(),
        indent=4
    )
)

print()


bad_response = response_builder.build(
    "Calculate 25 / 0",
    bad_call,
    bad_result
)


print(
    bad_response
)

print()


# ==================================================
# 32. TEST 11: COMPLETE PIPELINE
# ==================================================

print("TEST 11: Complete Agent Pipeline")
print()


pipeline_message = (
    "Calculate 125 / 5 + 7"
)


pipeline_session = (
    sessions.create_session()
)


pipeline_call = agent.build_tool_call(
    pipeline_message
)


pipeline_result = executor.execute(
    pipeline_session,
    pipeline_call
)


pipeline_response = response_builder.build(
    pipeline_message,
    pipeline_call,
    pipeline_result
)


pipeline_record = {
    "user":
        pipeline_message,

    "tool_call":
        pipeline_call.to_dict(),

    "tool_result":
        pipeline_result.to_dict(),

    "final_response":
        pipeline_response
}


print(
    json.dumps(
        pipeline_record,
        indent=4
    )
)

print()


# ==================================================
# 33. TEST 12: SESSION PERSISTENCE
# ==================================================

print("TEST 12: Session Persistence")
print()


sessions.add_message(
    pipeline_session,
    "user",
    pipeline_message
)


sessions.add_message(
    pipeline_session,
    "assistant",
    pipeline_response
)


saved_messages = sessions.get_messages(
    pipeline_session
)


print(
    json.dumps(
        saved_messages,
        indent=4
    )
)

print()


# ==================================================
# 34. TEST 13: TOOL CALL HISTORY
# ==================================================

print("TEST 13: Tool Call History")
print()


with database.connection() as connection:

    rows = connection.execute(
        """
        SELECT *
        FROM tool_calls
        ORDER BY id DESC
            LIMIT 10
        """
    ).fetchall()


for row in rows:

    print(
        json.dumps(
            dict(row),
            indent=4,
            default=str
        )
    )

    print()


# ==================================================
# 35. AGENT PROTOCOL
# ==================================================

print("STRUCTURED AGENT PROTOCOL")
print()

print("User Request")
print("     ↓")
print("Intent Detection")
print("     ↓")
print("Tool Call")
print("     ↓")
print("Schema Validation")
print("     ↓")
print("Tool Execution")
print("     ↓")
print("Tool Result")
print("     ↓")
print("Response Builder")
print("     ↓")
print("Reliable Response")

print()


# ==================================================
# 36. LLM INTEGRATION POINT
# ==================================================

print("FUTURE LLM INTEGRATION")
print()

print(
    "The current agent creates ToolCall objects "
    "deterministically."
)

print()

print(
    "A stronger instruction-tuned LLM can later "
    "produce the same structured ToolCall contract."
)

print()

print(
    "The ToolExecutor does not need to change "
    "when the source of the ToolCall changes."
)

print()


# ==================================================
# 37. WHY THIS ARCHITECTURE IS IMPORTANT
# ==================================================

print("WHY THIS ARCHITECTURE MATTERS")
print()

print(
    "The system no longer requires the language "
    "model to directly perform calculations."
)

print()

print(
    "The system can verify and log the operation "
    "before producing a response."
)

print()

print(
    "A better language model can later replace "
    "only the decision/response layers."
)

print()


# ==================================================
# 38. FUTURE TOOL-CALLING FLOW
# ==================================================

print("FUTURE TOOL-CALLING FLOW")
print()

print("User")
print(" ↓")
print("Instruction-Tuned LLM")
print(" ↓")
print("Structured Tool Call")
print(" ↓")
print("Validator")
print(" ↓")
print("Permission Manager")
print(" ↓")
print("Tool Executor")
print(" ↓")
print("Tool Result")
print(" ↓")
print("LLM Response Generator")
print(" ↓")
print("User")

print()


# ==================================================
# 39. FUTURE TOOL CATEGORIES
# ==================================================

print("FUTURE SILVERWING TOOLS")
print()

future_tools = [
    "calculator",
    "machine analysis",
    "filesystem",
    "web",
    "browser",
    "Python",
    "shell",
    "database",
    "ML services",
    "LLM services",
    "vision",
    "speech",
    "GitHub",
    "deployment",
    "monitoring",
    "research",
    "computer control"
]


for item in future_tools:

    print(
        "-",
        item
    )

print()


# ==================================================
# 40. CURRENT SILVERWING PROGRESS
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
print("Structured Tool Calls")
print(" ↓")
print("Tool Execution")
print(" ↓")
print("Reliable Response")
print(" ↓")
print("Conversational API")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 64 COMPLETE ===")
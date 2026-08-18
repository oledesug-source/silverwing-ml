# Silverwing ML
# Phase 4 - Lesson 62
# Persistent Conversational Memory and Semantic Retrieval
#
# This lesson integrates:
# - FastAPI
# - persistent conversation storage
# - sentence embeddings
# - semantic memory retrieval
# - local LLM generation
# - conversation context


import json
import sqlite3
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed
)

from threading import Thread


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 62")
print("Persistent Conversational Memory and Semantic Retrieval")
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
        / "silverwing_conversation.db"
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

print(
    "LLM:",
    MODEL_NAME
)

print(
    "Embedding model:",
    EMBEDDING_MODEL_NAME
)

print(
    "Database:",
    DATABASE_FILE
)

print(
    "Device:",
    DEVICE
)

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


print(
    "Tokenizer loaded."
)

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


print(
    "LLM loaded."
)

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
    embedding_model
    .get_sentence_embedding_dimension()
)


print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION
)

print()


# ==================================================
# 5. DATABASE CONNECTION
# ==================================================

print("TEST 5: Database")
print()


connection = sqlite3.connect(
    DATABASE_FILE,
    check_same_thread=False
)

connection.row_factory = sqlite3.Row


cursor = connection.cursor()


# ==================================================
# 6. DATABASE SCHEMA
# ==================================================

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


connection.commit()


print(
    "Persistent memory database ready."
)

print()


# ==================================================
# 7. TIME FUNCTION
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 8. EMBEDDING FUNCTIONS
# ==================================================

def create_embedding(
        text: str
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
        data
):

    return np.frombuffer(
        data,
        dtype=np.float32
    )


# ==================================================
# 9. COSINE SIMILARITY
# ==================================================

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
# 10. MEMORY MANAGER
# ==================================================

class PersistentMemoryManager:

    # ----------------------------------------------
    # Session
    # ----------------------------------------------

    def create_session(self):

        session_id = str(
            uuid.uuid4()
        )


        timestamp = utc_now()


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
                timestamp,
                timestamp
            )
        )


        connection.commit()


        return session_id


    def session_exists(
            self,
            session_id
    ):

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


        connection.commit()


    # ----------------------------------------------
    # Messages
    # ----------------------------------------------

    def add_message(
            self,
            session_id,
            role,
            content
    ):

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


        self.touch_session(
            session_id
        )


    def get_messages(
            self,
            session_id,
            limit=12
    ):

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


        rows = list(
            reversed(
                rows
            )
        )


        return [
            dict(row)
            for row in rows
        ]


    # ----------------------------------------------
    # Long-term memory
    # ----------------------------------------------

    def store_memory(
            self,
            session_id,
            content,
            memory_type="conversation",
            importance=0.7
    ):

        vector = create_embedding(
            content
        )


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
                    vector
                ),
                utc_now()
            )
        )


        connection.commit()


    # ----------------------------------------------
    # Semantic search
    # ----------------------------------------------

    def search_memory(
            self,
            query,
            limit=5
    ):

        query_vector = create_embedding(
            query
        )


        rows = connection.execute(
            """
            SELECT *
            FROM memories
            ORDER BY id ASC
            """
        ).fetchall()


        results = []


        for row in rows:

            memory_vector = (
                deserialize_embedding(
                    row["embedding"]
                )
            )


            similarity = cosine_similarity(
                query_vector,
                memory_vector
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


    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    def statistics(self):

        tables = [
            "sessions",
            "messages",
            "memories"
        ]


        result = {}


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


memory = PersistentMemoryManager()


# ==================================================
# 11. TEST PERSISTENT STORAGE
# ==================================================

print("TEST 6: Persistent Storage")
print()


test_session_id = (
    memory.create_session()
)


memory.add_message(
    test_session_id,
    "user",
    "I am building Silverwing."
)


memory.add_message(
    test_session_id,
    "assistant",
    (
        "Silverwing is being developed "
        "as a personal AI system."
    )
)


memory.store_memory(
    test_session_id,
    (
        "The user is building Silverwing "
        "as a personal general-purpose AI."
    ),
    memory_type="project",
    importance=1.0
)


print(
    "Session:",
    test_session_id
)

print(
    "Messages:",
    len(
        memory.get_messages(
            test_session_id
        )
    )
)

print()


# ==================================================
# 12. TEST SEMANTIC MEMORY
# ==================================================

print("TEST 7: Semantic Memory Search")
print()


memory.store_memory(
    test_session_id,
    (
        "Silverwing should support "
        "multiple simultaneous tasks."
    ),
    memory_type="capability",
    importance=0.95
)


memory.store_memory(
    test_session_id,
    (
        "Silverwing should maintain "
        "long-term memory across sessions."
    ),
    memory_type="memory",
    importance=0.95
)


memory.store_memory(
    test_session_id,
    (
        "The AI should be able to use "
        "different models and services."
    ),
    memory_type="architecture",
    importance=0.9
)


search_query = (
    "How should Silverwing remember "
    "important things about the user?"
)


memory_results = memory.search_memory(
    search_query,
    limit=3
)


print(
    "Query:",
    search_query
)

print()


for result in memory_results:

    print(
        "Similarity:",
        round(
            result["similarity"],
            4
        )
    )

    print(
        "Memory:",
        result["content"]
    )

    print()


# ==================================================
# 13. CONTEXT BUILDER
# ==================================================

def build_context(
        session_id,
        user_message
):

    messages = memory.get_messages(
        session_id,
        limit=12
    )


    memories = memory.search_memory(
        user_message,
        limit=4
    )


    lines = []


    lines.append(
        "SYSTEM:"
    )

    lines.append(
        (
            "You are Silverwing, a personal "
            "AI assistant."
        )
    )


    lines.append(
        ""
    )


    lines.append(
        "RELEVANT LONG-TERM MEMORY:"
    )


    for item in memories:

        lines.append(
            (
                f"- {item['content']}"
            )
        )


    lines.append(
        ""
    )


    lines.append(
        "RECENT CONVERSATION:"
    )


    for message in messages:

        role = message[
            "role"
        ].upper()


        lines.append(
            f"{role}: "
            f"{message['content']}"
        )


    lines.append(
        ""
    )


    lines.append(
        "LATEST USER MESSAGE:"
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
# 14. CONTEXT TEST
# ==================================================

print("TEST 8: Context Builder")
print()


context = build_context(
    test_session_id,
    (
        "What do you know about the "
        "Silverwing project?"
    )
)


print(
    context
)

print()


# ==================================================
# 15. TOKEN COUNT
# ==================================================

print("TEST 9: Context Token Count")
print()


encoded_context = tokenizer(
    context,
    return_tensors="pt"
)


token_count = (
    encoded_context[
        "input_ids"
    ].shape[1]
)


print(
    "Context tokens:",
    token_count
)

print()


# ==================================================
# 16. NON-STREAMING GENERATION
# ==================================================

def generate_response(
        prompt,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        max_tokens=40
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


    input_tokens = (
        inputs[
            "input_ids"
        ].shape[1]
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
    title=(
        "Silverwing Persistent "
        "Conversational API"
    ),
    description=(
        "Persistent conversation + "
        "semantic memory + local LLM."
    ),
    version="1.0.0"
)


# ==================================================
# 18. API REQUEST MODELS
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
        default=40,
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
            62,

        "service":
            "persistent-conversational-api",

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

        "memory":
            memory.statistics()
    }


# ==================================================
# 21. CREATE SESSION
# ==================================================

@app.post("/sessions")
def create_session_endpoint():

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
# 22. GET SESSION
# ==================================================

@app.get(
    "/sessions/{session_id}"
)
def get_session_endpoint(
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
        limit=100
    )


    return {
        "session_id":
            session_id,

        "messages":
            messages
    }


# ==================================================
# 23. MEMORY SEARCH ENDPOINT
# ==================================================

@app.post(
    "/memory/search"
)
def memory_search_endpoint(
        query: str
):

    results = memory.search_memory(
        query,
        limit=10
    )


    return {
        "query":
            query,

        "results":
            results
    }


# ==================================================
# 24. CHAT
# ==================================================

@app.post("/chat")
def chat(
        request: ChatRequest
):

    session_id = request.session_id


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


    # Store user message.

    memory.add_message(
        session_id,
        "user",
        request.message
    )


    # Build context using BOTH recent
    # conversation and semantic memory.

    prompt = build_context(
        session_id,
        request.message
    )


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


    response_text = result[
        "text"
    ]


    # Store assistant response.

    memory.add_message(
        session_id,
        "assistant",
        response_text
    )


    # Store useful conversation memory.

    memory.store_memory(
        session_id,
        (
            f"User said: "
            f"{request.message}"
        ),
        memory_type="conversation",
        importance=0.6
    )


    memory.store_memory(
        session_id,
        (
            f"Assistant responded: "
            f"{response_text}"
        ),
        memory_type="conversation",
        importance=0.5
    )


    return {
        "session_id":
            session_id,

        "request_id":
            str(uuid.uuid4()),

        "message":
            response_text,

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
# 25. STREAMING CHAT
# ==================================================

@app.post(
    "/chat/stream"
)
def chat_stream(
        request: ChatRequest
):

    session_id = request.session_id


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


    memory.add_message(
        session_id,
        "user",
        request.message
    )


    prompt = build_context(
        session_id,
        request.message
    )


    def stream_generator():

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )


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


        generation_kwargs = {
            **inputs,

            "max_new_tokens":
                request.max_tokens,

            "do_sample":
                True,

            "temperature":
                request.temperature,

            "top_k":
                request.top_k,

            "top_p":
                request.top_p,

            "pad_token_id":
                tokenizer.pad_token_id,

            "streamer":
                streamer
        }


        thread = Thread(
            target=model.generate,
            kwargs=generation_kwargs
        )


        thread.start()


        collected = []


        request_id = str(
            uuid.uuid4()
        )


        for text in streamer:

            collected.append(
                text
            )


            yield (
                    json.dumps(
                        {
                            "type":
                                "token",

                            "request_id":
                                request_id,

                            "session_id":
                                session_id,

                            "text":
                                text
                        },
                        ensure_ascii=False
                    )
                    +
                    "\n"
            )


        thread.join()


        final_text = "".join(
            collected
        )


        # Persist assistant response.

        memory.add_message(
            session_id,
            "assistant",
            final_text
        )


        memory.store_memory(
            session_id,
            (
                f"User asked: "
                f"{request.message}"
            ),
            memory_type="conversation",
            importance=0.6
        )


        memory.store_memory(
            session_id,
            (
                f"Assistant answered: "
                f"{final_text}"
            ),
            memory_type="conversation",
            importance=0.5
        )


        yield (
                json.dumps(
                    {
                        "type":
                            "done",

                        "request_id":
                            request_id,

                        "session_id":
                            session_id,

                        "text":
                            final_text
                    },
                    ensure_ascii=False
                )
                +
                "\n"
        )


    return StreamingResponse(
        stream_generator(),
        media_type=(
            "application/x-ndjson"
        )
    )


# ==================================================
# 26. CONTEXT ENDPOINT
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


    recent_user_message = ""


    if messages:

        recent_user_message = (
            messages[-1]["content"]
        )


    prompt = build_context(
        session_id,
        recent_user_message
    )


    tokens = tokenizer(
        prompt,
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
            prompt
    }


# ==================================================
# 27. MEMORY STATISTICS
# ==================================================

@app.get(
    "/memory/statistics"
)
def memory_statistics():

    return memory.statistics()


# ==================================================
# 28. CAPABILITIES
# ==================================================

@app.get("/capabilities")
def capabilities():

    return {
        "service":
            "silverwing_persistent_conversational_api",

        "capabilities": [
            "persistent_sessions",
            "conversation_history",
            "semantic_memory",
            "semantic_retrieval",
            "local_llm",
            "streaming_generation",
            "context_building"
        ],

        "endpoints": [
            "/chat",
            "/chat/stream",
            "/sessions",
            "/memory/search",
            "/memory/statistics",
            "/health",
            "/model"
        ]
    }


# ==================================================
# 29. MODEL INFORMATION
# ==================================================

@app.get("/model")
def model_info():

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    return {
        "provider":
            "local",

        "model":
            MODEL_NAME,

        "architecture":
            model.config.model_type,

        "parameters":
            parameter_count,

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "embedding_dimension":
            EMBEDDING_DIMENSION,

        "device":
            str(
                DEVICE
            )
    }


# ==================================================
# 30. LOCAL API TEST
# ==================================================

print("TEST 10: API Configuration")
print()


for route in app.routes:

    if hasattr(
            route,
            "path"
    ):

        print(
            route.path
        )


print()


# ==================================================
# 31. PERSISTENCE ARCHITECTURE
# ==================================================

print("PERSISTENT CONVERSATION ARCHITECTURE")
print()

print("User")
print(" ↓")
print("FastAPI")
print(" ↓")
print("Session Manager")
print(" ↓")
print("SQLite")
print(" ↓")
print("Recent Conversation")
print("       +")
print("Semantic Memory")
print("       ↓")
print("Context Builder")
print("       ↓")
print("AI Gateway / LLM")
print("       ↓")
print("Response")
print("       ↓")
print("Persistent Memory")

print()


# ==================================================
# 32. MEMORY PIPELINE
# ==================================================

print("MEMORY PIPELINE")
print()

print("Conversation")
print("     ↓")
print("Store message")
print("     ↓")
print("Create embedding")
print("     ↓")
print("Persistent memory")
print("     ↓")
print("Future semantic search")
print("     ↓")
print("Relevant context")
print("     ↓")
print("LLM")

print()


# ==================================================
# 33. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "The semantic search in this lesson scans "
    "all stored embeddings."
)

print()

print(
    "That is appropriate for learning and small "
    "datasets but will eventually need a vector "
    "index for large-scale memory."
)

print()

print(
    "The next memory architecture can reuse "
    "the FAISS approach from Lesson 46."
)

print()


# ==================================================
# 34. CURRENT SILVERWING PROGRESS
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
print("Tools")
print(" ↓")
print("Planning")
print(" ↓")
print("Multitasking")
print(" ↓")
print("Verification")
print(" ↓")
print("Persistent Jobs")
print(" ↓")
print("Service Communication")
print(" ↓")
print("Service Discovery")
print(" ↓")
print("Model Routing")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Streaming")
print(" ↓")
print("Persistent Conversational API")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 62 COMPLETE ===")
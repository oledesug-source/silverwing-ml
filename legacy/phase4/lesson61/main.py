# Silverwing ML
# Phase 4 - Lesson 61
# Silverwing Conversational API
#
# Goal:
# Integrate conversation state, local LLM inference,
# structured requests, and streaming into one API.


import json
import time
import uuid

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed
)

from threading import Thread


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 61")
print("Silverwing Conversational API")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "sshleifer/tiny-gpt2"

SEED = 42

set_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Seed:", SEED)

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
# 3. LOAD MODEL
# ==================================================

print("TEST 3: Load LLM")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Model loaded."
)

print()


# ==================================================
# 4. APPLICATION
# ==================================================

app = FastAPI(
    title="Silverwing Conversational API",
    description=(
        "Phase 4 Lesson 61 conversational AI API."
    ),
    version="1.0.0"
)


# ==================================================
# 5. IN-MEMORY SESSION STORE
# ==================================================

@dataclass
class ConversationSession:

    session_id: str

    messages: List[Dict[str, str]] = field(
        default_factory=list
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )


sessions: Dict[
    str,
    ConversationSession
] = {}


# ==================================================
# 6. REQUEST MODELS
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


class SessionResponse(BaseModel):

    session_id: str

    message_count: int


# ==================================================
# 7. SESSION MANAGEMENT
# ==================================================

def create_session():

    session_id = str(
        uuid.uuid4()
    )


    sessions[session_id] = (
        ConversationSession(
            session_id=session_id
        )
    )


    return sessions[
        session_id
    ]


def get_session(
        session_id: Optional[str]
):

    if session_id is None:

        return create_session()


    session = sessions.get(
        session_id
    )


    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation session not found."
            )
        )


    return session


# ==================================================
# 8. SYSTEM IDENTITY
# ==================================================

SYSTEM_MESSAGE = (
    "You are Silverwing, a personal AI "
    "assistant developed as part of the "
    "Silverwing ML project."
)


# ==================================================
# 9. CONTEXT BUILDER
# ==================================================

def build_prompt(
        session: ConversationSession
):

    lines = [
        "System: "
        + SYSTEM_MESSAGE
    ]


    # Keep recent context so the prompt does
    # not grow indefinitely.

    recent_messages = (
        session.messages[-12:]
    )


    for message in recent_messages:

        role = message[
            "role"
        ]

        content = message[
            "content"
        ]


        if role == "user":

            lines.append(
                "User: "
                +
                content
            )


        elif role == "assistant":

            lines.append(
                "Assistant: "
                +
                content
            )


    lines.append(
        "Assistant:"
    )


    return "\n".join(
        lines
    )


# ==================================================
# 10. NON-STREAMING GENERATION
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
# 11. STREAMING GENERATION
# ==================================================

def stream_response(
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


    request_id = str(
        uuid.uuid4()
    )


    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )


    generation_kwargs = {
        **inputs,

        "max_new_tokens":
            max_tokens,

        "do_sample":
            True,

        "temperature":
            temperature,

        "top_k":
            top_k,

        "top_p":
            top_p,

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


    for text in streamer:

        collected.append(
            text
        )


        event = {
            "type":
                "token",

            "request_id":
                request_id,

            "text":
                text
        }


        yield (
                json.dumps(
                    event,
                    ensure_ascii=False
                )
                +
                "\n"
        )


    thread.join()


    final_text = "".join(
        collected
    )


    final_event = {
        "type":
            "done",

        "request_id":
            request_id,

        "text":
            final_text
    }


    yield (
            json.dumps(
                final_event,
                ensure_ascii=False
            )
            +
            "\n"
    )


# ==================================================
# 12. ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "project":
            "Silverwing ML",

        "phase":
            4,

        "lesson":
            61,

        "service":
            "conversational-api",

        "status":
            "running"
    }


# ==================================================
# 13. HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model":
            MODEL_NAME,

        "device":
            str(
                DEVICE
            ),

        "active_sessions":
            len(
                sessions
            )
    }


# ==================================================
# 14. MODEL ENDPOINT
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

        "vocabulary":
            len(tokenizer),

        "device":
            str(
                DEVICE
            ),

        "capabilities": [
            "text_generation",
            "conversation",
            "streaming"
        ]
    }


# ==================================================
# 15. CREATE SESSION ENDPOINT
# ==================================================

@app.post(
    "/sessions",
    response_model=SessionResponse
)
def create_chat_session():

    session = create_session()


    return SessionResponse(
        session_id=session.session_id,
        message_count=len(
            session.messages
        )
    )


# ==================================================
# 16. GET SESSION ENDPOINT
# ==================================================

@app.get(
    "/sessions/{session_id}"
)
def get_chat_session(
        session_id: str
):

    session = get_session(
        session_id
    )


    return {
        "session_id":
            session.session_id,

        "message_count":
            len(
                session.messages
            ),

        "messages":
            session.messages
    }


# ==================================================
# 17. DELETE SESSION
# ==================================================

@app.delete(
    "/sessions/{session_id}"
)
def delete_chat_session(
        session_id: str
):

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )


    del sessions[
        session_id
    ]


    return {
        "status":
            "deleted",

        "session_id":
            session_id
    }


# ==================================================
# 18. CHAT ENDPOINT
# ==================================================

@app.post("/chat")
def chat(
        request: ChatRequest
):

    session = get_session(
        request.session_id
    )


    session.messages.append(
        {
            "role":
                "user",

            "content":
                request.message
        }
    )


    session.updated_at = (
        time.time()
    )


    prompt = build_prompt(
        session
    )


    start = time.perf_counter()


    result = generate_response(
        prompt=prompt,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )


    latency_ms = (
                         time.perf_counter()
                         -
                         start
                 ) * 1000


    assistant_text = (
        result["text"]
    )


    session.messages.append(
        {
            "role":
                "assistant",

            "content":
                assistant_text
        }
    )


    session.updated_at = (
        time.time()
    )


    return {
        "session_id":
            session.session_id,

        "request_id":
            str(
                uuid.uuid4()
            ),

        "message":
            assistant_text,

        "usage": {
            "input_tokens":
                result["input_tokens"],

            "generated_tokens":
                result["generated_tokens"]
        },

        "latency_ms":
            round(
                latency_ms,
                3
            )
    }


# ==================================================
# 19. STREAMING CHAT ENDPOINT
# ==================================================

@app.post("/chat/stream")
def chat_stream(
        request: ChatRequest
):

    session = get_session(
        request.session_id
    )


    session.messages.append(
        {
            "role":
                "user",

            "content":
                request.message
        }
    )


    session.updated_at = (
        time.time()
    )


    prompt = build_prompt(
        session
    )


    def stream_generator():

        collected = []


        for event in stream_response(
                prompt=prompt,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                max_tokens=request.max_tokens
        ):

            yield event


            try:

                parsed = json.loads(
                    event
                )


                if (
                        parsed.get("type")
                        ==
                        "token"
                ):

                    collected.append(
                        parsed.get(
                            "text",
                            ""
                        )
                    )


                elif (
                        parsed.get("type")
                        ==
                        "done"
                ):

                    final_text = "".join(
                        collected
                    )


                    session.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                final_text
                        }
                    )


                    session.updated_at = (
                        time.time()
                    )


            except json.JSONDecodeError:

                continue


    return StreamingResponse(
        stream_generator(),
        media_type=(
            "application/x-ndjson"
        )
    )


# ==================================================
# 20. SESSION CONTEXT ENDPOINT
# ==================================================

@app.get(
    "/sessions/{session_id}/context"
)
def session_context(
        session_id: str
):

    session = get_session(
        session_id
    )


    prompt = build_prompt(
        session
    )


    tokenized = tokenizer(
        prompt,
        return_tensors="pt"
    )


    token_count = (
        tokenized[
            "input_ids"
        ].shape[1]
    )


    return {
        "session_id":
            session_id,

        "message_count":
            len(
                session.messages
            ),

        "context_tokens":
            token_count,

        "prompt":
            prompt
    }


# ==================================================
# 21. API SELF-DESCRIPTION
# ==================================================

@app.get("/capabilities")
def capabilities():

    return {
        "service":
            "silverwing_conversational_api",

        "capabilities": [
            "chat",
            "conversation_sessions",
            "streaming",
            "local_llm",
            "context_management"
        ],

        "endpoints": [
            "/chat",
            "/chat/stream",
            "/sessions",
            "/health",
            "/model",
            "/capabilities"
        ]
    }


# ==================================================
# 22. LOCAL TEST FUNCTIONS
# ==================================================

def run_local_tests():

    print("TEST 4: API Configuration")
    print()

    print(
        "Routes:"
    )

    for route in app.routes:

        if hasattr(
                route,
                "path"
        ):

            print(
                route.path
            )

    print()


    print("TEST 5: Model Check")
    print()


    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    print(
        "Parameters:",
        parameter_count
    )

    print(
        "Device:",
        DEVICE
    )

    print()


# ==================================================
# 23. RUN LOCAL TESTS
# ==================================================

run_local_tests()


# ==================================================
# 24. REQUEST/RESPONSE ARCHITECTURE
# ==================================================

print("SILVERWING API ARCHITECTURE")
print()

print("Client")
print(" ↓")
print("FastAPI")
print(" ↓")
print("Session Manager")
print(" ↓")
print("Context Builder")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Local LLM")
print(" ↓")
print("Response")
print(" ↓")
print("Client")

print()


# ==================================================
# 25. STREAMING ARCHITECTURE
# ==================================================

print("STREAMING API ARCHITECTURE")
print()

print("Client")
print(" ↓")
print("POST /chat/stream")
print(" ↓")
print("FastAPI")
print(" ↓")
print("LLM")
print(" ↓")
print("Token Stream")
print(" ↓")
print("NDJSON")
print(" ↓")
print("Client")

print()


# ==================================================
# 26. FUTURE PRODUCTION ARCHITECTURE
# ==================================================

print("FUTURE SILVERWING API")
print()

print("User Interface")
print("      ↓")
print("API Gateway")
print("      ↓")
print("Authentication")
print("      ↓")
print("Rate Limiting")
print("      ↓")
print("Conversation Manager")
print("      ↓")
print("Memory Manager")
print("      ↓")
print("Agent")
print("      ↓")
print("AI Gateway")
print("      ↓")
print("Model Router")
print("      ↓")
print("LLM / Tools / Services")

print()


# ==================================================
# 27. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This lesson uses in-memory conversation "
    "sessions."
)

print()

print(
    "The sessions disappear when the API process "
    "stops."
)

print()

print(
    "The next architecture step is connecting "
    "this API to the persistent memory system "
    "from Lesson 44 and semantic memory from "
    "Lessons 45-46."
)

print()


# ==================================================
# 28. CURRENT SILVERWING PROGRESS
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
print("Real Local LLM")
print(" ↓")
print("Streaming")
print(" ↓")
print("Conversational API")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 61 COMPLETE ===")
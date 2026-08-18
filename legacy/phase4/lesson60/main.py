# Silverwing ML
# Phase 4 - Lesson 60
# Streaming LLM Responses
#
# Goal:
# Generate and display local LLM output incrementally.
# This introduces the streaming architecture needed
# for responsive conversational AI.


import json
import sys
import time
import uuid

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed
)

from threading import Thread


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 60")
print("Streaming LLM Responses")
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

print(
    "Model:",
    MODEL_NAME
)

print(
    "Device:",
    DEVICE
)

print(
    "Seed:",
    SEED
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
    "Tokenizer loaded:"
)

print(
    type(tokenizer).__name__
)

print()


# ==================================================
# 3. LOAD MODEL
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
    "Model loaded."
)

print()


# ==================================================
# 4. AI REQUEST
# ==================================================

@dataclass
class AIRequest:

    prompt: str

    temperature: float = 0.8

    top_k: int = 50

    top_p: float = 0.95

    max_tokens: int = 40

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 5. AI RESPONSE
# ==================================================

@dataclass
class AIResponse:

    text: str

    request_id: str

    input_tokens: int

    generated_tokens: int

    latency_ms: float

    first_token_latency_ms: Optional[
        float
    ]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 6. LOCAL STREAMING PROVIDER
# ==================================================

class StreamingLocalProvider:
    """
    Local Hugging Face provider that exposes
    both normal generation and streaming generation.
    """

    def __init__(
            self,
            model,
            tokenizer,
            device
    ):

        self.model = model

        self.tokenizer = tokenizer

        self.device = device


    # ----------------------------------------------
    # Health
    # ----------------------------------------------

    def health(self):

        return {
            "status":
                "healthy",

            "device":
                str(
                    self.device
                ),

            "model":
                MODEL_NAME
        }


    # ----------------------------------------------
    # Standard generation
    # ----------------------------------------------

    def generate(
            self,
            request: AIRequest
    ):

        start = time.perf_counter()


        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(
                self.device
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

                max_new_tokens=(
                    request.max_tokens
                ),

                do_sample=True,

                temperature=(
                    request.temperature
                ),

                top_k=(
                    request.top_k
                ),

                top_p=(
                    request.top_p
                ),

                pad_token_id=(
                    self.tokenizer.pad_token_id
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


        full_text = (
            self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )
        )


        if full_text.startswith(
                request.prompt
        ):

            response_text = (
                full_text[
                    len(request.prompt):
                ]
            ).strip()

        else:

            response_text = full_text


        duration = (
                           time.perf_counter()
                           -
                           start
                   ) * 1000


        return AIResponse(
            text=response_text,

            request_id=str(
                uuid.uuid4()
            ),

            input_tokens=input_tokens,

            generated_tokens=generated_tokens,

            latency_ms=round(
                duration,
                3
            ),

            first_token_latency_ms=None,

            metadata={
                "mode":
                    "complete"
            }
        )


    # ----------------------------------------------
    # Streaming generation
    # ----------------------------------------------

    def stream(
            self,
            request: AIRequest
    ):
        """
        Yield generated text fragments as they
        become available.
        """

        start = time.perf_counter()

        request_id = str(
            uuid.uuid4()
        )


        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }


        input_tokens = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )


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
                self.tokenizer.pad_token_id,

            "streamer":
                streamer
        }


        # Hugging Face generation is blocking, so
        # generation runs in another thread while
        # the main thread consumes streamed text.

        generation_thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )


        generation_thread.start()


        first_token_time = None

        collected_chunks = []


        for text_chunk in streamer:

            current_time = (
                time.perf_counter()
            )


            if first_token_time is None:

                first_token_time = (
                                           current_time
                                           -
                                           start
                                   ) * 1000


            collected_chunks.append(
                text_chunk
            )


            yield {
                "type":
                    "token",

                "request_id":
                    request_id,

                "text":
                    text_chunk,

                "timestamp":
                    time.time()
            }


        generation_thread.join()


        total_duration = (
                                 time.perf_counter()
                                 -
                                 start
                         ) * 1000


        full_text = "".join(
            collected_chunks
        )


        generated_token_count = len(
            self.tokenizer(
                full_text,
                add_special_tokens=False
            )[
                "input_ids"
            ]
        )


        yield {
            "type":
                "done",

            "request_id":
                request_id,

            "text":
                full_text,

            "input_tokens":
                input_tokens,

            "generated_tokens":
                generated_token_count,

            "latency_ms":
                round(
                    total_duration,
                    3
                ),

            "first_token_latency_ms":
                round(
                    first_token_time,
                    3
                )
                if first_token_time
                   is not None
                else None
        }


provider = StreamingLocalProvider(
    model,
    tokenizer,
    DEVICE
)


# ==================================================
# 7. PROVIDER HEALTH
# ==================================================

print("TEST 4: Provider Health")
print()


print(
    json.dumps(
        provider.health(),
        indent=4
    )
)

print()


# ==================================================
# 8. STANDARD GENERATION
# ==================================================

print("TEST 5: Standard Generation")
print()


normal_request = AIRequest(
    prompt=(
        "Silverwing AI can"
    ),
    max_tokens=25
)


normal_response = provider.generate(
    normal_request
)


print(
    "Response:"
)

print(
    normal_response.text
)

print()

print(
    "Latency:",
    normal_response.latency_ms,
    "ms"
)

print()


# ==================================================
# 9. STREAMING GENERATION
# ==================================================

print("TEST 6: Streaming Generation")
print()


stream_request = AIRequest(
    prompt=(
        "Silverwing AI can"
    ),
    max_tokens=30
)


print(
    "Stream:"
)

print()

stream_events = []

stream_start = time.perf_counter()


for event in provider.stream(
        stream_request
):

    stream_events.append(
        event
    )


    if event["type"] == "token":

        print(
            event["text"],
            end="",
            flush=True
        )


    elif event["type"] == "done":

        print()
        print()

        print(
            "Streaming complete."
        )


print()


# ==================================================
# 10. STREAM SUMMARY
# ==================================================

print("TEST 7: Stream Summary")
print()


done_event = None


for event in stream_events:

    if event["type"] == "done":

        done_event = event


if done_event:

    print(
        json.dumps(
            done_event,
            indent=4
        )
    )


print()


# ==================================================
# 11. FIRST-TOKEN LATENCY
# ==================================================

print("TEST 8: Time to First Token")
print()


if done_event:

    print(
        "First-token latency:",
        done_event[
            "first_token_latency_ms"
        ],
        "ms"
    )

    print(
        "Total latency:",
        done_event[
            "latency_ms"
        ],
        "ms"
    )

print()


# ==================================================
# 12. STANDARD VS STREAMING
# ==================================================

print("TEST 9: Complete vs Streaming")
print()


comparison = {
    "complete_generation": {
        "wait_for_full_response":
            True,

        "progressive_output":
            False
    },

    "streaming_generation": {
        "wait_for_full_response":
            False,

        "progressive_output":
            True
    }
}


print(
    json.dumps(
        comparison,
        indent=4
    )
)

print()


# ==================================================
# 13. STREAM EVENT TYPES
# ==================================================

print("TEST 10: Stream Event Types")
print()


event_types = [
    "token",
    "done"
]


for event_type in event_types:

    print(
        "-",
        event_type
    )

print()


# ==================================================
# 14. MESSAGE STREAM FORMAT
# ==================================================

print("TEST 11: Streaming Message Format")
print()


sample_token_event = {
    "type":
        "token",

    "request_id":
        str(
            uuid.uuid4()
        ),

    "text":
        "Hello",

    "timestamp":
        time.time()
}


sample_done_event = {
    "type":
        "done",

    "request_id":
        sample_token_event[
            "request_id"
        ],

    "text":
        "Hello world",

    "generated_tokens":
        2
}


print(
    "Token event:"
)

print(
    json.dumps(
        sample_token_event,
        indent=4
    )
)

print()

print(
    "Done event:"
)

print(
    json.dumps(
        sample_done_event,
        indent=4
    )
)

print()


# ==================================================
# 15. STREAM BUFFER
# ==================================================

class StreamBuffer:

    def __init__(self):

        self.parts = []


    def append(
            self,
            text
    ):

        self.parts.append(
            text
        )


    def text(self):

        return "".join(
            self.parts
        )


    def clear(self):

        self.parts.clear()


buffer = StreamBuffer()


# ==================================================
# 16. BUFFER DEMONSTRATION
# ==================================================

print("TEST 12: Stream Buffer")
print()


for fragment in [
    "Silverwing ",
    "is ",
    "learning."
]:

    buffer.append(
        fragment
    )


print(
    "Buffered text:"
)

print(
    buffer.text()
)

print()


# ==================================================
# 17. STREAM CALLBACK
# ==================================================

def print_stream(
        provider,
        request
):

    print(
        "Assistant:",
        end=" ",
        flush=True
    )


    final_text = []


    for event in provider.stream(
            request
    ):

        if event["type"] == "token":

            text = event["text"]

            final_text.append(
                text
            )


            print(
                text,
                end="",
                flush=True
            )


        elif event["type"] == "done":

            print()

            return {
                "request_id":
                    event["request_id"],

                "text":
                    "".join(
                        final_text
                    ),

                "latency_ms":
                    event["latency_ms"],

                "first_token_latency_ms":
                    event[
                        "first_token_latency_ms"
                    ]
            }


# ==================================================
# 18. CALLBACK DEMONSTRATION
# ==================================================

print("TEST 13: Streaming Callback")
print()


callback_result = print_stream(
    provider,
    AIRequest(
        prompt=(
            "Machine learning is"
        ),
        max_tokens=20
    )
)


print()

print(
    json.dumps(
        callback_result,
        indent=4
    )
)

print()


# ==================================================
# 19. CONVERSATION STREAM
# ==================================================

class StreamingConversation:

    def __init__(
            self,
            provider
    ):

        self.provider = provider

        self.messages = []


    def add_user(
            self,
            message
    ):

        self.messages.append(
            {
                "role":
                    "user",

                "content":
                    message
            }
        )


    def add_assistant(
            self,
            message
    ):

        self.messages.append(
            {
                "role":
                    "assistant",

                "content":
                    message
            }
        )


    def build_prompt(self):

        lines = []


        for message in (
                self.messages
        ):

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


            else:

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


    def stream_response(
            self,
            max_tokens=30
    ):

        request = AIRequest(
            prompt=(
                self.build_prompt()
            ),
            max_tokens=max_tokens
        )


        collected = []


        for event in (
                self.provider.stream(
                    request
                )
        ):

            if event["type"] == "token":

                text = event["text"]

                collected.append(
                    text
                )


                yield event


            elif event["type"] == "done":

                final_text = "".join(
                    collected
                )


                self.add_assistant(
                    final_text
                )


                yield event


conversation = StreamingConversation(
    provider
)


# ==================================================
# 20. CONVERSATIONAL STREAMING
# ==================================================

print("TEST 14: Conversational Streaming")
print()


conversation.add_user(
    "What is machine learning?"
)


print(
    "Assistant:",
    end=" ",
    flush=True
)


conversation_text = []


for event in (
        conversation.stream_response(
            max_tokens=30
        )
):

    if event["type"] == "token":

        print(
            event["text"],
            end="",
            flush=True
        )

        conversation_text.append(
            event["text"]
        )


print()

print()


print(
    "Stored assistant response:"
)

if conversation.messages:

    print(
        conversation.messages[-1][
            "content"
        ]
    )

print()


# ==================================================
# 21. MULTI-TURN CONTEXT
# ==================================================

print("TEST 15: Multi-Turn Context")
print()


conversation.add_user(
    "And why is it useful?"
)


print(
    "Current context:"
)

print(
    conversation.build_prompt()
)

print()


# ==================================================
# 22. STREAMING ARCHITECTURE
# ==================================================

print("STREAMING ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation Manager")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Model Provider")
print(" ↓")
print("LLM")
print(" ↓")
print("Token Stream")
print(" ↓")
print("Gateway Stream")
print(" ↓")
print("User Interface")

print()


# ==================================================
# 23. REAL-TIME COMMUNICATION
# ==================================================

print("REAL-TIME COMMUNICATION")
print()

print(
    "Streaming is especially useful for "
    "interactive interfaces."
)

print()

print(
    "The UI can display generated text "
    "while generation is still happening."
)

print()

print(
    "This reduces perceived waiting time."
)

print()


# ==================================================
# 24. FUTURE STREAM TRANSPORTS
# ==================================================

print("FUTURE STREAM TRANSPORTS")
print()

transports = [
    "HTTP streaming",
    "Server-Sent Events",
    "WebSocket",
    "WebRTC",
    "terminal streams"
]


for transport in transports:

    print(
        "-",
        transport
    )

print()


# ==================================================
# 25. IMPORTANT DISTINCTION
# ==================================================

print("IMPORTANT DISTINCTION")
print()

print(
    "Streaming changes how the response is "
    "delivered; it does not make the underlying "
    "language model more intelligent."
)

print()

print(
    "The model still generates the same type "
    "of token sequence."
)

print()

print(
    "The difference is that the application "
    "receives pieces of that sequence incrementally."
)

print()


# ==================================================
# 26. FUTURE SILVERWING INTERFACE
# ==================================================

print("FUTURE SILVERWING INTERFACE")
print()

print("User types")
print("   ↓")
print("Silverwing receives request")
print("   ↓")
print("Agent reasons")
print("   ↓")
print("LLM starts generation")
print("   ↓")
print("First token")
print("   ↓")
print("First visible response")
print("   ↓")
print("More tokens")
print("   ↓")
print("Complete response")

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
print("Streaming Generation")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 60 COMPLETE ===")
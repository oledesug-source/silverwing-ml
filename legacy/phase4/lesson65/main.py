# Silverwing ML
# Phase 4 - Lesson 65
# Real Instruction-Tuned LLM Integration
#
# Goal:
# Replace the tiny pretrained GPT-2 with a configurable
# instruction-tuned language model while preserving
# Silverwing's agent/tool architecture.


import json
import os
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

MODEL_NAME = os.getenv(
    "SILVERWING_LLM_MODEL",
    DEFAULT_MODEL
)

SEED = 42

set_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 65")
print("Real Instruction-Tuned LLM Integration")
print()

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
    "Random seed:",
    SEED
)

print()


# ==================================================
# TOKENIZER
# ==================================================

print("TEST 2: Load Instruction-Tuned Tokenizer")
print()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    "Tokenizer:",
    type(tokenizer).__name__
)

print(
    "Vocabulary size:",
    len(tokenizer)
)

print()


# ==================================================
# MODEL
# ==================================================

print("TEST 3: Load Instruction-Tuned Model")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Model class:",
    type(model).__name__
)

print(
    "Model loaded successfully."
)

print()


# ==================================================
# MODEL PARAMETERS
# ==================================================

print("TEST 4: Model Parameters")
print()


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "Total parameters:",
    total_parameters
)

print(
    "Trainable parameters:",
    trainable_parameters
)

print()


# ==================================================
# CHAT MESSAGE
# ==================================================

@dataclass
class ChatMessage:

    role: str

    content: str


# ==================================================
# MODEL REQUEST
# ==================================================

@dataclass
class ModelRequest:

    messages: List[ChatMessage]

    temperature: float = 0.7

    top_p: float = 0.9

    top_k: int = 40

    max_new_tokens: int = 128


# ==================================================
# MODEL RESPONSE
# ==================================================

@dataclass
class ModelResponse:

    text: str

    model: str

    request_id: str

    input_tokens: int

    generated_tokens: int

    latency_ms: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# INSTRUCTION-TUNED PROVIDER
# ==================================================

class InstructionTunedProvider:
    """
    Generic Hugging Face instruction-following
    provider.

    The rest of Silverwing communicates with this
    interface rather than directly depending on
    Transformers internals.
    """

    def __init__(
            self,
            model,
            tokenizer,
            model_name,
            device
    ):

        self.model = model

        self.tokenizer = tokenizer

        self.model_name = model_name

        self.device = device


    @property
    def provider_name(self):

        return "huggingface_local"


    def health(self):

        return {
            "provider":
                self.provider_name,

            "status":
                "healthy",

            "model":
                self.model_name,

            "device":
                str(
                    self.device
                )
        }


    def format_messages(
            self,
            messages
    ):

        chat_messages = [
            {
                "role":
                    message.role,

                "content":
                    message.content
            }
            for message
            in messages
        ]


        # Qwen-style chat templates are used when
        # the tokenizer provides one.

        if hasattr(
                self.tokenizer,
                "apply_chat_template"
        ):

            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )


        # Generic fallback.

        parts = []


        for message in chat_messages:

            parts.append(
                f"{message['role']}: "
                f"{message['content']}"
            )


        parts.append(
            "assistant:"
        )


        return "\n".join(
            parts
        )


    def generate(
            self,
            request: ModelRequest
    ):

        request_id = str(
            uuid.uuid4()
        )


        prompt = self.format_messages(
            request.messages
        )


        start = time.perf_counter()


        inputs = self.tokenizer(
            prompt,
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

            output_ids = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=(
                        request.max_new_tokens
                    ),

                    do_sample=True,

                    temperature=(
                        request.temperature
                    ),

                    top_p=(
                        request.top_p
                    ),

                    top_k=(
                        request.top_k
                    ),

                    pad_token_id=(
                        self.tokenizer.pad_token_id
                    )
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


        # Decode only generated tokens rather
        # than decoding the entire prompt.

        generated_ids = (
            output_ids[
                0,
                input_tokens:
            ]
        )


        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()


        latency = (
                          time.perf_counter()
                          -
                          start
                  ) * 1000


        return ModelResponse(
            text=text,

            model=self.model_name,

            request_id=request_id,

            input_tokens=input_tokens,

            generated_tokens=generated_tokens,

            latency_ms=round(
                latency,
                3
            ),

            metadata={
                "provider":
                    self.provider_name,

                "device":
                    str(
                        self.device
                    )
            }
        )


provider = InstructionTunedProvider(
    model=model,
    tokenizer=tokenizer,
    model_name=MODEL_NAME,
    device=DEVICE
)


# ==================================================
# HEALTH TEST
# ==================================================

print("TEST 5: Provider Health")
print()


print(
    json.dumps(
        provider.health(),
        indent=4
    )
)

print()


# ==================================================
# BASIC GENERATION
# ==================================================

print("TEST 6: Instruction Following")
print()


basic_request = ModelRequest(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are Silverwing, a precise "
                "personal AI assistant."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "Explain machine learning "
                "in one sentence."
            )
        )
    ],

    temperature=0.3,

    top_p=0.9,

    max_new_tokens=64
)


basic_response = provider.generate(
    basic_request
)


print(
    "Response:"
)

print(
    basic_response.text
)

print()

print(
    "Latency:",
    basic_response.latency_ms,
    "ms"
)

print(
    "Generated tokens:",
    basic_response.generated_tokens
)

print()


# ==================================================
# MULTI-TURN CONVERSATION
# ==================================================

print("TEST 7: Multi-Turn Conversation")
print()


conversation_request = ModelRequest(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are Silverwing, a helpful "
                "technical assistant."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "What is Python?"
            )
        ),

        ChatMessage(
            role="assistant",
            content=(
                "Python is a general-purpose "
                "programming language."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "Why is it useful for AI?"
            )
        )
    ],

    temperature=0.4,

    max_new_tokens=96
)


conversation_response = (
    provider.generate(
        conversation_request
    )
)


print(
    conversation_response.text
)

print()


# ==================================================
# TOOL RESULT CONTEXT
# ==================================================

print("TEST 8: Tool Result + LLM")
print()


tool_result = {
    "tool":
        "calculator",

    "arguments": {
        "expression":
            "25 * 8"
    },

    "result": {
        "value":
            200
    }
}


tool_request = ModelRequest(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are Silverwing. "
                "Use verified tool results. "
                "Do not invent calculations."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "Calculate 25 * 8."
            )
        ),

        ChatMessage(
            role="tool",
            content=json.dumps(
                tool_result
            )
        )
    ],

    temperature=0.2,

    max_new_tokens=64
)


tool_response = provider.generate(
    tool_request
)


print(
    "LLM response:"
)

print(
    tool_response.text
)

print()


# ==================================================
# SYSTEM INSTRUCTIONS
# ==================================================

print("TEST 9: System Instructions")
print()


instruction_request = ModelRequest(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are Silverwing. "
                "Answer clearly. "
                "Do not invent facts. "
                "When a tool result is supplied, "
                "use it exactly."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "The calculator reported 200. "
                "What is the answer?"
            )
        )
    ],

    temperature=0.2,

    max_new_tokens=48
)


instruction_response = provider.generate(
    instruction_request
)


print(
    instruction_response.text
)

print()


# ==================================================
# PROVIDER REGISTRY
# ==================================================

class ProviderRegistry:

    def __init__(self):

        self.providers = {}


    def register(
            self,
            provider
    ):

        self.providers[
            provider.provider_name
        ] = provider


    def get(
            self,
            name
    ):

        return self.providers.get(
            name
        )


    def list_providers(self):

        return list(
            self.providers.values()
        )


registry = ProviderRegistry()


registry.register(
    provider
)


# ==================================================
# MODEL ROUTER
# ==================================================

class ModelRouter:

    def __init__(
            self,
            provider_registry
    ):

        self.provider_registry = (
            provider_registry
        )


    def select(
            self,
            task_type="text_generation"
    ):

        providers = (
            self.provider_registry
            .list_providers()
        )


        if not providers:

            return None


        # Current lesson has one real provider.
        # Later this router can rank many providers.

        return providers[0]


router = ModelRouter(
    registry
)


# ==================================================
# AI GATEWAY
# ==================================================

class AIGateway:

    def __init__(
            self,
            router
    ):

        self.router = router


    def generate(
            self,
            request
    ):

        selected_provider = (
            self.router.select()
        )


        if selected_provider is None:

            raise RuntimeError(
                "No AI provider available."
            )


        return selected_provider.generate(
            request
        )


gateway = AIGateway(
    router
)


# ==================================================
# GATEWAY TEST
# ==================================================

print("TEST 10: Unified AI Gateway")
print()


gateway_request = ModelRequest(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are Silverwing, a personal "
                "technical assistant."
            )
        ),

        ChatMessage(
            role="user",
            content=(
                "Explain what an API is."
            )
        )
    ],

    temperature=0.3,

    max_new_tokens=72
)


gateway_response = gateway.generate(
    gateway_request
)


print(
    "Provider:",
    gateway_response.metadata[
        "provider"
    ]
)

print(
    "Model:",
    gateway_response.model
)

print(
    "Response:"
)

print(
    gateway_response.text
)

print()


# ==================================================
# REAL TOOL + REAL INSTRUCTION MODEL
# ==================================================

def calculator(
        expression
):

    allowed = (
        "0123456789+-*/(). "
    )


    if any(
            char not in allowed
            for char in expression
    ):

        raise ValueError(
            "Unsupported characters."
        )


    return eval(
        expression,
        {
            "__builtins__":
                {}
        },
        {}
    )


print("TEST 11: Tool-Augmented Instruction Model")
print()


expression = "125 / 5 + 7"

calculation = calculator(
    expression
)


messages = [
    ChatMessage(
        role="system",
        content=(
            "You are Silverwing. "
            "A trusted tool supplied the result. "
            "Use the result exactly."
        )
    ),

    ChatMessage(
        role="user",
        content=(
            f"Calculate {expression}."
        )
    ),

    ChatMessage(
        role="tool",
        content=json.dumps(
            {
                "tool":
                    "calculator",

                "expression":
                    expression,

                "result":
                    calculation
            }
        )
    )
]


tool_augmented_response = (
    gateway.generate(
        ModelRequest(
            messages=messages,
            temperature=0.2,
            max_new_tokens=64
        )
    )
)


print(
    "Tool result:",
    calculation
)

print()

print(
    "Assistant:"
)

print(
    tool_augmented_response.text
)

print()


# ==================================================
# RESPONSE QUALITY TEST
# ==================================================

print("TEST 12: Response Quality Structure")
print()


response_record = {
    "request_id":
        tool_augmented_response.request_id,

    "provider":
        tool_augmented_response.metadata[
            "provider"
        ],

    "model":
        tool_augmented_response.model,

    "response":
        tool_augmented_response.text,

    "input_tokens":
        tool_augmented_response.input_tokens,

    "generated_tokens":
        tool_augmented_response.generated_tokens,

    "latency_ms":
        tool_augmented_response.latency_ms
}


print(
    json.dumps(
        response_record,
        indent=4
    )
)

print()


# ==================================================
# MODEL SWITCHING
# ==================================================

print("TEST 13: Model Configuration")
print()


print(
    "Current model:",
    MODEL_NAME
)


print(
    "Override with environment variable:"
)

print(
    "SILVERWING_LLM_MODEL=<model-name>"
)

print()


# ==================================================
# PERSONAL AI BEHAVIOR
# ==================================================

print("TEST 14: Personal AI System Prompt")
print()


personal_system_prompt = """
You are Silverwing, a personal AI system.

Core behavior:
- Be useful.
- Be precise.
- Distinguish facts from uncertainty.
- Use tools when capabilities are required.
- Never invent tool results.
- Preserve relevant context.
- Prefer verified information.
- Respect permissions.
- Explain important actions clearly.
""".strip()


print(
    personal_system_prompt
)

print()


# ==================================================
# ARCHITECTURE
# ==================================================

print("INSTRUCTION-TUNED SILVERWING ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation Manager")
print(" ↓")
print("Agent")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Model Router")
print(" ↓")
print("Instruction-Tuned LLM")
print(" ↓")
print("Reasoning / Response")
print(" ↓")
print("User")

print()

print(
    "Tool path:"
)

print("Agent")
print(" ↓")
print("Structured Tool Call")
print(" ↓")
print("Tool Executor")
print(" ↓")
print("Verified Tool Result")
print(" ↓")
print("Instruction-Tuned LLM")
print(" ↓")
print("Final Response")

print()


# ==================================================
# WHY INSTRUCTION TUNING MATTERS
# ==================================================

print("WHY INSTRUCTION-TUNING MATTERS")
print()

print(
    "A pretrained language model learns "
    "language patterns."
)

print()

print(
    "An instruction-tuned model is additionally "
    "trained to follow user and system instructions."
)

print()

print(
    "That makes it much more suitable for "
    "agentic conversational systems."
)

print()


# ==================================================
# MODEL SELECTION PRINCIPLE
# ==================================================

print("MODEL SELECTION PRINCIPLE")
print()

print(
    "Silverwing should treat the model as a "
    "replaceable capability provider."
)

print()

print(
    "The AI Gateway should hide provider-specific "
    "implementation details."
)

print()

print(
    "This permits local, cloud, and specialist "
    "models to coexist."
)

print()


# ==================================================
# CURRENT SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("LLM")
print(" ↓")
print("Instruction-Tuned LLM")
print(" ↓")
print("Conversation")
print(" ↓")
print("Persistent Memory")
print(" ↓")
print("Semantic Retrieval")
print(" ↓")
print("Agent")
print(" ↓")
print("Structured Tool Calls")
print(" ↓")
print("Tool Execution")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Instruction-Aware Response")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 65 COMPLETE ===")
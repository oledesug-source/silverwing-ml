# Silverwing ML
# Phase 4 - Lesson 43
# Prompt and Context Engineering


import json
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 43")
print("Prompt and Context Engineering")
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


BASE_DIR = Path(__file__).resolve().parent

MEMORY_FILE = (
        BASE_DIR / "conversation_memory.json"
)


print("TEST 1: Configuration")
print()

print("Model:", MODEL_NAME)
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


print(
    "Tokenizer loaded."
)

print()


# ==================================================
# 3. LOAD MODEL
# ==================================================

print("TEST 3: Load Model")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Language model loaded."
)

print()


# ==================================================
# 4. SYSTEM CONFIGURATION
# ==================================================

print("TEST 4: AI Identity")
print()


SYSTEM_CONFIG = {
    "name": "Silverwing",
    "role": (
        "Machine-learning and AI engineering "
        "assistant."
    ),
    "personality": (
        "Clear, technically precise, helpful, "
        "and educational."
    ),
    "goals": [
        "Explain technical concepts.",
        "Help analyze machine-learning problems.",
        "Maintain conversational context.",
        "State uncertainty rather than invent facts."
    ],
    "limitations": [
        "Do not claim to have performed actions "
        "that were not actually performed.",
        "Do not invent data.",
        "Do not treat predictions as guaranteed facts."
    ]
}


print(
    json.dumps(
        SYSTEM_CONFIG,
        indent=4
    )
)

print()


# ==================================================
# 5. CONVERSATION MANAGER
# ==================================================

class ConversationManager:
    """
    Stores the conversation state.
    """

    def __init__(self):

        self.messages = []


    def add_message(
            self,
            role,
            content
    ):

        self.messages.append({
            "role": role,
            "content": content
        })


    def get_messages(self):

        return list(
            self.messages
        )


    def clear(self):

        self.messages.clear()


# ==================================================
# 6. CREATE CONVERSATION
# ==================================================

conversation = ConversationManager()


conversation.add_message(
    "user",
    "I am building a machine learning system."
)


conversation.add_message(
    "assistant",
    (
        "We are building it progressively from "
        "Python and machine learning foundations."
    )
)


conversation.add_message(
    "user",
    "I also want the system to communicate naturally."
)


print("TEST 5: Conversation")
print()


for message in (
        conversation.get_messages()
):

    print(
        message["role"].upper(),
        ":",
        message["content"]
    )

print()


# ==================================================
# 7. TASK CONTEXT
# ==================================================

TASK_CONTEXT = {
    "project": "Silverwing ML",
    "current_phase": "LLM Engineering",
    "current_goal": (
        "Build a communicative AI architecture."
    ),
    "available_capabilities": [
        "Machine-learning prediction API",
        "Conversation state",
        "Persistent JSON memory",
        "Local language-model inference"
    ]
}


print("TEST 6: Task Context")
print()

print(
    json.dumps(
        TASK_CONTEXT,
        indent=4
    )
)

print()


# ==================================================
# 8. CONTEXT BUILDER
# ==================================================

class ContextBuilder:
    """
    Builds structured model context from identity,
    instructions, task information, and messages.
    """

    def __init__(
            self,
            system_config,
            task_context
    ):

        self.system_config = (
            system_config
        )

        self.task_context = (
            task_context
        )


    def build(
            self,
            messages
    ):

        sections = []


        # ------------------------------------------
        # Identity
        # ------------------------------------------

        sections.append(
            "SYSTEM IDENTITY\n"
            f"Name: "
            f"{self.system_config['name']}\n"
            f"Role: "
            f"{self.system_config['role']}\n"
            f"Personality: "
            f"{self.system_config['personality']}"
        )


        # ------------------------------------------
        # Goals
        # ------------------------------------------

        goals = "\n".join(
            f"- {goal}"
            for goal
            in self.system_config["goals"]
        )


        sections.append(
            "SYSTEM GOALS\n"
            + goals
        )


        # ------------------------------------------
        # Limitations
        # ------------------------------------------

        limitations = "\n".join(
            f"- {item}"
            for item
            in self.system_config[
                "limitations"
            ]
        )


        sections.append(
            "SYSTEM LIMITATIONS\n"
            + limitations
        )


        # ------------------------------------------
        # Task context
        # ------------------------------------------

        capabilities = "\n".join(
            f"- {item}"
            for item
            in self.task_context[
                "available_capabilities"
            ]
        )


        sections.append(
            "TASK CONTEXT\n"
            f"Project: "
            f"{self.task_context['project']}\n"
            f"Phase: "
            f"{self.task_context['current_phase']}\n"
            f"Goal: "
            f"{self.task_context['current_goal']}\n"
            f"Capabilities:\n"
            f"{capabilities}"
        )


        # ------------------------------------------
        # Conversation
        # ------------------------------------------

        conversation_lines = []


        for message in messages:

            role = message["role"]
            content = message["content"]


            conversation_lines.append(
                f"{role.upper()}: {content}"
            )


        sections.append(
            "CONVERSATION\n"
            +
            "\n".join(
                conversation_lines
            )
        )


        # ------------------------------------------
        # Response instruction
        # ------------------------------------------

        sections.append(
            "RESPONSE INSTRUCTION\n"
            "Respond to the latest user message."
        )


        return "\n\n".join(
            sections
        )


context_builder = ContextBuilder(
    SYSTEM_CONFIG,
    TASK_CONTEXT
)


# ==================================================
# 9. BUILD STRUCTURED CONTEXT
# ==================================================

print("TEST 7: Structured Context")
print()


structured_context = (
    context_builder.build(
        conversation.get_messages()
    )
)


print(
    structured_context
)

print()


# ==================================================
# 10. TOKEN COUNT
# ==================================================

print("TEST 8: Context Token Count")
print()


encoded_context = tokenizer(
    structured_context,
    return_tensors="pt"
)


context_token_count = (
    encoded_context[
        "input_ids"
    ].shape[1]
)


print(
    "Context tokens:",
    context_token_count
)

print()


# ==================================================
# 11. CONTEXT WINDOW MANAGEMENT
# ==================================================

print("TEST 9: Context Window Management")
print()


def truncate_text(
        text,
        max_tokens
):
    """
    Keep only the last max_tokens from
    a tokenized text sequence.
    """

    encoded = tokenizer(
        text,
        return_tensors="pt"
    )


    input_ids = encoded[
        "input_ids"
    ][0]


    if len(input_ids) <= max_tokens:

        return text


    truncated_ids = input_ids[
        -max_tokens:
    ]


    return tokenizer.decode(
        truncated_ids,
        skip_special_tokens=True
    )


limited_context = truncate_text(
    structured_context,
    max_tokens=200
)


print(
    "Original context length:",
    context_token_count
)

print()

print(
    "Limited context:"
)

print(
    limited_context
)

print()


# ==================================================
# 12. PROMPT FORMATTER
# ==================================================

class PromptFormatter:
    """
    Converts structured context into a
    model-ready prompt.
    """

    def format(
            self,
            context,
            user_message
    ):

        return (
                context
                + "\n\n"
                + "LATEST USER MESSAGE\n"
                + user_message
                + "\n\n"
                + "ASSISTANT:"
        )


prompt_formatter = PromptFormatter()


# ==================================================
# 13. CREATE MODEL PROMPT
# ==================================================

print("TEST 10: Model Prompt")
print()


latest_user_message = (
    "Why do we need an LLM in the system?"
)


model_prompt = prompt_formatter.format(
    limited_context,
    latest_user_message
)


print(
    model_prompt
)

print()


# ==================================================
# 14. GENERATION FUNCTION
# ==================================================

def generate_response(
        prompt,
        max_new_tokens=40,
        temperature=0.8,
        top_k=50,
        top_p=0.95
):
    """
    Generate a response from the model.
    """

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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=(
                tokenizer.pad_token_id
            )
        )


    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


    if generated_text.startswith(
            prompt
    ):

        response = (
            generated_text[
                len(prompt):
            ]
        )

    else:

        response = generated_text


    return response.strip()


# ==================================================
# 15. MODEL RESPONSE
# ==================================================

print("TEST 11: LLM Response")
print()


response = generate_response(
    model_prompt
)


print(
    "Assistant:"
)

print(
    response
)

print()


# ==================================================
# 16. STRUCTURED RESPONSE OBJECT
# ==================================================

print("TEST 12: Response Object")
print()


response_object = {
    "model": MODEL_NAME,
    "response": response,
    "context_tokens": context_token_count,
    "generation": {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95
    }
}


print(
    json.dumps(
        response_object,
        indent=4
    )
)

print()


# ==================================================
# 17. CONTEXT RELEVANCE
# ==================================================

print("TEST 13: Context Relevance")
print()


relevant_context = [
    {
        "source": "conversation",
        "content": (
            "The user is building an ML system."
        ),
        "relevance": 1.0
    },

    {
        "source": "project",
        "content": (
            "The project is Silverwing ML."
        ),
        "relevance": 0.95
    },

    {
        "source": "current_task",
        "content": (
            "The current goal is communicative AI."
        ),
        "relevance": 1.0
    }
]


for item in relevant_context:

    print(
        item["source"],
        "->",
        item["relevance"]
    )

    print(
        item["content"]
    )

    print()


# ==================================================
# 18. CONTEXT PRIORITIZATION
# ==================================================

print("TEST 14: Context Prioritization")
print()


def prioritize_context(
        context_items,
        minimum_relevance=0.5
):

    filtered = [
        item
        for item in context_items
        if item["relevance"]
           >= minimum_relevance
    ]


    filtered.sort(
        key=lambda item:
        item["relevance"],
        reverse=True
    )


    return filtered


prioritized = prioritize_context(
    relevant_context
)


for item in prioritized:

    print(
        item["relevance"],
        "|",
        item["content"]
    )

print()


# ==================================================
# 19. PROMPT INJECTION CONCEPT
# ==================================================

print("TEST 15: Instruction Boundary")
print()


print(
    "System instructions and user content "
    "should be treated as different sources."
)

print()

print(
    "A user message should not automatically "
    "replace higher-priority system constraints."
)

print()

print(
    "Production AI systems need explicit "
    "instruction hierarchy and validation."
)

print()


# ==================================================
# 20. CONTEXT SOURCES
# ==================================================

print("TEST 16: Context Sources")
print()

context_sources = [
    "system instructions",
    "conversation history",
    "long-term memory",
    "retrieved knowledge",
    "tool results",
    "current user request"
]


for source in context_sources:

    print(
        "-",
        source
    )

print()


# ==================================================
# 21. FUTURE CONTEXT BUILDER
# ==================================================

print("FUTURE CONTEXT BUILDER")
print()

print("System Identity")
print("       +")
print("Conversation State")
print("       +")
print("Long-Term Memory")
print("       +")
print("Retrieved Knowledge")
print("       +")
print("Tool Results")
print("       +")
print("Current User Request")
print("       ↓")
print("Context Builder")
print("       ↓")
print("LLM")

print()


# ==================================================
# 22. COMMUNICATIVE AI ARCHITECTURE
# ==================================================

print("COMMUNICATIVE AI ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation Manager")
print(" ↓")
print("Memory Manager")
print(" ↓")
print("Context Builder")
print(" ↓")
print("LLM")
print(" ↓")
print("Response Processor")
print(" ↓")
print("User")

print()


# ==================================================
# 23. WHY PROMPT ENGINEERING MATTERS
# ==================================================

print("WHY CONTEXT ENGINEERING MATTERS")
print()

print(
    "A language model only processes the context "
    "presented to it."
)

print()

print(
    "Better context construction can improve "
    "consistency, relevance, and controllability."
)

print()

print(
    "The model itself should not be responsible "
    "for maintaining every application-level state."
)

print()


# ==================================================
# 24. CURRENT SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("Programming")
print("      ↓")
print("Machine Learning")
print("      ↓")
print("Deep Learning")
print("      ↓")
print("Transformers")
print("      ↓")
print("LLM Inference")
print("      ↓")
print("Conversation Memory")
print("      ↓")
print("Context Engineering")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 43 COMPLETE ===")

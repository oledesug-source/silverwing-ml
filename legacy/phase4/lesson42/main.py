# Silverwing ML
# Phase 4 - Lesson 42
# Conversation State and Chat Memory

import json
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 42")
print("Conversation State and Chat Memory")
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
print("Memory file:", MEMORY_FILE)

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

print("TEST 3: Load Language Model")
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
# 4. CONVERSATION MANAGER
# ==================================================

class ConversationManager:
    """
    Stores and manages the current conversation.
    """

    def __init__(
            self,
            memory_file,
            system_message
    ):

        self.memory_file = Path(
            memory_file
        )

        self.messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]


    def add_message(
            self,
            role,
            content
    ):

        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )


    def get_messages(self):

        return list(
            self.messages
        )


    def clear(self):

        self.messages = [
            self.messages[0]
        ]


    def build_prompt(self):

        lines = []

        for message in self.messages:

            role = message["role"]
            content = message["content"]

            if role == "system":

                lines.append(
                    "System: "
                    + content
                )

            elif role == "user":

                lines.append(
                    "User: "
                    + content
                )

            elif role == "assistant":

                lines.append(
                    "Assistant: "
                    + content
                )


        lines.append(
            "Assistant:"
        )


        return "\n".join(
            lines
        )


    def save(self):

        with open(
                self.memory_file,
                "w",
                encoding="utf-8"
        ) as file:

            json.dump(
                self.messages,
                file,
                indent=4,
                ensure_ascii=False
            )


    def load(self):

        if not self.memory_file.exists():

            return False


        with open(
                self.memory_file,
                "r",
                encoding="utf-8"
        ) as file:

            self.messages = json.load(
                file
            )


        return True


# ==================================================
# 5. CREATE CONVERSATION
# ==================================================

print("TEST 4: Create Conversation")
print()


conversation = ConversationManager(
    MEMORY_FILE,
    (
        "You are Silverwing, an AI assistant "
        "for the Silverwing ML project."
    )
)


print(
    "Conversation created."
)

print()


# ==================================================
# 6. ADD USER MESSAGE
# ==================================================

print("TEST 5: Add User Message")
print()


conversation.add_message(
    "user",
    "Hello Silverwing."
)


print(
    conversation.get_messages()
)

print()


# ==================================================
# 7. BUILD CONTEXT
# ==================================================

print("TEST 6: Build Conversation Context")
print()


prompt = conversation.build_prompt()


print(
    prompt
)

print()


# ==================================================
# 8. GENERATE RESPONSE
# ==================================================

def generate_response(
        prompt,
        max_new_tokens=40,
        temperature=0.8,
        top_k=50,
        top_p=0.95
):
    """
    Generate a response using the loaded LLM.
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


    # Remove the prompt from the generated
    # response when possible.

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
# 9. FIRST RESPONSE
# ==================================================

print("TEST 7: First Response")
print()


response = generate_response(
    conversation.build_prompt()
)


conversation.add_message(
    "assistant",
    response
)


print(
    "Assistant:"
)

print(
    response
)

print()


# ==================================================
# 10. SECOND USER MESSAGE
# ==================================================

print("TEST 8: Continue Conversation")
print()


conversation.add_message(
    "user",
    "What project are you helping me build?"
)


print(
    "Current context:"
)

print(
    conversation.build_prompt()
)

print()


# ==================================================
# 11. SECOND RESPONSE
# ==================================================

response = generate_response(
    conversation.build_prompt()
)


conversation.add_message(
    "assistant",
    response
)


print(
    "Assistant:"
)

print(
    response
)

print()


# ==================================================
# 12. DISPLAY FULL CONVERSATION
# ==================================================

print("TEST 9: Conversation History")
print()


for message in conversation.get_messages():

    print(
        message["role"].upper(),
        ":",
        message["content"]
    )

    print()


# ==================================================
# 13. SAVE MEMORY
# ==================================================

print("TEST 10: Save Conversation")
print()


conversation.save()


print(
    "Conversation saved to:"
)

print(
    MEMORY_FILE
)

print()


# ==================================================
# 14. VERIFY MEMORY FILE
# ==================================================

print("TEST 11: Verify Memory")
print()


if MEMORY_FILE.exists():

    print(
        "Memory file exists."
    )

    print(
        "File size:",
        MEMORY_FILE.stat().st_size,
        "bytes"
    )

else:

    print(
        "Memory file was not created."
    )

print()


# ==================================================
# 15. CREATE NEW MANAGER
# ==================================================

print("TEST 12: Simulate Application Restart")
print()


new_conversation = ConversationManager(
    MEMORY_FILE,
    (
        "You are Silverwing, an AI assistant "
        "for the Silverwing ML project."
    )
)


loaded = new_conversation.load()


print(
    "Memory loaded:",
    loaded
)

print()


# ==================================================
# 16. VERIFY RESTORED HISTORY
# ==================================================

print("TEST 13: Restored Conversation")
print()


for message in (
        new_conversation.get_messages()
):

    print(
        message["role"].upper(),
        ":",
        message["content"]
    )

    print()


# ==================================================
# 17. ADD ANOTHER MESSAGE AFTER RESTART
# ==================================================

print("TEST 14: Continue After Restart")
print()


new_conversation.add_message(
    "user",
    "What are we learning now?"
)


restored_prompt = (
    new_conversation.build_prompt()
)


print(
    "Restored context:"
)

print(
    restored_prompt
)

print()


# ==================================================
# 18. GENERATE AFTER MEMORY RESTORE
# ==================================================

response_after_restart = (
    generate_response(
        restored_prompt
    )
)


new_conversation.add_message(
    "assistant",
    response_after_restart
)


print(
    "Assistant:"
)

print(
    response_after_restart
)

print()


# ==================================================
# 19. CONTEXT SIZE
# ==================================================

print("TEST 15: Context Size")
print()


context_text = (
    new_conversation.build_prompt()
)


context_tokens = tokenizer(
    context_text,
    return_tensors="pt"
)


token_count = (
    context_tokens[
        "input_ids"
    ].shape[1]
)


print(
    "Context token count:",
    token_count
)

print()


# ==================================================
# 20. WHY CONTEXT MANAGEMENT MATTERS
# ==================================================

print("TEST 16: Context Management")
print()

print(
    "Conversation history grows as messages "
    "are added."
)

print()

print(
    "LLMs have finite context windows, so a "
    "production system must eventually manage "
    "how much history is sent to the model."
)

print()


# ==================================================
# 21. SIMPLE CONTEXT LIMITER
# ==================================================

print("TEST 17: Context Limiting")
print()


def limit_messages(
        messages,
        max_messages
):

    if len(messages) <= max_messages:

        return list(messages)


    system_message = messages[0]

    recent_messages = messages[
        -(max_messages - 1):
    ]


    return [
        system_message,
        *recent_messages
    ]


limited_messages = limit_messages(
    new_conversation.get_messages(),
    max_messages=5
)


print(
    "Original message count:",
    len(
        new_conversation.get_messages()
    )
)

print(
    "Limited message count:",
    len(
        limited_messages
    )
)

print()


# ==================================================
# 22. SHORT-TERM VS LONG-TERM MEMORY
# ==================================================

print("TEST 18: Memory Concepts")
print()

print(
    "Short-term memory:"
)

print(
    "Current conversation context."
)

print()

print(
    "Long-term memory:"
)

print(
    "Information stored for future conversations."
)

print()

print(
    "Our JSON file is a simple demonstration "
    "of persistent long-term storage."
)

print()


# ==================================================
# 23. MEMORY ARCHITECTURE
# ==================================================

print("MEMORY ARCHITECTURE")
print()

print("User message")
print("      ↓")
print("Conversation Manager")
print("      ↓")
print("Short-term context")
print("      ↓")
print("LLM")
print("      ↓")
print("Response")
print("      ↓")
print("Persistent Memory")

print()


# ==================================================
# 24. FUTURE MEMORY SYSTEM
# ==================================================

print("FUTURE SILVERWING MEMORY")
print()

print("Conversation Memory")
print("        +")
print("Task History")
print("        +")
print("User Preferences")
print("        +")
print("Knowledge Retrieval")
print("        +")
print("Vector Search")
print("        ↓")
print("Memory Manager")
print("        ↓")
print("Context Builder")
print("        ↓")
print("LLM")

print()


# ==================================================
# 25. COMMUNICATIVE AI ARCHITECTURE
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
print("Response")
print(" ↓")
print("Memory Update")
print(" ↓")
print("User")

print()


# ==================================================
# 26. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "The tiny GPT-2 model used here is not "
    "a capable conversational model."
)

print()

print(
    "The important achievement in this lesson "
    "is the conversation architecture around "
    "the model."
)

print()


# ==================================================
# 27. SILVERWING PROGRESS
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
print("Conversation State")
print("      ↓")
print("Persistent Memory")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 42 COMPLETE ===")

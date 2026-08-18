# Silverwing ML
# Phase 4 - Lesson 41
# Loading and Using a Pretrained Language Model
#
# This lesson introduces practical LLM inference.
# It uses a tiny pretrained GPT-2-compatible model
# for learning purposes.


import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 41")
print("Pretrained Language Model Inference")
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
print("Random seed:", SEED)

print()


# ==================================================
# 2. LOAD TOKENIZER
# ==================================================

print("TEST 2: Load Tokenizer")
print()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "Tokenizer loaded successfully."
)

print()

print(
    "Tokenizer class:",
    type(tokenizer).__name__
)

print()


# ==================================================
# 3. HANDLE PAD TOKEN
# ==================================================

print("TEST 3: Tokenizer Configuration")
print()


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    "PAD token:",
    tokenizer.pad_token
)

print(
    "EOS token:",
    tokenizer.eos_token
)

print(
    "Vocabulary size:",
    len(tokenizer)
)

print()


# ==================================================
# 4. LOAD PRETRAINED MODEL
# ==================================================

print("TEST 4: Load Language Model")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Model loaded successfully."
)

print()

print(
    "Model class:",
    type(model).__name__
)

print()


# ==================================================
# 5. COUNT PARAMETERS
# ==================================================

print("TEST 5: Model Parameters")
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
# 6. TOKENIZE A PROMPT
# ==================================================

print("TEST 6: Tokenization")
print()


prompt = (
    "The machine temperature is"
)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}


print(
    "Prompt:",
    prompt
)

print()

print(
    "Input IDs:",
    inputs["input_ids"]
)

print()

print(
    "Attention mask:",
    inputs["attention_mask"]
)

print()


# ==================================================
# 7. DECODE INPUT IDS
# ==================================================

print("TEST 7: Decode Input")
print()


decoded_input = tokenizer.decode(
    inputs["input_ids"][0],
    skip_special_tokens=True
)


print(
    "Decoded text:",
    decoded_input
)

print()


# ==================================================
# 8. FORWARD PASS
# ==================================================

print("TEST 8: Forward Pass")
print()


with torch.no_grad():

    outputs = model(
        **inputs
    )


print(
    "Output object:",
    type(outputs).__name__
)

print()

print(
    "Logits shape:",
    outputs.logits.shape
)

print()


# ==================================================
# 9. NEXT-TOKEN LOGITS
# ==================================================

print("TEST 9: Next-Token Logits")
print()


next_token_logits = (
    outputs.logits[:, -1, :]
)


print(
    "Next-token logits shape:",
    next_token_logits.shape
)

print()


# ==================================================
# 10. CONVERT LOGITS TO PROBABILITIES
# ==================================================

print("TEST 10: Next-Token Probabilities")
print()


probabilities = torch.softmax(
    next_token_logits,
    dim=-1
)


print(
    "Probability shape:",
    probabilities.shape
)

print()

print(
    "Probability sum:",
    probabilities.sum().item()
)

print()


# ==================================================
# 11. GREEDY NEXT TOKEN
# ==================================================

print("TEST 11: Greedy Next Token")
print()


next_token_id = torch.argmax(
    probabilities,
    dim=-1
)


next_token = tokenizer.decode(
    next_token_id,
    skip_special_tokens=True
)


print(
    "Next token ID:",
    next_token_id.item()
)

print(
    "Next token:",
    repr(next_token)
)

print()


# ==================================================
# 12. TOP-K TOKEN ANALYSIS
# ==================================================

print("TEST 12: Top-K Tokens")
print()


top_k = 10


top_probabilities, top_ids = torch.topk(
    probabilities,
    k=top_k,
    dim=-1
)


for token_id, probability in zip(
        top_ids[0],
        top_probabilities[0]
):

    token_text = tokenizer.decode(
        token_id.unsqueeze(0),
        skip_special_tokens=True
    )


    print(
        repr(token_text),
        "->",
        round(
            probability.item(),
            6
        )
    )


print()


# ==================================================
# 13. GREEDY GENERATION
# ==================================================

print("TEST 13: Greedy Generation")
print()


greedy_output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id
)


greedy_text = tokenizer.decode(
    greedy_output[0],
    skip_special_tokens=True
)


print(
    "Generated text:"
)

print(
    greedy_text
)

print()


# ==================================================
# 14. SAMPLING GENERATION
# ==================================================

print("TEST 14: Sampling Generation")
print()


set_seed(SEED)


sampled_output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)


sampled_text = tokenizer.decode(
    sampled_output[0],
    skip_special_tokens=True
)


print(
    "Sampled text:"
)

print(
    sampled_text
)

print()


# ==================================================
# 15. HIGHER TEMPERATURE
# ==================================================

print("TEST 15: Higher Temperature")
print()


set_seed(SEED)


creative_output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=1.2,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)


creative_text = tokenizer.decode(
    creative_output[0],
    skip_special_tokens=True
)


print(
    "Higher-temperature output:"
)

print(
    creative_text
)

print()


# ==================================================
# 16. MULTIPLE PROMPTS
# ==================================================

print("TEST 16: Multiple Prompts")
print()


prompts = [
    "The machine temperature is",
    "The pump is",
    "The engine is",
    "Machine maintenance requires",
    "Artificial intelligence can"
]


for prompt_text in prompts:

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )


    prompt_inputs = {
        key: value.to(DEVICE)
        for key, value
        in prompt_inputs.items()
    }


    with torch.no_grad():

        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
            )
        )


    text = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


    print(
        "Prompt:",
        prompt_text
    )

    print(
        "Output:",
        text
    )

    print()


# ==================================================
# 17. INPUT TOKEN COUNT
# ==================================================

print("TEST 17: Token Counts")
print()


input_token_count = (
    inputs["input_ids"].shape[1]
)


generated_token_count = (
        greedy_output.shape[1]
        -
        input_token_count
)


print(
    "Input tokens:",
    input_token_count
)

print(
    "Generated tokens:",
    generated_token_count
)

print()


# ==================================================
# 18. MODEL CONFIGURATION
# ==================================================

print("TEST 18: Model Configuration")
print()


print(
    "Model architecture:"
)

print(
    model.config.model_type
)

print()

print(
    "Hidden size:",
    getattr(
        model.config,
        "n_embd",
        "unknown"
    )
)

print()

print(
    "Number of layers:",
    getattr(
        model.config,
        "n_layer",
        "unknown"
    )
)

print()

print(
    "Attention heads:",
    getattr(
        model.config,
        "n_head",
        "unknown"
    )
)

print()


# ==================================================
# 19. MODEL INFERENCE FUNCTION
# ==================================================

print("TEST 19: Reusable LLM Function")
print()


def generate_text(
        prompt_text,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.95
):
    """
    Generate text with the loaded language model.
    """

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )


    prompt_inputs = {
        key: value.to(DEVICE)
        for key, value
        in prompt_inputs.items()
    }


    with torch.no_grad():

        output_ids = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=(
                tokenizer.pad_token_id
            )
        )


    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


result = generate_text(
    "Silverwing AI can"
)


print(
    result
)

print()


# ==================================================
# 20. MODEL ABSTRACTION CLASS
# ==================================================

print("TEST 20: LLM Service Abstraction")
print()


class LocalLLM:
    """
    Simple abstraction around a local
    Hugging Face causal language model.
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


    def generate(
            self,
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
            top_p=0.95
    ):

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


        with torch.no_grad():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                )
            )


        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )


llm = LocalLLM(
    model,
    tokenizer,
    DEVICE
)


print(
    llm.generate(
        "Silverwing ML is"
    )
)

print()


# ==================================================
# 21. COMMUNICATIVE AI INTERFACE
# ==================================================

print("TEST 21: Communicative Interface Concept")
print()


def respond_to_user(
        user_message
):
    """
    Basic language-model response function.

    This is not yet a full chatbot because
    it has no memory, tools, or conversation
    management.
    """

    prompt = (
            "User: "
            + user_message
            + "\nAssistant:"
    )


    response = llm.generate(
        prompt,
        max_new_tokens=40,
        temperature=0.7
    )


    return response


user_message = (
    "Tell me something about machine learning."
)


response = respond_to_user(
    user_message
)


print(
    "User:"
)

print(
    user_message
)

print()

print(
    "Model:"
)

print(
    response
)

print()


# ==================================================
# 22. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This tiny pretrained model is being used "
    "to demonstrate inference mechanics."
)

print()

print(
    "It is not a modern conversational assistant "
    "and should not be expected to reason reliably "
    "or follow instructions like a contemporary "
    "instruction-tuned chat model."
)

print()


# ==================================================
# 23. WHAT AN INSTRUCTION-TUNED MODEL ADDS
# ==================================================

print("INSTRUCTION-TUNED MODEL CONCEPT")
print()

print(
    "A chat-oriented system generally needs "
    "a model and prompting/training setup "
    "designed for instruction following."
)

print()

print(
    "A production communicative AI also needs "
    "conversation management, memory, safety, "
    "tool use, retrieval, and evaluation."
)

print()


# ==================================================
# 24. LLM ENGINEERING PIPELINE
# ==================================================

print("LLM ENGINEERING PIPELINE")
print()

print("Pretrained Model")
print("      ↓")
print("Tokenizer")
print("      ↓")
print("Prompt")
print("      ↓")
print("Model Inference")
print("      ↓")
print("Logits")
print("      ↓")
print("Decoding")
print("      ↓")
print("Generated Text")

print()


# ==================================================
# 25. FUTURE SILVERWING ARCHITECTURE
# ==================================================

print("FUTURE SILVERWING ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation Manager")
print(" ↓")
print("Prompt / Context Builder")
print(" ↓")
print("LLM")
print(" ↓")
print("Tool / Memory / ML Services")
print(" ↓")
print("Response Generator")
print(" ↓")
print("User")

print()


# ==================================================
# 26. CURRENT PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("Phase 1: Programming")
print("Phase 2: Machine Learning")
print("Phase 3: Deep Learning + Transformers")
print("Phase 4: LLM Engineering")
print()

print(
    "Lesson 41: Pretrained LLM inference"
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 41 COMPLETE ===")

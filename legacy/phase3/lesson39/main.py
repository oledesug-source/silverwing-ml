# Silverwing ML
# Phase 3 - Lesson 39
# Language Generation and Sampling
#
# Goal:
# Understand how language models convert token
# probabilities into generated text.


import random

import torch


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 39")
print("Language Generation and Sampling")
print()


# ==================================================
# 1. RANDOM SEEDS
# ==================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


print("TEST 1: Reproducibility")
print()

print("Seed:", SEED)

print()


# ==================================================
# 2. SMALL VOCABULARY
# ==================================================

vocabulary = [
    "machine",
    "temperature",
    "pressure",
    "is",
    "normal",
    "high",
    "warning",
    "critical",
    "detected",
    "stable",
    "inspection"
]


token_count = len(vocabulary)


print("TEST 2: Vocabulary")
print()

for index, token in enumerate(vocabulary):

    print(
        index,
        "->",
        token
    )

print()


# ==================================================
# 3. CREATE EXAMPLE LOGITS
# ==================================================

print("TEST 3: Raw Model Logits")
print()


logits = torch.tensor([
    3.5,
    2.2,
    1.1,
    2.8,
    0.5,
    3.2,
    2.4,
    1.0,
    1.8,
    0.7,
    0.3
])


print(
    "Logits:"
)

print(
    logits
)

print()


# ==================================================
# 4. CONVERT LOGITS TO PROBABILITIES
# ==================================================

print("TEST 4: Softmax Probabilities")
print()


probabilities = torch.softmax(
    logits,
    dim=-1
)


for token, probability in zip(
        vocabulary,
        probabilities
):

    print(
        token,
        "->",
        round(
            probability.item(),
            4
        )
    )


print()


print(
    "Probability sum:",
    probabilities.sum().item()
)

print()


# ==================================================
# 5. GREEDY DECODING
# ==================================================

print("TEST 5: Greedy Decoding")
print()


greedy_index = torch.argmax(
    probabilities
).item()


greedy_token = vocabulary[
    greedy_index
]


print(
    "Selected token:",
    greedy_token
)

print(
    "Probability:",
    probabilities[
        greedy_index
    ].item()
)

print()


# ==================================================
# 6. TEMPERATURE
# ==================================================

print("TEST 6: Temperature")
print()


def temperature_distribution(
        logits,
        temperature
):

    if temperature <= 0:

        raise ValueError(
            "Temperature must be greater than zero."
        )


    scaled_logits = (
            logits
            /
            temperature
    )


    return torch.softmax(
        scaled_logits,
        dim=-1
    )


temperatures = [
    0.3,
    0.7,
    1.0,
    1.5,
    2.0
]


for temperature in temperatures:

    distribution = (
        temperature_distribution(
            logits,
            temperature
        )
    )


    print(
        "Temperature:",
        temperature
    )


    for token, probability in zip(
            vocabulary,
            distribution
    ):

        if probability.item() >= 0.05:

            print(
                " ",
                token,
                "->",
                round(
                    probability.item(),
                    4
                )
            )


    print()


# ==================================================
# 7. TOP-K FILTERING
# ==================================================

print("TEST 7: Top-K Filtering")
print()


def top_k_filter(
        logits,
        k
):

    if k <= 0:

        raise ValueError(
            "k must be greater than zero."
        )


    k = min(
        k,
        logits.numel()
    )


    values, indices = torch.topk(
        logits,
        k=k
    )


    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )


    filtered_logits[
        indices
    ] = values


    return filtered_logits


for k in [
    1,
    3,
    5
]:

    filtered = top_k_filter(
        logits,
        k
    )


    probabilities_k = torch.softmax(
        filtered,
        dim=-1
    )


    print(
        "Top-K:",
        k
    )


    for token, probability in zip(
            vocabulary,
            probabilities_k
    ):

        if probability.item() > 0:

            print(
                " ",
                token,
                "->",
                round(
                    probability.item(),
                    4
                )
            )


    print()


# ==================================================
# 8. TOP-P / NUCLEUS FILTERING
# ==================================================

print("TEST 8: Top-P Filtering")
print()


def top_p_filter(
        logits,
        p
):

    if not 0 < p <= 1:

        raise ValueError(
            "p must be between 0 and 1."
        )


    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )


    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    )


    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1
    )


    # Mark tokens that would push the cumulative
    # probability beyond the selected threshold.

    remove_mask = (
            cumulative_probabilities > p
    )


    # Clone before shifting the mask.
    # This avoids overlapping-memory assignment
    # errors in PyTorch.

    shifted_mask = remove_mask.clone()

    shifted_mask[1:] = remove_mask[:-1]

    shifted_mask[0] = False

    remove_mask = shifted_mask


    filtered_sorted_logits = sorted_logits.clone()

    filtered_sorted_logits[
        remove_mask
    ] = float("-inf")


    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )


    filtered_logits[
        sorted_indices
    ] = filtered_sorted_logits


    return filtered_logits


for p in [
    0.50,
    0.75,
    0.90,
    0.95
]:

    filtered = top_p_filter(
        logits,
        p
    )


    probabilities_p = torch.softmax(
        filtered,
        dim=-1
    )


    print(
        "Top-P:",
        p
    )


    for token, probability in zip(
            vocabulary,
            probabilities_p
    ):

        if probability.item() > 0:

            print(
                " ",
                token,
                "->",
                round(
                    probability.item(),
                    4
                )
            )


    print()


# ==================================================
# 9. RANDOM SAMPLING
# ==================================================

print("TEST 9: Random Sampling")
print()


def sample_token(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=None
):

    if temperature <= 0:

        raise ValueError(
            "Temperature must be greater than zero."
        )


    adjusted_logits = (
            logits
            /
            temperature
    )


    if top_k is not None:

        adjusted_logits = (
            top_k_filter(
                adjusted_logits,
                top_k
            )
        )


    if top_p is not None:

        adjusted_logits = (
            top_p_filter(
                adjusted_logits,
                top_p
            )
        )


    probabilities = torch.softmax(
        adjusted_logits,
        dim=-1
    )


    token_index = torch.multinomial(
        probabilities,
        num_samples=1
    ).item()


    return (
        token_index,
        probabilities
    )


token_index, sampling_probabilities = (
    sample_token(
        logits,
        temperature=1.0
    )
)


print(
    "Sampled token:",
    vocabulary[token_index]
)

print(
    "Probability:",
    sampling_probabilities[
        token_index
    ].item()
)

print()


# ==================================================
# 10. COMPARE SAMPLING SETTINGS
# ==================================================

print("TEST 10: Sampling Settings")
print()


settings = [
    {
        "name": "Greedy-like",
        "temperature": 0.3,
        "top_k": 1,
        "top_p": None
    },

    {
        "name": "Focused",
        "temperature": 0.7,
        "top_k": 5,
        "top_p": None
    },

    {
        "name": "Balanced",
        "temperature": 1.0,
        "top_k": 5,
        "top_p": 0.9
    },

    {
        "name": "Creative",
        "temperature": 1.5,
        "top_k": None,
        "top_p": 0.95
    }
]


for setting in settings:

    index, probabilities_setting = (
        sample_token(
            logits,
            temperature=setting[
                "temperature"
            ],
            top_k=setting[
                "top_k"
            ],
            top_p=setting[
                "top_p"
            ]
        )
    )


    print(
        setting["name"],
        "->",
        vocabulary[index]
    )

    print(
        " temperature:",
        setting["temperature"]
    )

    print(
        " top_k:",
        setting["top_k"]
    )

    print(
        " top_p:",
        setting["top_p"]
    )

    print()


# ==================================================
# 11. MULTIPLE SAMPLES
# ==================================================

print("TEST 11: Generate Multiple Samples")
print()


for sample_number in range(
        1,
        11
):

    index, _ = sample_token(
        logits,
        temperature=1.0,
        top_p=0.9
    )


    print(
        "Sample",
        sample_number,
        ":",
        vocabulary[index]
    )


print()


# ==================================================
# 12. TOKEN GENERATION
# ==================================================

print("TEST 12: Token-by-Token Generation")
print()


def generate_tokens(
        logits,
        number_of_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None
):

    generated = []


    for _ in range(
            number_of_tokens
    ):

        index, _ = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )


        generated.append(
            vocabulary[index]
        )


    return generated


generated_tokens = generate_tokens(
    logits,
    number_of_tokens=8,
    temperature=1.0,
    top_k=5,
    top_p=0.9
)


print(
    "Generated tokens:"
)

print(
    generated_tokens
)

print()


# ==================================================
# 13. TEMPERATURE EFFECT
# ==================================================

print("TEST 13: Temperature Comparison")
print()


for temperature in [
    0.3,
    0.7,
    1.0,
    1.5
]:

    generated = generate_tokens(
        logits,
        number_of_tokens=8,
        temperature=temperature,
        top_p=0.9
    )


    print(
        "Temperature:",
        temperature
    )

    print(
        " ".join(generated)
    )

    print()


# ==================================================
# 14. TOP-K EFFECT
# ==================================================

print("TEST 14: Top-K Comparison")
print()


for k in [
    1,
    3,
    5,
    8
]:

    generated = generate_tokens(
        logits,
        number_of_tokens=8,
        temperature=1.0,
        top_k=k
    )


    print(
        "Top-K:",
        k
    )

    print(
        " ".join(generated)
    )

    print()


# ==================================================
# 15. TOP-P EFFECT
# ==================================================

print("TEST 15: Top-P Comparison")
print()


for p in [
    0.50,
    0.75,
    0.90,
    0.95
]:

    generated = generate_tokens(
        logits,
        number_of_tokens=8,
        temperature=1.0,
        top_p=p
    )


    print(
        "Top-P:",
        p
    )

    print(
        " ".join(generated)
    )

    print()


# ==================================================
# 16. ENTROPY
# ==================================================

print("TEST 16: Probability Entropy")
print()


def distribution_entropy(
        probabilities
):

    safe_probabilities = (
        probabilities[
            probabilities > 0
            ]
    )


    return -torch.sum(
        safe_probabilities
        *
        torch.log(
            safe_probabilities
        )
    )


for temperature in [
    0.3,
    0.7,
    1.0,
    1.5
]:

    distribution = (
        temperature_distribution(
            logits,
            temperature
        )
    )


    entropy = distribution_entropy(
        distribution
    )


    print(
        "Temperature:",
        temperature,
        "| Entropy:",
        round(
            entropy.item(),
            4
        )
    )


print()


# ==================================================
# 17. GREEDY VS SAMPLING
# ==================================================

print("TEST 17: Greedy vs Sampling")
print()


greedy_token = vocabulary[
    torch.argmax(
        probabilities
    ).item()
]


sampled_tokens = generate_tokens(
    logits,
    number_of_tokens=10,
    temperature=1.0,
    top_p=0.9
)


print(
    "Greedy token:"
)

print(
    greedy_token
)

print()

print(
    "Sampled sequence:"
)

print(
    " ".join(
        sampled_tokens
    )
)

print()


# ==================================================
# 18. LANGUAGE MODEL INTERPRETATION
# ==================================================

print("TEST 18: Language Model Interpretation")
print()

print(
    "A model produces logits."
)

print()

print(
    "Softmax converts logits into probabilities."
)

print()

print(
    "A decoding strategy chooses the next token."
)

print()

print(
    "The chosen token is appended to the sequence."
)

print()

print(
    "The model predicts another token."
)

print()

print(
    "The process repeats until a stopping condition."
)

print()


# ==================================================
# 19. WHY TEMPERATURE MATTERS
# ==================================================

print("WHY TEMPERATURE MATTERS")
print()

print(
    "Lower temperature makes the distribution "
    "more concentrated around high-probability tokens."
)

print()

print(
    "Higher temperature makes the distribution "
    "flatter and increases the chance of lower-"
    "probability tokens being selected."
)

print()


# ==================================================
# 20. WHY TOP-K MATTERS
# ==================================================

print("WHY TOP-K MATTERS")
print()

print(
    "Top-K limits sampling to the K highest-scoring "
    "candidate tokens."
)

print()


# ==================================================
# 21. WHY TOP-P MATTERS
# ==================================================

print("WHY TOP-P MATTERS")
print()

print(
    "Top-P keeps the smallest candidate set whose "
    "cumulative probability reaches the chosen threshold."
)

print()


# ==================================================
# 22. COMMUNICATIVE AI CONNECTION
# ==================================================

print("COMMUNICATIVE AI CONNECTION")
print()

print(
    "Generation settings affect how your future "
    "communicative AI sounds."
)

print()

print(
    "Lower temperature can favor consistency."
)

print(
    "Higher temperature can increase variation."
)

print(
    "Top-K and Top-P control which candidates "
    "remain available during sampling."
)

print()


# ==================================================
# 23. LLM GENERATION PIPELINE
# ==================================================

print("LLM GENERATION PIPELINE")
print()

print("Prompt")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Transformer")
print(" ↓")
print("Logits")
print(" ↓")
print("Temperature / Sampling")
print(" ↓")
print("Next Token")
print(" ↓")
print("Append Token")
print(" ↓")
print("Repeat")
print(" ↓")
print("Generated Response")

print()


# ==================================================
# 24. SILVERWING AI PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Transformer")
print("      ↓")
print("Language Model")
print("      ↓")
print("Logits")
print("      ↓")
print("Probability Distribution")
print("      ↓")
print("Decoding")
print("      ↓")
print("Generated Text")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 39 COMPLETE ===")

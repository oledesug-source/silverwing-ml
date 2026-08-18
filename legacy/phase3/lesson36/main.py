# Silverwing ML
# Phase 3 - Lesson 36
# Attention Mechanisms
#
# Goal:
# Understand the basic mathematical idea behind
# attention before implementing Transformers.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 36")
print("Attention Mechanisms")
print()


# ==================================================
# 1. PYTORCH INFORMATION
# ==================================================

print("TEST 1: PyTorch Information")
print()

print(
    "PyTorch version:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print()


# ==================================================
# 2. CREATE SEQUENCE REPRESENTATIONS
# ==================================================

print("TEST 2: Sequence Representations")
print()

# Imagine these are vector representations
# of four tokens.

sequence = torch.tensor([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0]
])


print("Sequence:")
print(sequence)

print()

print(
    "Sequence shape:",
    sequence.shape
)

print()


# ==================================================
# 3. CREATE QUERY, KEY AND VALUE
# ==================================================

print("TEST 3: Query, Key and Value")
print()


query = torch.tensor([
    [1.0, 0.0, 1.0, 0.0]
])


key = sequence

value = sequence


print("Query:")
print(query)

print()

print("Keys:")
print(key)

print()

print("Values:")
print(value)

print()


# ==================================================
# 4. CALCULATE ATTENTION SCORES
# ==================================================

print("TEST 4: Attention Scores")
print()


scores = torch.matmul(
    query,
    key.T
)


print(
    "Raw attention scores:"
)

print(
    scores
)

print()


# ==================================================
# 5. SCALE THE SCORES
# ==================================================

print("TEST 5: Scaled Attention Scores")
print()


dimension = key.shape[-1]


scaled_scores = (
        scores
        /
        math.sqrt(dimension)
)


print(
    "Scaled scores:"
)

print(
    scaled_scores
)

print()


# ==================================================
# 6. SOFTMAX ATTENTION WEIGHTS
# ==================================================

print("TEST 6: Attention Weights")
print()


attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)


print(
    "Attention weights:"
)

print(
    attention_weights
)

print()


print(
    "Sum of attention weights:",
    attention_weights.sum().item()
)

print()


# ==================================================
# 7. CALCULATE CONTEXT VECTOR
# ==================================================

print("TEST 7: Context Vector")
print()


context = torch.matmul(
    attention_weights,
    value
)


print(
    "Context vector:"
)

print(
    context
)

print()

print(
    "Context shape:",
    context.shape
)

print()


# ==================================================
# 8. SHOW WHICH TOKENS RECEIVED ATTENTION
# ==================================================

print("TEST 8: Attention Distribution")
print()


token_names = [
    "machine",
    "temperature",
    "warning",
    "detected"
]


for token, weight in zip(
        token_names,
        attention_weights[0]
):

    print(
        token,
        "->",
        round(
            weight.item(),
            4
        )
    )

print()


# ==================================================
# 9. SELF-ATTENTION
# ==================================================

print("TEST 9: Self-Attention")
print()


# In self-attention, the same sequence is used
# to construct queries, keys and values.

self_query = sequence
self_key = sequence
self_value = sequence


self_scores = torch.matmul(
    self_query,
    self_key.T
)


self_scaled_scores = (
        self_scores
        /
        math.sqrt(
            sequence.shape[-1]
        )
)


self_weights = F.softmax(
    self_scaled_scores,
    dim=-1
)


self_context = torch.matmul(
    self_weights,
    self_value
)


print(
    "Self-attention scores:"
)

print(
    self_scores
)

print()

print(
    "Self-attention weights:"
)

print(
    self_weights
)

print()

print(
    "Self-attention context:"
)

print(
    self_context
)

print()


# ==================================================
# 10. BUILD ATTENTION MODULE
# ==================================================

print("TEST 10: Build Attention Module")
print()


class SelfAttention(nn.Module):
    """
    Simple scaled dot-product self-attention.
    """

    def __init__(
            self,
            embedding_dimension
    ):

        super().__init__()

        self.query_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.key_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.value_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )


    def forward(self, x):

        query = self.query_projection(x)

        key = self.key_projection(x)

        value = self.value_projection(x)


        scores = torch.matmul(
            query,
            key.transpose(
                -2,
                -1
            )
        )


        scale = math.sqrt(
            query.shape[-1]
        )


        scores = scores / scale


        weights = F.softmax(
            scores,
            dim=-1
        )


        output = torch.matmul(
            weights,
            value
        )


        return output, weights


attention_model = SelfAttention(
    embedding_dimension=4
)


print(
    attention_model
)

print()


# ==================================================
# 11. RUN ATTENTION MODULE
# ==================================================

print("TEST 11: Attention Module Output")
print()


attention_output, weights = (
    attention_model(
        sequence
    )
)


print(
    "Attention output:"
)

print(
    attention_output
)

print()

print(
    "Attention weight shape:",
    weights.shape
)

print()


# ==================================================
# 12. MULTI-HEAD ATTENTION IDEA
# ==================================================

print("TEST 12: Multi-Head Attention Concept")
print()


print(
    "A single attention mechanism learns "
    "one type of relationship."
)

print()

print(
    "Multi-head attention uses several attention "
    "heads working in parallel."
)

print()

print(
    "Different heads can learn different "
    "relationships in the sequence."
)

print()


# ==================================================
# 13. USE PYTORCH MULTIHEAD ATTENTION
# ==================================================

print("TEST 13: PyTorch Multi-Head Attention")
print()


multi_head_attention = nn.MultiheadAttention(
    embed_dim=8,
    num_heads=2,
    batch_first=True
)


multi_head_input = torch.tensor([
    [
        [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0
        ],

        [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0
        ],

        [
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0
        ]
    ]
])


multi_output, multi_weights = (
    multi_head_attention(
        multi_head_input,
        multi_head_input,
        multi_head_input
    )
)


print(
    "Input shape:",
    multi_head_input.shape
)

print()

print(
    "Output shape:",
    multi_output.shape
)

print()

print(
    "Attention weights shape:",
    multi_weights.shape
)

print()


# ==================================================
# 14. CAUSAL ATTENTION
# ==================================================

print("TEST 14: Causal Attention")
print()


print(
    "Autoregressive language models must not "
    "look into future tokens when predicting "
    "the current token."
)

print()


sequence_length = 4


causal_mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length
    ),
    diagonal=1
).bool()


print(
    "Causal mask:"
)

print(
    causal_mask
)

print()


# ==================================================
# 15. APPLY CAUSAL MASK
# ==================================================

print("TEST 15: Masked Attention")
print()


example_scores = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]
])


masked_scores = example_scores.masked_fill(
    causal_mask,
    float("-inf")
)


masked_weights = F.softmax(
    masked_scores,
    dim=-1
)


print(
    "Original scores:"
)

print(
    example_scores
)

print()

print(
    "Masked scores:"
)

print(
    masked_scores
)

print()

print(
    "Masked attention weights:"
)

print(
    masked_weights
)

print()


# ==================================================
# 16. WHY CAUSAL MASKING MATTERS
# ==================================================

print("TEST 16: Causal Language Modeling")
print()

print(
    "For:"
)

print(
    "The machine temperature is high"
)

print()

print(
    "When predicting 'temperature', "
    "the model cannot use future tokens "
    "such as 'is' or 'high'."
)

print()

print(
    "This prevents future information from "
    "leaking into the prediction."
)

print()


# ==================================================
# 17. ATTENTION PIPELINE
# ==================================================

print("ATTENTION PIPELINE")
print()

print("Token representations")
print("        ↓")
print("Queries / Keys / Values")
print("        ↓")
print("Query-Key scores")
print("        ↓")
print("Scaling")
print("        ↓")
print("Softmax")
print("        ↓")
print("Attention weights")
print("        ↓")
print("Weighted values")
print("        ↓")
print("Context representation")

print()


# ==================================================
# 18. TRANSFORMER CONNECTION
# ==================================================

print("TRANSFORMER CONNECTION")
print()

print(
    "Transformers use attention as a central "
    "mechanism for processing token sequences."
)

print()

print(
    "Modern Transformer blocks combine attention "
    "with feed-forward neural networks, residual "
    "connections, normalization, and other components."
)

print()


# ==================================================
# 19. LLM CONNECTION
# ==================================================

print("LLM CONNECTION")
print()

print(
    "Large language models based on Transformers "
    "use attention to relate token representations."
)

print()

print(
    "This helps the model process relationships "
    "between tokens that may be far apart in a sequence."
)

print()


# ==================================================
# 20. COMMUNICATIVE AI PATH
# ==================================================

print("COMMUNICATIVE AI PATH")
print()

print("Text")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Self-Attention")
print(" ↓")
print("Transformer Blocks")
print(" ↓")
print("Language Model")
print(" ↓")
print("Next Token")
print(" ↓")
print("Generated Response")

print()


# ==================================================
# 21. SILVERWING PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Neural Networks")
print("      ↓")
print("Embeddings")
print("      ↓")
print("Sequence Modeling")
print("      ↓")
print("Attention")
print("      ↓")
print("Multi-Head Attention")
print("      ↓")
print("Transformers")
print("      ↓")
print("Language Models")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 36 COMPLETE ===")
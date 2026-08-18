# Silverwing ML
# Phase 5 - Lesson 69R
# Silverwing Own Foundation Model
# Own Self-Attention Mechanism
#
# Corrected version:
# - fixes TEST 20 syntax error
# - keeps attention implementation explicit
# - verifies causal masking
# - verifies gradients
# - saves configuration


import json
import math

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 69R")
print("Silverwing Own Foundation Model")
print("Own Self-Attention Mechanism")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_CONFIG_FILE = (
        BASE_DIR.parent
        / "lesson67R"
        / "silverwing_embedding_config.json"
)

POSITION_CONFIG_FILE = (
        BASE_DIR.parent
        / "lesson68R"
        / "silverwing_position_config.json"
)

ATTENTION_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_attention_config.json"
)

MODEL_DIMENSION = 128
NUMBER_OF_HEADS = 8
DROPOUT = 0.0
MAX_SEQUENCE_LENGTH = 256
SEED = 42

torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print("Model dimension:", MODEL_DIMENSION)
print("Attention heads:", NUMBER_OF_HEADS)
print("Dropout:", DROPOUT)
print("Maximum sequence length:", MAX_SEQUENCE_LENGTH)
print("Device:", DEVICE)

print()


# ==================================================
# 2. VERIFY PREVIOUS COMPONENTS
# ==================================================

print("TEST 2: Verify Previous Components")
print()

if not EMBEDDING_CONFIG_FILE.exists():

    raise FileNotFoundError(
        "Lesson 67R configuration was not found. "
        "Run Lesson 67R first."
    )

if not POSITION_CONFIG_FILE.exists():

    raise FileNotFoundError(
        "Lesson 68R configuration was not found. "
        "Run Lesson 68R first."
    )


with open(
        EMBEDDING_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    embedding_config = json.load(file)


with open(
        POSITION_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    position_config = json.load(file)


print("Embedding configuration found.")
print("Position configuration found.")

print()


# ==================================================
# 3. DIMENSION VALIDATION
# ==================================================

print("TEST 3: Attention Dimension Validation")
print()

if MODEL_DIMENSION % NUMBER_OF_HEADS != 0:

    raise ValueError(
        "Model dimension must be divisible "
        "by number of attention heads."
    )


HEAD_DIMENSION = (
        MODEL_DIMENSION
        // NUMBER_OF_HEADS
)


print("Model dimension:", MODEL_DIMENSION)
print("Number of heads:", NUMBER_OF_HEADS)
print("Dimension per head:", HEAD_DIMENSION)

print()


# ==================================================
# 4. CORE ATTENTION MATHEMATICS
# ==================================================

class SilverwingAttentionMath:

    @staticmethod
    def scaled_dot_product(
            queries,
            keys,
            values,
            mask=None
    ):
        """
        Attention(Q,K,V)
        =
        softmax(QK^T / sqrt(d))V
        """

        dimension = queries.shape[-1]

        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1)
        )

        scores = (
                scores
                /
                math.sqrt(dimension)
        )

        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            weights,
            values
        )

        return (
            output,
            weights,
            scores
        )


# ==================================================
# 5. SINGLE HEAD ATTENTION
# ==================================================

class SilverwingSingleHeadAttention(
    nn.Module
):

    def __init__(
            self,
            input_dimension
    ):

        super().__init__()

        self.input_dimension = (
            input_dimension
        )

        self.query_projection = nn.Linear(
            input_dimension,
            input_dimension,
            bias=False
        )

        self.key_projection = nn.Linear(
            input_dimension,
            input_dimension,
            bias=False
        )

        self.value_projection = nn.Linear(
            input_dimension,
            input_dimension,
            bias=False
        )

        self.output_projection = nn.Linear(
            input_dimension,
            input_dimension,
            bias=False
        )


    def forward(
            self,
            x,
            mask=None,
            return_attention=False
    ):

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        output, weights, scores = (
            SilverwingAttentionMath.scaled_dot_product(
                queries,
                keys,
                values,
                mask
            )
        )

        output = self.output_projection(
            output
        )

        if return_attention:

            return (
                output,
                weights,
                scores
            )

        return output


# ==================================================
# 6. MULTI-HEAD SELF-ATTENTION
# ==================================================

class SilverwingMultiHeadAttention(
    nn.Module
):
    """
    Silverwing's multi-head self-attention.
    """

    def __init__(
            self,
            model_dimension,
            number_of_heads,
            dropout=0.0
    ):

        super().__init__()

        if (
                model_dimension
                %
                number_of_heads
                !=
                0
        ):

            raise ValueError(
                "model_dimension must be divisible "
                "by number_of_heads."
            )

        self.model_dimension = (
            model_dimension
        )

        self.number_of_heads = (
            number_of_heads
        )

        self.head_dimension = (
                model_dimension
                //
                number_of_heads
        )

        self.query_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )

        self.key_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )

        self.value_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )

        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def split_heads(
            self,
            x
    ):

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        x = x.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension
        )

        return x.transpose(
            1,
            2
        )


    def merge_heads(
            self,
            x
    ):

        batch_size = x.shape[0]
        sequence_length = x.shape[2]

        x = x.transpose(
            1,
            2
        ).contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.model_dimension
        )


    def forward(
            self,
            x,
            mask=None,
            return_attention=False
    ):

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        queries = self.split_heads(
            queries
        )

        keys = self.split_heads(
            keys
        )

        values = self.split_heads(
            values
        )

        output, weights, scores = (
            SilverwingAttentionMath.scaled_dot_product(
                queries,
                keys,
                values,
                mask
            )
        )

        output = self.merge_heads(
            output
        )

        output = self.output_projection(
            output
        )

        output = self.dropout(
            output
        )

        if return_attention:

            return (
                output,
                weights,
                scores
            )

        return output


# ==================================================
# 7. CREATE ATTENTION MODULE
# ==================================================

print("TEST 4: Create Multi-Head Attention")
print()

attention = SilverwingMultiHeadAttention(
    model_dimension=MODEL_DIMENSION,
    number_of_heads=NUMBER_OF_HEADS,
    dropout=DROPOUT
).to(DEVICE)


print(
    "Attention module created."
)

print()


# ==================================================
# 8. PARAMETER COUNT
# ==================================================

print("TEST 5: Attention Parameters")
print()

attention_parameters = sum(
    parameter.numel()
    for parameter in attention.parameters()
)

print(
    "Trainable parameters:",
    attention_parameters
)

print()


# ==================================================
# 9. RANDOM INPUT
# ==================================================

print("TEST 6: Input Representation")
print()

batch_size = 2
sequence_length = 8

input_representation = torch.randn(
    batch_size,
    sequence_length,
    MODEL_DIMENSION,
    device=DEVICE
)

print(
    "Input shape:",
    tuple(
        input_representation.shape
    )
)

print()


# ==================================================
# 10. FORWARD PASS
# ==================================================

print("TEST 7: Attention Forward Pass")
print()

attention_output, attention_weights, attention_scores = (
    attention(
        input_representation,
        return_attention=True
    )
)

print(
    "Output shape:",
    tuple(
        attention_output.shape
    )
)

print(
    "Attention weights shape:",
    tuple(
        attention_weights.shape
    )
)

print(
    "Attention scores shape:",
    tuple(
        attention_scores.shape
    )
)

print()


# ==================================================
# 11. HEAD STRUCTURE
# ==================================================

print("TEST 8: Attention Head Structure")
print()

print("Batch:", batch_size)
print("Heads:", NUMBER_OF_HEADS)
print("Sequence:", sequence_length)
print("Head dimension:", HEAD_DIMENSION)

print()


# ==================================================
# 12. WEIGHT NORMALIZATION
# ==================================================

print("TEST 9: Attention Weight Normalization")
print()

weight_sums = attention_weights.sum(
    dim=-1
)

maximum_deviation = torch.max(
    torch.abs(
        weight_sums - 1.0
    )
)

print(
    "Maximum deviation from 1:",
    float(
        maximum_deviation
    )
)

print()


# ==================================================
# 13. CAUSAL MASK
# ==================================================

def create_causal_mask(
        sequence_length,
        device
):

    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bool
        )
    )

    return mask.unsqueeze(
        0
    ).unsqueeze(
        0
    )


print("TEST 10: Causal Mask")
print()

causal_mask = create_causal_mask(
    sequence_length,
    DEVICE
)

print(
    causal_mask[
        0,
        0
    ].int()
)

print()


# ==================================================
# 14. CAUSAL ATTENTION
# ==================================================

print("TEST 11: Causal Self-Attention")
print()

causal_output, causal_weights, causal_scores = (
    attention(
        input_representation,
        mask=causal_mask,
        return_attention=True
    )
)

print(
    "Causal output shape:",
    tuple(
        causal_output.shape
    )
)

print(
    "Causal attention shape:",
    tuple(
        causal_weights.shape
    )
)

print()


# ==================================================
# 15. FUTURE TOKEN PROTECTION
# ==================================================

print("TEST 12: Future Token Protection")
print()

future_attention = (
    causal_weights[
        0,
        0
    ]
)

upper_triangle = torch.triu(
    future_attention,
    diagonal=1
)

maximum_future_attention = torch.max(
    torch.abs(
        upper_triangle
    )
)

print(
    "Maximum future-token attention:",
    float(
        maximum_future_attention
    )
)

print()


# ==================================================
# 16. ATTENTION EXAMPLE
# ==================================================

print("TEST 13: Attention Example")
print()

for head in range(
        min(
            NUMBER_OF_HEADS,
            4
        )
):

    matrix = causal_weights[
        0,
        head
    ]

    print(
        "Head:",
        head
    )

    for position in range(
            min(
                sequence_length,
                5
            )
    ):

        print(
            " position",
            position,
            "->",
            matrix[
                position
            ].detach().cpu().tolist()
        )

    print()


# ==================================================
# 17. Q/K/V SHAPES
# ==================================================

print("TEST 14: Q / K / V")
print()

projected_queries = (
    attention.split_heads(
        attention.query_projection(
            input_representation
        )
    )
)

projected_keys = (
    attention.split_heads(
        attention.key_projection(
            input_representation
        )
    )
)

projected_values = (
    attention.split_heads(
        attention.value_projection(
            input_representation
        )
    )
)

print(
    "Q:",
    tuple(
        projected_queries.shape
    )
)

print(
    "K:",
    tuple(
        projected_keys.shape
    )
)

print(
    "V:",
    tuple(
        projected_values.shape
    )
)

print()


# ==================================================
# 18. RAW SCORE MATRIX
# ==================================================

print("TEST 15: Raw Attention Scores")
print()

first_head_queries = (
    projected_queries[
        0,
        0
    ]
)

first_head_keys = (
    projected_keys[
        0,
        0
    ]
)

raw_scores = (
        torch.matmul(
            first_head_queries,
            first_head_keys.transpose(
                0,
                1
            )
        )
        /
        math.sqrt(
            HEAD_DIMENSION
        )
)

print(
    "Score matrix shape:",
    tuple(
        raw_scores.shape
    )
)

print()


# ==================================================
# 19. SOFTMAX
# ==================================================

print("TEST 16: Attention Softmax")
print()

softmax_scores = F.softmax(
    raw_scores,
    dim=-1
)

print(
    "First score row:",
    softmax_scores[
        0
    ].detach().cpu().tolist()
)

print()


# ==================================================
# 20. VALUE AGGREGATION
# ==================================================

print("TEST 17: Value Aggregation")
print()

aggregated_values = torch.matmul(
    softmax_scores,
    projected_values[
        0,
        0
    ]
)

print(
    "Aggregated shape:",
    tuple(
        aggregated_values.shape
    )
)

print()


# ==================================================
# 21. HEAD MERGING
# ==================================================

print("TEST 18: Head Merging")
print()

separated_output = torch.matmul(
    causal_weights,
    projected_values
)

merged_output = attention.merge_heads(
    separated_output
)

print(
    "Separated heads:",
    tuple(
        separated_output.shape
    )
)

print(
    "Merged representation:",
    tuple(
        merged_output.shape
    )
)

print()


# ==================================================
# 22. OUTPUT PROJECTION
# ==================================================

print("TEST 19: Output Projection")
print()

projected_output = (
    attention.output_projection(
        merged_output
    )
)

print(
    "Projected output shape:",
    tuple(
        projected_output.shape
    )
)

print()


# ==================================================
# 23. FULL ATTENTION RECONSTRUCTION
# ==================================================

print("TEST 20: Full Attention Reconstruction")
print()

reconstructed = attention(
    input_representation,
    mask=causal_mask
)

reconstruction_difference = torch.norm(
    reconstructed
    -
    causal_output
)

print(
    "Reconstruction difference:",
    float(
        reconstruction_difference
    )
)

print()


# ==================================================
# 24. INPUT SENSITIVITY
# ==================================================

print("TEST 21: Input Sensitivity")
print()

modified_input = (
    input_representation.clone()
)

modified_input[
    0,
    3
] += 2.0

original_output = attention(
    input_representation,
    mask=causal_mask
)

modified_output = attention(
    modified_input,
    mask=causal_mask
)

difference = torch.norm(
    original_output
    -
    modified_output
)

print(
    "Output difference:",
    float(
        difference
    )
)

print()


# ==================================================
# 25. GRADIENT FLOW
# ==================================================

print("TEST 22: Gradient Flow")
print()

training_attention = (
    SilverwingMultiHeadAttention(
        MODEL_DIMENSION,
        NUMBER_OF_HEADS,
        DROPOUT
    ).to(DEVICE)
)

training_input = torch.randn(
    2,
    8,
    MODEL_DIMENSION,
    device=DEVICE,
    requires_grad=True
)

training_mask = create_causal_mask(
    8,
    DEVICE
)

training_output = (
    training_attention(
        training_input,
        mask=training_mask
    )
)

loss = training_output.pow(
    2
).mean()

loss.backward()

gradient_norms = [
    float(
        parameter.grad.norm()
    )
    for parameter
    in training_attention.parameters()
    if parameter.grad is not None
]

print(
    "Loss:",
    float(
        loss.detach()
    )
)

print(
    "Gradient tensors:",
    len(
        gradient_norms
    )
)

print(
    "Maximum gradient norm:",
    max(
        gradient_norms
    )
)

print()


# ==================================================
# 26. PARAMETER BREAKDOWN
# ==================================================

print("TEST 23: Parameter Breakdown")
print()

q_parameters = (
        MODEL_DIMENSION
        *
        MODEL_DIMENSION
)

k_parameters = q_parameters
v_parameters = q_parameters
output_parameters = q_parameters

total_projection_parameters = (
        q_parameters
        +
        k_parameters
        +
        v_parameters
        +
        output_parameters
)

print(
    "Query projection:",
    q_parameters
)

print(
    "Key projection:",
    k_parameters
)

print(
    "Value projection:",
    v_parameters
)

print(
    "Output projection:",
    output_parameters
)

print(
    "Total projection parameters:",
    total_projection_parameters
)

print()


# ==================================================
# 27. CONFIGURATION SAVE
# ==================================================

print("TEST 24: Save Attention Configuration")
print()

attention_config = {
    "model":
        "Silverwing-Attention-v1",

    "model_dimension":
        MODEL_DIMENSION,

    "number_of_heads":
        NUMBER_OF_HEADS,

    "head_dimension":
        HEAD_DIMENSION,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "dropout":
        DROPOUT,

    "causal":
        True,

    "device":
        str(
            DEVICE
        )
}


with open(
        ATTENTION_CONFIG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        attention_config,
        file,
        indent=4
    )

print(
    "Saved:",
    ATTENTION_CONFIG_FILE
)

print()


# ==================================================
# 28. ATTENTION FORMULA
# ==================================================

print("SILVERWING ATTENTION FORMULA")
print()

print(
    "Attention(Q, K, V)"
)

print(
    " = softmax(QK^T / sqrt(d_k))V"
)

print()


# ==================================================
# 29. CAUSAL LANGUAGE MODEL
# ==================================================

print("CAUSAL LANGUAGE MODEL ATTENTION")
print()

print("Past/current tokens")
print("       ↓")
print("Self-Attention")
print("       ↓")
print("Future tokens masked")
print("       ↓")
print("Contextual representation")
print("       ↓")
print("Next-token prediction")

print()


# ==================================================
# 30. WHY MULTIPLE HEADS
# ==================================================

print("WHY MULTIPLE ATTENTION HEADS")
print()

print(
    "Different heads provide separate learned "
    "projection spaces."
)

print()

print(
    "The system can therefore learn multiple "
    "types of relationships simultaneously."
)

print()

print(
    "The meaning of each head is learned during "
    "training rather than manually assigned."
)

print()


# ==================================================
# 31. SILVERWING REPRESENTATION PIPELINE
# ==================================================

print("SILVERWING REPRESENTATION PIPELINE")
print()

print("Text")
print(" ↓")
print("Own BPE Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Own Token Embeddings")
print(" ↓")
print("Own Position Encoding")
print(" ↓")
print("Q / K / V")
print(" ↓")
print("Scaled Dot-Product Attention")
print(" ↓")
print("Multi-Head Attention")
print(" ↓")
print("Contextual Representation")

print()


# ==================================================
# 32. BIO-INSPIRED CONNECTION
# ==================================================

print("BIO-INSPIRED CONNECTION")
print()

print(
    "Attention is a selective information-routing "
    "mechanism."
)

print()

print(
    "Future Silverwing systems can reuse this "
    "concept for memory retrieval, event relationships, "
    "temporal reasoning and multimodal interaction."
)

print()


# ==================================================
# 33. ENGINEERING PRINCIPLE
# ==================================================

print("ENGINEERING PRINCIPLE")
print()

print(
    "Self-attention is a mathematical mechanism."
)

print()

print(
    "It does not independently create autonomous "
    "intelligence."
)

print()

print(
    "Silverwing's larger cognitive capabilities "
    "will emerge from the interaction of the model "
    "with training, memory, tools, evaluation and "
    "adaptive control systems."
)

print()


# ==================================================
# 34. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "The attention mechanism now produces "
    "contextual representations."
)

print()

print(
    "The next lesson combines attention with:"
)

print(
    "- residual connections"
)

print(
    "- normalization"
)

print(
    "- feed-forward transformation"
)

print(
    "- dropout infrastructure"
)

print()

print(
    "Together these form a reusable "
    "Silverwing Transformer Block."
)

print()


# ==================================================
# 35. FOUNDATION MODEL PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own BPE Tokenizer")
print(" ↓")
print("Own Subword Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("Own Embedding System")
print(" ↓")
print("Own Position Encoding")
print(" ↓")
print("OWN SELF-ATTENTION  ← COMPLETE")
print(" ↓")
print("Transformer Block  ← NEXT")
print(" ↓")
print("Decoder Architecture")
print(" ↓")
print("Language Model")
print(" ↓")
print("Training")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Cognitive Integration")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 69R COMPLETE ===")
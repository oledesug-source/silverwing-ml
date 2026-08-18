# Silverwing ML
# Phase 5 - Lesson 70R
# Silverwing Own Foundation Model
# Own Transformer Block


import json
import math

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 70R")
print("Silverwing Own Foundation Model")
print("Own Transformer Block")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

ATTENTION_CONFIG_FILE = (
        BASE_DIR.parent
        / "lesson69R"
        / "silverwing_attention_config.json"
)

TRANSFORMER_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_transformer_config.json"
)

MODEL_DIMENSION = 128

NUMBER_OF_HEADS = 8

FEED_FORWARD_DIMENSION = 512

DROPOUT = 0.0

MAX_SEQUENCE_LENGTH = 256

SEED = 42


torch.manual_seed(
    SEED
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print(
    "Model dimension:",
    MODEL_DIMENSION
)

print(
    "Attention heads:",
    NUMBER_OF_HEADS
)

print(
    "Feed-forward dimension:",
    FEED_FORWARD_DIMENSION
)

print(
    "Dropout:",
    DROPOUT
)

print(
    "Maximum sequence length:",
    MAX_SEQUENCE_LENGTH
)

print(
    "Device:",
    DEVICE
)

print()


# ==================================================
# 2. VERIFY ATTENTION ARTIFACT
# ==================================================

print("TEST 2: Verify Attention Component")
print()


if not ATTENTION_CONFIG_FILE.exists():

    raise FileNotFoundError(
        "Lesson 69R attention configuration "
        "was not found.\n"
        "Run Lesson 69R first."
    )


with open(
        ATTENTION_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    attention_config = json.load(
        file
    )


print(
    "Attention configuration found."
)

print(
    "Attention model:",
    attention_config.get(
        "model"
    )
)

print()


# ==================================================
# 3. DIMENSION VALIDATION
# ==================================================

print("TEST 3: Dimension Validation")
print()


if (
        MODEL_DIMENSION
        %
        NUMBER_OF_HEADS
        !=
        0
):

    raise ValueError(
        "Model dimension must be divisible "
        "by number of attention heads."
    )


HEAD_DIMENSION = (
        MODEL_DIMENSION
        //
        NUMBER_OF_HEADS
)


if FEED_FORWARD_DIMENSION <= (
        MODEL_DIMENSION
):

    raise ValueError(
        "Feed-forward dimension should be "
        "greater than model dimension."
    )


print(
    "Head dimension:",
    HEAD_DIMENSION
)

print(
    "Dimensions valid."
)

print()


# ==================================================
# 4. CORE ATTENTION MATH
# ==================================================

class SilverwingAttentionMath:

    @staticmethod
    def scaled_dot_product(
            queries,
            keys,
            values,
            mask=None
    ):

        dimension = queries.shape[-1]


        scores = torch.matmul(
            queries,
            keys.transpose(
                -2,
                -1
            )
        )


        scores = (
                scores
                /
                math.sqrt(
                    dimension
                )
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
            weights
        )


# ==================================================
# 5. MULTI-HEAD SELF-ATTENTION
# ==================================================

class SilverwingMultiHeadAttention(
    nn.Module
):

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

        queries = self.query_projection(
            x
        )


        keys = self.key_projection(
            x
        )


        values = self.value_projection(
            x
        )


        queries = self.split_heads(
            queries
        )


        keys = self.split_heads(
            keys
        )


        values = self.split_heads(
            values
        )


        output, weights = (
            SilverwingAttentionMath
            .scaled_dot_product(
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
                weights
            )


        return output


# ==================================================
# 6. FEED-FORWARD NETWORK
# ==================================================

class SilverwingFeedForward(
    nn.Module
):
    """
    Position-wise feed-forward transformation.

    x
    ↓
    Linear
    ↓
    GELU
    ↓
    Dropout
    ↓
    Linear
    """

    def __init__(
            self,
            model_dimension,
            hidden_dimension,
            dropout=0.0
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            model_dimension,
            hidden_dimension
        )


        self.output_projection = nn.Linear(
            hidden_dimension,
            model_dimension
        )


        self.activation = nn.GELU()


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
            self,
            x
    ):

        x = self.input_projection(
            x
        )


        x = self.activation(
            x
        )


        x = self.dropout(
            x
        )


        x = self.output_projection(
            x
        )


        return x


# ==================================================
# 7. TRANSFORMER BLOCK
# ==================================================

class SilverwingTransformerBlock(
    nn.Module
):
    """
    One complete decoder-style transformer block.

    Architecture:

        input
          ↓
      self-attention
          ↓
      residual + norm
          ↓
      feed-forward
          ↓
      residual + norm
          ↓
        output
    """

    def __init__(
            self,
            model_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout=0.0
    ):

        super().__init__()


        self.attention = (
            SilverwingMultiHeadAttention(
                model_dimension,
                number_of_heads,
                dropout
            )
        )


        self.feed_forward = (
            SilverwingFeedForward(
                model_dimension,
                feed_forward_dimension,
                dropout
            )
        )


        self.norm_after_attention = (
            nn.LayerNorm(
                model_dimension
            )
        )


        self.norm_after_feed_forward = (
            nn.LayerNorm(
                model_dimension
            )
        )


        self.dropout_attention = (
            nn.Dropout(
                dropout
            )
        )


        self.dropout_feed_forward = (
            nn.Dropout(
                dropout
            )
        )


    def forward(
            self,
            x,
            mask=None,
            return_attention=False
    ):

        # ------------------------------------------
        # Attention
        # ------------------------------------------

        attention_output, attention_weights = (
            self.attention(
                x,
                mask=mask,
                return_attention=True
            )
        )


        # ------------------------------------------
        # Residual connection
        # ------------------------------------------

        x = (
                x
                +
                self.dropout_attention(
                    attention_output
                )
        )


        # ------------------------------------------
        # Normalization
        # ------------------------------------------

        x = self.norm_after_attention(
            x
        )


        # ------------------------------------------
        # Feed-forward
        # ------------------------------------------

        feed_forward_output = (
            self.feed_forward(
                x
            )
        )


        # ------------------------------------------
        # Residual connection
        # ------------------------------------------

        x = (
                x
                +
                self.dropout_feed_forward(
                    feed_forward_output
                )
        )


        # ------------------------------------------
        # Normalization
        # ------------------------------------------

        x = self.norm_after_feed_forward(
            x
        )


        if return_attention:

            return (
                x,
                attention_weights
            )


        return x


# ==================================================
# 8. CREATE BLOCK
# ==================================================

print("TEST 4: Create Transformer Block")
print()


transformer_block = (
    SilverwingTransformerBlock(
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_dimension=(
            FEED_FORWARD_DIMENSION
        ),
        dropout=DROPOUT
    )
)


transformer_block = (
    transformer_block.to(
        DEVICE
    )
)


print(
    "Transformer block created."
)

print()


# ==================================================
# 9. PARAMETER COUNT
# ==================================================

print("TEST 5: Transformer Parameters")
print()


total_parameters = sum(
    parameter.numel()
    for parameter
    in transformer_block.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in transformer_block.parameters()
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
# 10. RANDOM INPUT
# ==================================================

print("TEST 6: Input Tensor")
print()


batch_size = 2

sequence_length = 16


input_tensor = torch.randn(
    batch_size,
    sequence_length,
    MODEL_DIMENSION,
    device=DEVICE
)


print(
    "Input shape:",
    tuple(
        input_tensor.shape
    )
)

print()


# ==================================================
# 11. CAUSAL MASK
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


causal_mask = create_causal_mask(
    sequence_length,
    DEVICE
)


# ==================================================
# 12. FORWARD PASS
# ==================================================

print("TEST 7: Transformer Forward Pass")
print()


output, attention_weights = (
    transformer_block(
        input_tensor,
        mask=causal_mask,
        return_attention=True
    )
)


print(
    "Output shape:",
    tuple(
        output.shape
    )
)

print(
    "Attention shape:",
    tuple(
        attention_weights.shape
    )
)

print()


# ==================================================
# 13. SHAPE PRESERVATION
# ==================================================

print("TEST 8: Shape Preservation")
print()


if (
        output.shape
        ==
        input_tensor.shape
):

    print(
        "PASS: block preserves tensor shape."
    )

else:

    raise RuntimeError(
        "Transformer block changed the expected "
        "tensor shape."
    )

print()


# ==================================================
# 14. RESIDUAL CONNECTION
# ==================================================

print("TEST 9: Residual Representation")
print()


attention_only = (
    transformer_block.attention(
        input_tensor,
        mask=causal_mask
    )
)


attention_residual = (
        input_tensor
        +
        attention_only
)


print(
    "Input norm:",
    float(
        input_tensor.norm()
    )
)

print(
    "Attention output norm:",
    float(
        attention_only.norm()
    )
)

print(
    "Residual norm:",
    float(
        attention_residual.norm()
    )
)

print()


# ==================================================
# 15. NORMALIZATION
# ==================================================

print("TEST 10: Layer Normalization")
print()


normalized = (
    transformer_block.norm_after_attention(
        attention_residual
    )
)


mean_value = (
    normalized.mean()
)


std_value = (
    normalized.std()
)


print(
    "Global normalized mean:",
    float(
        mean_value
    )
)

print(
    "Global normalized standard deviation:",
    float(
        std_value
    )
)

print()


# ==================================================
# 16. FEED-FORWARD NETWORK
# ==================================================

print("TEST 11: Feed-Forward Network")
print()


feed_forward_output = (
    transformer_block.feed_forward(
        normalized
    )
)


print(
    "Input shape:",
    tuple(
        normalized.shape
    )
)

print(
    "Output shape:",
    tuple(
        feed_forward_output.shape
    )
)

print()


# ==================================================
# 17. FULL BLOCK STAGES
# ==================================================

print("TEST 12: Transformer Block Stages")
print()

print("Stage 1:")
print("Input")

print()

print("Stage 2:")
print("Self-Attention")

print()

print("Stage 3:")
print("Residual + Normalization")

print()

print("Stage 4:")
print("Feed-Forward Network")

print()

print("Stage 5:")
print("Residual + Normalization")

print()

print("Stage 6:")
print("Block Output")

print()


# ==================================================
# 18. ATTENTION WEIGHT NORMALIZATION
# ==================================================

print("TEST 13: Attention Weight Validation")
print()


weight_sums = (
    attention_weights.sum(
        dim=-1
    )
)


maximum_deviation = torch.max(
    torch.abs(
        weight_sums
        -
        1.0
    )
)


print(
    "Maximum weight normalization deviation:",
    float(
        maximum_deviation
    )
)

print()


# ==================================================
# 19. CAUSAL VALIDATION
# ==================================================

print("TEST 14: Causal Mask Validation")
print()


first_head_weights = (
    attention_weights[
        0,
        0
    ]
)


future_attention = torch.triu(
    first_head_weights,
    diagonal=1
)


maximum_future_weight = torch.max(
    torch.abs(
        future_attention
    )
)


print(
    "Maximum future-token attention:",
    float(
        maximum_future_weight
    )
)

print()


# ==================================================
# 20. FEED-FORWARD PARAMETER COUNT
# ==================================================

print("TEST 15: Feed-Forward Parameters")
print()


feed_forward_parameters = sum(
    parameter.numel()
    for parameter
    in transformer_block
    .feed_forward
    .parameters()
)


print(
    "Feed-forward parameters:",
    feed_forward_parameters
)

print()


# ==================================================
# 21. ATTENTION PARAMETER COUNT
# ==================================================

print("TEST 16: Attention Parameters")
print()


attention_parameters = sum(
    parameter.numel()
    for parameter
    in transformer_block
    .attention
    .parameters()
)


print(
    "Attention parameters:",
    attention_parameters
)

print()


# ==================================================
# 22. NORMALIZATION PARAMETERS
# ==================================================

print("TEST 17: Normalization Parameters")
print()


normalization_parameters = (
        sum(
            parameter.numel()
            for parameter
            in transformer_block
            .norm_after_attention
            .parameters()
        )
        +
        sum(
            parameter.numel()
            for parameter
            in transformer_block
            .norm_after_feed_forward
            .parameters()
        )
)


print(
    "Normalization parameters:",
    normalization_parameters
)

print()


# ==================================================
# 23. BLOCK PARAMETER ACCOUNTING
# ==================================================

print("TEST 18: Parameter Accounting")
print()


accounted_parameters = (
        attention_parameters
        +
        feed_forward_parameters
        +
        normalization_parameters
)


print(
    "Accounted parameters:",
    accounted_parameters
)

print(
    "Actual parameters:",
    total_parameters
)

print(
    "Accounting difference:",
    total_parameters
    -
    accounted_parameters
)

print()


# ==================================================
# 24. GRADIENT FLOW
# ==================================================

print("TEST 19: Gradient Flow")
print()


training_block = (
    SilverwingTransformerBlock(
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_dimension=(
            FEED_FORWARD_DIMENSION
        ),
        dropout=DROPOUT
    ).to(DEVICE)
)


training_input = torch.randn(
    2,
    12,
    MODEL_DIMENSION,
    device=DEVICE,
    requires_grad=True
)


training_mask = create_causal_mask(
    12,
    DEVICE
)


training_output = training_block(
    training_input,
    mask=training_mask
)


loss = (
    training_output
    .pow(2)
    .mean()
)


loss.backward()


gradient_norms = [
    float(
        parameter.grad.norm()
    )
    for parameter
    in training_block.parameters()
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
# 25. OPTIMIZER UPDATE
# ==================================================

print("TEST 20: Parameter Update")
print()


optimizer = torch.optim.AdamW(
    training_block.parameters(),
    lr=1e-3
)


parameters_before = [
    parameter.detach().clone()
    for parameter
    in training_block.parameters()
]


optimizer.step()


parameters_after = [
    parameter.detach().clone()
    for parameter
    in training_block.parameters()
]


total_parameter_change = 0.0


for before, after in zip(
        parameters_before,
        parameters_after
):

    total_parameter_change += float(
        torch.sum(
            torch.abs(
                after
                -
                before
            )
        )
    )


print(
    "Total parameter change:",
    total_parameter_change
)

print()


# ==================================================
# 26. TRAINING MODE
# ==================================================

print("TEST 21: Training Mode")
print()


training_block.train()

print(
    "Training mode:",
    training_block.training
)


training_block.eval()

print(
    "Evaluation mode:",
    training_block.training
)

print()


# ==================================================
# 27. BLOCK REUSABILITY
# ==================================================

print("TEST 22: Reusable Block")
print()


second_block = (
    SilverwingTransformerBlock(
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_dimension=(
            FEED_FORWARD_DIMENSION
        ),
        dropout=DROPOUT
    ).to(DEVICE)
)


first_output = transformer_block(
    input_tensor,
    mask=causal_mask
)


second_output = second_block(
    first_output,
    mask=causal_mask
)


print(
    "First block output:",
    tuple(
        first_output.shape
    )
)

print(
    "Second block output:",
    tuple(
        second_output.shape
    )
)

print()


# ==================================================
# 28. MULTI-BLOCK STACK
# ==================================================

class SilverwingTransformerStack(
    nn.Module
):

    def __init__(
            self,
            number_of_layers,
            model_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout=0.0
    ):

        super().__init__()


        self.layers = nn.ModuleList(
            [
                SilverwingTransformerBlock(
                    model_dimension,
                    number_of_heads,
                    feed_forward_dimension,
                    dropout
                )
                for _ in range(
                number_of_layers
            )
            ]
        )


    def forward(
            self,
            x,
            mask=None
    ):

        attention_maps = []


        for layer in self.layers:

            x, attention = layer(
                x,
                mask=mask,
                return_attention=True
            )


            attention_maps.append(
                attention
            )


        return (
            x,
            attention_maps
        )


# ==================================================
# 29. STACK TEST
# ==================================================

print("TEST 23: Transformer Stack")
print()


NUMBER_OF_LAYERS = 4


stack = (
    SilverwingTransformerStack(
        number_of_layers=NUMBER_OF_LAYERS,
        model_dimension=MODEL_DIMENSION,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_dimension=(
            FEED_FORWARD_DIMENSION
        ),
        dropout=DROPOUT
    ).to(DEVICE)
)


stack_output, attention_maps = stack(
    input_tensor,
    mask=causal_mask
)


print(
    "Layers:",
    NUMBER_OF_LAYERS
)

print(
    "Output shape:",
    tuple(
        stack_output.shape
    )
)

print(
    "Attention maps:",
    len(
        attention_maps
    )
)

print()


# ==================================================
# 30. STACK PARAMETER COUNT
# ==================================================

print("TEST 24: Stack Parameter Count")
print()


stack_parameters = sum(
    parameter.numel()
    for parameter
    in stack.parameters()
)


print(
    "Stack parameters:",
    stack_parameters
)

print(
    "Parameters per block approximately:",
    stack_parameters
    //
    NUMBER_OF_LAYERS
)

print()


# ==================================================
# 31. CONTEXTUAL TRANSFORMATION
# ==================================================

print("TEST 25: Contextual Transformation")
print()


input_norm = float(
    input_tensor.norm()
)

output_norm = float(
    stack_output.norm()
)


print(
    "Input norm:",
    input_norm
)

print(
    "Stack output norm:",
    output_norm
)

print()


# ==================================================
# 32. TRANSFORMER BLOCK ARCHITECTURE
# ==================================================

print("SILVERWING TRANSFORMER BLOCK")
print()

print("Input")
print("  │")
print("  ├───────────────┐")
print("  ↓               │")
print("Self-Attention    │")
print("  ↓               │")
print("Residual Add  ←───┘")
print("  ↓")
print("LayerNorm")
print("  ↓")
print("Feed-Forward")
print("  ↓")
print("Residual Add")
print("  ↓")
print("LayerNorm")
print("  ↓")
print("Output")

print()


# ==================================================
# 33. WHY THE FEED-FORWARD NETWORK
# ==================================================

print("WHY THE FEED-FORWARD NETWORK")
print()

print(
    "Attention mixes information between sequence "
    "positions."
)

print()

print(
    "The feed-forward network transforms the "
    "representation at each position."
)

print()

print(
    "Together they provide contextual communication "
    "and nonlinear feature transformation."
)

print()


# ==================================================
# 34. WHY RESIDUAL CONNECTIONS
# ==================================================

print("WHY RESIDUAL CONNECTIONS")
print()

print(
    "Residual paths provide a direct route for "
    "information and gradients through the network."
)

print()

print(
    "They become increasingly important as many "
    "transformer blocks are stacked."
)

print()


# ==================================================
# 35. WHY NORMALIZATION
# ==================================================

print("WHY NORMALIZATION")
print()

print(
    "Normalization helps keep internal activations "
    "in a more stable numerical regime."
)

print()

print(
    "This supports optimization of deeper networks."
)

print()


# ==================================================
# 36. BIO-INSPIRED CONNECTION
# ==================================================

print("BIO-INSPIRED CONNECTION")
print()

print(
    "A transformer block can be viewed as a "
    "repeatable computational processing unit."
)

print()

print(
    "Many such units create a deeper hierarchy "
    "of learned transformations."
)

print()

print(
    "This modularity will later be useful for "
    "experimentation and controlled architecture growth."
)

print()


# ==================================================
# 37. MODEL SCALING
# ==================================================

print("MODEL SCALING")
print()

print(
    "A foundation model can increase capacity by "
    "scaling several dimensions:"
)

print()

print(
    "1. Vocabulary size"
)

print(
    "2. Model dimension"
)

print(
    "3. Attention heads"
)

print(
    "4. Feed-forward dimension"
)

print(
    "5. Number of transformer layers"
)

print(
    "6. Training data"
)

print()


# ==================================================
# 38. SILVERWING FOUNDATION PIPELINE
# ==================================================

print("SILVERWING FOUNDATION MODEL PIPELINE")
print()

print("Raw Corpus")
print("   ↓")
print("Own BPE Tokenizer")
print("   ↓")
print("Own Token IDs")
print("   ↓")
print("Own Embeddings")
print("   ↓")
print("Own Position Encoding")
print("   ↓")
print("Transformer Block × N")
print("   ↓")
print("Language Modeling Head")
print("   ↓")
print("Token Probabilities")

print()


# ==================================================
# 39. CURRENT LIMITATION
# ==================================================

print("CURRENT LIMITATION")
print()

print(
    "The transformer block is not yet a complete "
    "language model."
)

print()

print(
    "It transforms representations but does not "
    "yet produce vocabulary logits."
)

print()

print(
    "The next major component is the decoder "
    "language-model architecture."
)

print()


# ==================================================
# 40. SAVE CONFIGURATION
# ==================================================

print("TEST 26: Save Transformer Configuration")
print()


transformer_config = {
    "model":
        "Silverwing-Transformer-v1",

    "model_dimension":
        MODEL_DIMENSION,

    "number_of_heads":
        NUMBER_OF_HEADS,

    "head_dimension":
        HEAD_DIMENSION,

    "feed_forward_dimension":
        FEED_FORWARD_DIMENSION,

    "dropout":
        DROPOUT,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "default_layers":
        NUMBER_OF_LAYERS,

    "causal":
        True,

    "device":
        str(
            DEVICE
        )
}


with open(
        TRANSFORMER_CONFIG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        transformer_config,
        file,
        indent=4
    )


print(
    "Saved:",
    TRANSFORMER_CONFIG_FILE
)

print()


# ==================================================
# 41. FOUNDATION PROGRESS
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
print("Own Self-Attention")
print(" ↓")
print("OWN TRANSFORMER BLOCK  ← COMPLETE")
print(" ↓")
print("Decoder Language Model  ← NEXT")
print(" ↓")
print("Training Objective")
print(" ↓")
print("Training Pipeline")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Memory / Agent Integration")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 70R COMPLETE ===")
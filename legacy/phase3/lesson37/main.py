# Silverwing ML
# Phase 3 - Lesson 37
# Building a Transformer Block
#
# Goal:
# Combine embeddings, multi-head self-attention,
# residual connections, layer normalization,
# and a feed-forward network.


import torch
import torch.nn as nn


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 37")
print("Building a Transformer Block")
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
# 2. BASIC CONFIGURATION
# ==================================================

vocabulary_size = 100
embedding_dimension = 32
number_of_heads = 4
feed_forward_dimension = 64
sequence_length = 8
batch_size = 2


print("TEST 2: Transformer Configuration")
print()

print(
    "Vocabulary size:",
    vocabulary_size
)

print(
    "Embedding dimension:",
    embedding_dimension
)

print(
    "Attention heads:",
    number_of_heads
)

print(
    "Feed-forward dimension:",
    feed_forward_dimension
)

print(
    "Sequence length:",
    sequence_length
)

print(
    "Batch size:",
    batch_size
)

print()


# ==================================================
# 3. CREATE TOKEN IDs
# ==================================================

print("TEST 3: Token IDs")
print()


token_ids = torch.randint(
    low=0,
    high=vocabulary_size,
    size=(
        batch_size,
        sequence_length
    )
)


print(
    "Token IDs:"
)

print(
    token_ids
)

print()

print(
    "Shape:",
    token_ids.shape
)

print()


# ==================================================
# 4. EMBEDDING LAYER
# ==================================================

print("TEST 4: Token Embeddings")
print()


embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)


token_embeddings = embedding(
    token_ids
)


print(
    "Embedding shape:",
    token_embeddings.shape
)

print()

print(
    "Expected:"
)

print(
    "(batch, sequence, embedding_dimension)"
)

print()


# ==================================================
# 5. POSITIONAL EMBEDDINGS
# ==================================================

print("TEST 5: Positional Embeddings")
print()


position_embedding = nn.Embedding(
    sequence_length,
    embedding_dimension
)


positions = torch.arange(
    sequence_length
)


position_vectors = position_embedding(
    positions
)


position_vectors = position_vectors.unsqueeze(
    0
)


print(
    "Position vector shape:",
    position_vectors.shape
)

print()


# ==================================================
# 6. COMBINE TOKEN + POSITION INFORMATION
# ==================================================

print("TEST 6: Input Representation")
print()


x = (
        token_embeddings
        +
        position_vectors
)


print(
    "Transformer input shape:",
    x.shape
)

print()


# ==================================================
# 7. MULTI-HEAD SELF-ATTENTION
# ==================================================

print("TEST 7: Multi-Head Self-Attention")
print()


self_attention = nn.MultiheadAttention(
    embed_dim=embedding_dimension,
    num_heads=number_of_heads,
    batch_first=True
)


attention_output, attention_weights = (
    self_attention(
        x,
        x,
        x
    )
)


print(
    "Attention output shape:",
    attention_output.shape
)

print()

print(
    "Attention weights shape:",
    attention_weights.shape
)

print()


# ==================================================
# 8. RESIDUAL CONNECTION
# ==================================================

print("TEST 8: Residual Connection")
print()


residual_1 = (
        x
        +
        attention_output
)


print(
    "Residual output shape:",
    residual_1.shape
)

print()


# ==================================================
# 9. LAYER NORMALIZATION
# ==================================================

print("TEST 9: Layer Normalization")
print()


layer_norm_1 = nn.LayerNorm(
    embedding_dimension
)


normalized_1 = layer_norm_1(
    residual_1
)


print(
    "Normalized output shape:",
    normalized_1.shape
)

print()


# ==================================================
# 10. FEED-FORWARD NETWORK
# ==================================================

print("TEST 10: Feed-Forward Network")
print()


feed_forward = nn.Sequential(

    nn.Linear(
        embedding_dimension,
        feed_forward_dimension
    ),

    nn.GELU(),

    nn.Linear(
        feed_forward_dimension,
        embedding_dimension
    )
)


feed_forward_output = feed_forward(
    normalized_1
)


print(
    "Feed-forward output shape:",
    feed_forward_output.shape
)

print()


# ==================================================
# 11. SECOND RESIDUAL CONNECTION
# ==================================================

print("TEST 11: Second Residual Connection")
print()


residual_2 = (
        normalized_1
        +
        feed_forward_output
)


print(
    "Residual output shape:",
    residual_2.shape
)

print()


# ==================================================
# 12. SECOND LAYER NORMALIZATION
# ==================================================

print("TEST 12: Second Layer Normalization")
print()


layer_norm_2 = nn.LayerNorm(
    embedding_dimension
)


transformer_output = layer_norm_2(
    residual_2
)


print(
    "Transformer output shape:"
)

print(
    transformer_output.shape
)

print()


# ==================================================
# 13. COMPLETE TRANSFORMER BLOCK
# ==================================================

print("TEST 13: Complete Transformer Block")
print()


class TransformerBlock(nn.Module):
    """
    Simplified Transformer encoder-style block.

    Components:
        1. Multi-head self-attention
        2. Residual connection
        3. Layer normalization
        4. Feed-forward network
        5. Residual connection
        6. Layer normalization
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            batch_first=True
        )


        self.layer_norm_1 = nn.LayerNorm(
            embedding_dimension
        )


        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_dimension,
                feed_forward_dimension
            ),

            nn.GELU(),

            nn.Linear(
                feed_forward_dimension,
                embedding_dimension
            )
        )


        self.layer_norm_2 = nn.LayerNorm(
            embedding_dimension
        )


    def forward(
            self,
            x
    ):

        attention_output, attention_weights = (
            self.attention(
                x,
                x,
                x,
                need_weights=True
            )
        )


        x = self.layer_norm_1(
            x
            +
            attention_output
        )


        feed_forward_output = (
            self.feed_forward(x)
        )


        x = self.layer_norm_2(
            x
            +
            feed_forward_output
        )


        return x, attention_weights


transformer_block = TransformerBlock(
    embedding_dimension=embedding_dimension,
    number_of_heads=number_of_heads,
    feed_forward_dimension=feed_forward_dimension
)


print(
    transformer_block
)

print()


# ==================================================
# 14. RUN THE TRANSFORMER BLOCK
# ==================================================

print("TEST 14: Run Transformer Block")
print()


block_output, block_attention = (
    transformer_block(x)
)


print(
    "Input shape:",
    x.shape
)

print()

print(
    "Output shape:",
    block_output.shape
)

print()

print(
    "Attention shape:",
    block_attention.shape
)

print()


# ==================================================
# 15. STACK MULTIPLE BLOCKS
# ==================================================

print("TEST 15: Multiple Transformer Blocks")
print()


number_of_blocks = 3


transformer_stack = nn.ModuleList([
    TransformerBlock(
        embedding_dimension=embedding_dimension,
        number_of_heads=number_of_heads,
        feed_forward_dimension=feed_forward_dimension
    )
    for _ in range(number_of_blocks)
])


stack_input = x


for index, block in enumerate(
        transformer_stack,
        start=1
):

    stack_input, _ = block(
        stack_input
    )

    print(
        "Block",
        index,
        "output shape:",
        stack_input.shape
    )


print()


# ==================================================
# 16. LANGUAGE-MODEL OUTPUT LAYER
# ==================================================

print("TEST 16: Language-Model Output")
print()


output_layer = nn.Linear(
    embedding_dimension,
    vocabulary_size
)


logits = output_layer(
    stack_input
)


print(
    "Logits shape:"
)

print(
    logits.shape
)

print()

print(
    "Expected:"
)

print(
    "(batch, sequence, vocabulary)"
)

print()


# ==================================================
# 17. NEXT-TOKEN PROBABILITIES
# ==================================================

print("TEST 17: Next-Token Probabilities")
print()


last_token_logits = logits[
    :,
    -1,
    :
]


probabilities = torch.softmax(
    last_token_logits,
    dim=-1
)


print(
    "Probability shape:"
)

print(
    probabilities.shape
)

print()


# ==================================================
# 18. CHOOSE NEXT TOKEN
# ==================================================

print("TEST 18: Next Token")
print()


next_token_ids = torch.argmax(
    probabilities,
    dim=-1
)


print(
    "Predicted next token IDs:"
)

print(
    next_token_ids
)

print()


# ==================================================
# 19. TOP-K TOKENS
# ==================================================

print("TEST 19: Top-K Predictions")
print()


top_k = 5


top_probabilities, top_token_ids = (
    torch.topk(
        probabilities,
        k=top_k,
        dim=-1
    )
)


print(
    "Top token IDs:"
)

print(
    top_token_ids
)

print()

print(
    "Top probabilities:"
)

print(
    top_probabilities
)

print()


# ==================================================
# 20. MODEL PARAMETER COUNT
# ==================================================

print("TEST 20: Parameter Count")
print()


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


print(
    "Transformer block parameters:",
    count_parameters(
        transformer_block
    )
)

print(
    "Three-block stack parameters:",
    count_parameters(
        transformer_stack
    )
)

print(
    "Output layer parameters:",
    count_parameters(
        output_layer
    )
)

print()


# ==================================================
# 21. WHAT EACH COMPONENT DOES
# ==================================================

print("TEST 21: Transformer Components")
print()

print(
    "Embedding:"
)

print(
    "Converts token IDs into vectors."
)

print()

print(
    "Positional information:"
)

print(
    "Provides information about token position."
)

print()

print(
    "Self-attention:"
)

print(
    "Lets token representations interact."
)

print()

print(
    "Residual connection:"
)

print(
    "Helps preserve and carry information "
    "through deep networks."
)

print()

print(
    "Layer normalization:"
)

print(
    "Helps stabilize the representations."
)

print()

print(
    "Feed-forward network:"
)

print(
    "Transforms each token representation "
    "after attention."
)

print()

print(
    "Output layer:"
)

print(
    "Converts representations into vocabulary logits."
)

print()


# ==================================================
# 22. TRANSFORMER BLOCK FLOW
# ==================================================

print("TRANSFORMER BLOCK FLOW")
print()

print("Token IDs")
print("   ↓")
print("Token Embeddings")
print("   +")
print("Positional Information")
print("   ↓")
print("Multi-Head Self-Attention")
print("   ↓")
print("Residual + Normalization")
print("   ↓")
print("Feed-Forward Network")
print("   ↓")
print("Residual + Normalization")
print("   ↓")
print("Transformer Output")

print()


# ==================================================
# 23. LANGUAGE MODEL FLOW
# ==================================================

print("LANGUAGE MODEL FLOW")
print()

print("Text")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Transformer Blocks")
print(" ↓")
print("Output Projection")
print(" ↓")
print("Vocabulary Logits")
print(" ↓")
print("Probabilities")
print(" ↓")
print("Next Token")

print()


# ==================================================
# 24. LLM CONNECTION
# ==================================================

print("LLM CONNECTION")
print()

print(
    "A modern Transformer language model "
    "contains many layers of Transformer blocks."
)

print()

print(
    "The actual architectures used by different "
    "LLMs vary in details, scale, normalization, "
    "position handling, and training objectives."
)

print()

print(
    "This lesson implements a simplified "
    "Transformer-style block for learning."
)

print()


# ==================================================
# 25. IMPORTANT DISTINCTION
# ==================================================

print("IMPORTANT DISTINCTION")
print()

print(
    "We have built a Transformer COMPONENT."
)

print()

print(
    "We have NOT built a useful LLM yet."
)

print()

print(
    "A real language model requires large-scale "
    "data, tokenization, training objectives, "
    "many parameters, extensive training, "
    "evaluation, and significant compute."
)

print()


# ==================================================
# 26. SILVERWING AI PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Python")
print(" ↓")
print("Mathematics")
print(" ↓")
print("Classical ML")
print(" ↓")
print("Neural Networks")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Sequence Modeling")
print(" ↓")
print("Attention")
print(" ↓")
print("Transformer Block")
print(" ↓")
print("Language Model")
print(" ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 37 COMPLETE ===")

# Silverwing ML
# Phase 5 - Lesson 67R
# Silverwing Own Foundation Model
# Own Embedding System
#
# Goal:
# Build Silverwing's own trainable token embedding
# and positional embedding architecture from scratch.
#
# This lesson does NOT use:
# - GPT-2 embeddings
# - Qwen embeddings
# - pretrained transformer embeddings
#
# PyTorch is used only as the numerical/tensor
# framework. The embedding architecture belongs
# to Silverwing.


import json
import math

from pathlib import Path
from typing import List

import torch
import torch.nn as nn


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 67R")
print("Silverwing Own Foundation Model")
print("Own Embedding System")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TOKENIZER_VOCABULARY = (
        Path(__file__).resolve().parent.parent
        / "lesson66R"
        / "silverwing_subword_vocabulary.json"
)

EMBEDDING_CONFIGURATION = (
        BASE_DIR / "silverwing_embedding_config.json"
)

MODEL_DIMENSION = 128

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
    "Tokenizer vocabulary:",
    TOKENIZER_VOCABULARY
)

print(
    "Embedding dimension:",
    MODEL_DIMENSION
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
# 2. LOAD SILVERWING VOCABULARY
# ==================================================

print("TEST 2: Load Silverwing Vocabulary")
print()


if not TOKENIZER_VOCABULARY.exists():

    raise FileNotFoundError(
        "Lesson 66R vocabulary was not found.\n"
        "Run phase5/lesson66R/main.py first."
    )


with open(
        TOKENIZER_VOCABULARY,
        "r",
        encoding="utf-8"
) as file:

    vocabulary_data = json.load(
        file
    )


TOKEN_TO_ID = {
    token: int(token_id)
    for token, token_id
    in vocabulary_data[
        "token_to_id"
    ].items()
}


ID_TO_TOKEN = {
    token_id: token
    for token, token_id
    in TOKEN_TO_ID.items()
}


VOCABULARY_SIZE = len(
    TOKEN_TO_ID
)


print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print()


# ==================================================
# 3. SPECIAL TOKEN IDS
# ==================================================

print("TEST 3: Special Token IDs")
print()


PAD_ID = TOKEN_TO_ID[
    "<PAD>"
]

UNK_ID = TOKEN_TO_ID[
    "<UNK>"
]

BOS_ID = TOKEN_TO_ID[
    "<BOS>"
]

EOS_ID = TOKEN_TO_ID[
    "<EOS>"
]


print(
    "PAD:",
    PAD_ID
)

print(
    "UNK:",
    UNK_ID
)

print(
    "BOS:",
    BOS_ID
)

print(
    "EOS:",
    EOS_ID
)

print()


# ==================================================
# 4. SILVERWING TOKEN EMBEDDING
# ==================================================

class SilverwingTokenEmbedding(
    nn.Module
):
    """
    Maps Silverwing token IDs to dense learned
    vectors.

    Each vocabulary entry owns one trainable
    embedding vector.
    """

    def __init__(
            self,
            vocabulary_size: int,
            embedding_dimension: int
    ):

        super().__init__()


        self.vocabulary_size = (
            vocabulary_size
        )

        self.embedding_dimension = (
            embedding_dimension
        )


        self.embedding = nn.Embedding(
            num_embeddings=(
                vocabulary_size
            ),

            embedding_dim=(
                embedding_dimension
            ),

            padding_idx=PAD_ID
        )


        self.reset_parameters()


    def reset_parameters(self):

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
        )


        with torch.no_grad():

            self.embedding.weight[
                PAD_ID
            ].zero_()


    def forward(
            self,
            token_ids
    ):

        return self.embedding(
            token_ids
        )


# ==================================================
# 5. CREATE TOKEN EMBEDDING
# ==================================================

print("TEST 4: Create Token Embedding")
print()


token_embedding = (
    SilverwingTokenEmbedding(
        VOCABULARY_SIZE,
        MODEL_DIMENSION
    )
)


token_embedding = (
    token_embedding.to(
        DEVICE
    )
)


print(
    "Embedding matrix shape:",
    tuple(
        token_embedding.embedding.weight.shape
    )
)

print()


# ==================================================
# 6. POSITIONAL EMBEDDING
# ==================================================

class SilverwingPositionalEmbedding(
    nn.Module
):
    """
    Learned positional representation.

    Token embeddings describe WHAT a token represents.
    Positional embeddings provide information about
    WHERE the token occurs in a sequence.
    """

    def __init__(
            self,
            maximum_sequence_length: int,
            embedding_dimension: int
    ):

        super().__init__()


        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.embedding_dimension = (
            embedding_dimension
        )


        self.embedding = nn.Embedding(
            num_embeddings=(
                maximum_sequence_length
            ),

            embedding_dim=(
                embedding_dimension
            )
        )


        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
        )


    def forward(
            self,
            sequence_length: int,
            device
    ):

        if sequence_length > (
                self.maximum_sequence_length
        ):

            raise ValueError(
                "Sequence length exceeds "
                "maximum configured length."
            )


        positions = torch.arange(
            sequence_length,
            device=device
        )


        return self.embedding(
            positions
        )


positional_embedding = (
    SilverwingPositionalEmbedding(
        MAX_SEQUENCE_LENGTH,
        MODEL_DIMENSION
    )
)


positional_embedding = (
    positional_embedding.to(
        DEVICE
    )
)


print("TEST 5: Positional Embedding")
print()

print(
    "Position matrix shape:",
    tuple(
        positional_embedding.embedding.weight.shape
    )
)

print()


# ==================================================
# 7. COMBINED EMBEDDING SYSTEM
# ==================================================

class SilverwingEmbeddingSystem(
    nn.Module
):
    """
    Combines token and positional representations.

    output = token_embedding + position_embedding
    """

    def __init__(
            self,
            vocabulary_size: int,
            embedding_dimension: int,
            maximum_sequence_length: int
    ):

        super().__init__()


        self.token_embedding = (
            SilverwingTokenEmbedding(
                vocabulary_size,
                embedding_dimension
            )
        )


        self.position_embedding = (
            SilverwingPositionalEmbedding(
                maximum_sequence_length,
                embedding_dimension
            )
        )


        self.embedding_dimension = (
            embedding_dimension
        )


        self.layer_norm = nn.LayerNorm(
            embedding_dimension
        )


        self.dropout = nn.Dropout(
            p=0.0
        )


    def forward(
            self,
            token_ids
    ):

        sequence_length = (
            token_ids.shape[1]
        )


        token_vectors = (
            self.token_embedding(
                token_ids
            )
        )


        position_vectors = (
            self.position_embedding(
                sequence_length,
                token_ids.device
            )
        )


        position_vectors = (
            position_vectors.unsqueeze(
                0
            )
        )


        combined = (
                token_vectors
                +
                position_vectors
        )


        combined = self.layer_norm(
            combined
        )


        combined = self.dropout(
            combined
        )


        return combined


embedding_system = (
    SilverwingEmbeddingSystem(
        VOCABULARY_SIZE,
        MODEL_DIMENSION,
        MAX_SEQUENCE_LENGTH
    )
)


embedding_system = (
    embedding_system.to(
        DEVICE
    )
)


# ==================================================
# 8. TOKEN ID EXAMPLE
# ==================================================

print("TEST 6: Token ID Input")
print()


sample_tokens = [
    "<BOS>",
    "silverwing",
    "learn",
    "s",
    "from",
    "data",
    ".",
    "<EOS>"
]


sample_ids = []


for token in sample_tokens:

    sample_ids.append(
        TOKEN_TO_ID.get(
            token,
            UNK_ID
        )
    )


input_ids = torch.tensor(
    [sample_ids],
    dtype=torch.long,
    device=DEVICE
)


print(
    "Tokens:",
    sample_tokens
)

print(
    "IDs:",
    input_ids.tolist()
)

print()


# ==================================================
# 9. EMBEDDING FORWARD PASS
# ==================================================

print("TEST 7: Embedding Forward Pass")
print()


with torch.no_grad():

    embeddings = embedding_system(
        input_ids
    )


print(
    "Input shape:",
    tuple(
        input_ids.shape
    )
)

print(
    "Embedding shape:",
    tuple(
        embeddings.shape
    )
)

print()


# ==================================================
# 10. SINGLE TOKEN VECTOR
# ==================================================

print("TEST 8: Single Token Representation")
print()


first_token_vector = (
    embeddings[0, 0]
)


print(
    "Token:",
    sample_tokens[0]
)

print(
    "Vector dimensions:",
    first_token_vector.shape[0]
)

print(
    "First values:",
    first_token_vector[
        :10
    ].detach().cpu().tolist()
)

print()


# ==================================================
# 11. TOKEN DIFFERENCE
# ==================================================

print("TEST 9: Token Representation Difference")
print()


vector_a = (
    embeddings[0, 1]
)


vector_b = (
    embeddings[0, 2]
)


distance = torch.norm(
    vector_a
    -
    vector_b
)


print(
    "Token A:",
    sample_tokens[1]
)

print(
    "Token B:",
    sample_tokens[2]
)

print(
    "Vector distance:",
    float(
        distance
    )
)

print()


# ==================================================
# 12. COSINE SIMILARITY
# ==================================================

print("TEST 10: Embedding Similarity")
print()


def cosine_similarity(
        vector_a,
        vector_b
):

    vector_a = vector_a.flatten()

    vector_b = vector_b.flatten()


    return torch.nn.functional.cosine_similarity(
        vector_a.unsqueeze(0),
        vector_b.unsqueeze(0)
    ).item()


similarity = cosine_similarity(
    vector_a,
    vector_b
)


print(
    "Cosine similarity:",
    similarity
)

print()


# ==================================================
# 13. SEQUENCE REPRESENTATION
# ==================================================

print("TEST 11: Sequence Representation")
print()


sequence_mean = (
    embeddings.mean(
        dim=1
    )
)


print(
    "Sequence representation shape:",
    tuple(
        sequence_mean.shape
    )
)

print()


# ==================================================
# 14. PADDING BEHAVIOR
# ==================================================

print("TEST 12: Padding")
print()


padded_ids = torch.tensor(
    [
        sample_ids,
        sample_ids[:-2]
        +
        [
            PAD_ID,
            PAD_ID
        ]
    ],
    dtype=torch.long,
    device=DEVICE
)


with torch.no_grad():

    padded_embeddings = (
        embedding_system(
            padded_ids
        )
    )


print(
    "Padded input shape:",
    tuple(
        padded_ids.shape
    )
)

print(
    "Output shape:",
    tuple(
        padded_embeddings.shape
    )
)

print()


# ==================================================
# 15. ATTENTION MASK
# ==================================================

print("TEST 13: Attention Mask")
print()


attention_mask = (
        padded_ids
        !=
        PAD_ID
).long()


print(
    attention_mask
)

print()


# ==================================================
# 16. MASKED SEQUENCE REPRESENTATION
# ==================================================

print("TEST 14: Masked Sequence Representation")
print()


mask = (
    attention_mask
    .unsqueeze(-1)
)


masked_embeddings = (
        padded_embeddings
        *
        mask
)


valid_token_counts = (
    attention_mask.sum(
        dim=1,
        keepdim=True
    ).clamp(
        min=1
    )
)


masked_mean = (
        masked_embeddings.sum(
            dim=1
        )
        /
        valid_token_counts
)


print(
    "Masked representation shape:",
    tuple(
        masked_mean.shape
    )
)

print()


# ==================================================
# 17. PARAMETER COUNT
# ==================================================

print("TEST 15: Embedding Parameters")
print()


total_parameters = sum(
    parameter.numel()
    for parameter
    in embedding_system.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in embedding_system.parameters()
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
# 18. PARAMETER BREAKDOWN
# ==================================================

print("TEST 16: Parameter Breakdown")
print()


token_parameter_count = (
        VOCABULARY_SIZE
        *
        MODEL_DIMENSION
)


position_parameter_count = (
        MAX_SEQUENCE_LENGTH
        *
        MODEL_DIMENSION
)


print(
    "Token embedding parameters:",
    token_parameter_count
)

print(
    "Position embedding parameters:",
    position_parameter_count
)

print()


# ==================================================
# 19. EMBEDDING UPDATE DEMONSTRATION
# ==================================================

print("TEST 17: Trainable Embeddings")
print()


target = torch.zeros_like(
    embeddings
)


loss_function = nn.MSELoss()


optimizer = torch.optim.Adam(
    embedding_system.parameters(),
    lr=1e-3
)


embedding_system.train()


optimizer.zero_grad()


prediction = embedding_system(
    input_ids
)


loss = loss_function(
    prediction,
    target
)


loss.backward()


optimizer.step()


embedding_system.eval()


print(
    "Demonstration loss:",
    float(
        loss.detach()
    )
)

print(
    "Gradient-based update completed."
)

print()


# ==================================================
# 20. EMBEDDING CHANGES
# ==================================================

print("TEST 18: Parameter Update Verification")
print()


with torch.no_grad():

    updated_embeddings = (
        embedding_system(
            input_ids
        )
    )


change = torch.mean(
    torch.abs(
        updated_embeddings
        -
        embeddings
    )
)


print(
    "Average embedding change:",
    float(
        change
    )
)

print()


# ==================================================
# 21. SAVE CONFIGURATION
# ==================================================

print("TEST 19: Save Embedding Configuration")
print()


configuration = {
    "model":
        "Silverwing-Embedding-v1",

    "vocabulary_size":
        VOCABULARY_SIZE,

    "embedding_dimension":
        MODEL_DIMENSION,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "device":
        str(
            DEVICE
        ),

    "padding_token_id":
        PAD_ID,

    "unknown_token_id":
        UNK_ID,

    "bos_token_id":
        BOS_ID,

    "eos_token_id":
        EOS_ID
}


with open(
        EMBEDDING_CONFIGURATION,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


print(
    "Saved:",
    EMBEDDING_CONFIGURATION
)

print()


# ==================================================
# 22. WHY EMBEDDINGS ARE LEARNED
# ==================================================

print("WHY EMBEDDINGS ARE LEARNED")
print()

print(
    "The initial vectors are random."
)

print()

print(
    "Training updates them according to model error."
)

print()

print(
    "Tokens appearing in useful contextual patterns "
    "can gradually develop useful representations."
)

print()

print(
    "The embedding space therefore becomes part "
    "of Silverwing's learned internal representation."
)

print()


# ==================================================
# 23. TOKENIZER → EMBEDDING CONNECTION
# ==================================================

print("TOKENIZER → EMBEDDING")
print()

print("Text")
print(" ↓")
print("Silverwing BPE")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Embedding Lookup")
print(" ↓")
print("Dense Vectors")

print()


# ==================================================
# 24. EMBEDDING → TRANSFORMER CONNECTION
# ==================================================

print("EMBEDDING → TRANSFORMER")
print()

print("Token IDs")
print(" ↓")
print("Token Embeddings")
print(" +")
print("Position Embeddings")
print(" ↓")
print("Layer Normalization")
print(" ↓")
print("Transformer")
print(" ↓")
print("Contextual Representations")

print()


# ==================================================
# 25. CONTEXTUAL VS STATIC REPRESENTATION
# ==================================================

print("CONTEXTUAL REPRESENTATION")
print()

print(
    "The token embedding is the initial representation."
)

print()

print(
    "The transformer will later modify that "
    "representation according to surrounding tokens."
)

print()

print(
    "Therefore the same token can acquire different "
    "contextual representations in different sentences."
)

print()


# ==================================================
# 26. FOUNDATION MODEL PIPELINE
# ==================================================

print("SILVERWING FOUNDATION MODEL")
print()

print("Corpus")
print(" ↓")
print("BPE Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Own Embedding Matrix")
print(" ↓")
print("Positional Representation")
print(" ↓")
print("Self-Attention")
print(" ↓")
print("Transformer Blocks")
print(" ↓")
print("Language Modeling Head")
print(" ↓")
print("Next Token")

print()


# ==================================================
# 27. SELF-GROWTH CONNECTION
# ==================================================

print("SELF-GROWTH CONNECTION")
print()

print(
    "Future Silverwing training can continuously "
    "refine its embedding space through validated "
    "training experiences."
)

print()

print(
    "However, model updates should remain versioned "
    "and evaluated rather than silently changing the "
    "production model."
)

print()


# ==================================================
# 28. ENGINEERING PRINCIPLE
# ==================================================

print("ENGINEERING PRINCIPLE")
print()

print(
    "Silverwing's learned representations should "
    "be treated as versioned model parameters."
)

print()

print(
    "Every training run should produce a reproducible "
    "checkpoint and measurable evaluation results."
)

print()


# ==================================================
# 29. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "The embedding layer knows what token "
    "representations exist."
)

print()

print(
    "The next lesson gives Silverwing a representation "
    "of sequence order."
)

print()

print(
    "That component is positional encoding."
)

print()


# ==================================================
# 30. SILVERWING FOUNDATION PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own BPE Tokenizer")
print(" ↓")
print("Own Subword Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("OWN EMBEDDING SYSTEM  ← COMPLETE")
print(" ↓")
print("Positional Encoding")
print(" ↓")
print("Self-Attention")
print(" ↓")
print("Transformer Block")
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

print("=== LESSON 67R COMPLETE ===")
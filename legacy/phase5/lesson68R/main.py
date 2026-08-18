# Silverwing ML
# Phase 5 - Lesson 68R
# Silverwing Own Foundation Model
# Positional Encoding System
#
# Goal:
# Build Silverwing's own sequence-position
# representation from scratch.
#
# No pretrained transformer positional system
# is used.


import json
import math

from pathlib import Path

import torch
import torch.nn as nn


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 68R")
print("Silverwing Own Foundation Model")
print("Positional Encoding System")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

VOCABULARY_FILE = (
        BASE_DIR.parent
        / "lesson66R"
        / "silverwing_subword_vocabulary.json"
)

EMBEDDING_CONFIG_FILE = (
        BASE_DIR.parent
        / "lesson67R"
        / "silverwing_embedding_config.json"
)

POSITION_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_position_config.json"
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
    "Vocabulary:",
    VOCABULARY_FILE
)

print(
    "Embedding configuration:",
    EMBEDDING_CONFIG_FILE
)

print(
    "Model dimension:",
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
# 2. VERIFY PREVIOUS LESSON ARTIFACTS
# ==================================================

print("TEST 2: Verify Previous Components")
print()


if not VOCABULARY_FILE.exists():

    raise FileNotFoundError(
        "Lesson 66R vocabulary was not found. "
        "Run Lesson 66R first."
    )


if not EMBEDDING_CONFIG_FILE.exists():

    raise FileNotFoundError(
        "Lesson 67R embedding configuration "
        "was not found. Run Lesson 67R first."
    )


print(
    "Lesson 66R vocabulary found."
)

print(
    "Lesson 67R embedding configuration found."
)

print()


# ==================================================
# 3. LOAD CONFIGURATION
# ==================================================

with open(
        EMBEDDING_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    embedding_config = json.load(
        file
    )


vocabulary_size = embedding_config[
    "vocabulary_size"
]


configured_dimension = embedding_config[
    "embedding_dimension"
]


if configured_dimension != (
        MODEL_DIMENSION
):

    raise ValueError(
        "Embedding dimension mismatch."
    )


print("TEST 3: Loaded Configuration")
print()

print(
    "Vocabulary size:",
    vocabulary_size
)

print(
    "Embedding dimension:",
    configured_dimension
)

print()


# ==================================================
# 4. POSITION INDEX
# ==================================================

print("TEST 4: Position Index")
print()


positions = torch.arange(
    MAX_SEQUENCE_LENGTH,
    dtype=torch.long
)


print(
    "First positions:",
    positions[:16].tolist()
)

print(
    "Last position:",
    int(
        positions[-1]
    )
)

print()


# ==================================================
# 5. LEARNED POSITIONAL EMBEDDING
# ==================================================

class SilverwingLearnedPositionEncoding(
    nn.Module
):
    """
    Trainable positional representation.

    Every sequence position owns a learnable
    vector.
    """

    def __init__(
            self,
            maximum_sequence_length: int,
            dimension: int
    ):

        super().__init__()


        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.dimension = dimension


        self.embedding = nn.Embedding(
            maximum_sequence_length,
            dimension
        )


        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
        )


    def forward(
            self,
            sequence_length,
            device
    ):

        if sequence_length > (
                self.maximum_sequence_length
        ):

            raise ValueError(
                "Sequence exceeds configured "
                "maximum length."
            )


        position_ids = torch.arange(
            sequence_length,
            device=device
        )


        return self.embedding(
            position_ids
        )


learned_position = (
    SilverwingLearnedPositionEncoding(
        MAX_SEQUENCE_LENGTH,
        MODEL_DIMENSION
    )
)


learned_position = (
    learned_position.to(
        DEVICE
    )
)


print("TEST 5: Learned Position Encoding")
print()

print(
    "Parameter shape:",
    tuple(
        learned_position.embedding.weight.shape
    )
)

print()


# ==================================================
# 6. SINUSOIDAL POSITIONAL ENCODING
# ==================================================

class SilverwingSinusoidalPositionEncoding(
    nn.Module
):
    """
    Deterministic sinusoidal positional encoding.

    The representation is calculated from position
    and dimension rather than learned directly.
    """

    def __init__(
            self,
            maximum_sequence_length: int,
            dimension: int
    ):

        super().__init__()


        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.dimension = dimension


        encoding = torch.zeros(
            maximum_sequence_length,
            dimension
        )


        position = torch.arange(
            maximum_sequence_length,
            dtype=torch.float32
        ).unsqueeze(
            1
        )


        div_term = torch.exp(
            torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32
            )
            *
            (
                    -math.log(
                        10000.0
                    )
                    /
                    dimension
            )
        )


        encoding[:, 0::2] = torch.sin(
            position * div_term
        )


        encoding[:, 1::2] = torch.cos(
            position * div_term
        )


        encoding = encoding.unsqueeze(
            0
        )


        self.register_buffer(
            "encoding",
            encoding
        )


    def forward(
            self,
            sequence_length
    ):

        if sequence_length > (
                self.maximum_sequence_length
        ):

            raise ValueError(
                "Sequence exceeds configured "
                "maximum length."
            )


        return self.encoding[
            :,
            :sequence_length,
            :
        ]


sinusoidal_position = (
    SilverwingSinusoidalPositionEncoding(
        MAX_SEQUENCE_LENGTH,
        MODEL_DIMENSION
    )
)


sinusoidal_position = (
    sinusoidal_position.to(
        DEVICE
    )
)


print("TEST 6: Sinusoidal Position Encoding")
print()

print(
    "Encoding shape:",
    tuple(
        sinusoidal_position.encoding.shape
    )
)

print()


# ==================================================
# 7. POSITION VECTOR INSPECTION
# ==================================================

print("TEST 7: Position Vector Inspection")
print()


sequence_length = 8


learned_vectors = (
    learned_position(
        sequence_length,
        DEVICE
    )
)


sinusoidal_vectors = (
    sinusoidal_position(
        sequence_length
    )
)


print(
    "Learned vector shape:",
    tuple(
        learned_vectors.shape
    )
)

print(
    "Sinusoidal vector shape:",
    tuple(
        sinusoidal_vectors.shape
    )
)

print()


# ==================================================
# 8. POSITION 0 VS POSITION 1
# ==================================================

print("TEST 8: Position Distinction")
print()


learned_distance = torch.norm(
    learned_vectors[0]
    -
    learned_vectors[1]
)


sinusoidal_distance = torch.norm(
    sinusoidal_vectors[0, 0]
    -
    sinusoidal_vectors[0, 1]
)


print(
    "Learned distance:",
    float(
        learned_distance
    )
)

print(
    "Sinusoidal distance:",
    float(
        sinusoidal_distance
    )
)

print()


# ==================================================
# 9. SINUSOIDAL PATTERN INSPECTION
# ==================================================

print("TEST 9: Sinusoidal Pattern")
print()


for position in range(5):

    vector = sinusoidal_vectors[
        0,
        position
    ]


    print(
        "Position",
        position,
        ":",
        vector[
            :8
        ].detach().cpu().tolist()
    )

print()


# ==================================================
# 10. POSITION INDEPENDENCE
# ==================================================

print("TEST 10: Position Representation Independence")
print()


position_a = sinusoidal_vectors[
    0,
    2
]


position_b = sinusoidal_vectors[
    0,
    2
]


same_position_difference = torch.norm(
    position_a
    -
    position_b
)


print(
    "Same-position difference:",
    float(
        same_position_difference
    )
)

print()


# ==================================================
# 11. COMBINE TOKEN + POSITION
# ==================================================

class SilverwingSequenceRepresentation(
    nn.Module
):
    """
    Converts token embeddings into ordered
    contextual input representations.
    """

    def __init__(
            self,
            dimension,
            maximum_sequence_length,
            mode="learned"
    ):

        super().__init__()


        self.dimension = dimension

        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.mode = mode


        if mode == "learned":

            self.position = (
                SilverwingLearnedPositionEncoding(
                    maximum_sequence_length,
                    dimension
                )
            )


        elif mode == "sinusoidal":

            self.position = (
                SilverwingSinusoidalPositionEncoding(
                    maximum_sequence_length,
                    dimension
                )
            )


        else:

            raise ValueError(
                "Position mode must be "
                "'learned' or 'sinusoidal'."
            )


        self.layer_norm = nn.LayerNorm(
            dimension
        )


    def forward(
            self,
            token_embeddings
    ):

        batch_size = (
            token_embeddings.shape[0]
        )

        sequence_length = (
            token_embeddings.shape[1]
        )


        if self.mode == "learned":

            position_vectors = (
                self.position(
                    sequence_length,
                    token_embeddings.device
                )
            )


            position_vectors = (
                position_vectors.unsqueeze(
                    0
                )
            )


        else:

            position_vectors = (
                self.position(
                    sequence_length
                )
            )


        if position_vectors.shape[0] == 1:

            position_vectors = (
                position_vectors.expand(
                    batch_size,
                    -1,
                    -1
                )
            )


        combined = (
                token_embeddings
                +
                position_vectors
        )


        return self.layer_norm(
            combined
        )


# ==================================================
# 12. TEST LEARNED SEQUENCE REPRESENTATION
# ==================================================

print("TEST 11: Learned Sequence Representation")
print()


token_embeddings = torch.randn(
    2,
    8,
    MODEL_DIMENSION,
    device=DEVICE
)


learned_sequence = (
    SilverwingSequenceRepresentation(
        MODEL_DIMENSION,
        MAX_SEQUENCE_LENGTH,
        mode="learned"
    ).to(DEVICE)
)


learned_output = learned_sequence(
    token_embeddings
)


print(
    "Input shape:",
    tuple(
        token_embeddings.shape
    )
)

print(
    "Output shape:",
    tuple(
        learned_output.shape
    )
)

print()


# ==================================================
# 13. TEST SINUSOIDAL SEQUENCE REPRESENTATION
# ==================================================

print("TEST 12: Sinusoidal Sequence Representation")
print()


sinusoidal_sequence = (
    SilverwingSequenceRepresentation(
        MODEL_DIMENSION,
        MAX_SEQUENCE_LENGTH,
        mode="sinusoidal"
    ).to(DEVICE)
)


sinusoidal_output = (
    sinusoidal_sequence(
        token_embeddings
    )
)


print(
    "Input shape:",
    tuple(
        token_embeddings.shape
    )
)

print(
    "Output shape:",
    tuple(
        sinusoidal_output.shape
    )
)

print()


# ==================================================
# 14. POSITION CHANGES REPRESENTATION
# ==================================================

print("TEST 13: Position Changes Input Representation")
print()


same_token = torch.ones(
    1,
    4,
    MODEL_DIMENSION,
    device=DEVICE
)


positioned = (
    sinusoidal_sequence(
        same_token
    )
)


difference = torch.norm(
    positioned[
        0,
        0
    ]
    -
    positioned[
        0,
        1
    ]
)


print(
    "Same token vector was placed at "
    "two different positions."
)

print(
    "Representation difference:",
    float(
        difference
    )
)

print()


# ==================================================
# 15. ORDER-SENSITIVITY
# ==================================================

print("TEST 14: Sequence Order Sensitivity")
print()


sequence_a = torch.zeros(
    1,
    4,
    MODEL_DIMENSION,
    device=DEVICE
)


sequence_b = torch.zeros(
    1,
    4,
    MODEL_DIMENSION,
    device=DEVICE
)


# Simulate the same token embeddings in
# different positions.

sequence_a[
    0,
    0
] = 1.0


sequence_a[
    0,
    1
] = 2.0


sequence_b[
    0,
    0
] = 2.0


sequence_b[
    0,
    1
] = 1.0


encoded_a = sinusoidal_sequence(
    sequence_a
)


encoded_b = sinusoidal_sequence(
    sequence_b
)


sequence_distance = torch.norm(
    encoded_a
    -
    encoded_b
)


print(
    "Order-sensitive distance:",
    float(
        sequence_distance
    )
)

print()


# ==================================================
# 16. LEARNED POSITION PARAMETERS
# ==================================================

print("TEST 15: Learned Position Parameters")
print()


learned_parameter_count = sum(
    parameter.numel()
    for parameter
    in learned_sequence.parameters()
)


print(
    "Learned positional parameters:",
    learned_parameter_count
)

print()


# ==================================================
# 17. SINUSOIDAL PARAMETERS
# ==================================================

print("TEST 16: Sinusoidal Parameters")
print()


sinusoidal_parameter_count = sum(
    parameter.numel()
    for parameter
    in sinusoidal_sequence.parameters()
)


print(
    "Trainable sinusoidal parameters:",
    sinusoidal_parameter_count
)

print()

print(
    "Sinusoidal encoding is deterministic."
)

print()


# ==================================================
# 18. MAXIMUM LENGTH
# ==================================================

print("TEST 17: Maximum Sequence Length")
print()


valid_sequence = sinusoidal_position(
    MAX_SEQUENCE_LENGTH
)


print(
    "Valid shape:",
    tuple(
        valid_sequence.shape
    )
)


try:

    sinusoidal_position(
        MAX_SEQUENCE_LENGTH + 1
    )

except ValueError as error:

    print(
        "Expected error:"
    )

    print(
        error
    )

print()


# ==================================================
# 19. POSITION CONFIGURATION
# ==================================================

print("TEST 18: Position Configuration")
print()


position_config = {
    "model":
        "Silverwing-Position-v1",

    "dimension":
        MODEL_DIMENSION,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "supported_modes": [
        "learned",
        "sinusoidal"
    ],

    "default_mode":
        "learned"
}


with open(
        POSITION_CONFIG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        position_config,
        file,
        indent=4
    )


print(
    "Saved:",
    POSITION_CONFIG_FILE
)

print()


# ==================================================
# 20. WHY POSITION MATTERS
# ==================================================

print("WHY POSITION MATTERS")
print()

print(
    "Self-attention can inspect relationships "
    "between tokens."
)

print()

print(
    "But token embeddings alone do not inherently "
    "identify where a token occurs in a sequence."
)

print()

print(
    "Position encoding gives the transformer "
    "sequence-order information."
)

print()


# ==================================================
# 21. TOKEN + POSITION + ATTENTION
# ==================================================

print("TOKEN + POSITION + ATTENTION")
print()

print("Text")
print(" ↓")
print("Silverwing Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Token Embeddings")
print(" +")
print("Position Encoding")
print(" ↓")
print("Ordered Representations")
print(" ↓")
print("Self-Attention")

print()


# ==================================================
# 22. LEARNED VS SINUSOIDAL
# ==================================================

print("LEARNED VS SINUSOIDAL")
print()

print(
    "Learned positions:"
)

print(
    "- trainable"
)

print(
    "- flexible"
)

print(
    "- tied to configured positions"
)

print()

print(
    "Sinusoidal positions:"
)

print(
    "- deterministic"
)

print(
    "- no learned position parameters"
)

print(
    "- mathematically generated"
)

print()


# ==================================================
# 23. SILVERWING DESIGN
# ==================================================

print("SILVERWING POSITION DESIGN")
print()

print(
    "The architecture keeps positional encoding "
    "as an explicit module."
)

print()

print(
    "This allows experimentation with different "
    "sequence-representation strategies without "
    "rewriting the transformer."
)

print()


# ==================================================
# 24. FUTURE POSITION RESEARCH
# ==================================================

print("FUTURE POSITION RESEARCH")
print()

future_methods = [
    "learned positions",
    "sinusoidal positions",
    "relative positions",
    "rotary representations",
    "linear attention position methods",
    "long-context position methods"
]


for method in future_methods:

    print(
        "-",
        method
    )

print()


# ==================================================
# 25. BIO-INSPIRED CONNECTION
# ==================================================

print("BIO-INSPIRED CONNECTION")
print()

print(
    "Sequence order is one form of structural context."
)

print()

print(
    "Future Silverwing architectures can represent "
    "not only token order but also time, events, "
    "memory chronology, and state transitions."
)

print()

print(
    "This becomes important for continual learning "
    "and temporal reasoning."
)

print()


# ==================================================
# 26. FOUNDATION MODEL PIPELINE
# ==================================================

print("SILVERWING FOUNDATION MODEL PIPELINE")
print()

print("Corpus")
print(" ↓")
print("Own BPE Tokenizer")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("Own Token Embeddings")
print(" ↓")
print("Own Position Representation")
print(" ↓")
print("SELF-ATTENTION")
print(" ↓")
print("Feed-Forward Network")
print(" ↓")
print("Transformer Block")
print(" ↓")
print("Decoder")
print(" ↓")
print("Language Model")

print()


# ==================================================
# 27. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "Silverwing can now represent:"
)

print(
    "1. token identity"
)

print(
    "2. sequence position"
)

print()

print(
    "The next lesson creates the mechanism that "
    "allows every token to selectively interact "
    "with every other relevant token."
)

print()

print(
    "That mechanism is Silverwing's own "
    "self-attention system."
)

print()


# ==================================================
# 28. FOUNDATION MODEL PROGRESS
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
print("OWN POSITION ENCODING  ← COMPLETE")
print(" ↓")
print("SELF-ATTENTION  ← NEXT")
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

print("=== LESSON 68R COMPLETE ===")
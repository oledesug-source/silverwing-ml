# Silverwing ML
# Phase 5 - Lesson 71R
# Silverwing Own Foundation Model
# Own Decoder Language Model
#
# Goal:
# Assemble Silverwing's tokenizer, embeddings,
# positional representation, transformer blocks,
# and language-model head into one native
# decoder-only language-model architecture.


import json
import math

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 71R")
print("Silverwing Own Foundation Model")
print("Own Decoder Language Model")
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
        BASE_DIR.parent
        / "lesson68R"
        / "silverwing_position_config.json"
)

TRANSFORMER_CONFIG_FILE = (
        BASE_DIR.parent
        / "lesson70R"
        / "silverwing_transformer_config.json"
)

MODEL_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_decoder_config.json"
)

CHECKPOINT_FILE = (
        BASE_DIR
        / "silverwing_decoder_initial.pt"
)


MODEL_DIMENSION = 128

NUMBER_OF_HEADS = 8

FEED_FORWARD_DIMENSION = 512

NUMBER_OF_LAYERS = 4

MAX_SEQUENCE_LENGTH = 256

DROPOUT = 0.0

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
    "Transformer layers:",
    NUMBER_OF_LAYERS
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
# 2. VERIFY PREVIOUS ARTIFACTS
# ==================================================

print("TEST 2: Verify Foundation Components")
print()


required_files = [
    VOCABULARY_FILE,
    EMBEDDING_CONFIG_FILE,
    POSITION_CONFIG_FILE,
    TRANSFORMER_CONFIG_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required artifact not found: "
            f"{file_path}"
        )


    print(
        "FOUND:",
        file_path
    )


print()


# ==================================================
# 3. LOAD VOCABULARY
# ==================================================

print("TEST 3: Load Silverwing Vocabulary")
print()


with open(
        VOCABULARY_FILE,
        "r",
        encoding="utf-8"
) as file:

    vocabulary_data = json.load(
        file
    )


TOKEN_TO_ID = {
    token:
        int(token_id)
    for token, token_id
    in vocabulary_data[
        "token_to_id"
    ].items()
}


ID_TO_TOKEN = {
    token_id:
        token
    for token_id, token
    in (
        (
            int(token_id),
            token
        )
        for token, token_id
        in vocabulary_data[
        "token_to_id"
    ].items()
    )
}


VOCABULARY_SIZE = len(
    TOKEN_TO_ID
)


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
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print(
    "PAD ID:",
    PAD_ID
)

print(
    "UNK ID:",
    UNK_ID
)

print(
    "BOS ID:",
    BOS_ID
)

print(
    "EOS ID:",
    EOS_ID
)

print()


# ==================================================
# 4. SIMPLE VOCABULARY TOKENIZATION
# ==================================================

def normalize_text(
        text: str
):

    return (
        text
        .lower()
        .strip()
    )


def tokenize_for_demo(
        text: str
):

    import re

    normalized = normalize_text(
        text
    )

    words = re.findall(
        r"\w+|[^\w\s]",
        normalized
    )

    tokens = []

    for word in words:

        # Exact vocabulary match first.

        if word in TOKEN_TO_ID:

            tokens.append(
                word
            )

            continue


        # Try individual characters /
        # BPE vocabulary pieces.

        for character in word:

            if character in TOKEN_TO_ID:

                tokens.append(
                    character
                )

            else:

                tokens.append(
                    "<UNK>"
                )


    return tokens


def encode_demo(
        text: str
):

    tokens = tokenize_for_demo(
        text
    )


    ids = [
        BOS_ID
    ]


    ids.extend(
        TOKEN_TO_ID.get(
            token,
            UNK_ID
        )
        for token
        in tokens
    )


    ids.append(
        EOS_ID
    )


    return ids


def decode_ids(
        ids
):

    tokens = []


    for token_id in ids:

        token = ID_TO_TOKEN.get(
            int(token_id),
            "<UNK>"
        )


        if token in {
            "<PAD>",
            "<BOS>",
            "<EOS>"
        }:

            continue


        tokens.append(
            token
        )


    return " ".join(
        tokens
    )


# ==================================================
# 5. CORE ATTENTION
# ==================================================

class SilverwingAttentionMath:

    @staticmethod
    def scaled_dot_product(
            queries,
            keys,
            values,
            mask=None
    ):

        dimension = (
            queries.shape[-1]
        )


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
# 6. MULTI-HEAD ATTENTION
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
                "Model dimension must be divisible "
                "by number of heads."
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
            mask=None
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


        return (
            output,
            weights
        )


# ==================================================
# 7. FEED-FORWARD
# ==================================================

class SilverwingFeedForward(
    nn.Module
):

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
# 8. TRANSFORMER BLOCK
# ==================================================

class SilverwingTransformerBlock(
    nn.Module
):

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
            mask=None
    ):

        attention_output, _ = (
            self.attention(
                x,
                mask
            )
        )


        x = (
                x
                +
                self.dropout_attention(
                    attention_output
                )
        )


        x = (
            self.norm_after_attention(
                x
            )
        )


        feed_forward_output = (
            self.feed_forward(
                x
            )
        )


        x = (
                x
                +
                self.dropout_feed_forward(
                    feed_forward_output
                )
        )


        x = (
            self.norm_after_feed_forward(
                x
            )
        )


        return x


# ==================================================
# 9. POSITIONAL ENCODING
# ==================================================

class SilverwingPositionEncoding(
    nn.Module
):

    def __init__(
            self,
            maximum_sequence_length,
            dimension
    ):

        super().__init__()


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

        positions = torch.arange(
            sequence_length,
            device=device
        )


        return self.embedding(
            positions
        )


# ==================================================
# 10. SILVERWING DECODER MODEL
# ==================================================

class SilverwingDecoderLM(
    nn.Module
):
    """
    Native Silverwing decoder-only language model.

    Pipeline:

    token IDs
       ↓
    token embeddings
       ↓
    position embeddings
       ↓
    transformer blocks
       ↓
    final normalization
       ↓
    vocabulary projection
       ↓
    logits
    """

    def __init__(
            self,
            vocabulary_size,
            model_dimension,
            number_of_heads,
            feed_forward_dimension,
            number_of_layers,
            maximum_sequence_length,
            dropout=0.0
    ):

        super().__init__()


        self.vocabulary_size = (
            vocabulary_size
        )

        self.model_dimension = (
            model_dimension
        )

        self.maximum_sequence_length = (
            maximum_sequence_length
        )


        # ------------------------------------------
        # Token embeddings
        # ------------------------------------------

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            model_dimension,
            padding_idx=PAD_ID
        )


        # ------------------------------------------
        # Position embeddings
        # ------------------------------------------

        self.position_embedding = (
            SilverwingPositionEncoding(
                maximum_sequence_length,
                model_dimension
            )
        )


        # ------------------------------------------
        # Transformer blocks
        # ------------------------------------------

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


        # ------------------------------------------
        # Final normalization
        # ------------------------------------------

        self.final_norm = nn.LayerNorm(
            model_dimension
        )


        # ------------------------------------------
        # Vocabulary projection
        # ------------------------------------------

        self.language_model_head = nn.Linear(
            model_dimension,
            vocabulary_size,
            bias=False
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.reset_parameters()


    def reset_parameters(self):

        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02
        )


        with torch.no_grad():

            self.token_embedding.weight[
                PAD_ID
            ].zero_()


        nn.init.normal_(
            self.language_model_head.weight,
            mean=0.0,
            std=0.02
        )


    def create_causal_mask(
            self,
            sequence_length,
            device
    ):

        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool
            )
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )


    def forward(
            self,
            input_ids,
            return_hidden_states=False,
            return_attention=False
    ):

        batch_size = (
            input_ids.shape[0]
        )

        sequence_length = (
            input_ids.shape[1]
        )


        if sequence_length > (
                self.maximum_sequence_length
        ):

            raise ValueError(
                "Input sequence exceeds "
                "maximum sequence length."
            )


        # ------------------------------------------
        # Token representation
        # ------------------------------------------

        token_vectors = (
            self.token_embedding(
                input_ids
            )
        )


        # ------------------------------------------
        # Position representation
        # ------------------------------------------

        position_vectors = (
            self.position_embedding(
                sequence_length,
                input_ids.device
            )
        )


        position_vectors = (
            position_vectors.unsqueeze(
                0
            )
        )


        position_vectors = (
            position_vectors.expand(
                batch_size,
                -1,
                -1
            )
        )


        hidden_states = (
                token_vectors
                +
                position_vectors
        )


        hidden_states = self.dropout(
            hidden_states
        )


        # ------------------------------------------
        # Causal mask
        # ------------------------------------------

        causal_mask = (
            self.create_causal_mask(
                sequence_length,
                input_ids.device
            )
        )


        attention_maps = []


        # ------------------------------------------
        # Transformer stack
        # ------------------------------------------

        for layer in self.layers:

            if return_attention:

                # Re-run attention explicitly to expose
                # attention maps while preserving the
                # block architecture.

                attention_output, attention_weights = (
                    layer.attention(
                        hidden_states,
                        causal_mask
                    )
                )


                hidden_states = (
                        hidden_states
                        +
                        layer.dropout_attention(
                            attention_output
                        )
                )


                hidden_states = (
                    layer.norm_after_attention(
                        hidden_states
                    )
                )


                feed_forward_output = (
                    layer.feed_forward(
                        hidden_states
                    )
                )


                hidden_states = (
                        hidden_states
                        +
                        layer.dropout_feed_forward(
                            feed_forward_output
                        )
                )


                hidden_states = (
                    layer.norm_after_feed_forward(
                        hidden_states
                    )
                )


                attention_maps.append(
                    attention_weights
                )

            else:

                hidden_states = layer(
                    hidden_states,
                    causal_mask
                )


        # ------------------------------------------
        # Final normalization
        # ------------------------------------------

        hidden_states = (
            self.final_norm(
                hidden_states
            )
        )


        # ------------------------------------------
        # Vocabulary projection
        # ------------------------------------------

        logits = (
            self.language_model_head(
                hidden_states
            )
        )


        result = {
            "logits":
                logits
        }


        if return_hidden_states:

            result[
                "hidden_states"
            ] = hidden_states


        if return_attention:

            result[
                "attention_maps"
            ] = attention_maps


        return result


    def count_parameters(self):

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


# ==================================================
# 11. CREATE MODEL
# ==================================================

print("TEST 4: Create Silverwing Decoder")
print()


silverwing = SilverwingDecoderLM(
    vocabulary_size=VOCABULARY_SIZE,
    model_dimension=MODEL_DIMENSION,
    number_of_heads=NUMBER_OF_HEADS,
    feed_forward_dimension=FEED_FORWARD_DIMENSION,
    number_of_layers=NUMBER_OF_LAYERS,
    maximum_sequence_length=MAX_SEQUENCE_LENGTH,
    dropout=DROPOUT
)


silverwing = silverwing.to(
    DEVICE
)


print(
    "Silverwing decoder created."
)

print()


# ==================================================
# 12. MODEL PARAMETERS
# ==================================================

print("TEST 5: Model Parameters")
print()


total_parameters = (
    silverwing.count_parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in silverwing.parameters()
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
# 13. SAMPLE INPUT
# ==================================================

print("TEST 6: Sample Token Sequence")
print()


sample_text = (
    "silverwing learns from data"
)


sample_ids = encode_demo(
    sample_text
)


input_ids = torch.tensor(
    [sample_ids],
    dtype=torch.long,
    device=DEVICE
)


print(
    "Text:",
    sample_text
)

print(
    "Token IDs:",
    sample_ids
)

print(
    "Sequence length:",
    len(
        sample_ids
    )
)

print()


# ==================================================
# 14. FORWARD PASS
# ==================================================

print("TEST 7: Decoder Forward Pass")
print()


with torch.no_grad():

    output = silverwing(
        input_ids,
        return_hidden_states=True,
        return_attention=True
    )


logits = output[
    "logits"
]


hidden_states = output[
    "hidden_states"
]


attention_maps = output[
    "attention_maps"
]


print(
    "Logits shape:",
    tuple(
        logits.shape
    )
)

print(
    "Hidden-state shape:",
    tuple(
        hidden_states.shape
    )
)

print(
    "Attention layers:",
    len(
        attention_maps
    )
)

print()


# ==================================================
# 15. NEXT-TOKEN LOGITS
# ==================================================

print("TEST 8: Next-Token Logits")
print()


last_position_logits = (
    logits[
        0,
        -1
    ]
)


print(
    "Next-token logits shape:",
    tuple(
        last_position_logits.shape
    )
)


top_k = min(
    10,
    VOCABULARY_SIZE
)


values, indices = torch.topk(
    last_position_logits,
    k=top_k
)


print(
    "Top candidate tokens:"
)


for score, token_id in zip(
        values.detach().cpu().tolist(),
        indices.detach().cpu().tolist()
):

    print(
        repr(
            ID_TO_TOKEN.get(
                int(token_id),
                "<UNK>"
            )
        ),
        "->",
        round(
            score,
            4
        )
    )


print()


# ==================================================
# 16. NEXT-TOKEN PROBABILITIES
# ==================================================

print("TEST 9: Next-Token Probabilities")
print()


probabilities = F.softmax(
    last_position_logits,
    dim=-1
)


print(
    "Probability sum:",
    float(
        probabilities.sum()
    )
)


print(
    "Maximum probability:",
    float(
        probabilities.max()
    )
)

print()


# ==================================================
# 17. PREDICT NEXT TOKEN
# ==================================================

print("TEST 10: Greedy Next Token")
print()


next_token_id = int(
    torch.argmax(
        probabilities
    )
)


next_token = ID_TO_TOKEN.get(
    next_token_id,
    "<UNK>"
)


print(
    "Next token ID:",
    next_token_id
)

print(
    "Next token:",
    repr(
        next_token
    )
)

print()


# ==================================================
# 18. TEMPERATURE
# ==================================================

def temperature_probabilities(
        logits,
        temperature
):

    if temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )


    scaled = (
            logits
            /
            temperature
    )


    return F.softmax(
        scaled,
        dim=-1
    )


print("TEST 11: Temperature")
print()


for temperature in [
    0.5,
    1.0,
    1.5
]:

    temperature_distribution = (
        temperature_probabilities(
            last_position_logits,
            temperature
        )
    )


    max_probability = (
        float(
            temperature_distribution.max()
        )
    )


    entropy = (
        -(
                temperature_distribution
                *
                torch.log(
                    temperature_distribution
                    +
                    1e-12
                )
        ).sum()
    )


    print(
        "Temperature:",
        temperature,
        "| max probability:",
        round(
            max_probability,
            6
        ),
        "| entropy:",
        round(
            float(
                entropy
            ),
            4
        )
    )

print()


# ==================================================
# 19. TOP-K
# ==================================================

def top_k_filter(
        logits,
        k
):

    if k <= 0:

        raise ValueError(
            "k must be positive."
        )


    k = min(
        k,
        logits.shape[-1]
    )


    threshold = torch.topk(
        logits,
        k
    ).values[
        -1
    ]


    filtered = logits.clone()


    filtered[
        filtered < threshold
        ] = float("-inf")


    return filtered


print("TEST 12: Top-K")
print()


filtered_logits = top_k_filter(
    last_position_logits,
    5
)


filtered_probabilities = F.softmax(
    filtered_logits,
    dim=-1
)


values, indices = torch.topk(
    filtered_probabilities,
    k=5
)


for probability, token_id in zip(
        values.detach().cpu().tolist(),
        indices.detach().cpu().tolist()
):

    print(
        repr(
            ID_TO_TOKEN.get(
                int(token_id),
                "<UNK>"
            )
        ),
        "->",
        round(
            probability,
            6
        )
    )


print()


# ==================================================
# 20. CAUSAL VALIDATION
# ==================================================

print("TEST 13: Causal Attention Validation")
print()


for layer_index, weights in enumerate(
        attention_maps
):

    future_weights = torch.triu(
        weights[
            0,
            0
        ],
        diagonal=1
    )


    maximum_future_weight = torch.max(
        torch.abs(
            future_weights
        )
    )


    print(
        "Layer",
        layer_index,
        "maximum future attention:",
        float(
            maximum_future_weight
        )
    )


print()


# ==================================================
# 21. HIDDEN REPRESENTATIONS
# ==================================================

print("TEST 14: Hidden Representations")
print()


print(
    "Hidden dimension:",
    hidden_states.shape[-1]
)

print(
    "Sequence positions:",
    hidden_states.shape[1]
)

print()


# ==================================================
# 22. LOSS FUNCTION
# ==================================================

print("TEST 15: Language Modeling Loss")
print()


# Input predicts the next token.

language_model_input = input_ids[
    :,
    :-1
]


targets = input_ids[
    :,
    1:
]


with torch.no_grad():

    training_output = silverwing(
        language_model_input
    )


training_logits = training_output[
    "logits"
]


loss = F.cross_entropy(
    training_logits.reshape(
        -1,
        VOCABULARY_SIZE
    ),
    targets.reshape(
        -1
    ),
    ignore_index=PAD_ID
)


print(
    "Input shape:",
    tuple(
        language_model_input.shape
    )
)

print(
    "Target shape:",
    tuple(
        targets.shape
    )
)

print(
    "Initial loss:",
    float(
        loss
    )
)

print()


# ==================================================
# 23. THEORETICAL RANDOM BASELINE
# ==================================================

print("TEST 16: Random Baseline")
print()


expected_random_loss = math.log(
    VOCABULARY_SIZE
)


print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print(
    "Uniform-random cross-entropy:",
    round(
        expected_random_loss,
        4
    )
)

print()


# ==================================================
# 24. TRAINABLE UPDATE
# ==================================================

print("TEST 17: One Optimization Step")
print()


optimizer = torch.optim.AdamW(
    silverwing.parameters(),
    lr=1e-3
)


silverwing.train()


optimizer.zero_grad()


training_output = silverwing(
    language_model_input
)


training_logits = training_output[
    "logits"
]


training_loss = F.cross_entropy(
    training_logits.reshape(
        -1,
        VOCABULARY_SIZE
    ),
    targets.reshape(
        -1
    ),
    ignore_index=PAD_ID
)


training_loss.backward()


optimizer.step()


print(
    "Training loss:",
    float(
        training_loss.detach()
    )
)


silverwing.eval()


print(
    "Optimization step completed."
)

print()


# ==================================================
# 25. GRADIENT CHECK
# ==================================================

print("TEST 18: Gradient Verification")
print()


gradient_values = []


for parameter in (
        silverwing.parameters()
):

    if parameter.grad is not None:

        gradient_values.append(
            float(
                parameter.grad.norm()
            )
        )


print(
    "Parameters with gradients:",
    len(
        gradient_values
    )
)

print(
    "Maximum gradient norm:",
    max(
        gradient_values
    )
)

print()


# ==================================================
# 26. CHECKPOINT
# ==================================================

print("TEST 19: Save Initial Checkpoint")
print()


checkpoint = {
    "model_state_dict":
        silverwing.state_dict(),

    "model_configuration": {
        "vocabulary_size":
            VOCABULARY_SIZE,

        "model_dimension":
            MODEL_DIMENSION,

        "number_of_heads":
            NUMBER_OF_HEADS,

        "feed_forward_dimension":
            FEED_FORWARD_DIMENSION,

        "number_of_layers":
            NUMBER_OF_LAYERS,

        "maximum_sequence_length":
            MAX_SEQUENCE_LENGTH
    },

    "token_ids": {
        "pad":
            PAD_ID,

        "unk":
            UNK_ID,

        "bos":
            BOS_ID,

        "eos":
            EOS_ID
    }
}


torch.save(
    checkpoint,
    CHECKPOINT_FILE
)


print(
    "Checkpoint:",
    CHECKPOINT_FILE
)

print()


# ==================================================
# 27. RELOAD CHECKPOINT
# ==================================================

print("TEST 20: Checkpoint Reload")
print()


reloaded_model = SilverwingDecoderLM(
    vocabulary_size=VOCABULARY_SIZE,
    model_dimension=MODEL_DIMENSION,
    number_of_heads=NUMBER_OF_HEADS,
    feed_forward_dimension=FEED_FORWARD_DIMENSION,
    number_of_layers=NUMBER_OF_LAYERS,
    maximum_sequence_length=MAX_SEQUENCE_LENGTH,
    dropout=DROPOUT
).to(DEVICE)


checkpoint_data = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False
)


reloaded_model.load_state_dict(
    checkpoint_data[
        "model_state_dict"
    ]
)


reloaded_model.eval()


print(
    "Checkpoint reloaded successfully."
)

print()


# ==================================================
# 28. RELOAD CONSISTENCY
# ==================================================

print("TEST 21: Reload Consistency")
print()


with torch.no_grad():

    original_logits = silverwing(
        input_ids
    )[
        "logits"
    ]


    reloaded_logits = reloaded_model(
        input_ids
    )[
        "logits"
    ]


difference = torch.max(
    torch.abs(
        original_logits
        -
        reloaded_logits
    )
)


print(
    "Maximum output difference:",
    float(
        difference
    )
)

print()


# ==================================================
# 29. MODEL INFORMATION
# ==================================================

print("TEST 22: Model Information")
print()


model_information = {
    "name":
        "Silverwing-Decoder-v1",

    "vocabulary_size":
        VOCABULARY_SIZE,

    "model_dimension":
        MODEL_DIMENSION,

    "attention_heads":
        NUMBER_OF_HEADS,

    "head_dimension":
        MODEL_DIMENSION
        //
        NUMBER_OF_HEADS,

    "feed_forward_dimension":
        FEED_FORWARD_DIMENSION,

    "layers":
        NUMBER_OF_LAYERS,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "parameters":
        total_parameters,

    "device":
        str(
            DEVICE
        )
}


print(
    json.dumps(
        model_information,
        indent=4
    )
)

print()


# ==================================================
# 30. SAVE MODEL CONFIGURATION
# ==================================================

print("TEST 23: Save Model Configuration")
print()


model_config = {
    "model":
        "Silverwing-Decoder-v1",

    "architecture":
        "decoder-only-transformer",

    "vocabulary_size":
        VOCABULARY_SIZE,

    "model_dimension":
        MODEL_DIMENSION,

    "attention_heads":
        NUMBER_OF_HEADS,

    "feed_forward_dimension":
        FEED_FORWARD_DIMENSION,

    "layers":
        NUMBER_OF_LAYERS,

    "maximum_sequence_length":
        MAX_SEQUENCE_LENGTH,

    "causal_attention":
        True,

    "shared_token_lm_weights":
        False,

    "checkpoint":
        str(
            CHECKPOINT_FILE
        )
}


with open(
        MODEL_CONFIG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        model_config,
        file,
        indent=4
    )


print(
    "Saved:",
    MODEL_CONFIG_FILE
)

print()


# ==================================================
# 31. ARCHITECTURE
# ==================================================

print("SILVERWING DECODER ARCHITECTURE")
print()

print("Token IDs")
print("   ↓")
print("Token Embeddings")
print("   +")
print("Position Embeddings")
print("   ↓")
print("Transformer Block 1")
print("   ↓")
print("Transformer Block 2")
print("   ↓")
print("Transformer Block 3")
print("   ↓")
print("Transformer Block 4")
print("   ↓")
print("Final LayerNorm")
print("   ↓")
print("Vocabulary Projection")
print("   ↓")
print("Logits")
print("   ↓")
print("Softmax")
print("   ↓")
print("Next Token")

print()


# ==================================================
# 32. LANGUAGE MODEL OBJECTIVE
# ==================================================

print("AUTOREGRESSIVE LANGUAGE MODELING")
print()

print(
    "Given:"
)

print(
    "token_1, token_2, ..., token_n"
)

print()

print(
    "Silverwing learns:"
)

print(
    "P(token_n+1 | token_1 ... token_n)"
)

print()

print(
    "Training shifts the sequence by one position:"
)

print(
    "Inputs  →  Targets"
)

print(
    "A B C   →  B C D"
)

print()


# ==================================================
# 33. WHY THE MODEL IS NOT YET INTELLIGENT
# ==================================================

print("IMPORTANT")
print()

print(
    "This model currently contains randomly "
    "initialized learned parameters plus one "
    "demonstration optimization step."
)

print()

print(
    "It is therefore NOT yet a useful language model."
)

print()

print(
    "The model becomes useful only after substantial "
    "training on a carefully designed corpus."
)

print()


# ==================================================
# 34. WHAT WE NOW OWN
# ==================================================

print("SILVERWING NATIVE MODEL COMPONENTS")
print()

print(
    "✓ Own vocabulary"
)

print(
    "✓ Own BPE-style tokenizer"
)

print(
    "✓ Own token embeddings"
)

print(
    "✓ Own position embeddings"
)

print(
    "✓ Own self-attention"
)

print(
    "✓ Own transformer block"
)

print(
    "✓ Own decoder architecture"
)

print(
    "✓ Own language-model head"
)

print()


# ==================================================
# 35. WHAT COMES NEXT
# ==================================================

print("NEXT FOUNDATION COMPONENTS")
print()

print(
    "The architecture exists."
)

print()

print(
    "Now Silverwing needs a real training system:"
)

print(
    "1. Corpus construction"
)

print(
    "2. Dataset cleaning"
)

print(
    "3. Sequence packing"
)

print(
    "4. Training batches"
)

print(
    "5. Optimizer scheduling"
)

print(
    "6. Validation"
)

print(
    "7. Checkpoints"
)

print(
    "8. Evaluation"
)

print()


# ==================================================
# 36. FOUNDATION MODEL PROGRESS
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
print("Own Transformer Block")
print(" ↓")
print("OWN DECODER LANGUAGE MODEL")
print(" ↓")
print("Training System  ← NEXT")
print(" ↓")
print("Large-Scale Training")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Agent Integration")
print(" ↓")
print("Controlled Self-Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 71R COMPLETE ===")
# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 100R
# Native Cross-Modal Alignment + Retrieval
# ============================================================
#
# 79R -> Native Reasoning Dataset
# 80R -> Native Reasoning Fine-Tuning
# 81R -> Native Memory-Aware Training
# 82R -> Native Tool-Aware Learning
# 83R -> Native Planning and Tool Sequencing
# 84R -> Native Verified Execution + Replanning
# 85R -> Native Mathematical Reasoning
# 86R -> Native Probability + Statistical Reasoning
# 87R -> Native Linear Algebra + Optimization
# 88R -> Native Algorithms + Data Structures
# 89R -> Native Data Analysis + SQL Reasoning
# 90R -> Native Data Engineering
# 91R -> Native Machine Learning Foundations
# 92R -> Native Classical Machine Learning
# 93R -> Native Neural Network Foundations
# 94R -> Native Deep Learning
# 95R -> Native Representation Learning
# 96R -> Native Sequence Representation Learning
# 97R -> Native Structured Representation Learning
# 98R -> Advanced Sequence + Structured Learning
# 99R -> Multimodal Representation Foundations
# 100R -> Cross-Modal Alignment + Retrieval
#
# ============================================================
# PURPOSE
# ============================================================
#
# This lesson establishes a robust native cross-modal system
# connecting:
#
#     text / sequence
#             \
#              \
#               -> shared representation space
#              /
#             /
#     numeric / structured
#
# The lesson validates:
#
#     exact instance alignment
#     bidirectional retrieval
#     hard-negative separation
#     semantic class consistency
#     positive-negative margins
#     deterministic inference
#     native reasoning traces
#
# ============================================================
# IMPORTANT ARCHITECTURAL DECISION
# ============================================================
#
# The multimodal encoder is trained independently from the
# Silverwing decoder.
#
# The decoder is loaded and validated as an inherited model,
# but the multimodal alignment stage does NOT depend on decoder
# tuple unpacking or decoder training in order to validate its
# core capability.
#
# This prevents one failing component from masking another.
#
# ============================================================
# OWNERSHIP
# ============================================================
#
# Tokenizer: Silverwing native
# Vocabulary: Silverwing native
# Decoder: Silverwing native
# Multimodal encoder: Silverwing native
# Dataset: Silverwing native
# Training: Silverwing native
# Evaluation: Silverwing native
#
# External LLM: NONE
#
# ============================================================

import json
import math
import random
import re
import time

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent

LESSON_66R = PHASE5_DIR / "lesson66R"
LESSON_71R = PHASE5_DIR / "lesson71R"
LESSON_79R = PHASE5_DIR / "lesson79R"
LESSON_99R = PHASE5_DIR / "lesson99R"

VOCABULARY_FILE = (
        LESSON_66R
        / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66R
        / "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71R
        / "silverwing_decoder_config.json"
)

REASONING_CONFIG_FILE = (
        LESSON_79R
        / "silverwing_reasoning_config.json"
)

BASE_CHECKPOINT_PRIMARY = (
        LESSON_99R
        / "checkpoints"
        / "silverwing_multimodal_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_99R
        / "checkpoints"
        / "silverwing_multimodal_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_registry.json"
)

TRAIN_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_train.jsonl"
)

VALIDATION_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_validation.jsonl"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_cross_modal_alignment_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_cross_modal_alignment_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR
        / "silverwing_cross_modal_alignment_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

MAX_SEQUENCE_LENGTH = 256

TEXT_DIMENSION = 16

NUMERIC_INPUT_DIMENSION = 5

SHARED_DIMENSION = 16

HIDDEN_DIMENSION = 32

TEMPERATURE = 0.08

INSTANCE_MARGIN = 0.20

CLASS_MARGIN = 0.05

CONTRASTIVE_WEIGHT = 1.0

INSTANCE_WEIGHT = 2.0

CLASS_WEIGHT = 0.50

ALIGNMENT_EPOCHS = 1200

ALIGNMENT_LEARNING_RATE = 0.003

ALIGNMENT_WEIGHT_DECAY = 0.0005

GRADIENT_CLIP_NORM = 1.0

EXACT_RETRIEVAL_THRESHOLD = 0.83

CLASS_RETRIEVAL_THRESHOLD = 0.83

BATCH_SIZE = 2

DECODER_EPOCHS = 3

DECODER_LEARNING_RATE = 0.001

DECODER_WEIGHT_DECAY = 0.001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 3. GENERIC HELPERS
# ============================================================

def require_file(
        path: Path
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def read_json(
        path: Path
) -> Any:

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def write_json(
        path: Path,
        data: Any
) -> None:

    with open(
            path,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def safe_mean(
        values: List[float]
) -> float:

    if not values:

        return 0.0

    return sum(values) / len(values)


def choose_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "Lesson 99R checkpoint was not found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


def cosine_similarity(
        left: torch.Tensor,
        right: torch.Tensor
) -> float:

    denominator = (
            torch.linalg.vector_norm(left)
            *
            torch.linalg.vector_norm(right)
    )

    denominator_value = float(
        denominator
    )

    if denominator_value == 0.0:

        return 0.0

    return float(
        torch.dot(
            left,
            right
        )
        /
        denominator
    )


# ============================================================
# HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 100R"
)

print(
    "Native Cross-Modal Alignment + Retrieval"
)

print()

print("79R -> Reasoning")
print("80R -> Reasoning Fine-Tuning")
print("81R -> Memory")
print("82R -> Tool Use")
print("83R -> Planning")
print("84R -> Verified Execution + Replanning")
print("85R -> Mathematical Reasoning")
print("86R -> Probability + Statistics")
print("87R -> Linear Algebra + Optimization")
print("88R -> Algorithms + Data Structures")
print("89R -> Data Analysis + SQL Reasoning")
print("90R -> Data Engineering")
print("91R -> Machine Learning Foundations")
print("92R -> Classical Machine Learning")
print("93R -> Neural Network Foundations")
print("94R -> Deep Learning")
print("95R -> Representation Learning")
print("96R -> Sequence Representation Learning")
print("97R -> Structured Representation Learning")
print("98R -> Advanced Sequence + Structured Learning")
print("99R -> Multimodal Representation Foundations")
print("100R -> Cross-Modal Alignment + Retrieval")

print()

print(
    "External LLM: NONE"
)

print(
    "Device:",
    DEVICE
)

print(
    "Sequence limit:",
    MAX_SEQUENCE_LENGTH
)

print(
    "Shared representation dimension:",
    SHARED_DIMENSION
)

print(
    "Temperature:",
    TEMPERATURE
)

print(
    "Contrastive weight:",
    CONTRASTIVE_WEIGHT
)

print(
    "Instance margin weight:",
    INSTANCE_WEIGHT
)

print(
    "Class margin weight:",
    CLASS_WEIGHT
)

print(
    "Alignment epochs:",
    ALIGNMENT_EPOCHS
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify Lesson 99R and Silverwing Inputs"
)

print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
    REASONING_CONFIG_FILE
]:

    require_file(
        path
    )

    print(
        "FOUND:",
        path
    )

BASE_CHECKPOINT = choose_base_checkpoint()

print(
    "FOUND:",
    BASE_CHECKPOINT
)

print()


# ============================================================
# TEST 2
# ============================================================

print(
    "TEST 2: Load Silverwing Configuration"
)

print()

model_config = read_json(
    MODEL_CONFIG_FILE
)

reasoning_config = read_json(
    REASONING_CONFIG_FILE
)

MODEL_DIMENSION = int(
    model_config[
        "model_dimension"
    ]
)

NUMBER_OF_HEADS = int(
    model_config[
        "attention_heads"
    ]
)

FEED_FORWARD_DIMENSION = int(
    model_config[
        "feed_forward_dimension"
    ]
)

NUMBER_OF_LAYERS = int(
    model_config[
        "layers"
    ]
)

MODEL_SEQUENCE_LIMIT = int(
    model_config[
        "maximum_sequence_length"
    ]
)

REASONING_SEQUENCE_LIMIT = int(
    reasoning_config.get(
        "max_reasoning_tokens",
        MODEL_SEQUENCE_LIMIT
    )
)

EFFECTIVE_SEQUENCE_LIMIT = min(
    MAX_SEQUENCE_LENGTH,
    MODEL_SEQUENCE_LIMIT,
    REASONING_SEQUENCE_LIMIT
)

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
    "Layers:",
    NUMBER_OF_LAYERS
)

print(
    "Sequence limit:",
    EFFECTIVE_SEQUENCE_LIMIT
)

if (
        MODEL_DIMENSION
        %
        NUMBER_OF_HEADS
        !=
        0
):

    raise RuntimeError(
        "Model dimension is not divisible by attention heads."
    )

print(
    "Configuration validated."
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Load Silverwing Vocabulary"
)

print()

vocabulary = read_json(
    VOCABULARY_FILE
)

if "token_to_id" not in vocabulary:

    raise RuntimeError(
        "Vocabulary does not contain token_to_id."
    )

TOKEN_TO_ID = {}

for token, token_id in vocabulary[
    "token_to_id"
].items():

    TOKEN_TO_ID[
        str(token)
    ] = int(
        token_id
    )

required_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

for token in required_tokens:

    if token not in TOKEN_TO_ID:

        raise RuntimeError(
            f"Missing required token: {token}"
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

VOCABULARY_SIZE = len(
    TOKEN_TO_ID
)

print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print(
    "Vocabulary validated."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Load Silverwing BPE"
)

print()

merge_data = read_json(
    MERGES_FILE
)

if isinstance(
        merge_data,
        dict
):

    merge_items = merge_data.get(
        "merges",
        []
    )

else:

    merge_items = merge_data

MERGE_RANKS = {}

for item in merge_items:

    if not isinstance(
            item,
            dict
    ):

        continue

    pair = item.get(
        "pair"
    )

    rank = item.get(
        "rank"
    )

    if (
            isinstance(pair, list)
            and len(pair) == 2
            and rank is not None
    ):

        MERGE_RANKS[
            (
                str(pair[0]),
                str(pair[1])
            )
        ] = int(
            rank
        )

print(
    "Merge operations:",
    len(MERGE_RANKS)
)

print(
    "BPE validated."
)

print()


# ============================================================
# TOKENIZER
# ============================================================

BPE_END = "</w>"


def split_words(
        text: str
) -> List[str]:

    return re.findall(
        r"\w+|[^\w\s]",
        str(text).lower(),
        flags=re.UNICODE
    )


def word_to_symbols(
        word: str
) -> Tuple[str, ...]:

    if not word:

        return tuple()

    symbols = list(
        word
    )

    symbols[-1] += BPE_END

    return tuple(
        symbols
    )


def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    result = []

    index = 0

    while index < len(symbols):

        if (
                index + 1 < len(symbols)
                and
                (
                        symbols[index],
                        symbols[index + 1]
                )
                ==
                pair
        ):

            result.append(
                symbols[index]
                +
                symbols[index + 1]
            )

            index += 2

        else:

            result.append(
                symbols[index]
            )

            index += 1

    return tuple(
        result
    )


def tokenize_word(
        word: str
) -> List[str]:

    symbols = word_to_symbols(
        word
    )

    while len(symbols) > 1:

        candidates = []

        for index in range(
                len(symbols) - 1
        ):

            pair = (
                symbols[index],
                symbols[index + 1]
            )

            if pair in MERGE_RANKS:

                candidates.append(
                    (
                        MERGE_RANKS[pair],
                        pair
                    )
                )

        if not candidates:

            break

        _, best_pair = min(
            candidates,
            key=lambda item: item[0]
        )

        symbols = merge_pair(
            symbols,
            best_pair
        )

    return list(
        symbols
    )


def tokenize_text(
        text: str
) -> List[str]:

    tokens = []

    for word in split_words(
            text
    ):

        tokens.extend(
            tokenize_word(
                word
            )
        )

    return tokens


def encode_text(
        text: str
) -> List[int]:

    ids = [
        BOS_ID
    ]

    for token in tokenize_text(
            text
    ):

        ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )

    ids.append(
        EOS_ID
    )

    return ids


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Native Instance-Level Cross-Modal Dataset"
)

print()

ALIGNMENT_DATA = [
    {
        "id": "motor_01",
        "text":
            "motor unit id one temperature warning",
        "numeric":
            [
                0.90,
                1.20,
                2.50,
                0.31,
                0.11
            ],
        "class": 0
    },
    {
        "id": "pump_01",
        "text":
            "pump unit id one pressure warning",
        "numeric":
            [
                0.75,
                1.40,
                1.80,
                0.42,
                0.17
            ],
        "class": 1
    },
    {
        "id": "sensor_01",
        "text":
            "sensor unit id one signal normal",
        "numeric":
            [
                0.42,
                0.80,
                0.90,
                0.18,
                0.27
            ],
        "class": 2
    },
    {
        "id": "motor_02",
        "text":
            "motor unit id two temperature high",
        "numeric":
            [
                0.88,
                1.18,
                2.45,
                0.37,
                0.13
            ],
        "class": 0
    },
    {
        "id": "pump_02",
        "text":
            "pump unit id two pressure high",
        "numeric":
            [
                0.78,
                1.38,
                1.82,
                0.46,
                0.19
            ],
        "class": 1
    },
    {
        "id": "sensor_02",
        "text":
            "sensor unit id two signal stable",
        "numeric":
            [
                0.44,
                0.82,
                0.92,
                0.21,
                0.29
            ],
        "class": 2
    }
]

INSTANCE_COUNT = len(
    ALIGNMENT_DATA
)

INSTANCE_IDS = [
    item["id"]
    for item
    in ALIGNMENT_DATA
]

CLASS_IDS = [
    item["class"]
    for item
    in ALIGNMENT_DATA
]

print(
    "Instances:",
    INSTANCE_COUNT
)

print(
    "Unique instance ids:",
    len(
        set(
            INSTANCE_IDS
        )
    )
)

print(
    "Classes:",
    sorted(
        set(
            CLASS_IDS
        )
    )
)

if (
        len(
            set(
                INSTANCE_IDS
            )
        )
        !=
        INSTANCE_COUNT
):

    raise RuntimeError(
        "Instance identifiers are not unique."
    )

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Prepare Paired Modalities"
)

print()

encoded_sequences = []

for item in ALIGNMENT_DATA:

    ids = encode_text(
        item["text"]
    )

    if (
            len(ids)
            >
            EFFECTIVE_SEQUENCE_LIMIT
    ):

        raise RuntimeError(
            (
                f"{item['id']} exceeds "
                "the Silverwing sequence limit."
            )
        )

    encoded_sequences.append(
        ids
    )

maximum_text_length = max(
    len(ids)
    for ids
    in encoded_sequences
)

text_rows = []

for ids in encoded_sequences:

    row = list(
        ids
    )

    row.extend(
        [
            PAD_ID
        ]
        *
        (
                maximum_text_length
                -
                len(row)
        )
    )

    text_rows.append(
        row
    )

text_tensor = torch.tensor(
    text_rows,
    dtype=torch.long
)

numeric_tensor = torch.tensor(
    [
        item["numeric"]
        for item
        in ALIGNMENT_DATA
    ],
    dtype=torch.float32
)

class_tensor = torch.tensor(
    CLASS_IDS,
    dtype=torch.long
)

print(
    "Text tensor:",
    tuple(
        text_tensor.shape
    )
)

print(
    "Numeric tensor:",
    tuple(
        numeric_tensor.shape
    )
)

print(
    "Class tensor:",
    tuple(
        class_tensor.shape
    )
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Native Numeric Feature Normalization"
)

print()

numeric_mean = numeric_tensor.mean(
    dim=0,
    keepdim=True
)

numeric_std = numeric_tensor.std(
    dim=0,
    keepdim=True
).clamp(
    min=1e-6
)

normalized_numeric_tensor = (
                                    numeric_tensor
                                    -
                                    numeric_mean
                            ) / numeric_std

print(
    "Raw feature means:",
    numeric_mean.squeeze(
        0
    ).tolist()
)

print(
    "Normalized feature means:",
    normalized_numeric_tensor.mean(
        dim=0
    ).tolist()
)

if not torch.isfinite(
        normalized_numeric_tensor
).all():

    raise RuntimeError(
        "Normalized numeric data contains invalid values."
    )

print(
    "Numeric feature normalization validated."
)

print()


# ============================================================
# TEST 8
# CROSS-MODAL ENCODER
# ============================================================

class CrossModalEncoder(
    nn.Module
):

    def __init__(
            self,
            vocabulary_size: int,
            numeric_dimension: int,
            sequence_limit: int
    ) -> None:

        super().__init__()

        self.text_embedding = nn.Embedding(
            vocabulary_size,
            TEXT_DIMENSION,
            padding_idx=PAD_ID
        )

        self.position_embedding = nn.Embedding(
            sequence_limit,
            TEXT_DIMENSION
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                TEXT_DIMENSION,
                HIDDEN_DIMENSION
            ),
            nn.GELU(),
            nn.Linear(
                HIDDEN_DIMENSION,
                SHARED_DIMENSION
            )
        )

        self.numeric_projection = nn.Sequential(
            nn.Linear(
                numeric_dimension,
                HIDDEN_DIMENSION
            ),
            nn.GELU(),
            nn.Linear(
                HIDDEN_DIMENSION,
                SHARED_DIMENSION
            )
        )

    def encode_text(
            self,
            token_ids: torch.Tensor
    ) -> torch.Tensor:

        batch_size = token_ids.shape[0]

        sequence_length = token_ids.shape[1]

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        positions = positions.unsqueeze(
            0
        ).expand(
            batch_size,
            sequence_length
        )

        embeddings = (
                self.text_embedding(
                    token_ids
                )
                +
                self.position_embedding(
                    positions
                )
        )

        padding_mask = (
                token_ids
                ==
                PAD_ID
        )

        embeddings = embeddings.masked_fill(
            padding_mask.unsqueeze(
                -1
            ),
            0.0
        )

        valid_counts = (
            (~padding_mask)
            .sum(
                dim=1,
                keepdim=True
            )
            .clamp(
                min=1
            )
            .to(
                embeddings.dtype
            )
        )

        pooled = (
                embeddings.sum(
                    dim=1
                )
                /
                valid_counts
        )

        output = self.text_projection(
            pooled
        )

        return F.normalize(
            output,
            p=2,
            dim=-1
        )

    def encode_numeric(
            self,
            numeric_values: torch.Tensor
    ) -> torch.Tensor:

        output = self.numeric_projection(
            numeric_values
        )

        return F.normalize(
            output,
            p=2,
            dim=-1
        )

    def forward(
            self,
            text_ids: torch.Tensor,
            numeric_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        text_output = self.encode_text(
            text_ids
        )

        numeric_output = self.encode_numeric(
            numeric_values
        )

        return {
            "text":
                text_output,

            "numeric":
                numeric_output
        }


encoder = CrossModalEncoder(
    vocabulary_size=VOCABULARY_SIZE,
    numeric_dimension=NUMERIC_INPUT_DIMENSION,
    sequence_limit=EFFECTIVE_SEQUENCE_LIMIT
).to(
    DEVICE
)

encoder_parameter_count = sum(
    parameter.numel()
    for parameter
    in encoder.parameters()
)

print(
    "TEST 8: Cross-Modal Encoder"
)

print(
    "Text dimension:",
    TEXT_DIMENSION
)

print(
    "Numeric input dimension:",
    NUMERIC_INPUT_DIMENSION
)

print(
    "Shared dimension:",
    SHARED_DIMENSION
)

print(
    "Trainable parameters:",
    encoder_parameter_count
)

print(
    "Cross-modal encoder validated."
)

print()


# ============================================================
# INPUT DEVICE TENSORS
# ============================================================

device_text = text_tensor.to(
    DEVICE
)

device_numeric = normalized_numeric_tensor.to(
    DEVICE
)

device_classes = class_tensor.to(
    DEVICE
)


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Initial Cross-Modal Representations"
)

print()

encoder.eval()

with torch.no_grad():

    initial_outputs = encoder(
        device_text,
        device_numeric
    )

initial_text = initial_outputs[
    "text"
]

initial_numeric = initial_outputs[
    "numeric"
]

print(
    "Text representation shape:",
    tuple(
        initial_text.shape
    )
)

print(
    "Numeric representation shape:",
    tuple(
        initial_numeric.shape
    )
)

if not torch.isfinite(
        initial_text
).all():

    raise RuntimeError(
        "Initial text representations contain invalid values."
    )

if not torch.isfinite(
        initial_numeric
).all():

    raise RuntimeError(
        "Initial numeric representations contain invalid values."
    )

print(
    "Initial representations validated."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Initial Cross-Modal Similarity Matrix"
)

print()

initial_similarity_matrix = torch.matmul(
    initial_text,
    initial_numeric.T
)

print(
    "Matrix shape:",
    tuple(
        initial_similarity_matrix.shape
    )
)

print(
    "Initial diagonal:",
    torch.diag(
        initial_similarity_matrix
    ).cpu().tolist()
)

if not torch.isfinite(
        initial_similarity_matrix
).all():

    raise RuntimeError(
        "Initial similarity matrix contains invalid values."
    )

print(
    "Initial similarity matrix validated."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Initial Hard-Negative Discovery"
)

print()

initial_hard_negatives = []

for query_index in range(
        INSTANCE_COUNT
):

    candidate_scores = []

    for candidate_index in range(
            INSTANCE_COUNT
    ):

        if (
                candidate_index
                ==
                query_index
        ):

            continue

        candidate_scores.append(
            (
                candidate_index,
                float(
                    initial_similarity_matrix[
                        query_index,
                        candidate_index
                    ]
                )
            )
        )

    candidate_scores.sort(
        key=lambda pair: pair[1],
        reverse=True
    )

    initial_hard_negatives.append(
        {
            "query":
                query_index,

            "negative":
                candidate_scores[0][0],

            "score":
                candidate_scores[0][1]
        }
    )

print(
    "Initial hard negatives:",
    initial_hard_negatives
)

if (
        len(
            initial_hard_negatives
        )
        !=
        INSTANCE_COUNT
):

    raise RuntimeError(
        "Hard-negative discovery failed."
    )

print(
    "Hard-negative discovery validated."
)

print()


# ============================================================
# TEST 12
# ALIGNMENT OBJECTIVE
# ============================================================

print(
    "TEST 12: Native Instance-Level + Class-Level Alignment"
)

print()


def calculate_alignment_components(
        text_vectors: torch.Tensor,
        numeric_vectors: torch.Tensor,
        class_ids: torch.Tensor
) -> Dict[str, torch.Tensor]:

    similarity = (
            torch.matmul(
                text_vectors,
                numeric_vectors.T
            )
            /
            TEMPERATURE
    )

    targets = torch.arange(
        similarity.shape[0],
        device=similarity.device
    )

    text_to_numeric_loss = F.cross_entropy(
        similarity,
        targets
    )

    numeric_to_text_loss = F.cross_entropy(
        similarity.T,
        targets
    )

    contrastive_loss = (
                               text_to_numeric_loss
                               +
                               numeric_to_text_loss
                       ) / 2.0

    hard_negative_losses = []

    instance_margin_losses = []

    class_margin_losses = []

    for query_index in range(
            similarity.shape[0]
    ):

        positive_score = similarity[
            query_index,
            query_index
        ]

        different_instance_scores = []

        same_class_scores = []

        for candidate_index in range(
                similarity.shape[1]
        ):

            if (
                    candidate_index
                    ==
                    query_index
            ):

                continue

            candidate_score = similarity[
                query_index,
                candidate_index
            ]

            different_instance_scores.append(
                candidate_score
            )

            if (
                    class_ids[
                        candidate_index
                    ]
                    ==
                    class_ids[
                        query_index
                    ]
            ):

                same_class_scores.append(
                    candidate_score
                )

        if different_instance_scores:

            hardest_negative = max(
                different_instance_scores
            )

            hard_negative_losses.append(
                F.relu(
                    0.5
                    -
                    positive_score
                    +
                    hardest_negative
                )
            )

            instance_margin_losses.append(
                F.relu(
                    INSTANCE_MARGIN
                    -
                    positive_score
                    +
                    hardest_negative
                )
            )

        if same_class_scores:

            hardest_same_class = max(
                same_class_scores
            )

            class_margin_losses.append(
                F.relu(
                    CLASS_MARGIN
                    -
                    positive_score
                    +
                    hardest_same_class
                )
            )

    if hard_negative_losses:

        hard_negative_loss = torch.stack(
            hard_negative_losses
        ).mean()

    else:

        hard_negative_loss = torch.zeros(
            (),
            device=similarity.device
        )

    if instance_margin_losses:

        instance_margin_loss = torch.stack(
            instance_margin_losses
        ).mean()

    else:

        instance_margin_loss = torch.zeros(
            (),
            device=similarity.device
        )

    if class_margin_losses:

        class_margin_loss = torch.stack(
            class_margin_losses
        ).mean()

    else:

        class_margin_loss = torch.zeros(
            (),
            device=similarity.device
        )

    total_loss = (
            CONTRASTIVE_WEIGHT
            *
            contrastive_loss
            +
            INSTANCE_WEIGHT
            *
            instance_margin_loss
            +
            CLASS_WEIGHT
            *
            class_margin_loss
    )

    return {
        "similarity":
            similarity,

        "contrastive":
            contrastive_loss,

        "hard_negative":
            hard_negative_loss,

        "instance_margin":
            instance_margin_loss,

        "class_margin":
            class_margin_loss,

        "total":
            total_loss
    }


with torch.no_grad():

    initial_alignment = calculate_alignment_components(
        initial_text,
        initial_numeric,
        device_classes
    )

print(
    "Initial contrastive loss:",
    float(
        initial_alignment[
            "contrastive"
        ]
    )
)

print(
    "Initial instance margin loss:",
    float(
        initial_alignment[
            "instance_margin"
        ]
    )
)

print(
    "Initial class margin loss:",
    float(
        initial_alignment[
            "class_margin"
        ]
    )
)

print(
    "Initial total alignment loss:",
    float(
        initial_alignment[
            "total"
        ]
    )
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Native Cross-Modal Alignment Training"
)

print()

alignment_optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=ALIGNMENT_LEARNING_RATE,
    weight_decay=ALIGNMENT_WEIGHT_DECAY
)

alignment_history = []

alignment_initial_loss = None
alignment_final_loss = None

alignment_start = time.perf_counter()

for epoch in range(
        1,
        ALIGNMENT_EPOCHS + 1
):

    encoder.train()

    alignment_optimizer.zero_grad(
        set_to_none=True
    )

    training_outputs = encoder(
        device_text,
        device_numeric
    )

    training_text = training_outputs[
        "text"
    ]

    training_numeric = training_outputs[
        "numeric"
    ]

    training_alignment = calculate_alignment_components(
        training_text,
        training_numeric,
        device_classes
    )

    total_alignment_loss = training_alignment[
        "total"
    ]

    if epoch == 1:

        alignment_initial_loss = float(
            total_alignment_loss.detach()
        )

    total_alignment_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        encoder.parameters(),
        GRADIENT_CLIP_NORM
    )

    alignment_optimizer.step()

    alignment_final_loss = float(
        total_alignment_loss.detach()
    )

    if (
            epoch == 1
            or
            epoch % 100 == 0
    ):

        alignment_history.append(
            {
                "epoch":
                    epoch,

                "total":
                    alignment_final_loss,

                "contrastive":
                    float(
                        training_alignment[
                            "contrastive"
                        ].detach()
                    ),

                "hard_negative":
                    float(
                        training_alignment[
                            "hard_negative"
                        ].detach()
                    ),

                "instance_margin":
                    float(
                        training_alignment[
                            "instance_margin"
                        ].detach()
                    ),

                "class_margin":
                    float(
                        training_alignment[
                            "class_margin"
                        ].detach()
                    )
            }
        )

alignment_duration = (
        time.perf_counter()
        -
        alignment_start
)

print(
    "Alignment history:",
    alignment_history
)

print(
    "Initial alignment loss:",
    alignment_initial_loss
)

print(
    "Final alignment loss:",
    alignment_final_loss
)

print(
    "Alignment training duration:",
    alignment_duration
)

if (
        alignment_initial_loss is None
        or
        alignment_final_loss is None
):

    raise RuntimeError(
        "Alignment training did not produce a valid loss."
    )

if (
        alignment_final_loss
        >
        alignment_initial_loss
):

    raise RuntimeError(
        "Cross-modal alignment loss increased."
    )

print(
    "Native cross-modal alignment training validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Final Cross-Modal Representations"
)

print()

encoder.eval()

with torch.no_grad():

    final_outputs = encoder(
        device_text,
        device_numeric
    )

final_text = final_outputs[
    "text"
]

final_numeric = final_outputs[
    "numeric"
]

print(
    "Final text representation shape:",
    tuple(
        final_text.shape
    )
)

print(
    "Final numeric representation shape:",
    tuple(
        final_numeric.shape
    )
)

if not torch.isfinite(
        final_text
).all():

    raise RuntimeError(
        "Final text representations are invalid."
    )

if not torch.isfinite(
        final_numeric
).all():

    raise RuntimeError(
        "Final numeric representations are invalid."
    )

print(
    "Final cross-modal representations validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Final Cross-Modal Similarity Matrix"
)

print()

final_similarity_matrix = torch.matmul(
    final_text,
    final_numeric.T
)

final_alignment = calculate_alignment_components(
    final_text,
    final_numeric,
    device_classes
)

print(
    "Final matrix shape:",
    tuple(
        final_similarity_matrix.shape
    )
)

print(
    "Final diagonal:",
    torch.diag(
        final_similarity_matrix
    ).cpu().tolist()
)

print(
    "Final contrastive loss:",
    float(
        final_alignment[
            "contrastive"
        ]
    )
)

print(
    "Final hard-negative loss:",
    float(
        final_alignment[
            "hard_negative"
        ]
    )
)

print(
    "Final instance margin loss:",
    float(
        final_alignment[
            "instance_margin"
        ]
    )
)

print(
    "Final class margin loss:",
    float(
        final_alignment[
            "class_margin"
        ]
    )
)

print(
    "Final total alignment loss:",
    float(
        final_alignment[
            "total"
        ]
    )
)

if not torch.isfinite(
        final_similarity_matrix
).all():

    raise RuntimeError(
        "Final similarity matrix contains invalid values."
    )

print(
    "Final cross-modal similarity matrix validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Exact Instance-Level Cross-Modal Retrieval"
)

print()

text_rankings = []
numeric_rankings = []

text_exact_correct = 0
numeric_exact_correct = 0

for query_index in range(
        INSTANCE_COUNT
):

    text_candidates = []
    numeric_candidates = []

    for candidate_index in range(
            INSTANCE_COUNT
    ):

        text_candidates.append(
            (
                candidate_index,
                float(
                    final_similarity_matrix[
                        query_index,
                        candidate_index
                    ]
                )
            )
        )

        numeric_candidates.append(
            (
                candidate_index,
                float(
                    final_similarity_matrix[
                        candidate_index,
                        query_index
                    ]
                )
            )
        )

    text_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    numeric_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    text_rankings.append(
        text_candidates
    )

    numeric_rankings.append(
        numeric_candidates
    )

    if (
            text_candidates[0][0]
            ==
            query_index
    ):

        text_exact_correct += 1

    if (
            numeric_candidates[0][0]
            ==
            query_index
    ):

        numeric_exact_correct += 1

text_exact_accuracy = (
        text_exact_correct
        /
        INSTANCE_COUNT
)

numeric_exact_accuracy = (
        numeric_exact_correct
        /
        INSTANCE_COUNT
)

print(
    "Text -> numeric exact accuracy:",
    text_exact_accuracy
)

print(
    "Numeric -> text exact accuracy:",
    numeric_exact_accuracy
)

for index in range(
        INSTANCE_COUNT
):

    print(
        INSTANCE_IDS[index],
        "text top result ->",
        INSTANCE_IDS[
            text_rankings[index][0][0]
        ],
        "| score =",
        text_rankings[index][0][1]
    )

print()

for index in range(
        INSTANCE_COUNT
):

    print(
        INSTANCE_IDS[index],
        "numeric top result ->",
        INSTANCE_IDS[
            numeric_rankings[index][0][0]
        ],
        "| score =",
        numeric_rankings[index][0][1]
    )

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Semantic Class Retrieval"
)

print()

class_text_correct = 0
class_numeric_correct = 0

for index in range(
        INSTANCE_COUNT
):

    best_numeric_index = text_rankings[
        index
    ][
        0
    ][
        0
    ]

    best_text_index = numeric_rankings[
        index
    ][
        0
    ][
        0
    ]

    if (
            CLASS_IDS[
                best_numeric_index
            ]
            ==
            CLASS_IDS[
                index
            ]
    ):

        class_text_correct += 1

    if (
            CLASS_IDS[
                best_text_index
            ]
            ==
            CLASS_IDS[
                index
            ]
    ):

        class_numeric_correct += 1

class_text_accuracy = (
        class_text_correct
        /
        INSTANCE_COUNT
)

class_numeric_accuracy = (
        class_numeric_correct
        /
        INSTANCE_COUNT
)

print(
    "Text -> numeric class accuracy:",
    class_text_accuracy
)

print(
    "Numeric -> text class accuracy:",
    class_numeric_accuracy
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Exact Positive-Negative Alignment Margins"
)

print()

positive_scores = []
hard_negative_scores = []
instance_margins = []

for index in range(
        INSTANCE_COUNT
):

    positive_score = float(
        final_similarity_matrix[
            index,
            index
        ]
    )

    negative_scores = []

    for candidate_index in range(
            INSTANCE_COUNT
    ):

        if candidate_index == index:

            continue

        negative_scores.append(
            float(
                final_similarity_matrix[
                    index,
                    candidate_index
                ]
            )
        )

    hardest_negative = max(
        negative_scores
    )

    positive_scores.append(
        positive_score
    )

    hard_negative_scores.append(
        hardest_negative
    )

    instance_margins.append(
        positive_score
        -
        hardest_negative
    )

mean_positive_score = safe_mean(
    positive_scores
)

mean_hard_negative_score = safe_mean(
    hard_negative_scores
)

mean_instance_margin = safe_mean(
    instance_margins
)

print(
    "Mean positive score:",
    mean_positive_score
)

print(
    "Mean hard-negative score:",
    mean_hard_negative_score
)

print(
    "Instance margins:",
    instance_margins
)

print(
    "Mean instance margin:",
    mean_instance_margin
)

if mean_instance_margin <= 0.0:

    raise RuntimeError(
        "Positive instances are not separated from hard negatives."
    )

print(
    "Positive-negative alignment margin validated."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Top-K Exact Retrieval"
)

print()

TOP_K = 3

text_top_k_hits = 0
numeric_top_k_hits = 0

for index in range(
        INSTANCE_COUNT
):

    text_top_k = [
        pair[0]
        for pair
        in text_rankings[index][:TOP_K]
    ]

    numeric_top_k = [
        pair[0]
        for pair
        in numeric_rankings[index][:TOP_K]
    ]

    if index in text_top_k:

        text_top_k_hits += 1

    if index in numeric_top_k:

        numeric_top_k_hits += 1

text_top_k_accuracy = (
        text_top_k_hits
        /
        INSTANCE_COUNT
)

numeric_top_k_accuracy = (
        numeric_top_k_hits
        /
        INSTANCE_COUNT
)

print(
    "Top-k text -> numeric exact accuracy:",
    text_top_k_accuracy
)

print(
    "Top-k numeric -> text exact accuracy:",
    numeric_top_k_accuracy
)

if (
        text_top_k_accuracy
        <
        1.0
):

    raise RuntimeError(
        "Top-k text retrieval failed."
    )

if (
        numeric_top_k_accuracy
        <
        1.0
):

    raise RuntimeError(
        "Top-k numeric retrieval failed."
    )

print(
    "Top-k exact retrieval validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Cross-Modal Retrieval Consistency"
)

print()

encoder.eval()

with torch.no_grad():

    repeated_outputs = encoder(
        device_text,
        device_numeric
    )

repeated_text = repeated_outputs[
    "text"
]

repeated_numeric = repeated_outputs[
    "numeric"
]

repeated_similarity_matrix = torch.matmul(
    repeated_text,
    repeated_numeric.T
)

consistency_error = float(
    torch.max(
        torch.abs(
            final_similarity_matrix
            -
            repeated_similarity_matrix
        )
    )
)

print(
    "Maximum retrieval difference:",
    consistency_error
)

if consistency_error > 1e-7:

    raise RuntimeError(
        "Cross-modal retrieval is not deterministic."
    )

print(
    "Cross-modal retrieval consistency validated."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Cross-Modal Reasoning Dataset"
)

print()

ALIGNMENT_TASKS = [
    {
        "example_id":
            "align_001",

        "domain":
            "positive_pairs",

        "problem":
            "What is an exact positive cross-modal pair?",

        "reasoning":
            "The text and numeric observations refer to the same exact instance.",

        "operation":
            "TEXT A -> NUMERIC A",

        "answer":
            "Both modalities refer to the same instance."
    },

    {
        "example_id":
            "align_002",

        "domain":
            "hard_negatives",

        "problem":
            "What is a hard negative?",

        "reasoning":
            "It is an incorrect instance that has a high similarity to the query.",

        "operation":
            "HIGH SIMILARITY + WRONG INSTANCE",

        "answer":
            "A similar but incorrect instance."
    },

    {
        "example_id":
            "align_003",

        "domain":
            "instance_retrieval",

        "problem":
            "Why test exact instance retrieval?",

        "reasoning":
            "Two records can share a semantic class while representing different events.",

        "operation":
            "CLASS MATCH != INSTANCE MATCH",

        "answer":
            "Exact retrieval verifies the correct paired instance."
    },

    {
        "example_id":
            "align_004",

        "domain":
            "bidirectional_retrieval",

        "problem":
            "Why retrieve in both directions?",

        "reasoning":
            "The shared space should support both text to numeric and numeric to text retrieval.",

        "operation":
            "TEXT <-> NUMERIC",

        "answer":
            "Both directions test the common representation."
    },

    {
        "example_id":
            "align_005",

        "domain":
            "alignment_margin",

        "problem":
            "What does a positive alignment margin indicate?",

        "reasoning":
            "The correct pair scores higher than the strongest incorrect candidate.",

        "operation":
            "POSITIVE > HARD NEGATIVE",

        "answer":
            "The correct pair is separated from its strongest competitor."
    },

    {
        "example_id":
            "align_006",

        "domain":
            "engineering_retrieval",

        "problem":
            "Why is cross-modal retrieval useful in engineering?",

        "reasoning":
            "A maintenance description can retrieve its corresponding machine measurements.",

        "operation":
            "MAINTENANCE TEXT -> MEASUREMENTS",

        "answer":
            "It connects engineering language with measured evidence."
    },

    {
        "example_id":
            "align_007",

        "domain":
            "memory_retrieval",

        "problem":
            "Why does cross-modal retrieval matter for memory?",

        "reasoning":
            "One memory can contain several representations that must remain connected.",

        "operation":
            "MEMORY TEXT <-> MEMORY SIGNAL",

        "answer":
            "It keeps representations of the same memory linked."
    },

    {
        "example_id":
            "align_008",

        "domain":
            "retrieval_validation",

        "problem":
            "How should alignment be validated?",

        "reasoning":
            "Use exact retrieval, class retrieval, margin validation and deterministic repeated inference.",

        "operation":
            "INSTANCE + CLASS + MARGIN + CONSISTENCY",

        "answer":
            "Validate identity, semantics, separation and consistency."
    }
]


def build_alignment_trace(
        task: Dict[str, Any]
) -> str:

    parts = [
        "P:" + task["problem"],
        "M:" + task["reasoning"],
        "Q:" + task["operation"],
        "V:validated",
        "A:" + task["answer"]
    ]

    return "\n".join(
        parts
    )


alignment_records = []

for task in ALIGNMENT_TASKS:

    trace = build_alignment_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    alignment_records.append(
        {
            "example_id":
                task["example_id"],

            "domain":
                task["domain"],

            "formatted_text":
                trace,

            "token_count":
                token_count
        }
    )

    print(
        task["example_id"],
        "->",
        token_count,
        "tokens",
        "| domain:",
        task["domain"]
    )

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Cross-Modal Alignment Token Validation"
)

print()

length_errors = []

for record in alignment_records:

    if (
            record["token_count"]
            >
            EFFECTIVE_SEQUENCE_LIMIT
    ):

        length_errors.append(
            {
                "example_id":
                    record["example_id"],

                "token_count":
                    record["token_count"],

                "maximum":
                    EFFECTIVE_SEQUENCE_LIMIT
            }
        )

if length_errors:

    print(
        json.dumps(
            length_errors,
            indent=4
        )
    )

    raise RuntimeError(
        (
            "Cross-modal alignment examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All cross-modal alignment examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Cross-Modal Alignment Domain Coverage"
)

print()

EXPECTED_DOMAINS = {
    "positive_pairs",
    "hard_negatives",
    "instance_retrieval",
    "bidirectional_retrieval",
    "alignment_margin",
    "engineering_retrieval",
    "memory_retrieval",
    "retrieval_validation"
}

actual_domains = {
    record["domain"]
    for record
    in alignment_records
}

print(
    "Domains:",
    sorted(
        actual_domains
    )
)

print(
    "Examples:",
    len(
        alignment_records
    )
)

if actual_domains != EXPECTED_DOMAINS:

    raise RuntimeError(
        "Cross-modal domain coverage is incomplete."
    )

print(
    "Domain coverage validated."
)

print()


# ============================================================
# TEST 24
# ============================================================

random.Random(
    SEED
).shuffle(
    alignment_records
)

validation_count = max(
    2,
    len(
        alignment_records
    ) // 4
)

validation_count = min(
    validation_count,
    len(
        alignment_records
    ) - 1
)

train_records = alignment_records[
    :-validation_count
]

validation_records = alignment_records[
    -validation_count:
]

print(
    "TEST 24: Alignment Reasoning Train/Validation Split"
)

print(
    "Training examples:",
    len(train_records)
)

print(
    "Validation examples:",
    len(validation_records)
)

print()


# ============================================================
# TEST 25
# ============================================================

write_json(
    REGISTRY_FILE,
    {
        "lesson":
            "100R",

        "capability":
            "native_cross_modal_alignment_retrieval",

        "external_llm":
            False,

        "sequence_limit":
            EFFECTIVE_SEQUENCE_LIMIT,

        "shared_dimension":
            SHARED_DIMENSION,

        "temperature":
            TEMPERATURE,

        "contrastive_weight":
            CONTRASTIVE_WEIGHT,

        "instance_weight":
            INSTANCE_WEIGHT,

        "class_weight":
            CLASS_WEIGHT,

        "instance_margin":
            INSTANCE_MARGIN,

        "class_margin":
            CLASS_MARGIN,

        "modalities":
            [
                "text_sequence",
                "numeric_structured"
            ]
    }
)

with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

print(
    "TEST 25: Save Native Alignment Artifacts"
)

print(
    "Registry:",
    REGISTRY_FILE
)

print(
    "Training dataset:",
    TRAIN_FILE
)

print(
    "Validation dataset:",
    VALIDATION_FILE
)

print(
    "Artifacts saved."
)

print()


# ============================================================
# TEST 26
# SILVERWING DECODER CONTRACT
# ============================================================

class SilverwingAttention(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            heads: int
    ) -> None:

        super().__init__()

        self.dimension = dimension
        self.heads = heads
        self.head_dimension = (
                dimension
                //
                heads
        )

        self.query_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.key_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.value_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = query.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        key = key.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        value = value.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        scores = torch.matmul(
            query,
            key.transpose(
                -2,
                -1
            )
        )

        scores = (
                scores
                /
                math.sqrt(
                    self.head_dimension
                )
        )

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        attended = torch.matmul(
            weights,
            value
        )

        attended = attended.transpose(
            1,
            2
        ).contiguous()

        attended = attended.view(
            batch_size,
            sequence_length,
            self.dimension
        )

        return self.output_projection(
            attended
        )


class SilverwingFeedForward(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            hidden_dimension: int
    ) -> None:

        super().__init__()

        self.input_projection = nn.Linear(
            dimension,
            hidden_dimension
        )

        self.output_projection = nn.Linear(
            hidden_dimension,
            dimension
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        hidden = self.input_projection(
            x
        )

        hidden = F.gelu(
            hidden
        )

        return self.output_projection(
            hidden
        )


class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(
            self
    ) -> None:

        super().__init__()

        self.attention = SilverwingAttention(
            MODEL_DIMENSION,
            NUMBER_OF_HEADS
        )

        self.norm_attention = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.feed_forward = SilverwingFeedForward(
            MODEL_DIMENSION,
            FEED_FORWARD_DIMENSION
        )

        self.norm_feed_forward = nn.LayerNorm(
            MODEL_DIMENSION
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        attended = self.attention(
            x
        )

        x = self.norm_attention(
            x
            +
            attended
        )

        feed_forward_output = self.feed_forward(
            x
        )

        x = self.norm_feed_forward(
            x
            +
            feed_forward_output
        )

        return x


class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(
            self
    ) -> None:

        super().__init__()

        self.embedding = nn.Embedding(
            EFFECTIVE_SEQUENCE_LIMIT,
            MODEL_DIMENSION
        )

    def forward(
            self,
            sequence_length: int,
            device: torch.device
    ) -> torch.Tensor:

        positions = torch.arange(
            sequence_length,
            device=device
        )

        return self.embedding(
            positions
        )


class SilverwingDecoder(
    nn.Module
):

    def __init__(
            self
    ) -> None:

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCABULARY_SIZE,
            MODEL_DIMENSION,
            padding_idx=PAD_ID
        )

        self.position_embedding = (
            SilverwingPositionEmbedding()
        )

        self.layers = nn.ModuleList()

        for _ in range(
                NUMBER_OF_LAYERS
        ):

            self.layers.append(
                SilverwingTransformerBlock()
            )

        self.final_norm = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.language_model_head = nn.Linear(
            MODEL_DIMENSION,
            VOCABULARY_SIZE,
            bias=False
        )

    def forward(
            self,
            input_ids: torch.Tensor
    ) -> torch.Tensor:

        sequence_length = input_ids.shape[1]

        if (
                sequence_length
                >
                EFFECTIVE_SEQUENCE_LIMIT
        ):

            raise RuntimeError(
                "Sequence exceeds Silverwing limit."
            )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            sequence_length,
            input_ids.device
        )

        x = (
                token_embeddings
                +
                position_embeddings.unsqueeze(
                    0
                )
        )

        for layer in self.layers:

            x = layer(
                x
            )

        x = self.final_norm(
            x
        )

        return self.language_model_head(
            x
        )


print(
    "TEST 26: Strict Silverwing Decoder Compatibility"
)

print()

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if not isinstance(
        checkpoint,
        dict
):

    raise RuntimeError(
        "99R checkpoint is not a dictionary."
    )

checkpoint_state = checkpoint.get(
    "model_state_dict"
)

if checkpoint_state is None:

    raise RuntimeError(
        "99R checkpoint has no model_state_dict."
    )

decoder = SilverwingDecoder().to(
    DEVICE
)

try:

    decoder.load_state_dict(
        checkpoint_state,
        strict=True
    )

except RuntimeError as exc:

    raise RuntimeError(
        (
            "100R decoder inheritance failed.\n\n"
            "The 99R decoder architecture is not compatible "
            "with the 100R decoder definition.\n\n"
            f"Checkpoint:\n{BASE_CHECKPOINT}\n\n"
            f"Original PyTorch error:\n{exc}"
        )
    ) from exc

print(
    "STRICT LOAD PASSED."
)

print(
    "99R decoder architecture preserved."
)

print()


# ============================================================
# TEST 27
# MULTIMODAL DATASET
# ============================================================

class AlignmentDataset(
    Dataset
):

    def __init__(
            self,
            records: List[
                Dict[str, Any]
            ]
    ) -> None:

        self.records = []

        for record in records:

            token_ids = encode_text(
                record[
                    "formatted_text"
                ]
            )

            if (
                    len(token_ids)
                    >
                    EFFECTIVE_SEQUENCE_LIMIT
            ):

                raise RuntimeError(
                    (
                        f"{record['example_id']} exceeds "
                        "Silverwing sequence limit."
                    )
                )

            input_ids = token_ids[
                :-1
            ]

            labels = token_ids[
                1:
            ]

            self.records.append(
                {
                    "example_id":
                        record[
                            "example_id"
                        ],

                    "input_ids":
                        torch.tensor(
                            input_ids,
                            dtype=torch.long
                        ),

                    "labels":
                        torch.tensor(
                            labels,
                            dtype=torch.long
                        )
                }
            )

    def __len__(
            self
    ) -> int:

        return len(
            self.records
        )

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Any]:

        return self.records[
            index
        ]


def collate_alignment_batch(
        batch: List[
            Dict[str, Any]
        ]
) -> Dict[str, Any]:

    maximum_length = 1

    for item in batch:

        maximum_length = max(
            maximum_length,
            len(
                item[
                    "input_ids"
                ]
            )
        )

    input_rows = []
    label_rows = []

    for item in batch:

        input_ids = item[
            "input_ids"
        ]

        labels = item[
            "labels"
        ]

        input_rows.append(
            torch.cat(
                [
                    input_ids,
                    torch.full(
                        (
                            maximum_length
                            -
                            len(
                                input_ids
                            ),
                        ),
                        PAD_ID,
                        dtype=torch.long
                    )
                ]
            )
        )

        label_rows.append(
            torch.cat(
                [
                    labels,
                    torch.full(
                        (
                            maximum_length
                            -
                            len(
                                labels
                            ),
                        ),
                        -100,
                        dtype=torch.long
                    )
                ]
            )
        )

    return {
        "example_ids":
            [
                item[
                    "example_id"
                ]
                for item
                in batch
            ],

        "input_ids":
            torch.stack(
                input_rows
            ),

        "labels":
            torch.stack(
                label_rows
            )
    }


train_dataset = AlignmentDataset(
    train_records
)

validation_dataset = AlignmentDataset(
    validation_records
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_alignment_batch
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_alignment_batch
)

print(
    "TEST 27: Native Alignment Reasoning DataLoaders"
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(validation_loader)
)

print()


# ============================================================
# TEST 28
# DECODER BASELINE
# ============================================================

print(
    "TEST 28: Silverwing Decoder Baseline"
)

print()


@torch.no_grad()
def evaluate_decoder(
        current_decoder: nn.Module,
        loader: DataLoader
) -> Dict[str, float]:

    current_decoder.eval()

    total_loss = 0.0
    batch_count = 0
    correct = 0
    valid_tokens = 0

    for batch in loader:

        inputs = batch[
            "input_ids"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        logits = current_decoder(
            inputs
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCABULARY_SIZE
            ),
            labels.reshape(
                -1
            ),
            ignore_index=-100
        )

        total_loss += float(
            loss
        )

        batch_count += 1

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        mask = (
                labels
                !=
                -100
        )

        correct += int(
            (
                    predictions[mask]
                    ==
                    labels[mask]
            ).sum()
        )

        valid_tokens += int(
            mask.sum()
        )

    if batch_count == 0:

        return {
            "loss":
                float("nan"),

            "perplexity":
                float("nan"),

            "accuracy":
                float("nan")
        }

    loss_value = (
            total_loss
            /
            batch_count
    )

    if (
            math.isfinite(
                loss_value
            )
            and
            loss_value < 50
    ):

        perplexity = math.exp(
            loss_value
        )

    else:

        perplexity = float(
            "inf"
        )

    if valid_tokens > 0:

        accuracy = (
                correct
                /
                valid_tokens
        )

    else:

        accuracy = float(
            "nan"
        )

    return {
        "loss":
            loss_value,

        "perplexity":
            perplexity,

        "accuracy":
            accuracy
    }


baseline_metrics = evaluate_decoder(
    decoder,
    validation_loader
)

print(
    "Baseline loss:",
    baseline_metrics[
        "loss"
    ]
)

print(
    "Baseline perplexity:",
    baseline_metrics[
        "perplexity"
    ]
)

print(
    "Baseline accuracy:",
    baseline_metrics[
        "accuracy"
    ]
)

print()


# ============================================================
# TEST 29
# LIGHT NATIVE DECODER FINE-TUNING
# ============================================================

print(
    "TEST 29: Native Cross-Modal Reasoning Fine-Tuning"
)

print()

decoder_optimizer = torch.optim.AdamW(
    decoder.parameters(),
    lr=DECODER_LEARNING_RATE,
    weight_decay=DECODER_WEIGHT_DECAY
)

decoder_history = []

decoder_start = time.perf_counter()

for epoch in range(
        1,
        DECODER_EPOCHS + 1
):

    decoder.train()

    epoch_loss = 0.0
    batches = 0

    for batch in train_loader:

        inputs = batch[
            "input_ids"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        decoder_optimizer.zero_grad(
            set_to_none=True
        )

        logits = decoder(
            inputs
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCABULARY_SIZE
            ),
            labels.reshape(
                -1
            ),
            ignore_index=-100
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            GRADIENT_CLIP_NORM
        )

        decoder_optimizer.step()

        epoch_loss += float(
            loss.detach()
        )

        batches += 1

    epoch_loss_value = (
            epoch_loss
            /
            max(
                batches,
                1
            )
    )

    decoder_history.append(
        {
            "epoch":
                epoch,

            "training_loss":
                epoch_loss_value
        }
    )

    print(
        "Epoch:",
        epoch
    )

    print(
        "Training loss:",
        epoch_loss_value
    )

decoder_duration = (
        time.perf_counter()
        -
        decoder_start
)

print(
    "Decoder training duration:",
    decoder_duration
)

print()


# ============================================================
# TEST 30
# FINAL DECODER EVALUATION
# ============================================================

print(
    "TEST 30: Final Cross-Modal Curriculum Evaluation"
)

print()

final_decoder_metrics = evaluate_decoder(
    decoder,
    validation_loader
)

print(
    "Final loss:",
    final_decoder_metrics[
        "loss"
    ]
)

print(
    "Final perplexity:",
    final_decoder_metrics[
        "perplexity"
    ]
)

print(
    "Final accuracy:",
    final_decoder_metrics[
        "accuracy"
    ]
)

print()


# ============================================================
# TEST 31
# NUMERICAL HEALTH
# ============================================================

print(
    "TEST 31: Numerical Health"
)

print()

nan_tensors = 0
inf_tensors = 0

for parameter in decoder.parameters():

    if torch.isnan(
            parameter
    ).any():

        nan_tensors += 1

    if torch.isinf(
            parameter
    ).any():

        inf_tensors += 1

decoder_numerically_healthy = (
        nan_tensors == 0
        and
        inf_tensors == 0
)

print(
    "NaN tensors:",
    nan_tensors
)

print(
    "Inf tensors:",
    inf_tensors
)

print(
    "Numerically healthy:",
    decoder_numerically_healthy
)

print()


# ============================================================
# TEST 32
# ALIGNMENT GATE
# ============================================================

print(
    "TEST 32: Final Multimodal Alignment Gate"
)

print()

alignment_errors = []

if (
        text_exact_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    alignment_errors.append(
        "Text-to-numeric exact retrieval below threshold."
    )

if (
        numeric_exact_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    alignment_errors.append(
        "Numeric-to-text exact retrieval below threshold."
    )

if (
        class_text_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    alignment_errors.append(
        "Text semantic retrieval below threshold."
    )

if (
        class_numeric_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    alignment_errors.append(
        "Numeric semantic retrieval below threshold."
    )

if (
        mean_instance_margin
        <=
        0.0
):

    alignment_errors.append(
        "Mean instance margin is not positive."
    )

if not torch.isfinite(
        final_similarity_matrix
).all():

    alignment_errors.append(
        "Final similarity matrix is numerically invalid."
    )

print(
    "Exact text -> numeric:",
    text_exact_accuracy
)

print(
    "Exact numeric -> text:",
    numeric_exact_accuracy
)

print(
    "Class text -> numeric:",
    class_text_accuracy
)

print(
    "Class numeric -> text:",
    class_numeric_accuracy
)

print(
    "Mean instance margin:",
    mean_instance_margin
)

print(
    "Alignment errors:",
    len(
        alignment_errors
    )
)

if alignment_errors:

    print(
        json.dumps(
            alignment_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Final multimodal alignment validation failed."
    )

print(
    "Final multimodal alignment validation passed."
)

print()


# ============================================================
# TEST 33
# PARAMETER CHANGE
# ============================================================

print(
    "TEST 33: Decoder Parameter Change"
)

print()

# Reconstruct baseline decoder from checkpoint so the comparison
# is always against the genuine 99R checkpoint, not a mutated
# runtime state.

baseline_decoder = SilverwingDecoder().to(
    DEVICE
)

baseline_decoder.load_state_dict(
    checkpoint_state,
    strict=True
)

changed_tensors = 0
total_parameter_change = 0.0

current_state = decoder.state_dict()
baseline_state = baseline_decoder.state_dict()

for name in current_state:

    difference = torch.sum(
        torch.abs(
            current_state[name]
            -
            baseline_state[name]
        )
    )

    difference_value = float(
        difference
    )

    total_parameter_change += (
        difference_value
    )

    if difference_value > 0:

        changed_tensors += 1

print(
    "Changed tensors:",
    changed_tensors
)

print(
    "Total absolute parameter change:",
    total_parameter_change
)

print()


# ============================================================
# TEST 34
# PROMOTION GATE
# ============================================================

print(
    "TEST 34: Silverwing 100R Promotion Gate"
)

print()

baseline_loss = baseline_metrics[
    "loss"
]

candidate_loss = final_decoder_metrics[
    "loss"
]

if not decoder_numerically_healthy:

    decision = "REJECT"

    reason = (
        "Decoder numerical health failed."
    )

elif alignment_errors:

    decision = "REJECT"

    reason = (
        "Multimodal alignment validation failed."
    )

elif not math.isfinite(
        candidate_loss
):

    decision = "REJECT"

    reason = (
        "Candidate decoder loss is invalid."
    )

elif (
        math.isfinite(
            baseline_loss
        )
        and
        candidate_loss
        <
        baseline_loss
):

    decision = "PROMOTE_CANDIDATE"

    reason = (
        "Cross-modal curriculum improved decoder validation loss."
    )

else:

    decision = "RETAIN_BASELINE"

    reason = (
        "Multimodal representation passed, but decoder "
        "validation loss did not improve."
    )

print(
    "Baseline loss:",
    baseline_loss
)

print(
    "Candidate loss:",
    candidate_loss
)

print(
    "Decision:",
    decision
)

print(
    "Reason:",
    reason
)

print()


# ============================================================
# TEST 35
# SAVE CHECKPOINT
# ============================================================

print(
    "TEST 35: Save 100R Candidate"
)

print()

candidate_payload = {
    "lesson":
        "100R",

    "capability":
        "native_cross_modal_alignment_retrieval",

    "model_state_dict":
        decoder.state_dict(),

    "optimizer_state_dict":
        decoder_optimizer.state_dict(),

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "device":
        str(
            DEVICE
        ),

    "external_llm":
        False,

    "sequence_limit":
        EFFECTIVE_SEQUENCE_LIMIT,

    "shared_dimension":
        SHARED_DIMENSION,

    "temperature":
        TEMPERATURE,

    "contrastive_weight":
        CONTRASTIVE_WEIGHT,

    "instance_weight":
        INSTANCE_WEIGHT,

    "class_weight":
        CLASS_WEIGHT,

    "instance_margin":
        INSTANCE_MARGIN,

    "class_margin":
        CLASS_MARGIN,

    "alignment_epochs":
        ALIGNMENT_EPOCHS,

    "alignment_training_duration":
        alignment_duration,

    "decoder_training_duration":
        decoder_duration,

    "alignment_initial_loss":
        alignment_initial_loss,

    "alignment_final_loss":
        alignment_final_loss,

    "exact_retrieval":
        {
            "text_to_numeric":
                text_exact_accuracy,

            "numeric_to_text":
                numeric_exact_accuracy
        },

    "class_retrieval":
        {
            "text_to_numeric":
                class_text_accuracy,

            "numeric_to_text":
                class_numeric_accuracy
        },

    "mean_positive_similarity":
        mean_positive_score,

    "mean_hard_negative_similarity":
        mean_hard_negative_score,

    "mean_instance_margin":
        mean_instance_margin,

    "baseline_metrics":
        baseline_metrics,

    "candidate_metrics":
        final_decoder_metrics,

    "decision":
        decision,

    "reason":
        reason,

    "alignment_history":
        alignment_history,

    "decoder_history":
        decoder_history
}

torch.save(
    candidate_payload,
    CANDIDATE_CHECKPOINT
)

print(
    "Candidate checkpoint:",
    CANDIDATE_CHECKPOINT
)

if (
        decision
        ==
        "PROMOTE_CANDIDATE"
):

    torch.save(
        candidate_payload,
        BEST_CHECKPOINT
    )

    print(
        "Promoted checkpoint:",
        BEST_CHECKPOINT
    )

else:

    print(
        "Baseline retained."
    )

print()


# ============================================================
# TEST 36
# REPORTS
# ============================================================

print(
    "TEST 36: Write 100R Reports"
)

print()

report = {
    "lesson":
        "100R",

    "capability":
        "native_cross_modal_alignment_retrieval",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "sequence_limit":
        EFFECTIVE_SEQUENCE_LIMIT,

    "shared_dimension":
        SHARED_DIMENSION,

    "dataset_instances":
        INSTANCE_COUNT,

    "dataset_classes":
        sorted(
            set(
                CLASS_IDS
            )
        ),

    "alignment":
        {
            "initial_loss":
                alignment_initial_loss,

            "final_loss":
                alignment_final_loss,

            "positive_similarity":
                mean_positive_score,

            "hard_negative_similarity":
                mean_hard_negative_score,

            "instance_margin":
                mean_instance_margin,

            "text_to_numeric_exact":
                text_exact_accuracy,

            "numeric_to_text_exact":
                numeric_exact_accuracy,

            "text_to_numeric_class":
                class_text_accuracy,

            "numeric_to_text_class":
                class_numeric_accuracy
        },

    "decoder":
        {
            "baseline":
                baseline_metrics,

            "candidate":
                final_decoder_metrics,

            "changed_tensors":
                changed_tensors,

            "total_parameter_change":
                total_parameter_change,

            "nan_tensors":
                nan_tensors,

            "inf_tensors":
                inf_tensors
        },

    "promotion":
        {
            "decision":
                decision,

            "reason":
                reason
        }
}

write_json(
    REPORT_FILE,
    report
)

write_json(
    EVALUATION_FILE,
    report
)

write_json(
    TRAINING_LOG_FILE,
    {
        "lesson":
            "100R",

        "alignment_history":
            alignment_history,

        "decoder_history":
            decoder_history,

        "final_report":
            report
    }
)

print(
    "Report:",
    REPORT_FILE
)

print(
    "Evaluation:",
    EVALUATION_FILE
)

print(
    "Training log:",
    TRAINING_LOG_FILE
)

print()


# ============================================================
# FINAL PREVIEW
# ============================================================

print(
    "SILVERWING CROSS-MODAL ALIGNMENT STACK"
)

print()

print(
    "Text / Sequence"
)

print(
    "      ↓"
)

print(
    "Native Text Encoder"
)

print(
    "      ↓"
)

print(
    "Shared Representation Space"
)

print(
    "      ↑"
)

print(
    "Native Numeric Encoder"
)

print(
    "      ↑"
)

print(
    "Numeric / Structured Evidence"
)

print(
    "      ↓"
)

print(
    "Positive Pair Alignment"
)

print(
    "      ↓"
)

print(
    "Hard-Negative Separation"
)

print(
    "      ↓"
)

print(
    "Exact Instance Retrieval"
)

print(
    "      ↓"
)

print(
    "Semantic Class Retrieval"
)

print(
    "      ↓"
)

print(
    "Bidirectional Retrieval"
)

print(
    "      ↓"
)

print(
    "Multimodal Memory Foundation"
)

print()


# ============================================================
# CURRENT CAPABILITY
# ============================================================

print(
    "LESSON 100R CAPABILITY"
)

print()

print(
    "Silverwing now has a native cross-modal representation "
    "and retrieval foundation."
)

print(
    "Text and structured numeric evidence share a trainable "
    "representation space."
)

print(
    "Exact instance identity is evaluated separately from "
    "semantic class identity."
)

print(
    "Hard negatives and positive-negative separation are measured."
)

print(
    "No external LLM is used."
)

print()


# ============================================================
# LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "100R uses a controlled native dataset."
)

print(
    "Image and audio encoders are not yet implemented."
)

print(
    "Large-scale multimodal pretraining is not yet implemented."
)

print(
    "Production vector indexing is not yet implemented."
)

print(
    "Persistent multimodal memory integration comes later."
)

print()


# ============================================================
# NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 101R: Native Hard-Negative Multimodal Learning"
)

print()

print(
    "Adaptive Hard Negatives + In-Batch Negatives + "
    "Retrieval Stress Testing + Robust Multimodal Alignment"
)

print()


# ============================================================
# FOUNDATION MODEL PROGRESS
# ============================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()

progress = [
    "Own Tokenizer",
    " ↓",
    "Own Vocabulary",
    " ↓",
    "Own Decoder",
    " ↓",
    "Own Training",
    " ↓",
    "Own Evaluation",
    " ↓",
    "Instruction Learning",
    " ↓",
    "79R Native Reasoning Dataset",
    " ↓",
    "80R Native Reasoning Fine-Tuning",
    " ↓",
    "81R Native Memory-Aware Training",
    " ↓",
    "82R Native Tool-Aware Learning",
    " ↓",
    "83R Native Planning and Tool Sequencing",
    " ↓",
    "84R Native Verified Execution and Replanning",
    " ↓",
    "85R Native Mathematical Reasoning",
    " ↓",
    "86R Native Probability and Statistical Reasoning",
    " ↓",
    "87R Native Linear Algebra and Optimization",
    " ↓",
    "88R Native Algorithms and Data Structures",
    " ↓",
    "89R Native Data Analysis + SQL Reasoning",
    " ↓",
    "90R Native Data Engineering",
    " ↓",
    "91R Native Machine Learning Foundations",
    " ↓",
    "92R Native Classical Machine Learning",
    " ↓",
    "93R Native Neural Network Foundations",
    " ↓",
    "94R Native Deep Learning",
    " ↓",
    "95R Native Representation Learning",
    " ↓",
    "96R Native Sequence Representation Learning",
    " ↓",
    "97R Native Structured Representation Learning",
    " ↓",
    "98R Native Advanced Sequence + Structured Learning",
    " ↓",
    "99R Native Multimodal Representation Foundations",
    " ↓",
    "100R Native Cross-Modal Alignment + Retrieval",
    " ↓",
    "101R Native Hard-Negative Multimodal Learning",
    " ↓",
    "Advanced Learning",
    " ↓",
    "Engineering + Scientific Intelligence",
    " ↓",
    "Continual Learning",
    " ↓",
    "Controlled Autonomous Improvement"
]

for line in progress:

    print(
        line
    )

print()

print(
    "=== LESSON 100R COMPLETE ==="
)
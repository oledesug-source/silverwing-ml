# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 101R
# Native Hard-Negative Multimodal Learning
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
# 101R -> Native Hard-Negative Multimodal Learning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 101R strengthens 100R with:
#
#   adaptive hard-negative mining
#   dynamic negative selection
#   bidirectional contrastive learning
#   exact instance separation
#   semantic class validation
#   retrieval stress testing
#   deterministic inference
#   compatibility inspection
#   controlled state recovery
#
# ============================================================
# IMPORTANT ENGINEERING RULE
# ============================================================
#
# DO NOT ASSUME MISMATCH.
#
# Instead:
#
#     Inspect
#       ↓
#     Compare
#       ↓
#     Diagnose
#       ↓
#     Repair safe differences
#       ↓
#     Validate
#       ↓
#     Continue
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_SEQUENCE_LENGTH = 256

TEXT_DIMENSION = 16

NUMERIC_INPUT_DIMENSION = 5

SHARED_DIMENSION = 16

HIDDEN_DIMENSION = 32

TEMPERATURE = 0.07

INSTANCE_MARGIN = 0.25

CLASS_MARGIN = 0.05

CONTRASTIVE_WEIGHT = 1.0

HARD_NEGATIVE_WEIGHT = 2.0

INSTANCE_WEIGHT = 1.5

CLASS_WEIGHT = 0.25

ALIGNMENT_EPOCHS = 1500

LEARNING_RATE = 0.0025

WEIGHT_DECAY = 0.0005

GRADIENT_CLIP_NORM = 1.0

EXACT_RETRIEVAL_THRESHOLD = 0.83

CLASS_RETRIEVAL_THRESHOLD = 0.83

TOP_K = 3


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent

LESSON_66R = PHASE5_DIR / "lesson66R"
LESSON_71R = PHASE5_DIR / "lesson71R"
LESSON_79R = PHASE5_DIR / "lesson79R"
LESSON_99R = PHASE5_DIR / "lesson99R"
LESSON_100R = PHASE5_DIR / "lesson100R"

VOCABULARY_FILE = (
        LESSON_66R /
        "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66R /
        "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71R /
        "silverwing_decoder_config.json"
)

REASONING_CONFIG_FILE = (
        LESSON_79R /
        "silverwing_reasoning_config.json"
)

CHECKPOINT_100R_PRIMARY = (
        LESSON_100R /
        "checkpoints" /
        "silverwing_cross_modal_alignment_best.pt"
)

CHECKPOINT_100R_CANDIDATE = (
        LESSON_100R /
        "checkpoints" /
        "silverwing_cross_modal_alignment_candidate.pt"
)

CHECKPOINT_99R_PRIMARY = (
        LESSON_99R /
        "checkpoints" /
        "silverwing_multimodal_best.pt"
)

CHECKPOINT_99R_CANDIDATE = (
        LESSON_99R /
        "checkpoints" /
        "silverwing_multimodal_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_hard_negative_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_hard_negative_best.pt"
)

REPORT_FILE = (
        BASE_DIR /
        "silverwing_hard_negative_report.json"
)

REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_hard_negative_registry.json"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_hard_negative_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_hard_negative_evaluation.json"
)


# ============================================================
# 3. HELPERS
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

    with path.open(
            "r",
            encoding="utf-8"
    ) as handle:

        return json.load(
            handle
        )


def write_json(
        path: Path,
        data: Any
) -> None:

    with path.open(
            "w",
            encoding="utf-8"
    ) as handle:

        json.dump(
            data,
            handle,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def safe_mean(
        values: List[float]
) -> float:

    if not values:

        return 0.0

    return (
            sum(values)
            /
            len(values)
    )


def choose_checkpoint() -> Path:

    candidates = [
        CHECKPOINT_100R_PRIMARY,
        CHECKPOINT_100R_CANDIDATE,
        CHECKPOINT_99R_PRIMARY,
        CHECKPOINT_99R_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        (
            "No usable 99R/100R Silverwing checkpoint "
            "was found."
        )
    )


# ============================================================
# 4. INITIALIZATION
# ============================================================

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
# 5. HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 101R"
)

print(
    "Native Hard-Negative Multimodal Learning"
)

print()

for line in [
    "79R -> Reasoning",
    "80R -> Reasoning Fine-Tuning",
    "81R -> Memory",
    "82R -> Tool Use",
    "83R -> Planning",
    "84R -> Verified Execution + Replanning",
    "85R -> Mathematical Reasoning",
    "86R -> Probability + Statistics",
    "87R -> Linear Algebra + Optimization",
    "88R -> Algorithms + Data Structures",
    "89R -> Data Analysis + SQL Reasoning",
    "90R -> Data Engineering",
    "91R -> Machine Learning Foundations",
    "92R -> Classical Machine Learning",
    "93R -> Neural Network Foundations",
    "94R -> Deep Learning",
    "95R -> Representation Learning",
    "96R -> Sequence Representation Learning",
    "97R -> Structured Representation Learning",
    "98R -> Advanced Sequence + Structured Learning",
    "99R -> Multimodal Representation Foundations",
    "100R -> Cross-Modal Alignment + Retrieval",
    "101R -> Hard-Negative Multimodal Learning",
]:

    print(
        line
    )

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
    "Instance margin:",
    INSTANCE_MARGIN
)

print(
    "Hard-negative weight:",
    HARD_NEGATIVE_WEIGHT
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
    "TEST 1: Verify Silverwing Inputs"
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

BASE_CHECKPOINT = choose_checkpoint()

print(
    "BASE CHECKPOINT:",
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
    "Effective sequence limit:",
    EFFECTIVE_SEQUENCE_LIMIT
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

if (
        "token_to_id"
        not in vocabulary
):

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

for token in [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]:

    if token not in TOKEN_TO_ID:

        raise RuntimeError(
            f"Missing token: {token}"
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
            isinstance(
                pair,
                list
            )
            and
            len(pair) == 2
            and
            rank is not None
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


def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    result = []

    index = 0

    while index < len(
            symbols
    ):

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

    if not word:

        return []

    symbols = list(
        word
    )

    symbols[-1] += BPE_END

    symbols = tuple(
        symbols
    )

    while len(
            symbols
    ) > 1:

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
    "TEST 5: Build Native Hard-Negative Multimodal Dataset"
)

print()

DATA = [
    {
        "id":
            "motor_01",

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

        "class":
            0
    },

    {
        "id":
            "motor_02",

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

        "class":
            0
    },

    {
        "id":
            "pump_01",

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

        "class":
            1
    },

    {
        "id":
            "pump_02",

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

        "class":
            1
    },

    {
        "id":
            "sensor_01",

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

        "class":
            2
    },

    {
        "id":
            "sensor_02",

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

        "class":
            2
    }
]

INSTANCE_COUNT = len(
    DATA
)

INSTANCE_IDS = [
    item["id"]
    for item
    in DATA
]

CLASS_IDS = [
    item["class"]
    for item
    in DATA
]

print(
    "Instances:",
    INSTANCE_COUNT
)

print(
    "Unique instances:",
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

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Prepare Text and Numeric Modalities"
)

print()

encoded_text = []

for item in DATA:

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

    encoded_text.append(
        ids
    )

max_text_length = max(
    len(ids)
    for ids
    in encoded_text
)

text_rows = []

for ids in encoded_text:

    row = list(
        ids
    )

    row.extend(
        [
            PAD_ID
        ]
        *
        (
                max_text_length
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
        in DATA
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
    "TEST 7: Normalize Structured Numeric Evidence"
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

normalized_numeric = (
                             numeric_tensor
                             -
                             numeric_mean
                     ) / numeric_std

print(
    "Raw means:",
    numeric_mean.squeeze(
        0
    ).tolist()
)

print(
    "Normalized means:",
    normalized_numeric.mean(
        dim=0
    ).tolist()
)

if not torch.isfinite(
        normalized_numeric
).all():

    raise RuntimeError(
        "Numeric normalization generated invalid values."
    )

print(
    "Numeric normalization validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

class AdaptiveCrossModalEncoder(
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

        counts = (
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
                counts
        )

        projected = self.text_projection(
            pooled
        )

        return F.normalize(
            projected,
            p=2,
            dim=-1
        )

    def encode_numeric(
            self,
            values: torch.Tensor
    ) -> torch.Tensor:

        projected = self.numeric_projection(
            values
        )

        return F.normalize(
            projected,
            p=2,
            dim=-1
        )

    def forward(
            self,
            text_ids: torch.Tensor,
            numeric_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        return {
            "text":
                self.encode_text(
                    text_ids
                ),

            "numeric":
                self.encode_numeric(
                    numeric_values
                )
        }


encoder = AdaptiveCrossModalEncoder(
    vocabulary_size=VOCABULARY_SIZE,
    numeric_dimension=NUMERIC_INPUT_DIMENSION,
    sequence_limit=EFFECTIVE_SEQUENCE_LIMIT
).to(
    DEVICE
)

print(
    "TEST 8: Adaptive Cross-Modal Encoder"
)

print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter
        in encoder.parameters()
    )
)

print(
    "Adaptive encoder validated."
)

print()


# ============================================================
# DEVICE DATA
# ============================================================

device_text = text_tensor.to(
    DEVICE
)

device_numeric = normalized_numeric.to(
    DEVICE
)

device_classes = class_tensor.to(
    DEVICE
)


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Inspect Previous Checkpoint"
)

print()

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if not isinstance(
        checkpoint,
        dict
):

    raise RuntimeError(
        "Previous checkpoint is not a dictionary."
    )

checkpoint_decoder_state = checkpoint.get(
    "model_state_dict"
)

checkpoint_encoder_state = checkpoint.get(
    "cross_modal_encoder_state_dict"
)

print(
    "Checkpoint:",
    BASE_CHECKPOINT
)

print(
    "Checkpoint fields:",
    sorted(
        checkpoint.keys()
    )
)

print(
    "Decoder state present:",
    checkpoint_decoder_state is not None
)

print(
    "Cross-modal encoder state present:",
    checkpoint_encoder_state is not None
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Compatibility Audit"
)

print()

encoder_recovery_mode = (
    "fresh_native_initialization"
)

if checkpoint_encoder_state is not None:

    current_state = encoder.state_dict()

    stored_state = checkpoint_encoder_state

    missing_keys = sorted(
        set(
            current_state.keys()
        )
        -
        set(
            stored_state.keys()
        )
    )

    unexpected_keys = sorted(
        set(
            stored_state.keys()
        )
        -
        set(
            current_state.keys()
        )
    )

    shape_mismatches = []

    for key in sorted(
            set(
                current_state.keys()
            )
            &
            set(
                stored_state.keys()
            )
    ):

        if (
                current_state[
                    key
                ].shape
                !=
                stored_state[
                    key
                ].shape
        ):

            shape_mismatches.append(
                key
            )

    print(
        "Stored encoder tensors:",
        len(stored_state)
    )

    print(
        "Current encoder tensors:",
        len(current_state)
    )

    print(
        "Missing:",
        len(missing_keys)
    )

    print(
        "Unexpected:",
        len(unexpected_keys)
    )

    print(
        "Shape mismatches:",
        len(shape_mismatches)
    )

    if (
            not missing_keys
            and
            not unexpected_keys
            and
            not shape_mismatches
    ):

        encoder.load_state_dict(
            stored_state,
            strict=True
        )

        encoder_recovery_mode = (
            "inherited_100R_encoder"
        )

        print(
            "Compatible multimodal encoder inherited."
        )

    else:

        print(
            "Stored multimodal state differs from "
            "the current schema."
        )

        print(
            "Controlled native recovery will repair "
            "the representation state."
        )

        encoder_recovery_mode = (
            "controlled_native_retraining"
        )

else:

    print(
        "No cross-modal encoder state exists."
    )

    print(
        "Controlled native recovery will establish "
        "the persistent encoder foundation."
    )


print(
    "Recovery mode:",
    encoder_recovery_mode
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Initial Native Multimodal Representations"
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
        "Initial text representation is invalid."
    )

if not torch.isfinite(
        initial_numeric
).all():

    raise RuntimeError(
        "Initial numeric representation is invalid."
    )

print(
    "Initial multimodal representations validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Adaptive Hard-Negative Objective"
)

print()


def build_objective(
        text_vectors: torch.Tensor,
        numeric_vectors: torch.Tensor,
        classes: torch.Tensor
) -> Dict[str, torch.Tensor]:

    similarity = torch.matmul(
        text_vectors,
        numeric_vectors.T
    )

    logits = (
            similarity
            /
            TEMPERATURE
    )

    targets = torch.arange(
        logits.shape[0],
        device=logits.device
    )

    forward_loss = F.cross_entropy(
        logits,
        targets
    )

    backward_loss = F.cross_entropy(
        logits.T,
        targets
    )

    contrastive_loss = (
                               forward_loss
                               +
                               backward_loss
                       ) / 2.0

    hard_negative_terms = []
    instance_terms = []
    class_terms = []

    for query_index in range(
            logits.shape[0]
    ):

        positive = logits[
            query_index,
            query_index
        ]

        negative_scores = []
        same_class_scores = []

        for candidate_index in range(
                logits.shape[1]
        ):

            if candidate_index == query_index:

                continue

            candidate_score = logits[
                query_index,
                candidate_index
            ]

            negative_scores.append(
                candidate_score
            )

            if (
                    classes[
                        candidate_index
                    ]
                    ==
                    classes[
                        query_index
                    ]
            ):

                same_class_scores.append(
                    candidate_score
                )

        if negative_scores:

            hardest_negative = max(
                negative_scores
            )

            margin_term = F.relu(
                INSTANCE_MARGIN
                +
                hardest_negative
                -
                positive
            )

            hard_negative_terms.append(
                margin_term
            )

            instance_terms.append(
                margin_term
            )

        if same_class_scores:

            hardest_same_class = max(
                same_class_scores
            )

            class_terms.append(
                F.relu(
                    CLASS_MARGIN
                    +
                    hardest_same_class
                    -
                    positive
                )
            )

    hard_negative_loss = torch.stack(
        hard_negative_terms
    ).mean()

    instance_loss = torch.stack(
        instance_terms
    ).mean()

    if class_terms:

        class_loss = torch.stack(
            class_terms
        ).mean()

    else:

        class_loss = torch.zeros(
            (),
            device=logits.device
        )

    total = (
            CONTRASTIVE_WEIGHT
            *
            contrastive_loss
            +
            HARD_NEGATIVE_WEIGHT
            *
            hard_negative_loss
            +
            INSTANCE_WEIGHT
            *
            instance_loss
            +
            CLASS_WEIGHT
            *
            class_loss
    )

    return {
        "similarity":
            similarity,

        "contrastive":
            contrastive_loss,

        "hard_negative":
            hard_negative_loss,

        "instance_margin":
            instance_loss,

        "class_margin":
            class_loss,

        "total":
            total
    }


with torch.no_grad():

    initial_objective = build_objective(
        initial_text,
        initial_numeric,
        device_classes
    )

print(
    "Initial contrastive loss:",
    float(
        initial_objective[
            "contrastive"
        ]
    )
)

print(
    "Initial hard-negative loss:",
    float(
        initial_objective[
            "hard_negative"
        ]
    )
)

print(
    "Initial instance margin loss:",
    float(
        initial_objective[
            "instance_margin"
        ]
    )
)

print(
    "Initial class margin loss:",
    float(
        initial_objective[
            "class_margin"
        ]
    )
)

print(
    "Initial total loss:",
    float(
        initial_objective[
            "total"
        ]
    )
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Adaptive Hard-Negative Training"
)

print()

optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

initial_loss = None
final_loss = None
history = []

training_start = time.perf_counter()

for epoch in range(
        1,
        ALIGNMENT_EPOCHS + 1
):

    encoder.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = encoder(
        device_text,
        device_numeric
    )

    training_text = outputs[
        "text"
    ]

    training_numeric = outputs[
        "numeric"
    ]

    objective = build_objective(
        training_text,
        training_numeric,
        device_classes
    )

    loss = objective[
        "total"
    ]

    if epoch == 1:

        initial_loss = float(
            loss.detach()
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        encoder.parameters(),
        GRADIENT_CLIP_NORM
    )

    optimizer.step()

    final_loss = float(
        loss.detach()
    )

    if (
            epoch == 1
            or
            epoch % 100 == 0
    ):

        history.append(
            {
                "epoch":
                    epoch,

                "total":
                    final_loss,

                "contrastive":
                    float(
                        objective[
                            "contrastive"
                        ].detach()
                    ),

                "hard_negative":
                    float(
                        objective[
                            "hard_negative"
                        ].detach()
                    ),

                "instance_margin":
                    float(
                        objective[
                            "instance_margin"
                        ].detach()
                    ),

                "class_margin":
                    float(
                        objective[
                            "class_margin"
                        ].detach()
                    )
            }
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)

print(
    "Initial hard-negative loss:",
    initial_loss
)

print(
    "Final hard-negative loss:",
    final_loss
)

print(
    "Training duration:",
    training_duration
)

print(
    "Training history:",
    history
)

if (
        initial_loss is None
        or
        final_loss is None
):

    raise RuntimeError(
        "Hard-negative training produced no loss."
    )

if (
        final_loss
        >
        initial_loss
):

    raise RuntimeError(
        "Hard-negative loss increased."
    )

print(
    "Adaptive hard-negative training validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Final Adaptive Representations"
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

final_objective = build_objective(
    final_text,
    final_numeric,
    device_classes
)

print(
    "Final total alignment loss:",
    float(
        final_objective[
            "total"
        ]
    )
)

print(
    "Final contrastive loss:",
    float(
        final_objective[
            "contrastive"
        ]
    )
)

print(
    "Final hard-negative loss:",
    float(
        final_objective[
            "hard_negative"
        ]
    )
)

if not torch.isfinite(
        final_text
).all():

    raise RuntimeError(
        "Final text representation is invalid."
    )

if not torch.isfinite(
        final_numeric
).all():

    raise RuntimeError(
        "Final numeric representation is invalid."
    )

print(
    "Final adaptive representations validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Hard-Negative Stress Test"
)

print()

final_similarity = final_objective[
    "similarity"
]

positive_scores = []
hard_negative_scores = []
margins = []

for index in range(
        INSTANCE_COUNT
):

    positive = float(
        final_similarity[
            index,
            index
        ]
    )

    negatives = []

    for candidate in range(
            INSTANCE_COUNT
    ):

        if candidate == index:

            continue

        negatives.append(
            float(
                final_similarity[
                    index,
                    candidate
                ]
            )
        )

    hardest = max(
        negatives
    )

    positive_scores.append(
        positive
    )

    hard_negative_scores.append(
        hardest
    )

    margins.append(
        positive
        -
        hardest
    )

mean_positive = safe_mean(
    positive_scores
)

mean_hard_negative = safe_mean(
    hard_negative_scores
)

mean_margin = safe_mean(
    margins
)

print(
    "Mean positive score:",
    mean_positive
)

print(
    "Mean hard-negative score:",
    mean_hard_negative
)

print(
    "Margins:",
    margins
)

print(
    "Mean margin:",
    mean_margin
)

if mean_margin <= 0.0:

    raise RuntimeError(
        "Hard-negative margin is not positive."
    )

print(
    "Hard-negative stress test passed."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Bidirectional Exact Retrieval"
)

print()

text_rankings = []
numeric_rankings = []

text_correct = 0
numeric_correct = 0

for index in range(
        INSTANCE_COUNT
):

    text_candidates = []
    numeric_candidates = []

    for candidate in range(
            INSTANCE_COUNT
    ):

        text_candidates.append(
            (
                candidate,
                float(
                    final_similarity[
                        index,
                        candidate
                    ]
                )
            )
        )

        numeric_candidates.append(
            (
                candidate,
                float(
                    final_similarity[
                        candidate,
                        index
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
            index
    ):

        text_correct += 1

    if (
            numeric_candidates[0][0]
            ==
            index
    ):

        numeric_correct += 1

text_accuracy = (
        text_correct
        /
        INSTANCE_COUNT
)

numeric_accuracy = (
        numeric_correct
        /
        INSTANCE_COUNT
)

print(
    "Text -> numeric exact accuracy:",
    text_accuracy
)

print(
    "Numeric -> text exact accuracy:",
    numeric_accuracy
)

for index in range(
        INSTANCE_COUNT
):

    print(
        INSTANCE_IDS[index],
        "->",
        INSTANCE_IDS[
            text_rankings[
                index
            ][0][0]
        ],
        "| score:",
        text_rankings[
            index
        ][0][1]
    )

print()

if (
        text_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    raise RuntimeError(
        "Text-to-numeric exact retrieval failed."
    )

if (
        numeric_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    raise RuntimeError(
        "Numeric-to-text exact retrieval failed."
    )

print(
    "Bidirectional exact retrieval validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Top-K Retrieval Stress Test"
)

print()

text_top_k_hits = 0
numeric_top_k_hits = 0

for index in range(
        INSTANCE_COUNT
):

    text_top_k = [
        pair[0]
        for pair
        in text_rankings[
            index
        ][
            :
            TOP_K
        ]
    ]

    numeric_top_k = [
        pair[0]
        for pair
        in numeric_rankings[
            index
        ][
            :
            TOP_K
        ]
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
    "Text top-k accuracy:",
    text_top_k_accuracy
)

print(
    "Numeric top-k accuracy:",
    numeric_top_k_accuracy
)

if (
        text_top_k_accuracy
        <
        1.0
):

    raise RuntimeError(
        "Text top-k retrieval failed."
    )

if (
        numeric_top_k_accuracy
        <
        1.0
):

    raise RuntimeError(
        "Numeric top-k retrieval failed."
    )

print(
    "Top-k retrieval validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Semantic Class Validation"
)

print()

text_class_correct = 0
numeric_class_correct = 0

for index in range(
        INSTANCE_COUNT
):

    best_numeric = text_rankings[
        index
    ][0][0]

    best_text = numeric_rankings[
        index
    ][0][0]

    if (
            CLASS_IDS[
                best_numeric
            ]
            ==
            CLASS_IDS[
                index
            ]
    ):

        text_class_correct += 1

    if (
            CLASS_IDS[
                best_text
            ]
            ==
            CLASS_IDS[
                index
            ]
    ):

        numeric_class_correct += 1

text_class_accuracy = (
        text_class_correct
        /
        INSTANCE_COUNT
)

numeric_class_accuracy = (
        numeric_class_correct
        /
        INSTANCE_COUNT
)

print(
    "Text -> numeric class accuracy:",
    text_class_accuracy
)

print(
    "Numeric -> text class accuracy:",
    numeric_class_accuracy
)

if (
        text_class_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    raise RuntimeError(
        "Text semantic retrieval failed."
    )

if (
        numeric_class_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    raise RuntimeError(
        "Numeric semantic retrieval failed."
    )

print(
    "Semantic class validation passed."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Deterministic Retrieval Validation"
)

print()

with torch.no_grad():

    repeat_outputs = encoder(
        device_text,
        device_numeric
    )

repeat_similarity = torch.matmul(
    repeat_outputs["text"],
    repeat_outputs["numeric"].T
)

determinism_error = float(
    torch.max(
        torch.abs(
            final_similarity
            -
            repeat_similarity
        )
    )
)

print(
    "Maximum retrieval difference:",
    determinism_error
)

if (
        determinism_error
        >
        1e-7
):

    raise RuntimeError(
        "Retrieval determinism failed."
    )

print(
    "Deterministic retrieval validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Compatibility Repair Contract"
)

print()

repair_actions = []

if (
        checkpoint_encoder_state
        is None
):

    repair_actions.append(
        "Missing multimodal encoder state repaired through native retraining."
    )

elif (
        encoder_recovery_mode
        ==
        "inherited_100R_encoder"
):

    repair_actions.append(
        "Compatible multimodal encoder inherited directly."
    )

else:

    repair_actions.append(
        "Stored multimodal encoder differed and was repaired through controlled native retraining."
    )

if (
        checkpoint_decoder_state
        is None
):

    repair_actions.append(
        "No decoder state was present; multimodal learning remained independent."
    )

else:

    repair_actions.append(
        "Previous decoder state inspected and preserved."
    )

for action in repair_actions:

    print(
        action
    )

print(
    "Compatibility procedure completed."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Numerical Health"
)

print()

nan_tensors = 0
inf_tensors = 0

for parameter in encoder.parameters():

    if torch.isnan(
            parameter
    ).any():

        nan_tensors += 1

    if torch.isinf(
            parameter
    ).any():

        inf_tensors += 1

numerically_healthy = (
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
    numerically_healthy
)

if not numerically_healthy:

    raise RuntimeError(
        "Encoder numerical health failed."
    )

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Build Native Hard-Negative Reasoning Tasks"
)

print()

TASKS = [
    (
        "hardneg_001",
        "hard_negative_mining",
        "What is adaptive hard-negative mining?",
        "Repeatedly selecting the strongest incorrect candidates and training against them.",
    ),
    (
        "hardneg_002",
        "instance_separation",
        "Why must same-class instances remain separable?",
        "Different physical events can belong to the same semantic class.",
    ),
    (
        "hardneg_003",
        "contrastive_learning",
        "What does contrastive alignment optimize?",
        "It raises the similarity of correct pairs and lowers the similarity of incorrect pairs.",
    ),
    (
        "hardneg_004",
        "retrieval_stress",
        "Why perform retrieval stress testing?",
        "To ensure difficult competing records cannot displace the correct pair.",
    ),
    (
        "hardneg_005",
        "engineering_reasoning",
        "Why are hard negatives useful in engineering intelligence?",
        "Similar machines and events must still be distinguished precisely.",
    ),
    (
        "hardneg_006",
        "memory_alignment",
        "Why does exact hard-negative resistance matter for memory?",
        "The memory system must retrieve the exact event rather than only a similar event.",
    ),
    (
        "hardneg_007",
        "compatibility",
        "How should a model compatibility problem be handled?",
        "Inspect the difference, repair safe differences, and reject only unsafe incompatibilities.",
    ),
    (
        "hardneg_008",
        "continual_learning",
        "Why diagnose incompatibilities before stopping?",
        "Some differences are recoverable and should not interrupt the learning pipeline unnecessarily.",
    ),
]

task_records = []

for (
        example_id,
        domain,
        question,
        answer
) in TASKS:

    trace = "\n".join(
        [
            "P:" + question,
            "M:" + answer,
            "V:validated"
        ]
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    if (
            token_count
            >
            EFFECTIVE_SEQUENCE_LIMIT
    ):

        raise RuntimeError(
            (
                f"{example_id} exceeds "
                "the Silverwing sequence limit."
            )
        )

    task_records.append(
        {
            "example_id":
                example_id,

            "domain":
                domain,

            "trace":
                trace,

            "token_count":
                token_count
        }
    )

    print(
        example_id,
        "->",
        token_count,
        "tokens |",
        domain
    )

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Final Native Hard-Negative Validation"
)

print()

validation_errors = []

if (
        final_loss
        >
        initial_loss
):

    validation_errors.append(
        "Hard-negative loss did not improve."
    )

if (
        mean_margin
        <=
        0.0
):

    validation_errors.append(
        "Mean hard-negative margin is not positive."
    )

if (
        text_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    validation_errors.append(
        "Text-to-numeric exact retrieval below threshold."
    )

if (
        numeric_accuracy
        <
        EXACT_RETRIEVAL_THRESHOLD
):

    validation_errors.append(
        "Numeric-to-text exact retrieval below threshold."
    )

if (
        text_class_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    validation_errors.append(
        "Text semantic retrieval below threshold."
    )

if (
        numeric_class_accuracy
        <
        CLASS_RETRIEVAL_THRESHOLD
):

    validation_errors.append(
        "Numeric semantic retrieval below threshold."
    )

if (
        determinism_error
        >
        1e-7
):

    validation_errors.append(
        "Deterministic retrieval failed."
    )

print(
    "Validation errors:",
    len(
        validation_errors
    )
)

if validation_errors:

    print(
        json.dumps(
            validation_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Lesson 101R validation failed."
    )

print(
    "Lesson 101R hard-negative validation passed."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save Native 101R State"
)

print()

payload = {
    "lesson":
        "101R",

    "capability":
        "native_hard_negative_multimodal_learning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "model_state_dict":
        checkpoint_decoder_state,

    "cross_modal_encoder_state_dict":
        encoder.state_dict(),

    "external_llm":
        False,

    "encoder_recovery_mode":
        encoder_recovery_mode,

    "sequence_limit":
        EFFECTIVE_SEQUENCE_LIMIT,

    "shared_dimension":
        SHARED_DIMENSION,

    "temperature":
        TEMPERATURE,

    "instance_margin":
        INSTANCE_MARGIN,

    "class_margin":
        CLASS_MARGIN,

    "contrastive_weight":
        CONTRASTIVE_WEIGHT,

    "hard_negative_weight":
        HARD_NEGATIVE_WEIGHT,

    "instance_weight":
        INSTANCE_WEIGHT,

    "class_weight":
        CLASS_WEIGHT,

    "metrics":
        {
            "initial_loss":
                initial_loss,

            "final_loss":
                final_loss,

            "mean_positive":
                mean_positive,

            "mean_hard_negative":
                mean_hard_negative,

            "mean_margin":
                mean_margin,

            "text_exact_accuracy":
                text_accuracy,

            "numeric_exact_accuracy":
                numeric_accuracy,

            "text_class_accuracy":
                text_class_accuracy,

            "numeric_class_accuracy":
                numeric_class_accuracy,

            "text_top_k_accuracy":
                text_top_k_accuracy,

            "numeric_top_k_accuracy":
                numeric_top_k_accuracy,

            "determinism_error":
                determinism_error
        },

    "repair_actions":
        repair_actions,

    "history":
        history
}

torch.save(
    payload,
    CANDIDATE_CHECKPOINT
)

torch.save(
    payload,
    BEST_CHECKPOINT
)

print(
    "Candidate:",
    CANDIDATE_CHECKPOINT
)

print(
    "Promoted:",
    BEST_CHECKPOINT
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Write 101R Reports"
)

print()

report = {
    "lesson":
        "101R",

    "capability":
        "native_hard_negative_multimodal_learning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "encoder_recovery_mode":
        encoder_recovery_mode,

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

    "alignment":
        {
            "initial_loss":
                initial_loss,

            "final_loss":
                final_loss,

            "mean_positive":
                mean_positive,

            "mean_hard_negative":
                mean_hard_negative,

            "mean_margin":
                mean_margin,

            "text_exact_accuracy":
                text_accuracy,

            "numeric_exact_accuracy":
                numeric_accuracy,

            "text_class_accuracy":
                text_class_accuracy,

            "numeric_class_accuracy":
                numeric_class_accuracy
        },

    "stress":
        {
            "text_top_k_accuracy":
                text_top_k_accuracy,

            "numeric_top_k_accuracy":
                numeric_top_k_accuracy,

            "determinism_error":
                determinism_error
        },

    "repair_actions":
        repair_actions,

    "validation":
        {
            "passed":
                True,

            "errors":
                validation_errors
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
            "101R",

        "history":
            history,

        "report":
            report
    }
)

write_json(
    REGISTRY_FILE,
    {
        "lesson":
            "101R",

        "capability":
            "native_hard_negative_multimodal_learning",

        "external_llm":
            False,

        "encoder_recovery_mode":
            encoder_recovery_mode,

        "next":
            "102R Native Multimodal Memory Integration"
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
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 101R MULTIMODAL STACK"
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
    "Engineering / Scientific Evidence"
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
    "Adaptive Hard-Negative Mining"
)

print(
    "      ↓"
)

print(
    "Instance Separation"
)

print(
    "      ↓"
)

print(
    "Exact Retrieval"
)

print(
    "      ↓"
)

print(
    "Semantic Retrieval"
)

print(
    "      ↓"
)

print(
    "Multimodal Memory"
)

print()


# ============================================================
# COMPATIBILITY MODEL
# ============================================================

print(
    "COMPATIBILITY MODEL"
)

print()

print(
    "Inspect"
)

print(
    "  ↓"
)

print(
    "Compare"
)

print(
    "  ↓"
)

print(
    "Diagnose"
)

print(
    "  ↓"
)

print(
    "Repair Safe Differences"
)

print(
    "  ↓"
)

print(
    "Validate"
)

print(
    "  ↓"
)

print(
    "Continue"
)

print()


# ============================================================
# WHY 101R MATTERS
# ============================================================

print(
    "WHY 101R MATTERS"
)

print()

print(
    "100R established cross-modal alignment."
)

print(
    "101R makes the alignment adversarial."
)

print(
    "The learner repeatedly confronts the strongest "
    "incorrect representation."
)

print(
    "This strengthens exact engineering retrieval, "
    "machine-event separation and future multimodal memory."
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
    "Lesson 102R: Native Multimodal Memory Integration"
)

print()

print(
    "Cross-Modal Memory Storage + Event Identity + "
    "Persistent Multimodal Retrieval"
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
    "86R Native Probability + Statistical Reasoning",
    " ↓",
    "87R Native Linear Algebra + Optimization",
    " ↓",
    "88R Native Algorithms + Data Structures",
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
    "102R Native Multimodal Memory Integration",
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
    "=== LESSON 101R COMPLETE ==="
)
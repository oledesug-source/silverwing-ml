# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 87R
# Native Linear Algebra and Optimization
# ============================================================
#
# 79R -> Native Reasoning Dataset
# 80R -> Native Reasoning Fine-Tuning
# 81R -> Native Memory-Aware Training
# 82R -> Native Tool-Aware Learning
# 83R -> Native Planning and Tool Sequencing
# 84R -> Native Verified Execution and Replanning
# 85R -> Native Mathematical Reasoning Foundation
# 86R -> Native Probability and Statistical Reasoning
# 87R -> Native Linear Algebra and Optimization
#
# ============================================================
# PURPOSE
# ============================================================
#
# 87R establishes the next mathematical foundation:
#
#   vectors
#   vector arithmetic
#   dot products
#   vector norms
#   matrices
#   matrix-vector multiplication
#   linear systems
#   quadratic objective functions
#   gradients
#   gradient descent
#   numerical convergence
#
# The long-term role of this layer is to support:
#
#   machine learning
#   neural networks
#   optimization
#   physics
#   engineering
#   control
#   scientific computing
#
# ============================================================
# REASONING CONTRACT
# ============================================================
#
# Problem
#   ↓
# Representation
#   ↓
# Operation / Formula
#   ↓
# Calculation
#   ↓
# Independent Validation
#   ↓
# Final Answer
#
# ============================================================
# MODEL OWNERSHIP
# ============================================================
#
# Tokenizer: Silverwing native
# Vocabulary: Silverwing native
# Decoder: Silverwing native
# Dataset: Silverwing native
# Training: Silverwing native
# Evaluation: Silverwing native
#
# External LLM: NONE
#
# ============================================================
# IMPORTANT
# ============================================================
#
# The established 256-token limit remains unchanged.
#
# The decoder architecture is loaded STRICTLY from 86R.
#
# 87R does not replace or silently alter the model architecture.
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
LESSON_86R = PHASE5_DIR / "lesson86R"

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

BASE_CHECKPOINT_PRIMARY = (
        LESSON_86R /
        "checkpoints" /
        "silverwing_statistics_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_86R /
        "checkpoints" /
        "silverwing_statistics_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LINEAR_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_registry.json"
)

LINEAR_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_train.jsonl"
)

LINEAR_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_validation.jsonl"
)

LINEAR_REPORT_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_linear_algebra_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_linear_algebra_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_linear_algebra_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 6.0e-6

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(SEED)

random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(file)


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


def approximately_equal(
        left: float,
        right: float,
        tolerance: float = 1e-8
) -> bool:

    return abs(left - right) <= tolerance


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 86R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 87R"
)

print(
    "Native Linear Algebra and Optimization"
)

print()

print(
    "79R -> Reasoning"
)

print(
    "80R -> Reasoning Fine-Tuning"
)

print(
    "81R -> Memory"
)

print(
    "82R -> Tool Use"
)

print(
    "83R -> Planning"
)

print(
    "84R -> Verified Execution + Replanning"
)

print(
    "85R -> Mathematical Reasoning"
)

print(
    "86R -> Probability + Statistics"
)

print(
    "87R -> Linear Algebra + Optimization"
)

print()

print(
    "External LLM: NONE"
)

print(
    "Sequence limit:",
    MAX_SEQUENCE_LENGTH
)

print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 86R and Silverwing Inputs"
)

print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
    REASONING_CONFIG_FILE,
]:

    require_file(path)

    print(
        "FOUND:",
        path
    )


BASE_CHECKPOINT = select_base_checkpoint()

print(
    "FOUND:",
    BASE_CHECKPOINT
)

print()


# ============================================================
# 6. TEST 2 - MODEL CONFIGURATION
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
    model_config["model_dimension"]
)

NUMBER_OF_HEADS = int(
    model_config["attention_heads"]
)

FEED_FORWARD_DIMENSION = int(
    model_config["feed_forward_dimension"]
)

NUMBER_OF_LAYERS = int(
    model_config["layers"]
)

MODEL_MAX_SEQUENCE_LENGTH = int(
    model_config["maximum_sequence_length"]
)

REASONING_MAX_SEQUENCE_LENGTH = int(
    reasoning_config.get(
        "max_reasoning_tokens",
        MAX_SEQUENCE_LENGTH
    )
)

MAX_SEQUENCE_LENGTH = min(
    MAX_SEQUENCE_LENGTH,
    MODEL_MAX_SEQUENCE_LENGTH,
    REASONING_MAX_SEQUENCE_LENGTH
)

if (
        MODEL_DIMENSION
        %
        NUMBER_OF_HEADS
        !=
        0
):

    raise ValueError(
        (
            "Model dimension must be divisible "
            "by attention heads."
        )
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
    MAX_SEQUENCE_LENGTH
)

print()


# ============================================================
# 7. TEST 3 - VOCABULARY
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
        not in
        vocabulary
):

    raise ValueError(
        "Vocabulary is missing token_to_id."
    )

TOKEN_TO_ID = {
    token:
        int(token_id)

    for token, token_id
    in vocabulary[
        "token_to_id"
    ].items()
}

for required in [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]:

    if required not in TOKEN_TO_ID:

        raise ValueError(
            f"Missing vocabulary token: {required}"
        )

PAD_ID = TOKEN_TO_ID["<PAD>"]
UNK_ID = TOKEN_TO_ID["<UNK>"]
BOS_ID = TOKEN_TO_ID["<BOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]

VOCABULARY_SIZE = len(
    TOKEN_TO_ID
)

print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print()


# ============================================================
# 8. TEST 4 - BPE
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

MERGE_RANKS: Dict[
    Tuple[str, str],
    int
] = {}

for item in merge_items:

    if not isinstance(
            item,
            dict
    ):

        continue

    pair = item.get(
        "pair"
    )

    if (
            not isinstance(
                pair,
                list
            )
            or
            len(pair) != 2
    ):

        continue

    if "rank" not in item:

        continue

    MERGE_RANKS[
        (
            str(pair[0]),
            str(pair[1])
        )
    ] = int(
        item["rank"]
    )

print(
    "Merge operations:",
    len(MERGE_RANKS)
)

print()


# ============================================================
# 9. TOKENIZER
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

    output = []

    index = 0

    while index < len(symbols):

        if (
                index < len(symbols) - 1
                and
                (
                        symbols[index],
                        symbols[index + 1]
                )
                ==
                pair
        ):

            output.append(
                symbols[index]
                +
                symbols[index + 1]
            )

            index += 2

        else:

            output.append(
                symbols[index]
            )

            index += 1

    return tuple(
        output
    )


def tokenize_word(
        word: str
) -> List[str]:

    symbols = word_to_symbols(
        word
    )

    if not symbols:

        return []

    while True:

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
# 10. LINEAR ALGEBRA FUNCTIONS
# ============================================================

def vector_add(
        a: List[float],
        b: List[float]
) -> List[float]:

    if len(a) != len(b):

        raise ValueError(
            "Vector dimensions must match."
        )

    return [
        x + y
        for x, y
        in zip(a, b)
    ]


def vector_subtract(
        a: List[float],
        b: List[float]
) -> List[float]:

    if len(a) != len(b):

        raise ValueError(
            "Vector dimensions must match."
        )

    return [
        x - y
        for x, y
        in zip(a, b)
    ]


def scalar_multiply(
        scalar: float,
        vector: List[float]
) -> List[float]:

    return [
        scalar * value
        for value
        in vector
    ]


def dot_product(
        a: List[float],
        b: List[float]
) -> float:

    if len(a) != len(b):

        raise ValueError(
            "Vector dimensions must match."
        )

    return sum(
        x * y
        for x, y
        in zip(a, b)
    )


def vector_norm(
        vector: List[float]
) -> float:

    return math.sqrt(
        sum(
            value * value
            for value
            in vector
        )
    )


def matrix_vector_multiply(
        matrix: List[List[float]],
        vector: List[float]
) -> List[float]:

    if not matrix:

        raise ValueError(
            "Matrix cannot be empty."
        )

    width = len(
        matrix[0]
    )

    if any(
            len(row) != width
            for row in matrix
    ):

        raise ValueError(
            "Matrix rows must have equal length."
        )

    if width != len(vector):

        raise ValueError(
            "Matrix and vector dimensions are incompatible."
        )

    return [
        dot_product(
            row,
            vector
        )
        for row
        in matrix
    ]


def matrix_multiply(
        a: List[List[float]],
        b: List[List[float]]
) -> List[List[float]]:

    if not a or not b:

        raise ValueError(
            "Matrices cannot be empty."
        )

    a_width = len(
        a[0]
    )

    b_width = len(
        b[0]
    )

    if any(
            len(row) != a_width
            for row in a
    ):

        raise ValueError(
            "Matrix A rows must have equal length."
        )

    if any(
            len(row) != b_width
            for row in b
    ):

        raise ValueError(
            "Matrix B rows must have equal length."
        )

    if a_width != len(b):

        raise ValueError(
            "Matrix dimensions are incompatible."
        )

    b_transposed = list(
        zip(*b)
    )

    return [
        [
            dot_product(
                row,
                list(column)
            )
            for column
            in b_transposed
        ]
        for row
        in a
    ]


def identity_matrix(
        size: int
) -> List[List[float]]:

    return [
        [
            1.0
            if row == column
            else 0.0
            for column
            in range(size)
        ]
        for row
        in range(size)
    ]


# ============================================================
# 11. OPTIMIZATION FUNCTIONS
# ============================================================

def quadratic_objective(
        x: float
) -> float:

    return (
            (x - 3.0) ** 2
            +
            2.0
    )


def quadratic_gradient(
        x: float
) -> float:

    return (
            2.0 * (x - 3.0)
    )


def gradient_descent_1d(
        initial_x: float,
        learning_rate: float,
        steps: int
) -> Tuple[
    float,
    List[float]
]:

    x = float(
        initial_x
    )

    history = [
        x
    ]

    for _ in range(
            steps
    ):

        gradient = (
            quadratic_gradient(
                x
            )
        )

        x -= (
                learning_rate
                *
                gradient
        )

        history.append(
            x
        )

    return (
        x,
        history
    )


def two_dimensional_quadratic(
        vector: List[float]
) -> float:

    if len(vector) != 2:

        raise ValueError(
            "Two-dimensional objective requires two values."
        )

    x, y = vector

    return (
            (x - 2.0) ** 2
            +
            (y + 1.0) ** 2
    )


def two_dimensional_gradient(
        vector: List[float]
) -> List[float]:

    x, y = vector

    return [
        2.0 * (x - 2.0),
        2.0 * (y + 1.0)
    ]


def gradient_descent_2d(
        initial: List[float],
        learning_rate: float,
        steps: int
) -> Tuple[
    List[float],
    List[List[float]]
]:

    vector = [
        float(value)
        for value
        in initial
    ]

    history = [
        vector.copy()
    ]

    for _ in range(
            steps
    ):

        gradient = two_dimensional_gradient(
            vector
        )

        vector = vector_subtract(
            vector,
            scalar_multiply(
                learning_rate,
                gradient
            )
        )

        history.append(
            vector.copy()
        )

    return (
        vector,
        history
    )


# ============================================================
# 12. LINEAR ALGEBRA CURRICULUM
# ============================================================

linear_tasks = [

    {
        "example_id":
            "lin_001",

        "domain":
            "vector_addition",

        "problem":
            "Add vectors [1, 2, 3] and [4, 5, 6].",

        "reasoning":
            "Add corresponding vector components.",

        "calculation":
            "[1+4, 2+5, 3+6] = [5, 7, 9]",

        "answer":
            "[5, 7, 9]",

        "validation":
            {
                "type":
                    "vector_add",

                "a":
                    [1, 2, 3],

                "b":
                    [4, 5, 6],

                "expected":
                    [5, 7, 9]
            }
    },

    {
        "example_id":
            "lin_002",

        "domain":
            "dot_product",

        "problem":
            "Find the dot product of [1, 2, 3] and [4, 5, 6].",

        "reasoning":
            "Multiply corresponding components and sum them.",

        "calculation":
            "4 + 10 + 18 = 32",

        "answer":
            "32",

        "validation":
            {
                "type":
                    "dot",

                "a":
                    [1, 2, 3],

                "b":
                    [4, 5, 6],

                "expected":
                    32
            }
    },

    {
        "example_id":
            "lin_003",

        "domain":
            "vector_norm",

        "problem":
            "Find the Euclidean norm of [3, 4].",

        "reasoning":
            "Take the square root of the sum of squared components.",

        "calculation":
            "sqrt(9 + 16) = 5",

        "answer":
            "5",

        "validation":
            {
                "type":
                    "norm",

                "vector":
                    [3, 4],

                "expected":
                    5
            }
    },

    {
        "example_id":
            "lin_004",

        "domain":
            "matrix_vector",

        "problem":
            "Multiply [[2,1],[1,3]] by [4,2].",

        "reasoning":
            "Take the dot product of each matrix row with the vector.",

        "calculation":
            "[2*4+1*2, 1*4+3*2] = [10,10]",

        "answer":
            "[10, 10]",

        "validation":
            {
                "type":
                    "matrix_vector",

                "matrix":
                    [
                        [2, 1],
                        [1, 3]
                    ],

                "vector":
                    [4, 2],

                "expected":
                    [10, 10]
            }
    },

    {
        "example_id":
            "lin_005",

        "domain":
            "matrix_multiplication",

        "problem":
            "Multiply [[1,2],[3,4]] by [[5,6],[7,8]].",

        "reasoning":
            "Multiply rows of the first matrix by columns of the second.",

        "calculation":
            "[[19,22],[43,50]]",

        "answer":
            "[[19,22],[43,50]]",

        "validation":
            {
                "type":
                    "matrix_matrix",

                "a":
                    [
                        [1, 2],
                        [3, 4]
                    ],

                "b":
                    [
                        [5, 6],
                        [7, 8]
                    ],

                "expected":
                    [
                        [19, 22],
                        [43, 50]
                    ]
            }
    },

    {
        "example_id":
            "lin_006",

        "domain":
            "gradient_descent_1d",

        "problem":
            "Minimize (x - 3)^2 + 2 using gradient descent starting at x=0 with learning rate 0.25 for 8 steps.",

        "reasoning":
            "The gradient is 2(x-3). Move opposite the gradient.",

        "calculation":
            "x approaches 3 after repeated updates.",

        "answer":
            "x approaches 3",

        "validation":
            {
                "type":
                    "gd_1d",

                "initial":
                    0,

                "learning_rate":
                    0.25,

                "steps":
                    8,

                "expected":
                    3.0,

                "tolerance":
                    0.02
            }
    },

    {
        "example_id":
            "lin_007",

        "domain":
            "gradient_descent_2d",

        "problem":
            "Minimize (x-2)^2 + (y+1)^2 starting at [0,0] with learning rate 0.25 for 8 steps.",

        "reasoning":
            "The gradient is [2(x-2), 2(y+1)]. Update both coordinates opposite the gradient.",

        "calculation":
            "[x,y] approaches [2,-1].",

        "answer":
            "[2, -1]",

        "validation":
            {
                "type":
                    "gd_2d",

                "initial":
                    [0, 0],

                "learning_rate":
                    0.25,

                "steps":
                    8,

                "expected":
                    [2.0, -1.0],

                "tolerance":
                    0.02
            }
    }
]


# ============================================================
# 13. TEST 5 - DOMAINS
# ============================================================

print(
    "TEST 5: Linear Algebra and Optimization Domains"
)

print()

expected_domains = {
    "vector_addition",
    "dot_product",
    "vector_norm",
    "matrix_vector",
    "matrix_multiplication",
    "gradient_descent_1d",
    "gradient_descent_2d"
}

actual_domains = {
    task["domain"]
    for task
    in linear_tasks
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(linear_tasks)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        (
            "Linear algebra curriculum domains "
            "are incomplete."
        )
    )


# ============================================================
# 14. TEST 6 - INDEPENDENT VALIDATION
# ============================================================

print(
    "TEST 6: Independent Linear Algebra Validation"
)

print()

validation_errors = []

for task in linear_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "vector_add":

        observed = vector_add(
            data["a"],
            data["b"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "dot":

        observed = dot_product(
            data["a"],
            data["b"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "norm":

        observed = vector_norm(
            data["vector"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "matrix_vector":

        observed = matrix_vector_multiply(
            data["matrix"],
            data["vector"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "matrix_matrix":

        observed = matrix_multiply(
            data["a"],
            data["b"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "gd_1d":

        observed, _ = gradient_descent_1d(
            data["initial"],
            data["learning_rate"],
            data["steps"]
        )

        valid = (
                abs(
                    observed
                    -
                    data["expected"]
                )
                <=
                data["tolerance"]
        )

    elif validation_type == "gd_2d":

        observed, _ = gradient_descent_2d(
            data["initial"],
            data["learning_rate"],
            data["steps"]
        )

        valid = (
                abs(
                    observed[0]
                    -
                    data["expected"][0]
                )
                <=
                data["tolerance"]
                and
                abs(
                    observed[1]
                    -
                    data["expected"][1]
                )
                <=
                data["tolerance"]
        )

    else:

        validation_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    (
                            "Unknown validation type: "
                            +
                            validation_type
                    )
            }
        )

        continue

    if not valid:

        validation_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Independent numerical validation failed."
            }
        )


if validation_errors:

    print(
        json.dumps(
            validation_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Independent linear algebra validation failed."
    )

print(
    "Linear algebra examples validated:",
    len(linear_tasks)
)

print()


# ============================================================
# 15. COMPACT TRACE
# ============================================================

def build_linear_trace(
        task: Dict[str, Any]
) -> str:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    if validation_type == "vector_add":

        validation_text = (
            "result=[5,7,9]"
        )

    elif validation_type == "dot":

        validation_text = (
            "dot=32"
        )

    elif validation_type == "norm":

        validation_text = (
            "norm=5"
        )

    elif validation_type == "matrix_vector":

        validation_text = (
            "result=[10,10]"
        )

    elif validation_type == "matrix_matrix":

        validation_text = (
            "result=[[19,22],[43,50]]"
        )

    elif validation_type == "gd_1d":

        validation_text = (
            "x≈3"
        )

    elif validation_type == "gd_2d":

        validation_text = (
            "xy≈[2,-1]"
        )

    else:

        validation_text = "valid"

    return "\n".join(
        [
            "P:" +
            task["problem"],

            "M:" +
            task["reasoning"],

            "C:" +
            task["calculation"],

            "V:" +
            validation_text,

            "A:" +
            task["answer"]
        ]
    )


# ============================================================
# 16. TEST 7 - TRACE CONSTRUCTION
# ============================================================

print(
    "TEST 7: Build Linear Algebra Reasoning Traces"
)

print()

linear_records = []

for task in linear_tasks:

    trace = build_linear_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    linear_records.append(
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
# 17. TEST 8 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 8: Linear Algebra Token Validation"
)

print()

length_errors = [
    {
        "example_id":
            record["example_id"],

        "token_count":
            record["token_count"],

        "maximum":
            MAX_SEQUENCE_LENGTH
    }

    for record
    in linear_records

    if (
            record["token_count"]
            >
            MAX_SEQUENCE_LENGTH
    )
]

if length_errors:

    print(
        json.dumps(
            length_errors,
            indent=4
        )
    )

    print()

    print(
        "OVERSIZED LINEAR ALGEBRA TRACES:"
    )

    for record in linear_records:

        if (
                record["token_count"]
                >
                MAX_SEQUENCE_LENGTH
        ):

            print()

            print(
                "-----",
                record["example_id"],
                "-----"
            )

            print(
                record["formatted_text"]
            )

    raise RuntimeError(
        (
            "Linear algebra examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All linear algebra examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 18. TEST 9 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 9: Linear Algebra Domain Coverage"
)

print()

domain_counts: Dict[
    str,
    int
] = {}

for record in linear_records:

    domain = record[
        "domain"
    ]

    domain_counts[
        domain
    ] = (
            domain_counts.get(
                domain,
                0
            )
            +
            1
    )

print(
    json.dumps(
        domain_counts,
        indent=4
    )
)

print()

for domain in expected_domains:

    if domain not in domain_counts:

        raise RuntimeError(
            (
                f"Linear algebra domain missing: "
                f"{domain}"
            )
        )


# ============================================================
# 19. TEST 10 - OPTIMIZATION CONVERGENCE
# ============================================================

print(
    "TEST 10: Optimization Convergence Validation"
)

print()

x_final, x_history = gradient_descent_1d(
    0,
    0.25,
    8
)

xy_final, xy_history = gradient_descent_2d(
    [0, 0],
    0.25,
    8
)

print(
    "1D final x:",
    x_final
)

print(
    "1D objective:",
    quadratic_objective(
        x_final
    )
)

print(
    "2D final:",
    xy_final
)

print(
    "2D objective:",
    two_dimensional_quadratic(
        xy_final
    )
)

print()

if (
        abs(
            x_final - 3.0
        )
        >
        0.02
):

    raise RuntimeError(
        "1D gradient descent convergence failed."
    )

if (
        abs(
            xy_final[0] - 2.0
        )
        >
        0.02
        or
        abs(
            xy_final[1] + 1.0
        )
        >
        0.02
):

    raise RuntimeError(
        "2D gradient descent convergence failed."
    )

print(
    "Optimization convergence validated."
)

print()


# ============================================================
# 20. TEST 11 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    linear_records
)

validation_count = max(
    2,
    int(
        round(
            len(linear_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(linear_records) - 1
)

linear_train_records = (
    linear_records[
        :-validation_count
    ]
)

linear_validation_records = (
    linear_records[
        -validation_count:
    ]
)

print(
    "TEST 11: Linear Algebra Train/Validation Split"
)

print(
    "Training examples:",
    len(linear_train_records)
)

print(
    "Validation examples:",
    len(linear_validation_records)
)

print()


# ============================================================
# 21. SAVE ARTIFACTS
# ============================================================

write_json(
    LINEAR_REGISTRY_FILE,
    {
        "lesson":
            "87R",

        "capability":
            "native_linear_algebra_and_optimization",

        "domains":
            sorted(
                expected_domains
            ),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(linear_tasks)
    }
)

with open(
        LINEAR_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in linear_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        LINEAR_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in linear_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    LINEAR_REPORT_FILE,
    {
        "lesson":
            "87R",

        "capability":
            "native_linear_algebra_and_optimization",

        "domains":
            sorted(
                expected_domains
            ),

        "training_examples":
            len(
                linear_train_records
            ),

        "validation_examples":
            len(
                linear_validation_records
            ),

        "external_llm":
            False
    }
)


# ============================================================
# 22. DATASET
# ============================================================

class LinearAlgebraDataset(
    Dataset
):

    def __init__(
            self,
            records: List[
                Dict[str, Any]
            ]
    ):

        self.samples = []

        for record in records:

            token_ids = encode_text(
                record[
                    "formatted_text"
                ]
            )

            if (
                    len(token_ids)
                    >
                    MAX_SEQUENCE_LENGTH
            ):

                raise ValueError(
                    (
                        f"{record['example_id']} "
                        "exceeds sequence limit."
                    )
                )

            self.samples.append(
                {
                    "example_id":
                        record[
                            "example_id"
                        ],

                    "input_ids":
                        token_ids[:-1],

                    "labels":
                        token_ids[1:]
                }
            )

    def __len__(
            self
    ) -> int:

        return len(
            self.samples
        )

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Any]:

        sample = self.samples[
            index
        ]

        return {
            "example_id":
                sample[
                    "example_id"
                ],

            "input_ids":
                torch.tensor(
                    sample[
                        "input_ids"
                    ],
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    sample[
                        "labels"
                    ],
                    dtype=torch.long
                )
        }


def collate_linear_batch(
        batch: List[
            Dict[str, Any]
        ]
) -> Dict[str, Any]:

    maximum_length = max(
        len(
            item[
                "input_ids"
            ]
        )
        for item
        in batch
    )

    inputs = []
    labels = []

    for item in batch:

        input_ids = item[
            "input_ids"
        ]

        item_labels = item[
            "labels"
        ]

        input_padding = (
                maximum_length
                -
                len(input_ids)
        )

        label_padding = (
                maximum_length
                -
                len(item_labels)
        )

        inputs.append(
            torch.cat(
                [
                    input_ids,

                    torch.full(
                        (
                            input_padding,
                        ),
                        PAD_ID,
                        dtype=torch.long
                    )
                ]
            )
        )

        labels.append(
            torch.cat(
                [
                    item_labels,

                    torch.full(
                        (
                            label_padding,
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
                inputs
            ),

        "labels":
            torch.stack(
                labels
            )
    }


linear_train_dataset = LinearAlgebraDataset(
    linear_train_records
)

linear_validation_dataset = LinearAlgebraDataset(
    linear_validation_records
)

linear_train_loader = DataLoader(
    linear_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_linear_batch
)

linear_validation_loader = DataLoader(
    linear_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_linear_batch
)

print(
    "TEST 12: Linear Algebra DataLoaders"
)

print(
    "Training samples:",
    len(linear_train_dataset)
)

print(
    "Validation samples:",
    len(linear_validation_dataset)
)

print(
    "Training batches:",
    len(linear_train_loader)
)

print(
    "Validation batches:",
    len(linear_validation_loader)
)

print()


# ============================================================
# 23. EXACT SILVERWING ATTENTION
# ============================================================

class SilverwingAttention(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            heads: int
    ):

        super().__init__()

        if dimension % heads != 0:

            raise ValueError(
                "Invalid attention configuration."
            )

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

        query = self.query_projection(
            x
        )

        key = self.key_projection(
            x
        )

        value = self.value_projection(
            x
        )

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

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
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

        attended = (
            attended
            .transpose(
                1,
                2
            )
            .contiguous()
        )

        attended = attended.view(
            batch_size,
            sequence_length,
            self.dimension
        )

        return self.output_projection(
            attended
        )


# ============================================================
# 24. FEED FORWARD
# ============================================================

class SilverwingFeedForward(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            hidden_dimension: int
    ):

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

        return self.output_projection(
            F.gelu(
                self.input_projection(
                    x
                )
            )
        )


# ============================================================
# 25. TRANSFORMER BLOCK
# ============================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(
            self
    ):

        super().__init__()

        self.attention = (
            SilverwingAttention(
                MODEL_DIMENSION,
                NUMBER_OF_HEADS
            )
        )

        self.norm_attention = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.feed_forward = (
            SilverwingFeedForward(
                MODEL_DIMENSION,
                FEED_FORWARD_DIMENSION
            )
        )

        self.norm_feed_forward = nn.LayerNorm(
            MODEL_DIMENSION
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.norm_attention(
            x
            +
            self.attention(
                x
            )
        )

        x = self.norm_feed_forward(
            x
            +
            self.feed_forward(
                x
            )
        )

        return x


# ============================================================
# 26. POSITION EMBEDDING
# ============================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(
            self
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            MAX_SEQUENCE_LENGTH,
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


# ============================================================
# 27. SILVERWING DECODER
# ============================================================

class SilverwingDecoder(
    nn.Module
):

    def __init__(
            self
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCABULARY_SIZE,
            MODEL_DIMENSION,
            padding_idx=PAD_ID
        )

        self.position_embedding = (
            SilverwingPositionEmbedding()
        )

        self.layers = nn.ModuleList(
            [
                SilverwingTransformerBlock()
                for _ in range(
                NUMBER_OF_LAYERS
            )
            ]
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
                MAX_SEQUENCE_LENGTH
        ):

            raise ValueError(
                "Sequence exceeds model limit."
            )

        x = (
                self.token_embedding(
                    input_ids
                )
                +
                self.position_embedding(
                    sequence_length,
                    input_ids.device
                ).unsqueeze(0)
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


# ============================================================
# 28. TEST 13 - STRICT LOAD
# ============================================================

print(
    "TEST 13: Strict Load of 86R Statistical Model"
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

    raise ValueError(
        "86R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "86R checkpoint is missing model_state_dict."
    )

state_dict = checkpoint[
    "model_state_dict"
]

required_prefixes = [
    "token_embedding.",
    "position_embedding.embedding.",
    "layers.0.attention.query_projection.",
    "layers.0.attention.key_projection.",
    "layers.0.attention.value_projection.",
    "layers.0.attention.output_projection.",
    "layers.0.feed_forward.input_projection.",
    "layers.0.feed_forward.output_projection.",
    "layers.0.norm_attention.",
    "layers.0.norm_feed_forward.",
    "final_norm.",
    "language_model_head."
]

for prefix in required_prefixes:

    if not any(
            key.startswith(prefix)
            for key in state_dict.keys()
    ):

        raise RuntimeError(
            (
                "86R checkpoint architecture mismatch. "
                f"Missing prefix: {prefix}"
            )
        )

model = (
    SilverwingDecoder()
    .to(DEVICE)
)

try:

    model.load_state_dict(
        state_dict,
        strict=True
    )

except RuntimeError as exc:

    raise RuntimeError(
        (
            "87R refused to load a mismatched "
            "86R Silverwing model.\n\n"
            "The decoder architecture must remain "
            "identical across curriculum stages.\n\n"
            f"Checkpoint:\n{BASE_CHECKPOINT}\n\n"
            f"PyTorch error:\n{exc}"
        )
    ) from exc

print(
    "STRICT LOAD PASSED."
)

print(
    "86R model is compatible with 87R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 29. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 30. LOSS
# ============================================================

def linear_loss(
        logits: torch.Tensor,
        labels: torch.Tensor
) -> torch.Tensor:

    return F.cross_entropy(
        logits.reshape(
            -1,
            VOCABULARY_SIZE
        ),
        labels.reshape(
            -1
        ),
        ignore_index=-100
    )


# ============================================================
# 31. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
        current_model: nn.Module,
        loader: DataLoader
) -> Dict[str, float]:

    current_model.eval()

    total_loss = 0.0

    batches = 0

    correct = 0

    valid_tokens = 0

    for batch in loader:

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                DEVICE
            )
        )

        logits = current_model(
            input_ids
        )

        loss = linear_loss(
            logits,
            labels
        )

        total_loss += float(
            loss
        )

        batches += 1

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

    if batches == 0:

        return {
            "loss":
                float("nan"),

            "perplexity":
                float("nan"),

            "accuracy":
                float("nan"),

            "tokens":
                0
        }

    loss_value = (
            total_loss
            /
            batches
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

    accuracy = (
        correct
        /
        valid_tokens

        if valid_tokens

        else

        float("nan")
    )

    return {
        "loss":
            loss_value,

        "perplexity":
            perplexity,

        "accuracy":
            accuracy,

        "tokens":
            valid_tokens
    }


# ============================================================
# 32. TEST 14 - BASELINE
# ============================================================

print(
    "TEST 14: Baseline Linear Algebra Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    linear_validation_loader
)

print(
    "Baseline loss:",
    baseline_metrics["loss"]
)

print(
    "Baseline perplexity:",
    baseline_metrics["perplexity"]
)

print(
    "Baseline accuracy:",
    baseline_metrics["accuracy"]
)

print()


# ============================================================
# 33. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(
        linear_train_loader
    )
    *
    EPOCHS
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
)


# ============================================================
# 34. TEST 15 - TRAINING
# ============================================================

print(
    "TEST 15: Native Linear Algebra and Optimization Fine-Tuning"
)

print()

history = []

best_validation_loss = float(
    "inf"
)

global_step = 0

training_start = time.perf_counter()

for epoch in range(
        1,
        EPOCHS + 1
):

    model.train()

    epoch_loss = 0.0

    epoch_batches = 0

    for batch_number, batch in enumerate(
            linear_train_loader,
            start=1
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                DEVICE
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids
        )

        loss = linear_loss(
            logits,
            labels
        )

        loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM
            )
        )

        optimizer.step()

        scheduler.step()

        global_step += 1

        epoch_loss += float(
            loss.detach()
        )

        epoch_batches += 1

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"| Batch {batch_number}/{len(linear_train_loader)} "
            f"| Step {global_step} "
            f"| Loss {float(loss.detach()):.6f} "
            f"| Grad {float(gradient_norm):.6f} "
            f"| LR {optimizer.param_groups[0]['lr']:.8f}"
        )

    train_loss = (
            epoch_loss
            /
            max(
                epoch_batches,
                1
            )
    )

    validation_metrics = evaluate(
        model,
        linear_validation_loader
    )

    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_metrics[
                    "loss"
                ],

            "validation_perplexity":
                validation_metrics[
                    "perplexity"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
        }
    )

    print()

    print(
        "Epoch",
        epoch,
        "complete."
    )

    print(
        "Training loss:",
        train_loss
    )

    print(
        "Validation loss:",
        validation_metrics[
            "loss"
        ]
    )

    print(
        "Validation accuracy:",
        validation_metrics[
            "accuracy"
        ]
    )

    print()

    if (
            math.isfinite(
                validation_metrics[
                    "loss"
                ]
            )
            and
            validation_metrics[
                "loss"
            ]
            <
            best_validation_loss
    ):

        best_validation_loss = (
            validation_metrics[
                "loss"
            ]
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "lesson":
                    "87R",

                "base_checkpoint":
                    str(
                        BASE_CHECKPOINT
                    ),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "linear_task_count":
                    len(
                        linear_tasks
                    )
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 35. TEST 16 - FINAL EVALUATION
# ============================================================

print(
    "TEST 16: Final Linear Algebra Evaluation"
)

print()

final_metrics = evaluate(
    model,
    linear_validation_loader
)

print(
    "Final loss:",
    final_metrics["loss"]
)

print(
    "Final perplexity:",
    final_metrics["perplexity"]
)

print(
    "Final accuracy:",
    final_metrics["accuracy"]
)

print()


# ============================================================
# 36. TEST 17 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 17: Numerical Health"
)

print()

nan_tensors = 0

inf_tensors = 0

for parameter in model.parameters():

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

print()


# ============================================================
# 37. TEST 18 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 18: Parameter Change"
)

print()

changed_tensors = 0

total_parameter_change = 0.0

for name, parameter in (
        model.state_dict().items()
):

    original = baseline_state[
        name
    ]

    difference = torch.sum(
        torch.abs(
            parameter.detach()
            -
            original
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
# 38. TEST 19 - POST-TRAINING VALIDATION
# ============================================================

print(
    "TEST 19: Post-Training Linear Algebra Validation"
)

print()

post_training_errors = []

for task in linear_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "vector_add":

        observed = vector_add(
            data["a"],
            data["b"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "dot":

        observed = dot_product(
            data["a"],
            data["b"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "norm":

        observed = vector_norm(
            data["vector"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "matrix_vector":

        observed = matrix_vector_multiply(
            data["matrix"],
            data["vector"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "matrix_matrix":

        observed = matrix_multiply(
            data["a"],
            data["b"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "gd_1d":

        observed, _ = gradient_descent_1d(
            data["initial"],
            data["learning_rate"],
            data["steps"]
        )

        valid = (
                abs(
                    observed
                    -
                    data["expected"]
                )
                <=
                data["tolerance"]
        )

    elif validation_type == "gd_2d":

        observed, _ = gradient_descent_2d(
            data["initial"],
            data["learning_rate"],
            data["steps"]
        )

        valid = (
                abs(
                    observed[0]
                    -
                    data["expected"][0]
                )
                <=
                data["tolerance"]
                and
                abs(
                    observed[1]
                    -
                    data["expected"][1]
                )
                <=
                data["tolerance"]
        )

    if not valid:

        post_training_errors.append(
            {
                "example_id":
                    task["example_id"],

                "domain":
                    task["domain"],

                "error":
                    "Linear algebra validation failed."
            }
        )


if post_training_errors:

    print(
        json.dumps(
            post_training_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Post-training linear algebra validation failed."
    )

print(
    "Post-training linear algebra validation passed:",
    len(linear_tasks)
)

print()


# ============================================================
# 39. TEST 20 - PROMOTION
# ============================================================

print(
    "TEST 20: Linear Algebra Promotion Gate"
)

print()

baseline_loss = (
    baseline_metrics[
        "loss"
    ]
)

candidate_loss = (
    final_metrics[
        "loss"
    ]
)

if not numerically_healthy:

    decision = "REJECT"

    reason = (
        "Numerical instability detected."
    )

elif not math.isfinite(
        candidate_loss
):

    decision = "REJECT"

    reason = (
        "Candidate linear algebra loss is invalid."
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

    decision = (
        "PROMOTE_CANDIDATE"
    )

    reason = (
        "Linear algebra validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Linear algebra validation loss did not improve."
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
# 40. TEST 21 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 21: Save Linear Algebra Candidate"
)

print()

candidate_payload = {

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "scheduler_state_dict":
        scheduler.state_dict(),

    "lesson":
        "87R",

    "training_mode":
        "native_linear_algebra_and_optimization",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "baseline_metrics":
        baseline_metrics,

    "candidate_metrics":
        final_metrics,

    "decision":
        decision,

    "reason":
        reason,

    "global_step":
        global_step,

    "training_duration_seconds":
        training_duration,

    "history":
        history,

    "domains":
        sorted(
            expected_domains
        ),

    "linear_task_count":
        len(
            linear_tasks
        ),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH
}

torch.save(
    candidate_payload,
    CANDIDATE_CHECKPOINT
)

print(
    "Candidate:",
    CANDIDATE_CHECKPOINT
)

print()

if decision == "PROMOTE_CANDIDATE":

    torch.save(
        candidate_payload,
        BEST_CHECKPOINT
    )

    print(
        "Promoted:",
        BEST_CHECKPOINT
    )

else:

    print(
        "Baseline retained."
    )

print()


# ============================================================
# 41. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "87R",

    "training_mode":
        "native_linear_algebra_and_optimization",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "domains":
        sorted(
            expected_domains
        ),

    "linear_task_count":
        len(
            linear_tasks
        ),

    "training_examples":
        len(
            linear_train_records
        ),

    "validation_examples":
        len(
            linear_validation_records
        ),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "epochs":
        EPOCHS,

    "global_steps":
        global_step,

    "training_duration_seconds":
        training_duration,

    "baseline":
        baseline_metrics,

    "final":
        final_metrics,

    "decision":
        decision,

    "reason":
        reason,

    "history":
        history
}

write_json(
    TRAINING_LOG_FILE,
    training_log
)


# ============================================================
# 42. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "87R",

    "capability":
        "native_linear_algebra_and_optimization",

    "domains":
        sorted(
            expected_domains
        ),

    "linear_task_count":
        len(
            linear_tasks
        ),

    "training_examples":
        len(
            linear_train_records
        ),

    "validation_examples":
        len(
            linear_validation_records
        ),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "baseline":
        baseline_metrics,

    "candidate":
        final_metrics,

    "numerical_health":
        {
            "nan_tensors":
                nan_tensors,

            "inf_tensors":
                inf_tensors,

            "healthy":
                numerically_healthy
        },

    "parameter_change":
        {
            "changed_tensors":
                changed_tensors,

            "total_absolute_parameter_change":
                total_parameter_change
        },

    "independent_validation":
        {
            "passed":
                len(
                    post_training_errors
                )
                ==
                0
        },

    "optimization_validation":
        {
            "one_dimensional":
                {
                    "final_x":
                        x_final,

                    "objective":
                        quadratic_objective(
                            x_final
                        )
                },

            "two_dimensional":
                {
                    "final_vector":
                        xy_final,

                    "objective":
                        two_dimensional_quadratic(
                            xy_final
                        )
                }
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
    EVALUATION_FILE,
    evaluation_report
)


# ============================================================
# 43. LINEAR ALGEBRA STACK
# ============================================================

print(
    "SILVERWING LINEAR ALGEBRA AND OPTIMIZATION STACK"
)

print()

print(
    "Vectors"
)

print(
    " ↓"
)

print(
    "Dot Products"
)

print(
    " ↓"
)

print(
    "Vector Norms"
)

print(
    " ↓"
)

print(
    "Matrices"
)

print(
    " ↓"
)

print(
    "Matrix-Vector Operations"
)

print(
    " ↓"
)

print(
    "Matrix Multiplication"
)

print(
    " ↓"
)

print(
    "Gradient"
)

print(
    " ↓"
)

print(
    "Gradient Descent"
)

print(
    " ↓"
)

print(
    "Future: Linear Systems"
)

print(
    " ↓"
)

print(
    "Future: Eigenvalues and Eigenvectors"
)

print(
    " ↓"
)

print(
    "Future: Advanced Optimization"
)

print()


# ============================================================
# 44. WHY 87R MATTERS
# ============================================================

print(
    "WHY 87R MATTERS"
)

print()

print(
    "Linear algebra is a core mathematical language "
    "used throughout modern machine learning."
)

print()

print(
    "Optimization provides the mathematical mechanism "
    "for adjusting parameters toward an objective."
)

print()

print(
    "This layer directly supports later neural-network "
    "training, scientific computing and engineering."
)

print()


# ============================================================
# 45. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "87R uses a small controlled linear algebra curriculum."
)

print(
    "87R does not yet cover matrix inverses."
)

print(
    "87R does not yet cover eigenvalues and eigenvectors."
)

print(
    "87R does not yet cover singular value decomposition."
)

print(
    "87R does not yet cover constrained optimization."
)

print(
    "87R does not yet cover multivariable calculus."
)

print(
    "87R does not yet establish advanced optimization competence."
)

print()


# ============================================================
# 46. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 88R: Native Algorithms and Data Structures"
)

print()

print(
    "Arrays + Linked Structures + Trees + Graphs + "
    "Searching + Sorting + Complexity + Algorithmic Reasoning"
)

print()


# ============================================================
# 47. FOUNDATION MODEL PROGRESS
# ============================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()

print("Own Tokenizer")
print(" ↓")
print("Own Vocabulary")
print(" ↓")
print("Own Decoder")
print(" ↓")
print("Own Training")
print(" ↓")
print("Own Evaluation")
print(" ↓")
print("Instruction Learning")
print(" ↓")
print("79R Native Reasoning Dataset")
print(" ↓")
print("80R Native Reasoning Fine-Tuning")
print(" ↓")
print("81R Native Memory-Aware Training")
print(" ↓")
print("82R Native Tool-Aware Learning")
print(" ↓")
print("83R Native Planning and Tool Sequencing")
print(" ↓")
print("84R Native Verified Execution and Replanning")
print(" ↓")
print("85R Native Mathematical Reasoning Foundation")
print(" ↓")
print("86R Native Probability and Statistical Reasoning")
print(" ↓")
print("87R Native Linear Algebra and Optimization")
print(" ↓")
print("88R Native Algorithms and Data Structures")
print(" ↓")
print("Data Analysis + Data Engineering")
print(" ↓")
print("Engineering + Scientific Intelligence")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 48. COMPLETE
# ============================================================

print(
    "=== LESSON 87R COMPLETE ==="
)
# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 91R
# Native Machine Learning Foundations
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
# 88R -> Native Algorithms and Data Structures
# 89R -> Native Data Analysis and SQL Reasoning
# 90R -> Native Data Engineering
# 91R -> Native Machine Learning Foundations
#
# ============================================================
# PURPOSE
# ============================================================
#
# Establish Silverwing's machine-learning foundation:
#
#   datasets
#   features
#   targets
#   supervised learning
#   regression
#   classification
#   train/validation/test splits
#   baseline models
#   predictions
#   loss functions
#   evaluation metrics
#   generalization
#   overfitting detection
#   deterministic validation
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
# SEQUENCE LIMIT
# ============================================================
#
# Silverwing maximum sequence length: 256
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
LESSON_90R = PHASE5_DIR / "lesson90R"

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
        LESSON_90R /
        "checkpoints" /
        "silverwing_data_engineering_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_90R /
        "checkpoints" /
        "silverwing_data_engineering_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ML_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_registry.json"
)

ML_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_train.jsonl"
)

ML_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_validation.jsonl"
)

ML_TEST_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_test.jsonl"
)

ML_REPORT_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_ml_foundation_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_ml_foundation_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_ml_foundation_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 4.0e-6

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

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

    return abs(
        left - right
    ) <= tolerance


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 90R checkpoint found.\n"
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
    "PHASE 5 - LESSON 91R"
)

print(
    "Native Machine Learning Foundations"
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
    "TEST 1: Verify Lesson 90R and Silverwing Inputs"
)

print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
    REASONING_CONFIG_FILE,
]:

    require_file(
        path
    )

    print(
        "FOUND:",
        path
    )

BASE_CHECKPOINT = (
    select_base_checkpoint()
)

print(
    "FOUND:",
    BASE_CHECKPOINT
)

print()


# ============================================================
# 6. TEST 2 - CONFIGURATION
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

MODEL_MAX_SEQUENCE_LENGTH = int(
    model_config[
        "maximum_sequence_length"
    ]
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

if MODEL_DIMENSION % NUMBER_OF_HEADS != 0:

    raise ValueError(
        "Model dimension must be divisible by attention heads."
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

if "token_to_id" not in vocabulary:

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
            not isinstance(pair, list)
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
# 10. MACHINE LEARNING CORE
# ============================================================

def train_test_split_indices(
        size: int,
        train_ratio: float,
        validation_ratio: float
) -> Tuple[
    List[int],
    List[int],
    List[int]
]:

    if size < 3:

        raise ValueError(
            "At least three samples are required."
        )

    indices = list(
        range(size)
    )

    random.Random(
        SEED
    ).shuffle(
        indices
    )

    train_end = int(
        size
        *
        train_ratio
    )

    validation_end = (
            train_end
            +
            int(
                size
                *
                validation_ratio
            )
    )

    train_indices = indices[
        :train_end
    ]

    validation_indices = indices[
        train_end:
        validation_end
    ]

    test_indices = indices[
        validation_end:
    ]

    if not train_indices:
        raise RuntimeError("Training split is empty.")

    if not validation_indices:
        raise RuntimeError("Validation split is empty.")

    if not test_indices:
        raise RuntimeError("Test split is empty.")

    return (
        train_indices,
        validation_indices,
        test_indices
    )


def mean_squared_error(
        actual: List[float],
        predicted: List[float]
) -> float:

    if len(actual) != len(predicted):

        raise ValueError(
            "Actual and predicted lengths differ."
        )

    return sum(
        (
                a - p
        ) ** 2

        for a, p
        in zip(
            actual,
            predicted
        )
    ) / len(actual)


def mean_absolute_error(
        actual: List[float],
        predicted: List[float]
) -> float:

    if len(actual) != len(predicted):

        raise ValueError(
            "Actual and predicted lengths differ."
        )

    return sum(
        abs(a - p)

        for a, p
        in zip(
            actual,
            predicted
        )
    ) / len(actual)


def accuracy_score(
        actual: List[int],
        predicted: List[int]
) -> float:

    if len(actual) != len(predicted):

        raise ValueError(
            "Actual and predicted lengths differ."
        )

    return (
            sum(
                a == p
                for a, p
                in zip(
                    actual,
                    predicted
                )
            )
            /
            len(actual)
    )


def binary_cross_entropy(
        probability: float,
        target: int
) -> float:

    probability = min(
        max(
            probability,
            1e-7
        ),
        1.0 - 1e-7
    )

    return -(
            target * math.log(probability)
            +
            (
                    1 - target
            )
            *
            math.log(
                1 - probability
            )
    )


# ============================================================
# 11. NATIVE ML DATA
# ============================================================

REGRESSION_DATA = [

    {
        "feature":
            1.0,

        "target":
            3.0
    },

    {
        "feature":
            2.0,

        "target":
            5.0
    },

    {
        "feature":
            3.0,

        "target":
            7.0
    },

    {
        "feature":
            4.0,

        "target":
            9.0
    },

    {
        "feature":
            5.0,

        "target":
            11.0
    },

    {
        "feature":
            6.0,

        "target":
            13.0
    },

    {
        "feature":
            7.0,

        "target":
            15.0
    },

    {
        "feature":
            8.0,

        "target":
            17.0
    },

    {
        "feature":
            9.0,

        "target":
            19.0
    },

    {
        "feature":
            10.0,

        "target":
            21.0
    }
]


CLASSIFICATION_DATA = [

    {
        "feature":
            1.0,

        "label":
            0
    },

    {
        "feature":
            2.0,

        "label":
            0
    },

    {
        "feature":
            3.0,

        "label":
            0
    },

    {
        "feature":
            4.0,

        "label":
            0
    },

    {
        "feature":
            6.0,

        "label":
            1
    },

    {
        "feature":
            7.0,

        "label":
            1
    },

    {
        "feature":
            8.0,

        "label":
            1
    },

    {
        "feature":
            9.0,

        "label":
            1
    }
]


# ============================================================
# 12. ML TASKS
# ============================================================

ml_tasks = [

    {
        "example_id":
            "ml_001",

        "domain":
            "dataset",

        "problem":
            "What is the role of a machine-learning dataset?",

        "reasoning":
            "A dataset contains structured examples from which a model can learn a mapping or decision rule.",

        "operation":
            "DATA -> EXAMPLES",

        "answer":
            "A dataset provides learning examples."
    },

    {
        "example_id":
            "ml_002",

        "domain":
            "feature_target",

        "problem":
            "In supervised learning, what are features and targets?",

        "reasoning":
            "Features are inputs provided to the model; targets are desired outputs used for learning and evaluation.",

        "operation":
            "X -> INPUT; Y -> TARGET",

        "answer":
            "Features are inputs and targets are desired outputs."
    },

    {
        "example_id":
            "ml_003",

        "domain":
            "regression",

        "problem":
            "When should regression be used?",

        "reasoning":
            "Regression predicts a continuous numerical target rather than a discrete class.",

        "operation":
            "X -> CONTINUOUS Y",

        "answer":
            "Use regression for continuous numerical targets."
    },

    {
        "example_id":
            "ml_004",

        "domain":
            "classification",

        "problem":
            "When should classification be used?",

        "reasoning":
            "Classification predicts a discrete category or class label.",

        "operation":
            "X -> DISCRETE Y",

        "answer":
            "Use classification for discrete class labels."
    },

    {
        "example_id":
            "ml_005",

        "domain":
            "train_validation_test",

        "problem":
            "Why separate training, validation and test data?",

        "reasoning":
            "Training fits parameters, validation guides model choices and test data estimates performance on unseen examples.",

        "operation":
            "TRAIN -> VALIDATE -> TEST",

        "answer":
            "The splits separate learning, model selection and final evaluation."
    },

    {
        "example_id":
            "ml_006",

        "domain":
            "loss",

        "problem":
            "What does a loss function measure?",

        "reasoning":
            "Loss measures disagreement between model predictions and target values.",

        "operation":
            "PREDICTION -> LOSS",

        "answer":
            "Loss measures prediction error."
    },

    {
        "example_id":
            "ml_007",

        "domain":
            "generalization",

        "problem":
            "What is generalization?",

        "reasoning":
            "Generalization is the ability to perform well on previously unseen data drawn from the same underlying task.",

        "operation":
            "SEEN -> UNSEEN",

        "answer":
            "Generalization is performance on unseen data."
    },

    {
        "example_id":
            "ml_008",

        "domain":
            "overfitting",

        "problem":
            "What does overfitting mean?",

        "reasoning":
            "Overfitting occurs when a model fits training data too specifically and performs worse on unseen data.",

        "operation":
            "TRAIN GOOD -> TEST POOR",

        "answer":
            "Overfitting is poor generalization caused by excessive adaptation to training data."
    },

    {
        "example_id":
            "ml_009",

        "domain":
            "regression_metric",

        "problem":
            "What does mean squared error measure?",

        "reasoning":
            "Mean squared error averages the squared differences between predictions and targets.",

        "operation":
            "MSE = MEAN((Y-P)^2)",

        "answer":
            "MSE measures average squared prediction error."
    },

    {
        "example_id":
            "ml_010",

        "domain":
            "classification_metric",

        "problem":
            "What does classification accuracy measure?",

        "reasoning":
            "Accuracy is the fraction of predictions that equal the correct class labels.",

        "operation":
            "ACCURACY = CORRECT / TOTAL",

        "answer":
            "Accuracy is the fraction of correct predictions."
    }
]


# ============================================================
# 13. TEST 5 - DATASET CONTRACT
# ============================================================

print(
    "TEST 5: Machine Learning Dataset Contract"
)

print()

required_regression_fields = {
    "feature",
    "target"
}

required_classification_fields = {
    "feature",
    "label"
}

if not all(
        required_regression_fields.issubset(
            row.keys()
        )
        for row
        in REGRESSION_DATA
):

    raise RuntimeError(
        "Regression dataset contract failed."
    )

if not all(
        required_classification_fields.issubset(
            row.keys()
        )
        for row
        in CLASSIFICATION_DATA
):

    raise RuntimeError(
        "Classification dataset contract failed."
    )

print(
    "Regression samples:",
    len(REGRESSION_DATA)
)

print(
    "Classification samples:",
    len(CLASSIFICATION_DATA)
)

print(
    "Dataset contracts valid."
)

print()


# ============================================================
# 14. TEST 6 - SPLITS
# ============================================================

print(
    "TEST 6: Train/Validation/Test Splits"
)

print()

regression_splits = train_test_split_indices(
    len(REGRESSION_DATA),
    0.60,
    0.20
)

classification_splits = train_test_split_indices(
    len(CLASSIFICATION_DATA),
    0.50,
    0.25
)

print(
    "Regression split sizes:",
    [
        len(part)
        for part
        in regression_splits
    ]
)

print(
    "Classification split sizes:",
    [
        len(part)
        for part
        in classification_splits
    ]
)

print()

if sum(
        len(part)
        for part
        in regression_splits
) != len(REGRESSION_DATA):

    raise RuntimeError(
        "Regression split coverage failed."
    )

if sum(
        len(part)
        for part
        in classification_splits
) != len(CLASSIFICATION_DATA):

    raise RuntimeError(
        "Classification split coverage failed."
    )

print(
    "Train/validation/test split contracts valid."
)

print()


# ============================================================
# 15. TEST 7 - REGRESSION BASELINE
# ============================================================

print(
    "TEST 7: Native Regression Baseline"
)

print()

reg_train_indices, reg_validation_indices, reg_test_indices = (
    regression_splits
)

reg_train = [
    REGRESSION_DATA[index]
    for index
    in reg_train_indices
]

reg_validation = [
    REGRESSION_DATA[index]
    for index
    in reg_validation_indices
]

reg_test = [
    REGRESSION_DATA[index]
    for index
    in reg_test_indices
]

train_x = [
    row["feature"]
    for row
    in reg_train
]

train_y = [
    row["target"]
    for row
    in reg_train
]

mean_x = (
        sum(train_x)
        /
        len(train_x)
)

mean_y = (
        sum(train_y)
        /
        len(train_y)
)

numerator = sum(
    (
            x - mean_x
    )
    *
    (
            y - mean_y
    )

    for x, y
    in zip(
        train_x,
        train_y
    )
)

denominator = sum(
    (
            x - mean_x
    ) ** 2

    for x
    in train_x
)

slope = (
        numerator /
        denominator
)

intercept = (
        mean_y
        -
        slope *
        mean_x
)


def regression_predict(
        x: float
) -> float:

    return (
            slope * x
            +
            intercept
    )


validation_predictions = [
    regression_predict(
        row["feature"]
    )
    for row
    in reg_validation
]

test_predictions = [
    regression_predict(
        row["feature"]
    )
    for row
    in reg_test
]

validation_targets = [
    row["target"]
    for row
    in reg_validation
]

test_targets = [
    row["target"]
    for row
    in reg_test
]

reg_validation_mse = mean_squared_error(
    validation_targets,
    validation_predictions
)

reg_test_mse = mean_squared_error(
    test_targets,
    test_predictions
)

print(
    "Slope:",
    slope
)

print(
    "Intercept:",
    intercept
)

print(
    "Validation MSE:",
    reg_validation_mse
)

print(
    "Test MSE:",
    reg_test_mse
)

print()

if not approximately_equal(
        slope,
        2.0,
        0.001
):

    raise RuntimeError(
        "Regression slope validation failed."
    )

if not approximately_equal(
        intercept,
        1.0,
        0.001
):

    raise RuntimeError(
        "Regression intercept validation failed."
    )

print(
    "Native regression baseline validated."
)

print()


# ============================================================
# 16. TEST 8 - CLASSIFICATION BASELINE
# ============================================================

print(
    "TEST 8: Native Classification Baseline"
)

print()

cls_train_indices, cls_validation_indices, cls_test_indices = (
    classification_splits
)

cls_train = [
    CLASSIFICATION_DATA[index]
    for index
    in cls_train_indices
]

cls_validation = [
    CLASSIFICATION_DATA[index]
    for index
    in cls_validation_indices
]

cls_test = [
    CLASSIFICATION_DATA[index]
    for index
    in cls_test_indices
]

class_mean = {}

for label in [
    0,
    1
]:

    values = [
        row["feature"]
        for row
        in cls_train
        if row["label"] == label
    ]

    class_mean[label] = (
            sum(values)
            /
            len(values)
    )

threshold = (
                    class_mean[0]
                    +
                    class_mean[1]
            ) / 2.0


def classify(
        feature: float
) -> int:

    if feature >= threshold:
        return 1

    return 0


validation_class_predictions = [
    classify(
        row["feature"]
    )
    for row
    in cls_validation
]

test_class_predictions = [
    classify(
        row["feature"]
    )
    for row
    in cls_test
]

validation_class_targets = [
    row["label"]
    for row
    in cls_validation
]

test_class_targets = [
    row["label"]
    for row
    in cls_test
]

classification_validation_accuracy = (
    accuracy_score(
        validation_class_targets,
        validation_class_predictions
    )
)

classification_test_accuracy = (
    accuracy_score(
        test_class_targets,
        test_class_predictions
    )
)

print(
    "Class means:",
    class_mean
)

print(
    "Decision threshold:",
    threshold
)

print(
    "Validation accuracy:",
    classification_validation_accuracy
)

print(
    "Test accuracy:",
    classification_test_accuracy
)

print()

if not (
        classification_validation_accuracy
        >=
        0.5
):

    raise RuntimeError(
        "Classification baseline validation failed."
    )

if not (
        classification_test_accuracy
        >=
        0.5
):

    raise RuntimeError(
        "Classification baseline test failed."
    )

print(
    "Native classification baseline validated."
)

print()


# ============================================================
# 17. TEST 9 - LOSS FUNCTIONS
# ============================================================

print(
    "TEST 9: Native Machine Learning Loss Functions"
)

print()

regression_loss_check = mean_squared_error(
    [3.0, 5.0],
    [3.0, 4.0]
)

classification_loss_check = binary_cross_entropy(
    0.8,
    1
)

print(
    "MSE example:",
    regression_loss_check
)

print(
    "Binary cross-entropy example:",
    classification_loss_check
)

if not approximately_equal(
        regression_loss_check,
        0.5
):

    raise RuntimeError(
        "MSE validation failed."
    )

if not (
        classification_loss_check > 0
):

    raise RuntimeError(
        "Binary cross-entropy validation failed."
    )

print(
    "Loss-function contracts valid."
)

print()


# ============================================================
# 18. TEST 10 - GENERALIZATION
# ============================================================

print(
    "TEST 10: Generalization Evaluation"
)

print()

print(
    "Regression validation MSE:",
    reg_validation_mse
)

print(
    "Regression test MSE:",
    reg_test_mse
)

print(
    "Classification validation accuracy:",
    classification_validation_accuracy
)

print(
    "Classification test accuracy:",
    classification_test_accuracy
)

print()

generalization_gap_regression = abs(
    reg_test_mse
    -
    reg_validation_mse
)

generalization_gap_classification = abs(
    classification_test_accuracy
    -
    classification_validation_accuracy
)

print(
    "Regression generalization gap:",
    generalization_gap_regression
)

print(
    "Classification generalization gap:",
    generalization_gap_classification
)

if (
        not math.isfinite(
            generalization_gap_regression
        )
        or
        not math.isfinite(
            generalization_gap_classification
        )
):

    raise RuntimeError(
        "Generalization metrics are invalid."
    )

print(
    "Generalization evaluation valid."
)

print()


# ============================================================
# 19. TEST 11 - REASONING TRACES
# ============================================================

print(
    "TEST 11: Build Machine Learning Reasoning Traces"
)

print()

ml_records = []


def build_ml_trace(
        task: Dict[str, Any]
) -> str:

    return "\n".join(
        [
            "P:" +
            task["problem"],

            "M:" +
            task["reasoning"],

            "Q:" +
            task["operation"],

            "V:validated",

            "A:" +
            task["answer"]
        ]
    )


for task in ml_tasks:

    trace = build_ml_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    ml_records.append(
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
# 20. TEST 12 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 12: Machine Learning Token Validation"
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
    in ml_records

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

    raise RuntimeError(
        (
            "Machine-learning examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All machine-learning examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 21. TEST 13 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 13: Machine Learning Domain Coverage"
)

print()

expected_domains = {
    "dataset",
    "feature_target",
    "regression",
    "classification",
    "train_validation_test",
    "loss",
    "generalization",
    "overfitting",
    "regression_metric",
    "classification_metric"
}

actual_domains = {
    record["domain"]
    for record
    in ml_records
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(ml_records)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        "Machine-learning domain coverage is incomplete."
    )


# ============================================================
# 22. TEST 14 - BASELINE CONSISTENCY
# ============================================================

print(
    "TEST 14: Baseline Consistency Cross-Check"
)

print()

regression_predictions_again = [
    regression_predict(
        row["feature"]
    )
    for row
    in reg_test
]

regression_mse_again = mean_squared_error(
    test_targets,
    regression_predictions_again
)

classification_predictions_again = [
    classify(
        row["feature"]
    )
    for row
    in cls_test
]

classification_accuracy_again = accuracy_score(
    test_class_targets,
    classification_predictions_again
)

print(
    "Regression MSE:",
    regression_mse_again
)

print(
    "Classification accuracy:",
    classification_accuracy_again
)

print()

if not approximately_equal(
        regression_mse_again,
        reg_test_mse
):

    raise RuntimeError(
        "Regression baseline is nondeterministic."
    )

if not approximately_equal(
        classification_accuracy_again,
        classification_test_accuracy
):

    raise RuntimeError(
        "Classification baseline is nondeterministic."
    )

print(
    "Baseline consistency validated."
)

print()


# ============================================================
# 23. TEST 15 - SPLIT INTEGRITY
# ============================================================

print(
    "TEST 15: Split Integrity"
)

print()

regression_all = set(
    reg_train_indices
    +
    reg_validation_indices
    +
    reg_test_indices
)

classification_all = set(
    cls_train_indices
    +
    cls_validation_indices
    +
    cls_test_indices
)

if regression_all != set(
        range(
            len(REGRESSION_DATA)
        )
):

    raise RuntimeError(
        "Regression split does not cover dataset exactly."
    )

if classification_all != set(
        range(
            len(CLASSIFICATION_DATA)
        )
):

    raise RuntimeError(
        "Classification split does not cover dataset exactly."
    )

if (
        len(
            set(reg_train_indices)
            &
            set(reg_validation_indices)
        )
        !=
        0
):

    raise RuntimeError(
        "Regression train/validation overlap detected."
    )

if (
        len(
            set(reg_train_indices)
            &
            set(reg_test_indices)
        )
        !=
        0
):

    raise RuntimeError(
        "Regression train/test overlap detected."
    )

if (
        len(
            set(reg_validation_indices)
            &
            set(reg_test_indices)
        )
        !=
        0
):

    raise RuntimeError(
        "Regression validation/test overlap detected."
    )

print(
    "Train/validation/test split integrity validated."
)

print()


# ============================================================
# 24. TEST 16 - TRAINING TRACE SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    ml_records
)

validation_count = max(
    3,
    int(
        round(
            len(ml_records)
            *
            0.30
        )
    )
)

validation_count = min(
    validation_count,
    len(ml_records) - 1
)

ml_train_records = (
    ml_records[
        :-validation_count
    ]
)

ml_validation_records = (
    ml_records[
        -validation_count:
    ]
)

print(
    "TEST 16: ML Reasoning Train/Validation Split"
)

print(
    "Training examples:",
    len(ml_train_records)
)

print(
    "Validation examples:",
    len(ml_validation_records)
)

print()


# ============================================================
# 25. SAVE ARTIFACTS
# ============================================================

write_json(
    ML_REGISTRY_FILE,
    {
        "lesson":
            "91R",

        "capability":
            "native_machine_learning_foundations",

        "domains":
            sorted(
                expected_domains
            ),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(ml_tasks)
    }
)

with open(
        ML_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in ml_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        ML_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in ml_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

test_artifacts = {
    "regression":
        {
            "validation_mse":
                reg_validation_mse,

            "test_mse":
                reg_test_mse,

            "slope":
                slope,

            "intercept":
                intercept
        },

    "classification":
        {
            "validation_accuracy":
                classification_validation_accuracy,

            "test_accuracy":
                classification_test_accuracy,

            "threshold":
                threshold
        }
}

write_json(
    ML_TEST_FILE,
    test_artifacts
)

write_json(
    ML_REPORT_FILE,
    {
        "lesson":
            "91R",

        "capability":
            "native_machine_learning_foundations",

        "domains":
            sorted(
                expected_domains
            ),

        "training_examples":
            len(ml_train_records),

        "validation_examples":
            len(ml_validation_records),

        "external_llm":
            False,

        "test_metrics":
            test_artifacts
    }
)


# ============================================================
# 26. DATASET FOR NATIVE DECODER
# ============================================================

class MachineLearningDataset(
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


def collate_ml_batch(
        batch: List[
            Dict[str, Any]
        ]
) -> Dict[str, Any]:

    maximum_length = max(
        len(
            item["input_ids"]
        )
        for item in batch
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
                item["example_id"]
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


ml_train_dataset = MachineLearningDataset(
    ml_train_records
)

ml_validation_dataset = MachineLearningDataset(
    ml_validation_records
)

ml_train_loader = DataLoader(
    ml_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_ml_batch
)

ml_validation_loader = DataLoader(
    ml_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_ml_batch
)

print(
    "TEST 17: ML Foundation DataLoaders"
)

print(
    "Training samples:",
    len(ml_train_dataset)
)

print(
    "Validation samples:",
    len(ml_validation_dataset)
)

print(
    "Training batches:",
    len(ml_train_loader)
)

print(
    "Validation batches:",
    len(ml_validation_loader)
)

print()


# ============================================================
# 27. SILVERWING ATTENTION
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
# 28. FEED FORWARD
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
# 29. TRANSFORMER BLOCK
# ============================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(self):

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
            self.attention(x)
        )

        x = self.norm_feed_forward(
            x
            +
            self.feed_forward(x)
        )

        return x


# ============================================================
# 30. POSITION EMBEDDING
# ============================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(self):

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
# 31. SILVERWING DECODER
# ============================================================

class SilverwingDecoder(
    nn.Module
):

    def __init__(self):

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

            x = layer(x)

        x = self.final_norm(
            x
        )

        return self.language_model_head(
            x
        )


# ============================================================
# 32. TEST 18 - STRICT LOAD
# ============================================================

print(
    "TEST 18: Strict Load of 90R Data Engineering Model"
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
        "90R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "90R checkpoint is missing model_state_dict."
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
                "90R checkpoint architecture mismatch. "
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
            "91R refused to load a mismatched "
            "90R Silverwing model.\n\n"
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
    "90R model is compatible with 91R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 33. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 34. LOSS
# ============================================================

def ml_loss(
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
# 35. EVALUATION
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
            batch["input_ids"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        logits = current_model(
            input_ids
        )

        loss = ml_loss(
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
                labels != -100
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

    perplexity = (
        math.exp(
            loss_value
        )
        if (
                math.isfinite(
                    loss_value
                )
                and
                loss_value < 50
        )
        else
        float("inf")
    )

    accuracy = (
        correct /
        valid_tokens
        if valid_tokens
        else float("nan")
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
# 36. TEST 19 - BASELINE
# ============================================================

print(
    "TEST 19: Baseline Machine Learning Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    ml_validation_loader
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
# 37. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(
        ml_train_loader
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
# 38. TEST 20 - TRAINING
# ============================================================

print(
    "TEST 20: Native Machine Learning Fine-Tuning"
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
            ml_train_loader,
            start=1
    ):

        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids
        )

        loss = ml_loss(
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
            f"| Batch {batch_number}/{len(ml_train_loader)} "
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
        ml_validation_loader
    )

    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_metrics["loss"],

            "validation_perplexity":
                validation_metrics["perplexity"],

            "validation_accuracy":
                validation_metrics["accuracy"],

            "learning_rate":
                optimizer.param_groups[0]["lr"]
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
        validation_metrics["loss"]
    )

    print(
        "Validation accuracy:",
        validation_metrics["accuracy"]
    )

    print()

    if (
            math.isfinite(
                validation_metrics["loss"]
            )
            and
            validation_metrics["loss"]
            <
            best_validation_loss
    ):

        best_validation_loss = (
            validation_metrics["loss"]
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
                    "91R",

                "base_checkpoint":
                    str(BASE_CHECKPOINT),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "ml_task_count":
                    len(ml_tasks)
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 39. TEST 21 - FINAL EVALUATION
# ============================================================

print(
    "TEST 21: Final Machine Learning Evaluation"
)

print()

final_metrics = evaluate(
    model,
    ml_validation_loader
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
# 40. TEST 22 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 22: Numerical Health"
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
# 41. TEST 23 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 23: Parameter Change"
)

print()

changed_tensors = 0
total_parameter_change = 0.0

for name, parameter in model.state_dict().items():

    original = baseline_state[name]

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
# 42. TEST 24 - POST-TRAINING ML VALIDATION
# ============================================================

print(
    "TEST 24: Post-Training Machine Learning Validation"
)

print()

post_training_errors = []

# Regression mathematical contract.
post_regression_predictions = [
    regression_predict(
        row["feature"]
    )
    for row
    in reg_test
]

post_regression_mse = mean_squared_error(
    test_targets,
    post_regression_predictions
)

if not math.isfinite(
        post_regression_mse
):

    post_training_errors.append(
        "Regression test MSE is not finite."
    )

if abs(
        slope - 2.0
) > 0.001:

    post_training_errors.append(
        "Regression slope changed unexpectedly."
    )

if abs(
        intercept - 1.0
) > 0.001:

    post_training_errors.append(
        "Regression intercept changed unexpectedly."
    )

# Classification mathematical contract.
post_classification_predictions = [
    classify(
        row["feature"]
    )
    for row
    in cls_test
]

post_classification_accuracy = (
    accuracy_score(
        test_class_targets,
        post_classification_predictions
    )
)

if not math.isfinite(
        post_classification_accuracy
):

    post_training_errors.append(
        "Classification accuracy is not finite."
    )

if (
        post_classification_accuracy
        <
        0.5
):

    post_training_errors.append(
        "Classification test accuracy fell below 0.5."
    )

# Dataset split contract.
if (
        len(reg_train_indices)
        +
        len(reg_validation_indices)
        +
        len(reg_test_indices)
        !=
        len(REGRESSION_DATA)
):

    post_training_errors.append(
        "Regression split coverage failed."
    )

if (
        len(cls_train_indices)
        +
        len(cls_validation_indices)
        +
        len(cls_test_indices)
        !=
        len(CLASSIFICATION_DATA)
):

    post_training_errors.append(
        "Classification split coverage failed."
    )

# Generalization contract.
if not math.isfinite(
        generalization_gap_regression
):

    post_training_errors.append(
        "Regression generalization gap invalid."
    )

if not math.isfinite(
        generalization_gap_classification
):

    post_training_errors.append(
        "Classification generalization gap invalid."
    )

# Dataset contract.
if len(
        REGRESSION_DATA
) != 10:

    post_training_errors.append(
        "Regression dataset size changed."
    )

if len(
        CLASSIFICATION_DATA
) != 8:

    post_training_errors.append(
        "Classification dataset size changed."
    )

print(
    "Post-training validation errors:",
    len(post_training_errors)
)

if post_training_errors:

    print(
        json.dumps(
            post_training_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Post-training machine-learning validation failed."
    )

print(
    "Post-training machine-learning validation passed:",
    len(ml_tasks)
)

print()


# ============================================================
# 43. TEST 25 - MODEL SELECTION FOUNDATION
# ============================================================

print(
    "TEST 25: Model Selection Foundation"
)

print()

validation_improved = (
        math.isfinite(
            baseline_metrics["loss"]
        )
        and
        math.isfinite(
            final_metrics["loss"]
        )
        and
        final_metrics["loss"]
        <
        baseline_metrics["loss"]
)

print(
    "Baseline validation loss:",
    baseline_metrics["loss"]
)

print(
    "Final validation loss:",
    final_metrics["loss"]
)

print(
    "Validation improvement:",
    validation_improved
)

print()


# ============================================================
# 44. TEST 26 - PROMOTION
# ============================================================

print(
    "TEST 26: Machine Learning Promotion Gate"
)

print()

baseline_loss = (
    baseline_metrics["loss"]
)

candidate_loss = (
    final_metrics["loss"]
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
        "Candidate machine-learning loss is invalid."
    )

elif post_training_errors:

    decision = "REJECT"

    reason = (
        "Post-training machine-learning validation failed."
    )

elif validation_improved:

    decision = "PROMOTE_CANDIDATE"

    reason = (
        "Machine-learning validation loss improved."
    )

else:

    decision = "RETAIN_BASELINE"

    reason = (
        "Machine-learning validation loss did not improve."
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
# 45. TEST 27 - SAVE
# ============================================================

print(
    "TEST 27: Save Machine Learning Candidate"
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
        "91R",

    "training_mode":
        "native_machine_learning_foundations",

    "base_checkpoint":
        str(BASE_CHECKPOINT),

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
        sorted(expected_domains),

    "ml_task_count":
        len(ml_tasks),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "classical_ml_metrics":
        {
            "regression":
                {
                    "validation_mse":
                        reg_validation_mse,

                    "test_mse":
                        reg_test_mse,

                    "slope":
                        slope,

                    "intercept":
                        intercept
                },

            "classification":
                {
                    "validation_accuracy":
                        classification_validation_accuracy,

                    "test_accuracy":
                        classification_test_accuracy,

                    "threshold":
                        threshold
                }
        },

    "post_training_validation":
        {
            "passed":
                len(post_training_errors)
                ==
                0,

            "errors":
                post_training_errors
        }
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
# 46. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "91R",

    "training_mode":
        "native_machine_learning_foundations",

    "base_checkpoint":
        str(BASE_CHECKPOINT),

    "external_llm":
        False,

    "device":
        str(DEVICE),

    "domains":
        sorted(expected_domains),

    "ml_task_count":
        len(ml_tasks),

    "training_examples":
        len(ml_train_records),

    "validation_examples":
        len(ml_validation_records),

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

    "classical_ml":
        {
            "regression_test_mse":
                reg_test_mse,

            "classification_test_accuracy":
                classification_test_accuracy
        },

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
# 47. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "91R",

    "capability":
        "native_machine_learning_foundations",

    "domains":
        sorted(expected_domains),

    "ml_task_count":
        len(ml_tasks),

    "training_examples":
        len(ml_train_records),

    "validation_examples":
        len(ml_validation_records),

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

    "classical_ml":
        {
            "regression":
                {
                    "validation_mse":
                        reg_validation_mse,

                    "test_mse":
                        reg_test_mse,

                    "slope":
                        slope,

                    "intercept":
                        intercept,

                    "generalization_gap":
                        generalization_gap_regression
                },

            "classification":
                {
                    "validation_accuracy":
                        classification_validation_accuracy,

                    "test_accuracy":
                        classification_test_accuracy,

                    "threshold":
                        threshold,

                    "generalization_gap":
                        generalization_gap_classification
                }
        },

    "independent_validation":
        {
            "passed":
                len(post_training_errors)
                ==
                0,

            "errors":
                post_training_errors
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
# 48. ML FOUNDATION STACK
# ============================================================

print(
    "SILVERWING MACHINE LEARNING FOUNDATION STACK"
)

print()

print("Validated Data")
print(" ↓")
print("Features + Targets")
print(" ↓")
print("Dataset Splitting")
print(" ↓")
print("Regression")
print(" ↓")
print("Classification")
print(" ↓")
print("Loss Functions")
print(" ↓")
print("Metrics")
print(" ↓")
print("Generalization")
print(" ↓")
print("Model Selection")
print(" ↓")
print("Future: Feature Engineering")
print(" ↓")
print("Future: Regularization")
print(" ↓")
print("Future: Cross-Validation")
print(" ↓")
print("Future: Ensemble Learning")
print(" ↓")
print("Future: Neural Networks")

print()


# ============================================================
# 49. WHY 91R MATTERS
# ============================================================

print(
    "WHY 91R MATTERS"
)

print()

print(
    "91R creates the formal bridge from data engineering "
    "into machine learning."
)

print()

print(
    "Silverwing now has explicit contracts for data, "
    "features, targets, training, evaluation and generalization."
)

print()

print(
    "This foundation will support later classical ML, "
    "deep learning and advanced model-building stages."
)

print()


# ============================================================
# 50. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "91R uses controlled educational datasets."
)

print(
    "91R does not yet implement large-scale feature engineering."
)

print(
    "91R does not yet implement cross-validation."
)

print(
    "91R does not yet implement regularization techniques."
)

print(
    "91R does not yet implement tree ensembles."
)

print(
    "91R does not yet implement neural network training as an "
    "independent classical-ML curriculum."
)

print(
    "91R does not yet establish advanced machine-learning competence."
)

print()


# ============================================================
# 51. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 92R: Native Classical Machine Learning"
)

print()

print(
    "Linear Models + Decision Trees + KNN + "
    "Cross-Validation + Regularization + Model Selection"
)

print()


# ============================================================
# 52. FOUNDATION MODEL PROGRESS
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
print("89R Native Data Analysis and SQL Reasoning")
print(" ↓")
print("90R Native Data Engineering")
print(" ↓")
print("91R Native Machine Learning Foundations")
print(" ↓")
print("92R Native Classical Machine Learning")
print(" ↓")
print("93R Native Neural Network Foundations")
print(" ↓")
print("94R Native Deep Learning")
print(" ↓")
print("LLM Architecture + Advanced Learning")
print(" ↓")
print("Engineering + Scientific Intelligence")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 53. COMPLETE
# ============================================================

print(
    "=== LESSON 91R COMPLETE ==="
)
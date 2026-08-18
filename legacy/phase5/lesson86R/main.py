# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 86R
# Native Probability and Statistical Reasoning
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# Deepen Silverwing's mathematical intelligence through:
#
#   probability
#   expected value
#   population variance
#   sample variability
#   standard error
#   confidence intervals
#
# The training contract is:
#
#   Problem
#      ↓
#   Representation
#      ↓
#   Method
#      ↓
#   Calculation
#      ↓
#   Independent Validation
#      ↓
#   Final Answer
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
# The 256-token model limit remains unchanged.
#
# The script deliberately keeps mathematical traces compact.
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
LESSON_85R = PHASE5_DIR / "lesson85R"

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
        LESSON_85R /
        "checkpoints" /
        "silverwing_math_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_85R /
        "checkpoints" /
        "silverwing_math_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATISTICS_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_statistics_registry.json"
)

STATISTICS_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_statistics_train.jsonl"
)

STATISTICS_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_statistics_validation.jsonl"
)

STATISTICS_REPORT_FILE = (
        BASE_DIR /
        "silverwing_statistics_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_statistics_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_statistics_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_statistics_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_statistics_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 6.5e-6

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
            "No Lesson 85R checkpoint found.\n"
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
    "PHASE 5 - LESSON 86R"
)

print(
    "Native Probability and Statistical Reasoning"
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
    "86R -> Probability + Statistical Reasoning"
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
    "TEST 1: Verify Lesson 85R and Silverwing Inputs"
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
# 10. STATISTICAL FUNCTIONS
# ============================================================

def population_mean(
        values: List[float]
) -> float:

    if not values:

        raise ValueError(
            "Cannot calculate mean of empty data."
        )

    return (
            sum(values)
            /
            len(values)
    )


def population_variance(
        values: List[float]
) -> float:

    average = population_mean(
        values
    )

    return (
            sum(
                (
                        value -
                        average
                ) ** 2

                for value
                in values
            )
            /
            len(values)
    )


def sample_variance(
        values: List[float]
) -> float:

    if len(values) < 2:

        raise ValueError(
            "Sample variance requires at least two observations."
        )

    average = population_mean(
        values
    )

    return (
            sum(
                (
                        value -
                        average
                ) ** 2

                for value
                in values
            )
            /
            (
                    len(values) - 1
            )
    )


def sample_standard_deviation(
        values: List[float]
) -> float:

    return math.sqrt(
        sample_variance(
            values
        )
    )


def expected_value(
        values: List[float],
        probabilities: List[float]
) -> float:

    if (
            len(values)
            !=
            len(probabilities)
    ):

        raise ValueError(
            "Values and probabilities must have equal length."
        )

    if not approximately_equal(
            sum(probabilities),
            1.0
    ):

        raise ValueError(
            "Probabilities must sum to 1."
        )

    return sum(
        value * probability

        for value, probability
        in zip(
            values,
            probabilities
        )
    )


def complement_probability(
        probability: float
) -> float:

    return 1.0 - probability


def standard_error(
        standard_deviation: float,
        sample_size: int
) -> float:

    if sample_size <= 0:

        raise ValueError(
            "Sample size must be positive."
        )

    return (
            standard_deviation
            /
            math.sqrt(
                sample_size
            )
    )


def confidence_interval_95(
        sample_mean: float,
        sample_standard_deviation_value: float,
        sample_size: int
) -> Tuple[
    float,
    float
]:

    se = standard_error(
        sample_standard_deviation_value,
        sample_size
    )

    margin = (
            1.96 * se
    )

    return (
        sample_mean - margin,
        sample_mean + margin
    )


# ============================================================
# 11. STATISTICAL CURRICULUM
# ============================================================

statistics_tasks = [

    {
        "example_id":
            "stat_001",

        "domain":
            "complement_probability",

        "problem":
            "If the probability of success is 0.7, find the probability of failure.",

        "reasoning":
            "Failure is the complement of success.",

        "calculation":
            "1 - 0.7 = 0.3",

        "answer":
            "0.3",

        "validation":
            {
                "type":
                    "complement",

                "probability":
                    0.7,

                "expected":
                    0.3
            }
    },

    {
        "example_id":
            "stat_002",

        "domain":
            "expected_value",

        "problem":
            "A variable takes values 1, 2 and 3 with probabilities 0.2, 0.5 and 0.3. Find its expected value.",

        "reasoning":
            "Multiply each outcome by its probability and sum.",

        "calculation":
            "0.2 + 1.0 + 0.9 = 2.1",

        "answer":
            "2.1",

        "validation":
            {
                "type":
                    "expected_value",

                "values":
                    [
                        1,
                        2,
                        3
                    ],

                "probabilities":
                    [
                        0.2,
                        0.5,
                        0.3
                    ],

                "expected":
                    2.1
            }
    },

    {
        "example_id":
            "stat_003",

        "domain":
            "population_variance",

        "problem":
            "Find the population variance of 2, 4, 6 and 8.",

        "reasoning":
            "The mean is 5. Squared deviations are 9, 1, 1 and 9.",

        "calculation":
            "20 / 4 = 5",

        "answer":
            "5",

        "validation":
            {
                "type":
                    "population_variance",

                "values":
                    [
                        2,
                        4,
                        6,
                        8
                    ],

                "expected":
                    5.0
            }
    },

    {
        "example_id":
            "stat_004",

        "domain":
            "sample_variability",

        "problem":
            "Find the sample standard deviation of 2, 4, 6 and 8.",

        "reasoning":
            "Use n - 1 for sample variance, then take the square root.",

        "calculation":
            "sample variance = 20/3; SD = sqrt(20/3)",

        "answer":
            "2.582",

        "validation":
            {
                "type":
                    "sample_standard_deviation",

                "values":
                    [
                        2,
                        4,
                        6,
                        8
                    ],

                "expected":
                    math.sqrt(
                        20 / 3
                    )
            }
    },

    {
        "example_id":
            "stat_005",

        "domain":
            "sampling",

        "problem":
            "A population has mean 50 and standard deviation 10. Find the standard error for a sample of 25.",

        "reasoning":
            "Standard error is standard deviation divided by square root of sample size.",

        "calculation":
            "10 / sqrt(25) = 2",

        "answer":
            "2",

        "validation":
            {
                "type":
                    "standard_error",

                "standard_deviation":
                    10,

                "sample_size":
                    25,

                "expected":
                    2.0
            }
    },

    {
        "example_id":
            "stat_006",

        "domain":
            "confidence_interval",

        "problem":
            "A sample mean is 100, sample standard deviation is 10, and sample size is 25. Find the approximate 95 percent confidence interval.",

        "reasoning":
            "Compute standard error, multiply by 1.96, then add and subtract the margin.",

        "calculation":
            "SE = 2; margin = 3.92; interval = [96.08, 103.92]",

        "answer":
            "[96.08, 103.92]",

        "validation":
            {
                "type":
                    "confidence_interval",

                "sample_mean":
                    100,

                "sample_standard_deviation":
                    10,

                "sample_size":
                    25,

                "expected_lower":
                    96.08,

                "expected_upper":
                    103.92
            }
    }
]


# ============================================================
# 12. TEST 5 - DOMAINS
# ============================================================

print(
    "TEST 5: Statistical Curriculum Domains"
)

print()

expected_domains = {
    "complement_probability",
    "expected_value",
    "population_variance",
    "sample_variability",
    "sampling",
    "confidence_interval"
}

actual_domains = {
    task["domain"]
    for task in statistics_tasks
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(statistics_tasks)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        "Statistical curriculum domains are incomplete."
    )


# ============================================================
# 13. TEST 6 - INDEPENDENT VALIDATION
# ============================================================

print(
    "TEST 6: Independent Statistical Validation"
)

print()

validation_errors = []

for task in statistics_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "complement":

        observed = (
            complement_probability(
                data["probability"]
            )
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "expected_value":

        observed = expected_value(
            data["values"],
            data["probabilities"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "population_variance":

        observed = population_variance(
            data["values"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "sample_standard_deviation":

        observed = sample_standard_deviation(
            data["values"]
        )

        valid = approximately_equal(
            observed,
            data["expected"],
            1e-6
        )

    elif validation_type == "standard_error":

        observed = standard_error(
            data["standard_deviation"],
            data["sample_size"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "confidence_interval":

        lower, upper = confidence_interval_95(
            data["sample_mean"],
            data["sample_standard_deviation"],
            data["sample_size"]
        )

        valid = (
                approximately_equal(
                    lower,
                    data["expected_lower"],
                    1e-6
                )
                and
                approximately_equal(
                    upper,
                    data["expected_upper"],
                    1e-6
                )
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
                    "Independent statistical validation failed."
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
        "Independent statistical validation failed."
    )

print(
    "Statistical examples validated:",
    len(statistics_tasks)
)

print()


# ============================================================
# 14. COMPACT TRACE
# ============================================================

def build_statistical_trace(
        task: Dict[str, Any]
) -> str:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    if validation_type == "complement":

        validation_text = (
            "1-0.7=0.3"
        )

    elif validation_type == "expected_value":

        validation_text = (
            "EV=2.1"
        )

    elif validation_type == "population_variance":

        validation_text = (
            "Var=5"
        )

    elif validation_type == "sample_standard_deviation":

        validation_text = (
            "SD=2.582"
        )

    elif validation_type == "standard_error":

        validation_text = (
            "SE=2"
        )

    elif validation_type == "confidence_interval":

        validation_text = (
            "CI=[96.08,103.92]"
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
# 15. TEST 7 - TRACE CONSTRUCTION
# ============================================================

print(
    "TEST 7: Build Statistical Reasoning Traces"
)

print()

statistics_records = []

for task in statistics_tasks:

    trace = build_statistical_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    statistics_records.append(
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
# 16. TEST 8 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 8: Statistical Token Validation"
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

    for record in statistics_records

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
        "OVERSIZED STATISTICAL TRACES:"
    )

    for record in statistics_records:

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
            "Statistical examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All statistical examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 17. TEST 9 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 9: Statistical Domain Coverage"
)

print()

domain_counts: Dict[
    str,
    int
] = {}

for record in statistics_records:

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
                f"Statistical domain missing: "
                f"{domain}"
            )
        )


# ============================================================
# 18. TEST 10 - SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    statistics_records
)

validation_count = max(
    2,
    int(
        round(
            len(statistics_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(statistics_records) - 1
)

statistics_train_records = (
    statistics_records[
        :-validation_count
    ]
)

statistics_validation_records = (
    statistics_records[
        -validation_count:
    ]
)

print(
    "TEST 10: Statistical Train/Validation Split"
)

print(
    "Training examples:",
    len(statistics_train_records)
)

print(
    "Validation examples:",
    len(statistics_validation_records)
)

print()


# ============================================================
# 19. SAVE ARTIFACTS
# ============================================================

write_json(
    STATISTICS_REGISTRY_FILE,
    {
        "lesson":
            "86R",

        "capability":
            "native_probability_and_statistical_reasoning",

        "domains":
            sorted(
                expected_domains
            ),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(statistics_tasks)
    }
)

with open(
        STATISTICS_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in statistics_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        STATISTICS_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in statistics_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    STATISTICS_REPORT_FILE,
    {
        "lesson":
            "86R",

        "capability":
            "native_probability_and_statistical_reasoning",

        "domains":
            sorted(
                expected_domains
            ),

        "training_examples":
            len(statistics_train_records),

        "validation_examples":
            len(statistics_validation_records),

        "external_llm":
            False
    }
)


# ============================================================
# 20. DATASET
# ============================================================

class StatisticalDataset(
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


def collate_statistical_batch(
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
                for item in batch
            ],

        "input_ids":
            torch.stack(inputs),

        "labels":
            torch.stack(labels)
    }


statistics_train_dataset = StatisticalDataset(
    statistics_train_records
)

statistics_validation_dataset = StatisticalDataset(
    statistics_validation_records
)

statistics_train_loader = DataLoader(
    statistics_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_statistical_batch
)

statistics_validation_loader = DataLoader(
    statistics_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_statistical_batch
)

print(
    "TEST 11: Statistical DataLoaders"
)

print(
    "Training samples:",
    len(statistics_train_dataset)
)

print(
    "Validation samples:",
    len(statistics_validation_dataset)
)

print(
    "Training batches:",
    len(statistics_train_loader)
)

print(
    "Validation batches:",
    len(statistics_validation_loader)
)

print()


# ============================================================
# 21. EXACT SILVERWING ATTENTION
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
# 22. FEED FORWARD
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
# 23. TRANSFORMER BLOCK
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
# 24. POSITION EMBEDDING
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
# 25. SILVERWING DECODER
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
# 26. TEST 12 - STRICT LOAD
# ============================================================

print(
    "TEST 12: Strict Load of 85R Mathematical Model"
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
        "85R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "85R checkpoint is missing model_state_dict."
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
            key.startswith(
                prefix
            )
            for key in state_dict.keys()
    ):

        raise RuntimeError(
            (
                "85R checkpoint architecture mismatch. "
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
            "86R refused to load a mismatched "
            "85R Silverwing model.\n\n"
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
    "85R model is compatible with 86R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 27. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()
    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 28. LOSS
# ============================================================

def statistical_loss(
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
# 29. EVALUATION
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
            .to(
                DEVICE
            )
        )

        labels = (
            batch["labels"]
            .to(
                DEVICE
            )
        )

        logits = current_model(
            input_ids
        )

        loss = statistical_loss(
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
# 30. TEST 13 - BASELINE
# ============================================================

print(
    "TEST 13: Baseline Statistical Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    statistics_validation_loader
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
# 31. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(
        statistics_train_loader
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
# 32. TEST 14 - TRAINING
# ============================================================

print(
    "TEST 14: Native Probability and Statistical Fine-Tuning"
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
            statistics_train_loader,
            start=1
    ):

        input_ids = (
            batch["input_ids"]
            .to(
                DEVICE
            )
        )

        labels = (
            batch["labels"]
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

        loss = statistical_loss(
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
            f"| Batch {batch_number}/{len(statistics_train_loader)} "
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
        statistics_validation_loader
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
                    "86R",

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

                "statistical_task_count":
                    len(
                        statistics_tasks
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
# 33. TEST 15 - FINAL EVALUATION
# ============================================================

print(
    "TEST 15: Final Statistical Evaluation"
)

print()

final_metrics = evaluate(
    model,
    statistics_validation_loader
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
# 34. TEST 16 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 16: Numerical Health"
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
# 35. TEST 17 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 17: Parameter Change"
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
# 36. TEST 18 - POST-TRAINING VALIDATION
# ============================================================

print(
    "TEST 18: Post-Training Statistical Validation"
)

print()

post_training_errors = []

for task in statistics_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "complement":

        observed = (
                1.0
                -
                data["probability"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "expected_value":

        observed = expected_value(
            data["values"],
            data["probabilities"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "population_variance":

        observed = population_variance(
            data["values"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "sample_standard_deviation":

        observed = sample_standard_deviation(
            data["values"]
        )

        valid = approximately_equal(
            observed,
            data["expected"],
            1e-6
        )

    elif validation_type == "standard_error":

        observed = standard_error(
            data["standard_deviation"],
            data["sample_size"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "confidence_interval":

        lower, upper = (
            confidence_interval_95(
                data["sample_mean"],
                data["sample_standard_deviation"],
                data["sample_size"]
            )
        )

        valid = (
                approximately_equal(
                    lower,
                    data["expected_lower"],
                    1e-6
                )
                and
                approximately_equal(
                    upper,
                    data["expected_upper"],
                    1e-6
                )
        )

    if not valid:

        post_training_errors.append(
            {
                "example_id":
                    task["example_id"],

                "domain":
                    task["domain"],

                "error":
                    "Statistical validation failed."
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
        "Post-training statistical validation failed."
    )

print(
    "Post-training statistical validation passed:",
    len(statistics_tasks)
)

print()


# ============================================================
# 37. TEST 19 - PROMOTION
# ============================================================

print(
    "TEST 19: Statistical Promotion Gate"
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
        "Candidate statistical loss is invalid."
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
        "Statistical validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Statistical validation loss did not improve."
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
# 38. TEST 20 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 20: Save Statistical Candidate"
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
        "86R",

    "training_mode":
        "native_probability_and_statistical_reasoning",

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

    "statistical_task_count":
        len(
            statistics_tasks
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
# 39. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "86R",

    "training_mode":
        "native_probability_and_statistical_reasoning",

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

    "statistical_task_count":
        len(
            statistics_tasks
        ),

    "training_examples":
        len(
            statistics_train_records
        ),

    "validation_examples":
        len(
            statistics_validation_records
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
# 40. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "86R",

    "capability":
        "native_probability_and_statistical_reasoning",

    "domains":
        sorted(
            expected_domains
        ),

    "statistical_task_count":
        len(
            statistics_tasks
        ),

    "training_examples":
        len(
            statistics_train_records
        ),

    "validation_examples":
        len(
            statistics_validation_records
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
# 41. STATISTICAL INTELLIGENCE STACK
# ============================================================

print(
    "SILVERWING STATISTICAL INTELLIGENCE STACK"
)

print()

print("Basic Probability")
print(" ↓")
print("Complementary Probability")
print(" ↓")
print("Expected Value")
print(" ↓")
print("Population Variance")
print(" ↓")
print("Sample Variability")
print(" ↓")
print("Standard Error")
print(" ↓")
print("Confidence Intervals")
print(" ↓")
print("Future: Probability Distributions")
print(" ↓")
print("Future: Hypothesis Testing")
print(" ↓")
print("Future: Regression")
print(" ↓")
print("Future: Statistical Learning")

print()


# ============================================================
# 42. WHY 86R MATTERS
# ============================================================

print(
    "WHY 86R MATTERS"
)

print()

print(
    "Probability and statistics provide a formal "
    "language for uncertainty, variability and evidence."
)

print()

print(
    "This layer supports machine learning, scientific "
    "reasoning, engineering measurement and decisions."
)

print()

print(
    "86R extends the mathematical foundation established by 85R."
)

print()


# ============================================================
# 43. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "86R uses a small controlled statistical curriculum."
)

print(
    "86R does not yet implement full probability distributions."
)

print(
    "86R does not yet implement hypothesis testing."
)

print(
    "86R does not yet implement regression."
)

print(
    "86R does not yet implement Bayesian inference."
)

print(
    "86R does not yet establish advanced statistical competence."
)

print()


# ============================================================
# 44. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 87R: Native Linear Algebra and Optimization"
)

print()

print(
    "Vectors + Matrices + Transformations + "
    "Gradients + Optimization + Numerical Validation"
)

print()


# ============================================================
# 45. FOUNDATION MODEL PROGRESS
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
print("Engineering + Scientific Intelligence")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 46. COMPLETE
# ============================================================

print(
    "=== LESSON 86R COMPLETE ==="
)
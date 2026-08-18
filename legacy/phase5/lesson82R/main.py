# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 82R
# Native Tool-Aware Learning Engine
# ============================================================
#
# CURRICULUM POSITION
#
# 79R -> Native reasoning dataset + evaluation
# 80R -> Native reasoning fine-tuning
# 81R -> Native memory-aware training
# 82R -> Native tool-aware learning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 82R teaches Silverwing the controlled relationship:
#
#     TASK
#       |
#       v
#     TOOL SELECTION
#       |
#       v
#     ARGUMENT CONSTRUCTION
#       |
#       v
#     TOOL EXECUTION
#       |
#       v
#     RESULT
#       |
#       v
#     VERIFICATION
#       |
#       v
#     FINAL ANSWER
#
# This is the foundation for later:
#
# - planning
# - multi-tool sequencing
# - verified execution
# - agentic behavior
# - continual learning
#
# ============================================================
# ASSUMPTIONS
# ============================================================
#
# This lesson assumes:
#
# 1. Silverwing owns its tokenizer.
# 2. Silverwing owns its vocabulary.
# 3. Silverwing owns its decoder.
# 4. 79R established reasoning data.
# 5. 80R established reasoning fine-tuning.
# 6. 81R established memory-aware training.
# 7. 81R produced a compatible checkpoint.
#
# ============================================================
# WHAT 82R DOES
# ============================================================
#
# - defines native tool schemas
# - validates tool schemas
# - executes deterministic local tools
# - independently verifies results
# - teaches tool selection
# - teaches argument construction
# - teaches result interpretation
# - teaches verification traces
# - fine-tunes the native Silverwing decoder
# - evaluates against baseline
# - applies promotion gate
#
# ============================================================
# WHAT 82R DOES NOT DO
# ============================================================
#
# - no GPT-2
# - no Qwen
# - no external LLM
# - no cloud reasoning model
# - no arbitrary code execution
# - no unrestricted internet agent
# - no autonomous self-modification
# - no decoder architecture replacement
#
# ============================================================
# TOKEN BUDGET DESIGN
# ============================================================
#
# Silverwing currently uses a 256-token sequence limit.
#
# The previous tool representation duplicated machine inputs
# inside the result object, causing tool_003 to reach 305 tokens.
#
# This lesson deliberately uses compact traces:
#
# Task
# Tool
# Arguments
# Result
# Verification
# Final Answer
#
# Result data does NOT repeat the argument data.
#
# The 256-token limit is preserved.
#
# ============================================================
# PROMOTION MODEL
# ============================================================
#
# baseline
#    |
#    v
# candidate
#    |
#    v
# validation
#    |
#    +----> reject
#    |
#    +----> retain baseline
#    |
#    +----> promote candidate
#
# ============================================================


import hashlib
import json
import math
import random
import re
import time

from dataclasses import dataclass
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
LESSON_81R = PHASE5_DIR / "lesson81R"

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
        LESSON_81R
        / "checkpoints"
        / "silverwing_memory_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_81R
        / "checkpoints"
        / "silverwing_memory_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR
        / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOOL_REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_tool_registry.json"
)

TOOL_TRAIN_FILE = (
        BASE_DIR
        / "silverwing_tool_train.jsonl"
)

TOOL_VALIDATION_FILE = (
        BASE_DIR
        / "silverwing_tool_validation.jsonl"
)

TOOL_REPORT_FILE = (
        BASE_DIR
        / "silverwing_tool_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_tool_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_tool_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_tool_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR
        / "silverwing_tool_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1.2e-5

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

MAX_TOOL_ARGUMENTS_LENGTH = 90

MAX_TOOL_RESULT_LENGTH = 80

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


def sha256_text(
        text: str
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 81R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 82R")
print("Native Tool-Aware Learning Engine")
print()

print("79R -> Native Reasoning Dataset")
print("80R -> Native Reasoning Fine-Tuning")
print("81R -> Native Memory-Aware Training")
print("82R -> Native Tool-Aware Learning")
print()

print("External LLM: NONE")
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - VERIFY INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 81R and Silverwing Inputs"
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

if MODEL_DIMENSION % NUMBER_OF_HEADS != 0:

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
    token: int(token_id)
    for token, token_id
    in vocabulary["token_to_id"].items()
}

required_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

missing_tokens = [
    token
    for token in required_tokens
    if token not in TOKEN_TO_ID
]

if missing_tokens:

    raise ValueError(
        f"Missing vocabulary tokens: {missing_tokens}"
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

    symbols = list(word)

    symbols[-1] += BPE_END

    return tuple(symbols)


def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    output: List[str] = []

    index = 0

    while index < len(symbols):

        if (
                index < len(symbols) - 1
                and
                (
                        symbols[index],
                        symbols[index + 1]
                )
                == pair
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

    return tuple(output)


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

    return list(symbols)


def tokenize_text(
        text: str
) -> List[str]:

    tokens: List[str] = []

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
# 10. TOOL DEFINITIONS
# ============================================================

@dataclass
class ToolDefinition:

    name: str

    description: str

    argument_schema: Dict[str, str]

    result_schema: Dict[str, str]

    risk_level: str

    deterministic: bool


# ============================================================
# 11. TOOL IMPLEMENTATIONS
# ============================================================

def calculator_tool(
        arguments: Dict[str, Any]
) -> Dict[str, Any]:

    expression = str(
        arguments.get(
            "expression",
            ""
        )
    ).strip()

    if not expression:

        raise ValueError(
            "Calculator requires expression."
        )

    if not re.fullmatch(
            r"[0-9+\-*/(). ]+",
            expression
    ):

        raise ValueError(
            "Calculator expression contains unsupported characters."
        )

    try:

        result = eval(
            expression,
            {
                "__builtins__":
                    {}
            },
            {}
        )

    except Exception as exc:

        raise ValueError(
            f"Calculator failed: {exc}"
        ) from exc

    return {
        "status":
            "success",

        "result":
            float(result),

        "expression":
            expression
    }


def unit_conversion_tool(
        arguments: Dict[str, Any]
) -> Dict[str, Any]:

    value = float(
        arguments.get(
            "value",
            0
        )
    )

    from_unit = str(
        arguments.get(
            "from_unit",
            ""
        )
    ).lower()

    to_unit = str(
        arguments.get(
            "to_unit",
            ""
        )
    ).lower()

    conversions = {
        (
            "minutes",
            "seconds"
        ):
            lambda x: x * 60,

        (
            "hours",
            "minutes"
        ):
            lambda x: x * 60,

        (
            "kilometers",
            "meters"
        ):
            lambda x: x * 1000,

        (
            "meters",
            "kilometers"
        ):
            lambda x: x / 1000
    }

    key = (
        from_unit,
        to_unit
    )

    if key not in conversions:

        raise ValueError(
            (
                f"Unsupported conversion: "
                f"{from_unit} -> {to_unit}"
            )
        )

    result = conversions[key](
        value
    )

    return {
        "status":
            "success",

        "result":
            result,

        "from_unit":
            from_unit,

        "to_unit":
            to_unit
    }


def machine_risk_tool(
        arguments: Dict[str, Any]
) -> Dict[str, Any]:

    temperature = float(
        arguments.get(
            "temperature",
            0
        )
    )

    pressure = float(
        arguments.get(
            "pressure",
            0
        )
    )

    rpm = float(
        arguments.get(
            "rpm",
            0
        )
    )

    operating_hours = float(
        arguments.get(
            "operating_hours",
            0
        )
    )

    risk_points = 0

    if temperature >= 100:

        risk_points += 2

    elif temperature >= 90:

        risk_points += 1

    if pressure >= 140:

        risk_points += 2

    elif pressure >= 120:

        risk_points += 1

    if rpm >= 3000:

        risk_points += 2

    elif rpm >= 2400:

        risk_points += 1

    if operating_hours >= 5000:

        risk_points += 2

    elif operating_hours >= 3000:

        risk_points += 1

    if risk_points >= 6:

        prediction = "CRITICAL"

    elif risk_points >= 3:

        prediction = "WARNING"

    else:

        prediction = "NORMAL"

    confidence = (
        0.95
        if prediction == "CRITICAL"
        else
        0.85
        if prediction == "WARNING"
        else
        0.90
    )

    return {
        "status":
            "success",

        "prediction":
            prediction,

        "confidence":
            confidence
    }


# ============================================================
# 12. TOOL REGISTRY
# ============================================================

tool_registry = {

    "calculator": {

        "definition":
            ToolDefinition(
                name="calculator",
                description=(
                    "Performs deterministic "
                    "arithmetic calculations."
                ),
                argument_schema={
                    "expression":
                        "string"
                },
                result_schema={
                    "result":
                        "number"
                },
                risk_level="low",
                deterministic=True
            ),

        "function":
            calculator_tool
    },

    "unit_conversion": {

        "definition":
            ToolDefinition(
                name="unit_conversion",
                description=(
                    "Converts supported "
                    "engineering units."
                ),
                argument_schema={
                    "value":
                        "number",

                    "from_unit":
                        "string",

                    "to_unit":
                        "string"
                },
                result_schema={
                    "result":
                        "number"
                },
                risk_level="low",
                deterministic=True
            ),

        "function":
            unit_conversion_tool
    },

    "machine_risk": {

        "definition":
            ToolDefinition(
                name="machine_risk",
                description=(
                    "Estimates machine risk "
                    "from operating measurements."
                ),
                argument_schema={
                    "temperature":
                        "number",

                    "pressure":
                        "number",

                    "rpm":
                        "number",

                    "operating_hours":
                        "number"
                },
                result_schema={
                    "prediction":
                        "string",

                    "confidence":
                        "number"
                },
                risk_level="medium",
                deterministic=True
            ),

        "function":
            machine_risk_tool
    }
}


# ============================================================
# 13. TEST 5 - TOOL REGISTRY
# ============================================================

print(
    "TEST 5: Native Tool Registry"
)

print()

for name, entry in tool_registry.items():

    definition = entry[
        "definition"
    ]

    print(
        name,
        "->",
        definition.description
    )

    print(
        " risk:",
        definition.risk_level
    )

    print(
        " deterministic:",
        definition.deterministic
    )

print()


# ============================================================
# 14. TEST 6 - SCHEMA VALIDATION
# ============================================================

print(
    "TEST 6: Tool Schema Validation"
)

print()

schema_errors = []

for name, entry in tool_registry.items():

    definition = entry[
        "definition"
    ]

    if (
            name
            !=
            definition.name
    ):

        schema_errors.append(
            f"Name mismatch: {name}"
        )

    if not definition.argument_schema:

        schema_errors.append(
            (
                    "Missing argument schema: "
                    +
                    name
            )
        )

    if not definition.result_schema:

        schema_errors.append(
            (
                    "Missing result schema: "
                    +
                    name
            )
        )

    if definition.risk_level not in {
        "low",
        "medium",
        "high"
    }:

        schema_errors.append(
            (
                    "Invalid risk level: "
                    +
                    name
            )
        )

if schema_errors:

    print(
        json.dumps(
            schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Tool schema validation failed."
    )

print(
    "Tool schemas valid:",
    len(tool_registry)
)

print()


# ============================================================
# 15. TEST 7 - TOOL EXECUTION
# ============================================================

print(
    "TEST 7: Deterministic Tool Execution"
)

print()

tool_tests = [

    (
        "calculator",
        {
            "expression":
                "25 * 8"
        }
    ),

    (
        "unit_conversion",
        {
            "value":
                3,

            "from_unit":
                "hours",

            "to_unit":
                "minutes"
        }
    ),

    (
        "machine_risk",
        {
            "temperature":
                97,

            "pressure":
                130,

            "rpm":
                2600,

            "operating_hours":
                3500
        }
    )
]

tool_execution_results = []

for tool_name, arguments in tool_tests:

    result = (
        tool_registry[
            tool_name
        ][
            "function"
        ](
            arguments
        )
    )

    print(
        tool_name,
        "->",
        result
    )

    tool_execution_results.append(
        {
            "tool":
                tool_name,

            "arguments":
                arguments,

            "result":
                result
        }
    )

print()


# ============================================================
# 16. TOOL RESULT VERIFICATION
# ============================================================

def verify_tool_result(
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any]
) -> Tuple[
    bool,
    str
]:

    if (
            result.get(
                "status"
            )
            !=
            "success"
    ):

        return (
            False,
            "Tool did not report success."
        )

    expected = (
        tool_registry[
            tool_name
        ][
            "function"
        ](
            arguments
        )
    )

    if tool_name == "calculator":

        if (
                expected.get(
                    "result"
                )
                ==
                result.get(
                    "result"
                )
        ):

            return (
                True,
                "Calculator result reproduced."
            )

        return (
            False,
            "Calculator result verification mismatch."
        )

    if tool_name == "unit_conversion":

        if (
                expected.get(
                    "result"
                )
                ==
                result.get(
                    "result"
                )
        ):

            return (
                True,
                "Conversion result reproduced."
            )

        return (
            False,
            "Conversion verification mismatch."
        )

    if tool_name == "machine_risk":

        if (
                expected.get(
                    "prediction"
                )
                ==
                result.get(
                    "prediction"
                )
                and
                expected.get(
                    "confidence"
                )
                ==
                result.get(
                    "confidence"
                )
        ):

            return (
                True,
                "Risk prediction reproduced."
            )

        return (
            False,
            "Risk prediction verification mismatch."
        )

    return (
        False,
        "Unknown tool."
    )


print(
    "TEST 8: Tool Result Verification"
)

print()

verified_results = []

for item in tool_execution_results:

    verified, verification = (
        verify_tool_result(
            item["tool"],
            item["arguments"],
            item["result"]
        )
    )

    print(
        item["tool"],
        "-> verified:",
        verified,
        "|",
        verification
    )

    if not verified:

        raise RuntimeError(
            (
                "Tool result verification failed for "
                f"{item['tool']}."
            )
        )

    verified_results.append(
        {
            "tool":
                item["tool"],

            "arguments":
                item["arguments"],

            "result":
                item["result"],

            "verified":
                verified,

            "verification":
                verification
        }
    )

print()


# ============================================================
# 17. TEST 9 - TRAINING TASKS
# ============================================================

print(
    "TEST 9: Build Native Tool-Aware Training Tasks"
)

print()

tool_tasks = [

    {
        "example_id":
            "tool_001",

        "task":
            "Calculate 25 multiplied by 8.",

        "tool":
            "calculator",

        "arguments":
            {
                "expression":
                    "25 * 8"
            },

        "answer":
            "The result is 200."
    },

    {
        "example_id":
            "tool_002",

        "task":
            "Convert 3 hours to minutes.",

        "tool":
            "unit_conversion",

        "arguments":
            {
                "value":
                    3,

                "from_unit":
                    "hours",

                "to_unit":
                    "minutes"
            },

        "answer":
            "3 hours equals 180 minutes."
    },

    {
        "example_id":
            "tool_003",

        "task":
            (
                "Assess machine risk for temperature 97, "
                "pressure 130, rpm 2600, and 3500 operating hours."
            ),

        "tool":
            "machine_risk",

        "arguments":
            {
                "temperature":
                    97,

                "pressure":
                    130,

                "rpm":
                    2600,

                "operating_hours":
                    3500
            },

        "answer":
            "The machine is classified as WARNING risk."
    },

    {
        "example_id":
            "tool_004",

        "task":
            "Calculate 144 divided by 12.",

        "tool":
            "calculator",

        "arguments":
            {
                "expression":
                    "144 / 12"
            },

        "answer":
            "The result is 12."
    },

    {
        "example_id":
            "tool_005",

        "task":
            "Convert 2500 meters to kilometers.",

        "tool":
            "unit_conversion",

        "arguments":
            {
                "value":
                    2500,

                "from_unit":
                    "meters",

                "to_unit":
                    "kilometers"
            },

        "answer":
            "2500 meters equals 2.5 kilometers."
    },

    {
        "example_id":
            "tool_006",

        "task":
            "Calculate 18 plus 24.",

        "tool":
            "calculator",

        "arguments":
            {
                "expression":
                    "18 + 24"
            },

        "answer":
            "The result is 42."
    }
]

print(
    "Tool-aware tasks:",
    len(tool_tasks)
)

print()


# ============================================================
# 18. COMPACT TOOL ARGUMENT FORMAT
# ============================================================

def format_tool_arguments(
        arguments: Dict[str, Any]
) -> str:

    parts = []

    for key, value in arguments.items():

        parts.append(
            f"{key}={value}"
        )

    return " ".join(
        parts
    )


# ============================================================
# 19. COMPACT TOOL RESULT FORMAT
# ============================================================

def format_tool_result(
        tool_name: str,
        result: Dict[str, Any]
) -> str:

    if tool_name == "calculator":

        return (
            f"result={result['result']}"
        )

    if tool_name == "unit_conversion":

        return (
            f"result={result['result']}"
        )

    if tool_name == "machine_risk":

        return (
            f"prediction={result['prediction']} "
            f"confidence={result['confidence']}"
        )

    compact_items = []

    for key, value in result.items():

        if key == "inputs":

            continue

        if key == "status":

            continue

        compact_items.append(
            f"{key}={value}"
        )

    return " ".join(
        compact_items
    )


# ============================================================
# 20. COMPACT TOOL TRACE
# ============================================================

def build_tool_trace(
        task: Dict[str, Any]
) -> str:

    tool_name = task[
        "tool"
    ]

    arguments = task[
        "arguments"
    ]

    result = (
        tool_registry[
            tool_name
        ][
            "function"
        ](
            arguments
        )
    )

    verified, verification = (
        verify_tool_result(
            tool_name,
            arguments,
            result
        )
    )

    if not verified:

        raise RuntimeError(
            (
                "Tool result failed verification for "
                f"{task['example_id']}."
            )
        )

    argument_text = (
        format_tool_arguments(
            arguments
        )
    )

    result_text = (
        format_tool_result(
            tool_name,
            result
        )
    )

    trace = "\n".join(
        [
            "Task:",
            task["task"],
            "",
            "Tool:",
            tool_name,
            "",
            "Arguments:",
            argument_text,
            "",
            "Result:",
            result_text,
            "",
            "Verification:",
            verification,
            "",
            "Final Answer:",
            task["answer"]
        ]
    )

    return trace


# ============================================================
# 21. TEST 10 - TOOL TRACE VALIDATION
# ============================================================

print(
    "TEST 10: Validate Compact Tool Traces"
)

print()

tool_records = []

for task in tool_tasks:

    full_text = build_tool_trace(
        task
    )

    token_count = len(
        encode_text(
            full_text
        )
    )

    argument_text = (
        format_tool_arguments(
            task["arguments"]
        )
    )

    result = (
        tool_registry[
            task["tool"]
        ][
            "function"
        ](
            task["arguments"]
        )
    )

    result_text = (
        format_tool_result(
            task["tool"],
            result
        )
    )

    if (
            len(argument_text)
            >
            MAX_TOOL_ARGUMENTS_LENGTH
    ):

        raise RuntimeError(
            (
                f"{task['example_id']} "
                "tool arguments are too long."
            )
        )

    if (
            len(result_text)
            >
            MAX_TOOL_RESULT_LENGTH
    ):

        raise RuntimeError(
            (
                f"{task['example_id']} "
                "tool result is too long."
            )
        )

    tool_records.append(
        {
            "example_id":
                task["example_id"],

            "task":
                task["task"],

            "tool":
                task["tool"],

            "arguments":
                task["arguments"],

            "formatted_text":
                full_text,

            "token_count":
                token_count
        }
    )

    print(
        task["example_id"],
        "->",
        token_count,
        "tokens",
        "| tool:",
        task["tool"]
    )

print()


# ============================================================
# 22. TEST 11 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 11: Tool-Aware Token Validation"
)

print()

tool_length_errors = []

for record in tool_records:

    if (
            record["token_count"]
            >
            MAX_SEQUENCE_LENGTH
    ):

        tool_length_errors.append(
            {
                "example_id":
                    record["example_id"],

                "token_count":
                    record["token_count"],

                "maximum":
                    MAX_SEQUENCE_LENGTH
            }
        )

if tool_length_errors:

    print(
        json.dumps(
            tool_length_errors,
            indent=4
        )
    )

    print()

    print(
        "OVERSIZED TRACES:"
    )

    for record in tool_records:

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
            "Tool-aware examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All tool-aware examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 23. TEST 12 - TOOL SELECTION
# ============================================================

print(
    "TEST 12: Tool Selection Validation"
)

print()

selection_errors = []

for record in tool_records:

    if (
            record["tool"]
            not in
            tool_registry
    ):

        selection_errors.append(
            {
                "example_id":
                    record["example_id"],

                "tool":
                    record["tool"]
            }
        )

if selection_errors:

    print(
        json.dumps(
            selection_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Tool selection validation failed."
    )

print(
    "Tool selections valid:",
    len(tool_records)
)

print()


# ============================================================
# 24. TEST 13 - ARGUMENT VALIDATION
# ============================================================

print(
    "TEST 13: Tool Argument Validation"
)

print()

argument_errors = []

for record in tool_records:

    # ToolDefinition is a dataclass, therefore its
    # fields must be accessed with dot notation.
    definition = (
        tool_registry[
            record["tool"]
        ][
            "definition"
        ]
    )

    required = set(
        definition.argument_schema.keys()
    )

    supplied = set(
        record["arguments"].keys()
    )

    missing = (
            required
            -
            supplied
    )

    if missing:

        argument_errors.append(
            {
                "example_id":
                    record["example_id"],

                "tool":
                    record["tool"],

                "missing":
                    sorted(missing)
            }
        )

if argument_errors:

    print(
        json.dumps(
            argument_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Tool argument validation failed."
    )

print(
    "Tool arguments valid:",
    len(tool_records)
)

print()


# ============================================================
# 25. TEST 14 - RESULT VERIFICATION CONTRACT
# ============================================================

print(
    "TEST 14: Tool Verification Contract"
)

print()

verification_errors = []

for record in tool_records:

    result = (
        tool_registry[
            record["tool"]
        ][
            "function"
        ](
            record["arguments"]
        )
    )

    verified, explanation = (
        verify_tool_result(
            record["tool"],
            record["arguments"],
            result
        )
    )

    if not verified:

        verification_errors.append(
            {
                "example_id":
                    record["example_id"],

                "verification":
                    explanation
            }
        )

if verification_errors:

    print(
        json.dumps(
            verification_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Tool verification contract failed."
    )

print(
    "All tool traces have verified results."
)

print()


# ============================================================
# 26. TEST 15 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    tool_records
)

validation_count = max(
    2,
    int(
        round(
            len(tool_records)
            *
            0.30
        )
    )
)

validation_count = min(
    validation_count,
    len(tool_records) - 1
)

tool_train_records = (
    tool_records[
        :-validation_count
    ]
)

tool_validation_records = (
    tool_records[
        -validation_count:
    ]
)

print(
    "TEST 15: Tool Train/Validation Split"
)

print(
    "Training examples:",
    len(tool_train_records)
)

print(
    "Validation examples:",
    len(tool_validation_records)
)

print()


# ============================================================
# 27. SAVE TOOL ARTIFACTS
# ============================================================

registry_serialized = {}

for name, entry in tool_registry.items():

    definition = entry[
        "definition"
    ]

    registry_serialized[
        name
    ] = {
        "name":
            definition.name,

        "description":
            definition.description,

        "argument_schema":
            definition.argument_schema,

        "result_schema":
            definition.result_schema,

        "risk_level":
            definition.risk_level,

        "deterministic":
            definition.deterministic
    }

write_json(
    TOOL_REGISTRY_FILE,
    registry_serialized
)

with open(
        TOOL_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in tool_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        TOOL_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in tool_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

tool_report = {
    "lesson":
        "82R",

    "capability":
        "native_tool_aware_learning",

    "tool_count":
        len(tool_registry),

    "training_examples":
        len(tool_train_records),

    "validation_examples":
        len(tool_validation_records),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "external_llm":
        False,

    "external_tool_model":
        False
}

write_json(
    TOOL_REPORT_FILE,
    tool_report
)


# ============================================================
# 28. DATASET
# ============================================================

class ToolDataset(
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
                record["formatted_text"]
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
                        record["example_id"],

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
                sample["example_id"],

            "input_ids":
                torch.tensor(
                    sample["input_ids"],
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    sample["labels"],
                    dtype=torch.long
                )
        }


def collate_tool_batch(
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


tool_train_dataset = ToolDataset(
    tool_train_records
)

tool_validation_dataset = ToolDataset(
    tool_validation_records
)

tool_train_loader = DataLoader(
    tool_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_tool_batch
)

tool_validation_loader = DataLoader(
    tool_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_tool_batch
)

print(
    "TEST 16: Tool DataLoaders"
)

print(
    "Training samples:",
    len(tool_train_dataset)
)

print(
    "Validation samples:",
    len(tool_validation_dataset)
)

print(
    "Training batches:",
    len(tool_train_loader)
)

print(
    "Validation batches:",
    len(tool_validation_loader)
)

print()


# ============================================================
# 29. EXACT SILVERWING ATTENTION
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

        if (
                dimension
                %
                heads
                !=
                0
        ):

            raise ValueError(
                "Invalid attention configuration."
            )

        self.dimension = dimension

        self.heads = heads

        self.head_dimension = (
                dimension // heads
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
# 30. EXACT SILVERWING FEED FORWARD
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
                self.input_projection(x)
            )
        )


# ============================================================
# 31. EXACT SILVERWING TRANSFORMER BLOCK
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
            self.attention(x)
        )

        x = self.norm_feed_forward(
            x
            +
            self.feed_forward(x)
        )

        return x


# ============================================================
# 32. EXACT POSITION EMBEDDING
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
# 33. EXACT SILVERWING DECODER
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

            x = layer(x)

        x = self.final_norm(x)

        return self.language_model_head(x)


# ============================================================
# 34. TEST 17 - STRICT LOAD
# ============================================================

print(
    "TEST 17: Strict Load of 81R Memory Model"
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
        "81R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "81R checkpoint is missing model_state_dict."
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
                "81R checkpoint architecture mismatch. "
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
            "82R refused to load a mismatched "
            "81R Silverwing model.\n\n"
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
    "81R model is compatible with 82R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 35. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()
    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 36. LOSS
# ============================================================

def tool_aware_loss(
        logits: torch.Tensor,
        labels: torch.Tensor
) -> torch.Tensor:

    return F.cross_entropy(
        logits.reshape(
            -1,
            VOCABULARY_SIZE
        ),
        labels.reshape(-1),
        ignore_index=-100
    )


# ============================================================
# 37. EVALUATION
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

        loss = tool_aware_loss(
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
# 38. TEST 18 - BASELINE
# ============================================================

print(
    "TEST 18: Baseline Tool-Aware Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    tool_validation_loader
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
# 39. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(tool_train_loader)
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
# 40. TEST 19 - TRAINING
# ============================================================

print(
    "TEST 19: Native Tool-Aware Fine-Tuning"
)

print()

history: List[
    Dict[str, Any]
] = []

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
            tool_train_loader,
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

        loss = tool_aware_loss(
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
            f"| Batch {batch_number}/{len(tool_train_loader)} "
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
        tool_validation_loader
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
                    "82R",

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

                "tool_count":
                    len(
                        tool_registry
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
# 41. TEST 20 - FINAL
# ============================================================

print(
    "TEST 20: Final Tool-Aware Evaluation"
)

print()

final_metrics = evaluate(
    model,
    tool_validation_loader
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
# 42. TEST 21 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 21: Numerical Health"
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
# 43. TEST 22 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 22: Parameter Change"
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
# 44. TEST 23 - PROMOTION
# ============================================================

print(
    "TEST 23: Tool-Aware Promotion Gate"
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
        "Candidate tool-aware loss is invalid."
    )

elif (
        math.isfinite(
            baseline_loss
        )
        and
        candidate_loss < baseline_loss
):

    decision = (
        "PROMOTE_CANDIDATE"
    )

    reason = (
        "Tool-aware validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Tool-aware validation loss did not improve."
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
# 45. TEST 24 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 24: Save Tool-Aware Candidate"
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
        "82R",

    "training_mode":
        "native_tool_aware_fine_tuning",

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

    "tool_count":
        len(
            tool_registry
        )
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
        "82R",

    "training_mode":
        "native_tool_aware_fine_tuning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "external_llm":
        False,

    "external_tool_model":
        False,

    "device":
        str(
            DEVICE
        ),

    "tool_count":
        len(
            tool_registry
        ),

    "training_examples":
        len(
            tool_train_records
        ),

    "validation_examples":
        len(
            tool_validation_records
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
# 47. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "82R",

    "capability":
        "native_tool_aware_learning",

    "tools":
        list(
            tool_registry.keys()
        ),

    "verified_tool_executions":
        verified_results,

    "training_examples":
        len(
            tool_train_records
        ),

    "validation_examples":
        len(
            tool_validation_records
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
# 48. TOOL-AWARE COGNITIVE LOOP
# ============================================================

print(
    "SILVERWING TOOL-AWARE COGNITIVE LOOP"
)

print()

print("Task")
print("  ↓")
print("Need Detection")
print("  ↓")
print("Tool Selection")
print("  ↓")
print("Argument Construction")
print("  ↓")
print("Tool Execution")
print("  ↓")
print("Result Inspection")
print("  ↓")
print("Result Verification")
print("  ↓")
print("Reasoned Answer")
print("  ↓")
print("Feedback")

print()


# ============================================================
# 49. TOOL SAFETY CONTRACT
# ============================================================

print(
    "TOOL SAFETY CONTRACT"
)

print()

print(
    "Every tool has an explicit schema."
)

print(
    "Arguments are validated."
)

print(
    "Deterministic results are reproducible."
)

print(
    "Tool output is independently verified."
)

print(
    "Only validated candidates can be promoted."
)

print()


# ============================================================
# 50. WHAT 82R ADDS
# ============================================================

print(
    "WHAT 82R ADDS"
)

print()

print(
    "81R gave Silverwing persistent contextual memory."
)

print()

print(
    "82R gives Silverwing a controlled interface "
    "for deterministic capability invocation."
)

print()

print(
    "This is the foundation for future planning, "
    "sequencing and agentic execution."
)

print()


# ============================================================
# 51. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "82R does not yet perform autonomous tool discovery."
)

print(
    "82R does not yet perform arbitrary code execution."
)

print(
    "82R does not yet perform long-horizon planning."
)

print(
    "82R does not yet autonomously modify its architecture."
)

print(
    "82R does not yet implement a complete autonomous agent."
)

print()


# ============================================================
# 52. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 83R: Native Planning and Tool Sequencing"
)

print()

print(
    "Reasoning + Memory + Multiple Tools + "
    "Ordered Actions + Verification"
)

print()


# ============================================================
# 53. FOUNDATION MODEL PROGRESS
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

print("Continual Learning")
print(" ↓")

print("Controlled Autonomous Improvement")

print()


# ============================================================
# 54. COMPLETE
# ============================================================

print(
    "=== LESSON 82R COMPLETE ==="
)
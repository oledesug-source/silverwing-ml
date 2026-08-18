# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 83R
# Native Planning and Tool Sequencing
# ============================================================
#
# 83R builds directly on:
#
# 79R -> Reasoning
# 80R -> Reasoning Fine-Tuning
# 81R -> Memory
# 82R -> Tool Use
# 83R -> Planning + Tool Sequencing
#
# PURPOSE
# -------
# Teach Silverwing to:
#
#   goal
#     ->
#   ordered tool steps
#     ->
#   dependencies
#     ->
#   intermediate results
#     ->
#   verification
#     ->
#   final answer
#
# IMPORTANT
# ---------
# Silverwing remains limited to the established 256-token
# sequence length.
#
# The execution engine can handle references such as:
#
#   $step_01.result
#
# and:
#
#   $step_01.result + 60
#
# The training trace is intentionally compact so that planning
# examples remain inside the established sequence budget.
#
# NO EXTERNAL LLM
# ---------------
# No GPT-2
# No Qwen
# No cloud model
#
# ============================================================

import ast
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
LESSON_82R = PHASE5_DIR / "lesson82R"

VOCABULARY_FILE = (
        LESSON_66R / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66R / "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71R / "silverwing_decoder_config.json"
)

REASONING_CONFIG_FILE = (
        LESSON_79R / "silverwing_reasoning_config.json"
)

BASE_CHECKPOINT_PRIMARY = (
        LESSON_82R
        / "checkpoints"
        / "silverwing_tool_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_82R
        / "checkpoints"
        / "silverwing_tool_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLAN_REGISTRY_FILE = (
        BASE_DIR / "silverwing_plan_registry.json"
)

PLAN_TRAIN_FILE = (
        BASE_DIR / "silverwing_plan_train.jsonl"
)

PLAN_VALIDATION_FILE = (
        BASE_DIR / "silverwing_plan_validation.jsonl"
)

PLAN_REPORT_FILE = (
        BASE_DIR / "silverwing_plan_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR / "silverwing_planning_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR / "silverwing_planning_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR / "silverwing_planning_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR / "silverwing_planning_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256
MAX_PLAN_STEPS = 2

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

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def read_json(path: Path) -> Any:
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


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():
        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():
        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 82R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 83R")
print("Native Planning and Tool Sequencing")
print()

print("79R -> Reasoning Dataset")
print("80R -> Reasoning Fine-Tuning")
print("81R -> Memory-Aware Training")
print("82R -> Tool-Aware Learning")
print("83R -> Planning and Tool Sequencing")
print()

print("External LLM: NONE")
print("Plan steps:", MAX_PLAN_STEPS)
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 82R and Silverwing Inputs"
)
print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
    REASONING_CONFIG_FILE,
]:
    require_file(path)
    print("FOUND:", path)

BASE_CHECKPOINT = select_base_checkpoint()

print("FOUND:", BASE_CHECKPOINT)
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
        "Model dimension must be divisible by attention heads."
    )

print("Model dimension:", MODEL_DIMENSION)
print("Attention heads:", NUMBER_OF_HEADS)
print("Feed-forward dimension:", FEED_FORWARD_DIMENSION)
print("Layers:", NUMBER_OF_LAYERS)
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
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

VOCABULARY_SIZE = len(TOKEN_TO_ID)

print("Vocabulary size:", VOCABULARY_SIZE)
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

merge_items = (
    merge_data.get("merges", [])
    if isinstance(merge_data, dict)
    else merge_data
)

MERGE_RANKS: Dict[
    Tuple[str, str],
    int
] = {}

for item in merge_items:

    if not isinstance(item, dict):
        continue

    pair = item.get("pair")

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
    ] = int(item["rank"])

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

    symbols = word_to_symbols(word)

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

    tokens = []

    for word in split_words(text):
        tokens.extend(
            tokenize_word(word)
        )

    return tokens


def encode_text(
        text: str
) -> List[int]:

    ids = [BOS_ID]

    for token in tokenize_text(text):

        ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )

    ids.append(EOS_ID)

    return ids


# ============================================================
# 10. SAFE ARITHMETIC
# ============================================================

def safe_arithmetic(
        expression: str
) -> float:

    cleaned = expression.replace(
        " ",
        ""
    )

    if not re.fullmatch(
            r"[0-9+\-*/().]+",
            cleaned
    ):
        raise ValueError(
            "Unsupported arithmetic expression."
        )

    tree = ast.parse(
        cleaned,
        mode="eval"
    )

    def evaluate_node(
            node: ast.AST
    ) -> float:

        if isinstance(
                node,
                ast.Expression
        ):
            return evaluate_node(
                node.body
            )

        if isinstance(
                node,
                ast.Constant
        ):

            if isinstance(
                    node.value,
                    (int, float)
            ):
                return float(node.value)

            raise ValueError(
                "Invalid arithmetic constant."
            )

        if isinstance(
                node,
                ast.UnaryOp
        ):

            value = evaluate_node(
                node.operand
            )

            if isinstance(
                    node.op,
                    ast.USub
            ):
                return -value

            if isinstance(
                    node.op,
                    ast.UAdd
            ):
                return value

            raise ValueError(
                "Unsupported unary operator."
            )

        if isinstance(
                node,
                ast.BinOp
        ):

            left = evaluate_node(
                node.left
            )

            right = evaluate_node(
                node.right
            )

            if isinstance(
                    node.op,
                    ast.Add
            ):
                return left + right

            if isinstance(
                    node.op,
                    ast.Sub
            ):
                return left - right

            if isinstance(
                    node.op,
                    ast.Mult
            ):
                return left * right

            if isinstance(
                    node.op,
                    ast.Div
            ):

                if right == 0:
                    raise ValueError(
                        "Division by zero."
                    )

                return left / right

            raise ValueError(
                "Unsupported binary operator."
            )

        raise ValueError(
            "Unsupported arithmetic node."
        )

    return evaluate_node(tree)


# ============================================================
# 11. TOOLS
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

    return {
        "status":
            "success",

        "result":
            float(
                safe_arithmetic(
                    expression
                )
            )
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
            "hours",
            "minutes"
        ):
            lambda x: x * 60,

        (
            "minutes",
            "seconds"
        ):
            lambda x: x * 60,

        (
            "meters",
            "kilometers"
        ):
            lambda x: x / 1000,

        (
            "kilometers",
            "meters"
        ):
            lambda x: x * 1000

    }

    key = (
        from_unit,
        to_unit
    )

    if key not in conversions:

        raise ValueError(
            (
                f"Unsupported conversion: "
                f"{from_unit}->{to_unit}"
            )
        )

    return {
        "status":
            "success",

        "result":
            conversions[key](value),

        "unit":
            to_unit
    }


TOOL_FUNCTIONS = {

    "calculator":
        calculator_tool,

    "unit_conversion":
        unit_conversion_tool

}


# ============================================================
# 12. TEST 5 - TOOLS
# ============================================================

print(
    "TEST 5: Verify Planning Tools"
)
print()

print(
    "Available tools:",
    list(TOOL_FUNCTIONS.keys())
)

print()


# ============================================================
# 13. PLAN SCHEMA
# ============================================================

@dataclass
class PlanStep:

    step_id: str

    tool: str

    arguments: Dict[str, Any]

    depends_on: List[str]


@dataclass
class PlanDefinition:

    plan_id: str

    goal: str

    steps: List[PlanStep]

    verification: str

    final_answer: str


# ============================================================
# 14. REFERENCE RESOLUTION
# ============================================================

REFERENCE_PATTERN = re.compile(
    r"\$([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)"
)


def resolve_value(
        value: Any,
        results: Dict[
            str,
            Dict[str, Any]
        ]
) -> Any:

    if not isinstance(
            value,
            str
    ):
        return value

    text = value.strip()

    exact = re.fullmatch(
        r"\$([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)",
        text
    )

    if exact:

        step_id = exact.group(1)
        result_key = exact.group(2)

        if step_id not in results:
            raise RuntimeError(
                f"Missing dependency result: {step_id}"
            )

        if result_key not in results[step_id]:
            raise RuntimeError(
                (
                    f"Missing result field "
                    f"{result_key} in {step_id}"
                )
            )

        return results[
            step_id
        ][
            result_key
        ]

    references = REFERENCE_PATTERN.findall(
        text
    )

    if not references:
        return value

    expression = text

    for step_id, result_key in references:

        if step_id not in results:
            raise RuntimeError(
                (
                    f"Missing dependency result: "
                    f"{step_id}"
                )
            )

        if result_key not in results[step_id]:
            raise RuntimeError(
                (
                    f"Missing result field "
                    f"{result_key} in {step_id}"
                )
            )

        resolved = results[
            step_id
        ][
            result_key
        ]

        expression = expression.replace(
            f"${step_id}.{result_key}",
            str(resolved)
        )

    return safe_arithmetic(
        expression
    )


# ============================================================
# 15. PLAN EXECUTION
# ============================================================

def execute_plan(
        plan: PlanDefinition
) -> Dict[str, Any]:

    results = {}
    verified_steps = []

    for step in plan.steps:

        if step.tool not in TOOL_FUNCTIONS:

            raise RuntimeError(
                (
                    f"Unknown tool: "
                    f"{step.tool}"
                )
            )

        for dependency in step.depends_on:

            if dependency not in results:

                raise RuntimeError(
                    (
                        f"Dependency {dependency} "
                        f"not completed before "
                        f"{step.step_id}."
                    )
                )

        resolved_arguments = {}

        for key, value in step.arguments.items():

            resolved_arguments[key] = (
                resolve_value(
                    value,
                    results
                )
            )

        result = TOOL_FUNCTIONS[
            step.tool
        ](
            resolved_arguments
        )

        if result.get(
                "status"
        ) != "success":

            raise RuntimeError(
                (
                    f"Tool failure in "
                    f"{step.step_id}"
                )
            )

        results[
            step.step_id
        ] = result

        verified_steps.append(
            step.step_id
        )

    return {
        "results":
            results,

        "verified_steps":
            verified_steps,

        "complete":
            len(verified_steps)
            ==
            len(plan.steps)
    }


# ============================================================
# 16. TEST 6 - BASIC PLAN EXECUTION
# ============================================================

print(
    "TEST 6: Basic Plan Execution"
)
print()

basic_plan = PlanDefinition(

    plan_id="plan_test_001",

    goal="Calculate 3 hours in seconds.",

    steps=[

        PlanStep(
            step_id="step_01",
            tool="unit_conversion",
            arguments={
                "value":
                    3,

                "from_unit":
                    "hours",

                "to_unit":
                    "minutes"
            },
            depends_on=[]
        ),

        PlanStep(
            step_id="step_02",
            tool="unit_conversion",
            arguments={
                "value":
                    "$step_01.result",

                "from_unit":
                    "minutes",

                "to_unit":
                    "seconds"
            },
            depends_on=[
                "step_01"
            ]
        )

    ],

    verification=(
        "Step 02 must consume the result of step 01."
    ),

    final_answer=
    "3 hours equals 10800 seconds."
)

basic_execution = execute_plan(
    basic_plan
)

print(
    "Plan complete:",
    basic_execution["complete"]
)

print(
    "Verified steps:",
    basic_execution["verified_steps"]
)

print(
    "Results:",
    basic_execution["results"]
)

print()

if not basic_execution["complete"]:

    raise RuntimeError(
        "Basic plan execution failed."
    )


# ============================================================
# 17. TEST 7 - DEPENDENCY VALIDATION
# ============================================================

print(
    "TEST 7: Plan Dependency Validation"
)
print()

dependency_errors = []

known_steps = {
    step.step_id
    for step in basic_plan.steps
}

for index, step in enumerate(
        basic_plan.steps
):

    previous_steps = {
        item.step_id
        for item in basic_plan.steps[:index]
    }

    for dependency in step.depends_on:

        if dependency not in known_steps:

            dependency_errors.append(
                {
                    "step":
                        step.step_id,

                    "dependency":
                        dependency
                }
            )

        elif dependency not in previous_steps:

            dependency_errors.append(
                {
                    "step":
                        step.step_id,

                    "error":
                        (
                            f"{dependency} "
                            "does not precede step."
                        )
                }
            )

if dependency_errors:

    print(
        json.dumps(
            dependency_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Plan dependency validation failed."
    )

print(
    "Plan dependencies valid."
)
print()


# ============================================================
# 18. TEST 8 - PLANNING TASKS
# ============================================================

print(
    "TEST 8: Build Native Planning Tasks"
)
print()

planning_tasks = [

    {
        "example_id":
            "plan_001",

        "goal":
            "Convert 3 hours to seconds.",

        "steps":
            [
                {
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

                    "depends_on":
                        []
                },

                {
                    "tool":
                        "unit_conversion",

                    "arguments":
                        {
                            "value":
                                "$step_01.result",

                            "from_unit":
                                "minutes",

                            "to_unit":
                                "seconds"
                        },

                    "depends_on":
                        [
                            "step_01"
                        ]
                }
            ],

        "verification":
            "step2 uses step1 result.",

        "answer":
            "3 hours = 10800 seconds."

    },

    {
        "example_id":
            "plan_002",

        "goal":
            "Convert 5 minutes to seconds and add 60.",

        "steps":
            [
                {
                    "tool":
                        "unit_conversion",

                    "arguments":
                        {
                            "value":
                                5,

                            "from_unit":
                                "minutes",

                            "to_unit":
                                "seconds"
                        },

                    "depends_on":
                        []
                },

                {
                    "tool":
                        "calculator",

                    "arguments":
                        {
                            "expression":
                                "$step_01.result + 60"
                        },

                    "depends_on":
                        [
                            "step_01"
                        ]
                }
            ],

        "verification":
            "step2 uses step1 result.",

        "answer":
            "5 minutes = 300 seconds; plus 60 = 360."

    },

    {
        "example_id":
            "plan_003",

        "goal":
            "Convert 2 km to meters and add 500.",

        "steps":
            [
                {
                    "tool":
                        "unit_conversion",

                    "arguments":
                        {
                            "value":
                                2,

                            "from_unit":
                                "kilometers",

                            "to_unit":
                                "meters"
                        },

                    "depends_on":
                        []
                },

                {
                    "tool":
                        "calculator",

                    "arguments":
                        {
                            "expression":
                                "$step_01.result + 500"
                        },

                    "depends_on":
                        [
                            "step_01"
                        ]
                }
            ],

        "verification":
            "step2 uses step1 result.",

        "answer":
            "2 km = 2000 m; plus 500 = 2500 m."

    },

    {
        "example_id":
            "plan_004",

        "goal":
            "Calculate 25 times 8, then add 100.",

        "steps":
            [
                {
                    "tool":
                        "calculator",

                    "arguments":
                        {
                            "expression":
                                "25 * 8"
                        },

                    "depends_on":
                        []
                },

                {
                    "tool":
                        "calculator",

                    "arguments":
                        {
                            "expression":
                                "$step_01.result + 100"
                        },

                    "depends_on":
                        [
                            "step_01"
                        ]
                }
            ],

        "verification":
            "step2 uses step1 result.",

        "answer":
            "25 x 8 = 200; plus 100 = 300."

    },

    {
        "example_id":
            "plan_005",

        "goal":
            "Convert 3 km to meters, then divide by 2.",

        "steps":
            [
                {
                    "tool":
                        "unit_conversion",

                    "arguments":
                        {
                            "value":
                                3,

                            "from_unit":
                                "kilometers",

                            "to_unit":
                                "meters"
                        },

                    "depends_on":
                        []
                },

                {
                    "tool":
                        "calculator",

                    "arguments":
                        {
                            "expression":
                                "$step_01.result / 2"
                        },

                    "depends_on":
                        [
                            "step_01"
                        ]
                }
            ],

        "verification":
            "step2 uses step1 result.",

        "answer":
            "3 km = 3000 m; half = 1500 m."

    }

]

print(
    "Planning tasks:",
    len(planning_tasks)
)
print()


# ============================================================
# 19. PLAN BUILDING
# ============================================================

def build_plan(
        task: Dict[str, Any]
) -> PlanDefinition:

    steps = []

    for index, data in enumerate(
            task["steps"],
            start=1
    ):

        steps.append(
            PlanStep(
                step_id=
                f"step_{index:02d}",

                tool=
                data["tool"],

                arguments=
                data["arguments"],

                depends_on=
                data["depends_on"]
            )
        )

    return PlanDefinition(
        plan_id=
        task["example_id"],

        goal=
        task["goal"],

        steps=
        steps,

        verification=
        task["verification"],

        final_answer=
        task["answer"]
    )


# ============================================================
# 20. COMPACT TRACE
# ============================================================
#
# IMPORTANT:
#
# The previous trace repeated too much natural-language
# scaffolding. This version represents planning compactly:
#
# G: goal
# P: step1 -> result; step2 -> result
# V: verification
# A: answer
#
# This still preserves:
#
# goal
# step ordering
# dependencies
# intermediate result
# verification
# final answer
#
# ============================================================

def compact_arguments(
        arguments: Dict[str, Any]
) -> str:

    return ";".join(
        f"{key}={value}"
        for key, value
        in arguments.items()
    )


def compact_result(
        result: Dict[str, Any]
) -> str:

    return ";".join(
        f"{key}={value}"
        for key, value
        in result.items()
        if key != "status"
    )


def build_plan_trace(
        plan: PlanDefinition
) -> str:

    execution = execute_plan(
        plan
    )

    lines = [
        "G:" + plan.goal
    ]

    step_chunks = []

    for step in plan.steps:

        deps = (
            "-"
            if not step.depends_on
            else
            ",".join(
                step.depends_on
            )
        )

        args = compact_arguments(
            step.arguments
        )

        result = compact_result(
            execution[
                "results"
            ][
                step.step_id
            ]
        )

        step_chunks.append(

            (
                f"{step.step_id}"
                f"={step.tool}"
                f"|d={deps}"
                f"|a={args}"
                f"|r={result}"
            )

        )

    lines.append(
        "P:" +
        " ".join(
            step_chunks
        )
    )

    lines.append(
        "V:" +
        plan.verification
    )

    lines.append(
        "A:" +
        plan.final_answer
    )

    return "\n".join(
        lines
    )


# ============================================================
# 21. TEST 9 - TRACE VALIDATION
# ============================================================

print(
    "TEST 9: Validate Compact Planning Traces"
)
print()

plan_records = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    if len(plan.steps) != MAX_PLAN_STEPS:

        raise RuntimeError(
            (
                f"{task['example_id']} must contain "
                f"{MAX_PLAN_STEPS} steps."
            )
        )

    trace = build_plan_trace(
        plan
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    plan_records.append(
        {
            "example_id":
                task["example_id"],

            "goal":
                task["goal"],

            "formatted_text":
                trace,

            "token_count":
                token_count,

            "step_count":
                len(plan.steps)
        }
    )

    print(
        task["example_id"],
        "->",
        token_count,
        "tokens",
        "| steps:",
        len(plan.steps)
    )

print()


# ============================================================
# 22. TEST 10 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 10: Planning Token Validation"
)
print()

plan_length_errors = [

    {
        "example_id":
            record["example_id"],

        "token_count":
            record["token_count"],

        "maximum":
            MAX_SEQUENCE_LENGTH
    }

    for record in plan_records

    if (
            record["token_count"]
            >
            MAX_SEQUENCE_LENGTH
    )
]

if plan_length_errors:

    print(
        json.dumps(
            plan_length_errors,
            indent=4
        )
    )

    raise RuntimeError(
        (
            "Planning examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All planning examples fit "
    "the Silverwing sequence limit."
)
print()


# ============================================================
# 23. TEST 11 - DEPENDENCY CONTRACT
# ============================================================

print(
    "TEST 11: Plan Dependency Contract"
)
print()

dependency_errors = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    known_steps = {
        step.step_id
        for step in plan.steps
    }

    for index, step in enumerate(
            plan.steps
    ):

        previous_steps = {
            previous.step_id
            for previous
            in plan.steps[:index]
        }

        for dependency in step.depends_on:

            if dependency not in known_steps:

                dependency_errors.append(
                    {
                        "example_id":
                            plan.plan_id,

                        "step":
                            step.step_id,

                        "error":
                            (
                                f"Unknown dependency "
                                f"{dependency}"
                            )
                    }
                )

            elif dependency not in previous_steps:

                dependency_errors.append(
                    {
                        "example_id":
                            plan.plan_id,

                        "step":
                            step.step_id,

                        "error":
                            (
                                f"{dependency} does not "
                                "precede this step."
                            )
                    }
                )

if dependency_errors:

    print(
        json.dumps(
            dependency_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Plan dependency validation failed."
    )

print(
    "Plan dependencies valid."
)
print()


# ============================================================
# 24. TEST 12 - REFERENCE VALIDATION
# ============================================================

print(
    "TEST 12: Intermediate Result Reference Validation"
)
print()

reference_errors = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    known_steps = {
        step.step_id
        for step in plan.steps
    }

    for step in plan.steps:

        for value in step.arguments.values():

            if not isinstance(
                    value,
                    str
            ):
                continue

            references = REFERENCE_PATTERN.findall(
                value
            )

            for step_id, result_key in references:

                if step_id not in known_steps:

                    reference_errors.append(
                        {
                            "example_id":
                                plan.plan_id,

                            "step":
                                step.step_id,

                            "reference":
                                value,

                            "error":
                                "Referenced step does not exist."
                        }
                    )

if reference_errors:

    print(
        json.dumps(
            reference_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Intermediate reference validation failed."
    )

print(
    "Intermediate result references valid."
)
print()


# ============================================================
# 25. TEST 13 - EXECUTION VERIFICATION
# ============================================================

print(
    "TEST 13: End-to-End Plan Execution Verification"
)
print()

execution_errors = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    try:

        execution = execute_plan(
            plan
        )

        if not execution[
            "complete"
        ]:

            execution_errors.append(
                {
                    "example_id":
                        plan.plan_id,

                    "error":
                        "Plan did not complete."
                }
            )

    except Exception as exc:

        execution_errors.append(
            {
                "example_id":
                    plan.plan_id,

                "error":
                    str(exc)
            }
        )

if execution_errors:

    print(
        json.dumps(
            execution_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Plan execution verification failed."
    )

print(
    "Plans executed and verified:",
    len(planning_tasks)
)
print()


# ============================================================
# 26. TEST 14 - RESULT DEPENDENCY SEMANTICS
# ============================================================

print(
    "TEST 14: Result Dependency Semantics"
)
print()

semantic_errors = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    execution = execute_plan(
        plan
    )

    if len(plan.steps) < 2:
        continue

    second_step = plan.steps[1]

    references = []

    for value in second_step.arguments.values():

        if isinstance(
                value,
                str
        ):

            references.extend(
                REFERENCE_PATTERN.findall(
                    value
                )
            )

    if not references:

        semantic_errors.append(
            {
                "example_id":
                    plan.plan_id,

                "error":
                    "Second step has no previous-step reference."
            }
        )

        continue

    for step_id, result_key in references:

        if step_id not in execution["results"]:

            semantic_errors.append(
                {
                    "example_id":
                        plan.plan_id,

                    "error":
                        (
                            f"Reference {step_id} "
                            "was not produced."
                        )
                }
            )

        elif result_key not in execution[
            "results"
        ][
            step_id
        ]:

            semantic_errors.append(
                {
                    "example_id":
                        plan.plan_id,

                    "error":
                        (
                            f"Result key {result_key} "
                            "was not produced."
                        )
                }
            )

if semantic_errors:

    print(
        json.dumps(
            semantic_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Result dependency semantics failed."
    )

print(
    "Result dependencies are semantically valid."
)
print()


# ============================================================
# 27. TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    plan_records
)

validation_count = max(
    2,
    int(
        round(
            len(plan_records)
            * 0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(plan_records) - 1
)

plan_train_records = (
    plan_records[
        :-validation_count
    ]
)

plan_validation_records = (
    plan_records[
        -validation_count:
    ]
)

print(
    "TEST 15: Planning Train/Validation Split"
)

print(
    "Training examples:",
    len(plan_train_records)
)

print(
    "Validation examples:",
    len(plan_validation_records)
)

print()


# ============================================================
# 28. SAVE ARTIFACTS
# ============================================================

plan_registry = {

    "lesson":
        "83R",

    "capability":
        "native_planning_and_tool_sequencing",

    "maximum_steps":
        MAX_PLAN_STEPS,

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "planning_examples":
        len(planning_tasks)

}

write_json(
    PLAN_REGISTRY_FILE,
    plan_registry
)

with open(
        PLAN_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in plan_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        PLAN_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in plan_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    PLAN_REPORT_FILE,
    {
        "lesson":
            "83R",

        "capability":
            "native_planning_and_tool_sequencing",

        "maximum_steps":
            MAX_PLAN_STEPS,

        "training_examples":
            len(plan_train_records),

        "validation_examples":
            len(plan_validation_records),

        "external_llm":
            False
    }
)


# ============================================================
# 29. DATASET
# ============================================================

class PlanningDataset(
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

    def __len__(self) -> int:
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


def collate_plan_batch(
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
                - len(input_ids)
        )

        label_padding = (
                maximum_length
                - len(item_labels)
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


plan_train_dataset = PlanningDataset(
    plan_train_records
)

plan_validation_dataset = PlanningDataset(
    plan_validation_records
)

plan_train_loader = DataLoader(
    plan_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_plan_batch
)

plan_validation_loader = DataLoader(
    plan_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_plan_batch
)

print(
    "TEST 16: Planning DataLoaders"
)

print(
    "Training samples:",
    len(plan_train_dataset)
)

print(
    "Validation samples:",
    len(plan_validation_dataset)
)

print(
    "Training batches:",
    len(plan_train_loader)
)

print(
    "Validation batches:",
    len(plan_validation_loader)
)

print()


# ============================================================
# 30. EXACT SILVERWING ATTENTION
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
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(1, 2)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
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
            .transpose(1, 2)
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
# 31. EXACT FEED FORWARD
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
# 32. EXACT BLOCK
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
            x + self.attention(x)
        )

        x = self.norm_feed_forward(
            x + self.feed_forward(x)
        )

        return x


# ============================================================
# 33. POSITION EMBEDDING
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
# 34. EXACT SILVERWING DECODER
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

        if sequence_length > MAX_SEQUENCE_LENGTH:

            raise ValueError(
                "Sequence exceeds model limit."
            )

        x = (
                self.token_embedding(input_ids)
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
# 35. TEST 17 - STRICT LOAD
# ============================================================

print(
    "TEST 17: Strict Load of 82R Tool Model"
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
        "82R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "82R checkpoint is missing model_state_dict."
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
                "82R checkpoint architecture mismatch. "
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
            "83R refused to load a mismatched "
            "82R Silverwing model.\n\n"
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
    "82R model is compatible with 83R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 36. BASELINE
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()
    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 37. LOSS
# ============================================================

def planning_loss(
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
# 38. EVALUATION
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

        loss = planning_loss(
            logits,
            labels
        )

        total_loss += float(loss)
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
            total_loss / batches
    )

    if (
            math.isfinite(loss_value)
            and
            loss_value < 50
    ):

        perplexity = math.exp(
            loss_value
        )

    else:

        perplexity = float("inf")

    accuracy = (
        correct / valid_tokens
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
# 39. TEST 18 - BASELINE
# ============================================================

print(
    "TEST 18: Baseline Planning Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    plan_validation_loader
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
# 40. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(plan_train_loader) * EPOCHS
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
)


# ============================================================
# 41. TEST 19 - TRAINING
# ============================================================

print(
    "TEST 19: Native Planning Fine-Tuning"
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
            plan_train_loader,
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

        loss = planning_loss(
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
            f"| Batch {batch_number}/{len(plan_train_loader)} "
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
        plan_validation_loader
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
                    "83R",

                "base_checkpoint":
                    str(BASE_CHECKPOINT),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "plan_count":
                    len(planning_tasks)
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 42. TEST 20 - FINAL EVALUATION
# ============================================================

print(
    "TEST 20: Final Planning Evaluation"
)

print()

final_metrics = evaluate(
    model,
    plan_validation_loader
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
# 43. TEST 21 - NUMERICAL HEALTH
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
# 44. TEST 22 - PARAMETER CHANGE
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
# 45. TEST 23 - BEHAVIOR VALIDATION
# ============================================================

print(
    "TEST 23: Planning Behavior Validation"
)

print()

behavior_errors = []

for task in planning_tasks:

    plan = build_plan(
        task
    )

    execution = execute_plan(
        plan
    )

    if not execution["complete"]:

        behavior_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Plan did not complete."
            }
        )

    if len(plan.steps) != MAX_PLAN_STEPS:

        behavior_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Incorrect plan step count."
            }
        )

if behavior_errors:

    print(
        json.dumps(
            behavior_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Planning behavior validation failed."
    )

print(
    "Planning behavior valid:",
    len(planning_tasks)
)

print()


# ============================================================
# 46. TEST 24 - PROMOTION
# ============================================================

print(
    "TEST 24: Planning Promotion Gate"
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
        "Candidate planning loss is invalid."
    )

elif (
        math.isfinite(baseline_loss)
        and
        candidate_loss < baseline_loss
):

    decision = (
        "PROMOTE_CANDIDATE"
    )

    reason = (
        "Planning validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Planning validation loss did not improve."
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
# 47. TEST 25 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 25: Save Planning Candidate"
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
        "83R",

    "training_mode":
        "native_planning_and_tool_sequencing",

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

    "plan_count":
        len(planning_tasks),

    "maximum_plan_steps":
        MAX_PLAN_STEPS,

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
# 48. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "83R",

    "training_mode":
        "native_planning_and_tool_sequencing",

    "base_checkpoint":
        str(BASE_CHECKPOINT),

    "external_llm":
        False,

    "device":
        str(DEVICE),

    "plan_count":
        len(planning_tasks),

    "maximum_plan_steps":
        MAX_PLAN_STEPS,

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "training_examples":
        len(plan_train_records),

    "validation_examples":
        len(plan_validation_records),

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
# 49. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "83R",

    "capability":
        "native_planning_and_tool_sequencing",

    "maximum_plan_steps":
        MAX_PLAN_STEPS,

    "planning_examples":
        len(planning_tasks),

    "training_examples":
        len(plan_train_records),

    "validation_examples":
        len(plan_validation_records),

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

    "planning_behavior":
        {
            "valid":
                len(planning_tasks)
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
# 50. PLANNING ARCHITECTURE
# ============================================================

print(
    "SILVERWING PLANNING ARCHITECTURE"
)

print()

print("Goal")
print(" ↓")
print("Plan Construction")
print(" ↓")
print("Step Ordering")
print(" ↓")
print("Dependency Resolution")
print(" ↓")
print("Tool Execution")
print(" ↓")
print("Intermediate Result")
print(" ↓")
print("Next Step")
print(" ↓")
print("Verification")
print(" ↓")
print("Final Answer")

print()


# ============================================================
# 51. PLANNING PRINCIPLES
# ============================================================

print(
    "PLANNING PRINCIPLES"
)

print()

print(
    "Dependent steps execute only after dependencies."
)

print(
    "Intermediate results can become later inputs."
)

print(
    "Result references are resolved before execution."
)

print(
    "Arithmetic references use a restricted evaluator."
)

print(
    "Every plan must complete successfully."
)

print(
    "Promotion requires validation."
)

print()


# ============================================================
# 52. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "83R currently supports short two-step plans."
)

print(
    "83R does not yet perform long-horizon planning."
)

print(
    "83R does not yet autonomously replan."
)

print(
    "83R does not yet dynamically discover tools."
)

print(
    "83R does not yet perform unrestricted autonomous action."
)

print(
    "83R does not yet modify its own architecture."
)

print()


# ============================================================
# 53. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 84R: Native Verified Execution and Replanning"
)

print()

print(
    "Plan + Execution + Failure Detection + "
    "Recovery + Replanning + Verification"
)

print()


# ============================================================
# 54. FOUNDATION MODEL PROGRESS
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
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 55. COMPLETE
# ============================================================

print(
    "=== LESSON 83R COMPLETE ==="
)
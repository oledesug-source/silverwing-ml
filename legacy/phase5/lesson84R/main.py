# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 84R
# Native Verified Execution and Replanning Engine
# ============================================================
#
# CURRICULUM
#
# 79R -> Native Reasoning Dataset
# 80R -> Native Reasoning Fine-Tuning
# 81R -> Native Memory-Aware Training
# 82R -> Native Tool-Aware Learning
# 83R -> Native Planning and Tool Sequencing
# 84R -> Native Verified Execution and Replanning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 83R established:
#
#     goal
#       ↓
#     ordered plan
#       ↓
#     dependencies
#       ↓
#     tool execution
#
# 84R adds:
#
#     execution monitoring
#       ↓
#     failure detection
#       ↓
#     recovery decision
#       ↓
#     replanning
#       ↓
#     re-execution
#       ↓
#     final verification
#
# This creates the first controlled closed-loop action cycle.
#
# ============================================================
# IMPORTANT
# ============================================================
#
# This is not unrestricted autonomous behavior.
#
# The environment and tools are deterministic and controlled.
#
# Silverwing learns the structure:
#
#     PLAN
#       ↓
#     ACT
#       ↓
#     OBSERVE
#       ↓
#     VERIFY
#       ↓
#     RECOVER
#       ↓
#     REPLAN
#
# ============================================================
# NO EXTERNAL LLM
# ============================================================
#
# GPT-2: NONE
# Qwen: NONE
# External LLM: NONE
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
# ============================================================
# SEQUENCE LIMIT
# ============================================================
#
# The established 256-token limit is preserved.
#
# Training traces are compact.
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
from typing import Any, Dict, List, Optional, Tuple

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
LESSON_83R = PHASE5_DIR / "lesson83R"

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
        LESSON_83R
        / "checkpoints"
        / "silverwing_planning_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_83R
        / "checkpoints"
        / "silverwing_planning_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXECUTION_REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_execution_registry.json"
)

EXECUTION_TRAIN_FILE = (
        BASE_DIR
        / "silverwing_execution_train.jsonl"
)

EXECUTION_VALIDATION_FILE = (
        BASE_DIR
        / "silverwing_execution_validation.jsonl"
)

EXECUTION_REPORT_FILE = (
        BASE_DIR
        / "silverwing_execution_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_execution_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_execution_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_execution_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR
        / "silverwing_execution_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 8.0e-6

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

MAX_EXECUTION_STEPS = 3

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


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 83R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 84R")
print("Native Verified Execution and Replanning")
print()

print("79R -> Reasoning")
print("80R -> Reasoning Fine-Tuning")
print("81R -> Memory")
print("82R -> Tool Use")
print("83R -> Planning")
print("84R -> Verified Execution + Replanning")
print()

print("External LLM: NONE")
print("Maximum execution steps:", MAX_EXECUTION_STEPS)
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 83R and Silverwing Inputs"
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

merge_items = (
    merge_data.get("merges", [])
    if isinstance(
        merge_data,
        dict
    )
    else merge_data
)

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

    ids = [BOS_ID]

    for token in tokenize_text(
            text
    ):

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

                return float(
                    node.value
                )

            raise ValueError(
                "Invalid constant."
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
            "Unsupported expression node."
        )

    return evaluate_node(
        tree
    )


# ============================================================
# 11. TOOL DEFINITIONS
# ============================================================

@dataclass
class ToolOutcome:

    success: bool

    result: Dict[str, Any]

    error: Optional[str] = None


def calculator_tool(
        arguments: Dict[str, Any]
) -> ToolOutcome:

    try:

        expression = str(
            arguments.get(
                "expression",
                ""
            )
        )

        value = safe_arithmetic(
            expression
        )

        return ToolOutcome(
            success=True,
            result={
                "value":
                    value
            }
        )

    except Exception as exc:

        return ToolOutcome(
            success=False,
            result={},
            error=str(exc)
        )


def unit_conversion_tool(
        arguments: Dict[str, Any]
) -> ToolOutcome:

    try:

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
                lambda x:
                x * 60,

            (
                "minutes",
                "seconds"
            ):
                lambda x:
                x * 60,

            (
                "meters",
                "kilometers"
            ):
                lambda x:
                x / 1000,

            (
                "kilometers",
                "meters"
            ):
                lambda x:
                x * 1000

        }

        key = (
            from_unit,
            to_unit
        )

        if key not in conversions:

            return ToolOutcome(
                success=False,
                result={},
                error=(
                    f"Unsupported conversion: "
                    f"{from_unit}->{to_unit}"
                )
            )

        result = conversions[
            key
        ](
            value
        )

        return ToolOutcome(
            success=True,
            result={
                "value":
                    result,

                "unit":
                    to_unit
            }
        )

    except Exception as exc:

        return ToolOutcome(
            success=False,
            result={},
            error=str(exc)
        )


TOOL_FUNCTIONS = {

    "calculator":
        calculator_tool,

    "unit_conversion":
        unit_conversion_tool

}


# ============================================================
# 12. CONTROLLED FAILURE TOOL WRAPPER
# ============================================================

def execute_tool(
        tool_name: str,
        arguments: Dict[str, Any],
        forced_failure: bool = False
) -> ToolOutcome:

    if forced_failure:

        return ToolOutcome(
            success=False,
            result={},
            error="Injected execution failure."
        )

    if tool_name not in TOOL_FUNCTIONS:

        return ToolOutcome(
            success=False,
            result={},
            error=f"Unknown tool: {tool_name}"
        )

    return TOOL_FUNCTIONS[
        tool_name
    ](
        arguments
    )


# ============================================================
# 13. PLAN SCHEMA
# ============================================================

@dataclass
class ExecutionStep:

    step_id: str

    tool: str

    arguments: Dict[str, Any]

    depends_on: List[str]


@dataclass
class ExecutionPlan:

    plan_id: str

    goal: str

    steps: List[ExecutionStep]

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
                f"Missing result: {step_id}"
            )

        if result_key not in results[
            step_id
        ]:

            raise RuntimeError(
                (
                    f"Missing field "
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
                    f"Missing dependency "
                    f"{step_id}"
                )
            )

        if result_key not in results[
            step_id
        ]:

            raise RuntimeError(
                (
                    f"Missing result field "
                    f"{result_key}"
                )
            )

        expression = expression.replace(
            f"${step_id}.{result_key}",
            str(
                results[
                    step_id
                ][
                    result_key
                ]
            )
        )

    return safe_arithmetic(
        expression
    )


# ============================================================
# 15. PLAN EXECUTION
# ============================================================

def execute_plan(
        plan: ExecutionPlan,
        failure_step: Optional[str] = None
) -> Dict[str, Any]:

    results: Dict[
        str,
        Dict[str, Any]
    ] = {}

    completed_steps = []

    failed_step = None

    failure_error = None

    for step in plan.steps:

        for dependency in step.depends_on:

            if dependency not in results:

                failed_step = step.step_id

                failure_error = (
                    f"Dependency {dependency} unavailable."
                )

                return {
                    "success":
                        False,

                    "results":
                        results,

                    "completed_steps":
                        completed_steps,

                    "failed_step":
                        failed_step,

                    "error":
                        failure_error
                }

        resolved_arguments = {}

        try:

            for key, value in step.arguments.items():

                resolved_arguments[
                    key
                ] = resolve_value(
                    value,
                    results
                )

        except Exception as exc:

            return {
                "success":
                    False,

                "results":
                    results,

                "completed_steps":
                    completed_steps,

                "failed_step":
                    step.step_id,

                "error":
                    str(exc)
            }

        forced_failure = (
                step.step_id
                ==
                failure_step
        )

        outcome = execute_tool(
            step.tool,
            resolved_arguments,
            forced_failure=forced_failure
        )

        if not outcome.success:

            return {
                "success":
                    False,

                "results":
                    results,

                "completed_steps":
                    completed_steps,

                "failed_step":
                    step.step_id,

                "error":
                    outcome.error
            }

        results[
            step.step_id
        ] = outcome.result

        completed_steps.append(
            step.step_id
        )

    return {
        "success":
            True,

        "results":
            results,

        "completed_steps":
            completed_steps,

        "failed_step":
            None,

        "error":
            None
    }


# ============================================================
# 16. REPLANNING
# ============================================================

def replan_after_failure(
        plan: ExecutionPlan,
        failed_step: str,
        error: str
) -> ExecutionPlan:

    if (
            plan.plan_id
            ==
            "exec_001"
    ):

        return ExecutionPlan(

            plan_id=
            "replan_001",

            goal=
            plan.goal,

            steps=[

                ExecutionStep(

                    step_id=
                    "step_01",

                    tool=
                    "calculator",

                    arguments=
                    {
                        "expression":
                            "25 * 8"
                    },

                    depends_on=[]

                )

            ],

            verification=
            (
                "Recovered by replacing the "
                "failed unit conversion with "
                "a deterministic calculation."
            ),

            final_answer=
            "25 multiplied by 8 equals 200."

        )

    return ExecutionPlan(

        plan_id=
        plan.plan_id
        +
        "_replanned",

        goal=
        plan.goal,

        steps=
        plan.steps,

        verification=
        (
            "Original plan was retried after "
            "failure analysis."
        ),

        final_answer=
        plan.final_answer

    )


# ============================================================
# 17. TEST 5 - TOOL AVAILABILITY
# ============================================================

print(
    "TEST 5: Verify Execution Tools"
)

print()

print(
    "Available tools:",
    list(
        TOOL_FUNCTIONS.keys()
    )
)

print()


# ============================================================
# 18. TEST 6 - NORMAL EXECUTION
# ============================================================

print(
    "TEST 6: Verified Normal Execution"
)

print()

normal_plan = ExecutionPlan(

    plan_id=
    "normal_001",

    goal=
    "Calculate 3 hours in seconds.",

    steps=[

        ExecutionStep(

            step_id=
            "step_01",

            tool=
            "unit_conversion",

            arguments=
            {
                "value":
                    3,

                "from_unit":
                    "hours",

                "to_unit":
                    "minutes"
            },

            depends_on=[]

        ),

        ExecutionStep(

            step_id=
            "step_02",

            tool=
            "unit_conversion",

            arguments=
            {
                "value":
                    "$step_01.value",

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

    verification=
    "Both execution steps must succeed.",

    final_answer=
    "3 hours equals 10800 seconds."

)

normal_execution = execute_plan(
    normal_plan
)

print(
    "Execution success:",
    normal_execution["success"]
)

print(
    "Completed:",
    normal_execution["completed_steps"]
)

print(
    "Results:",
    normal_execution["results"]
)

print()


if not normal_execution[
    "success"
]:

    raise RuntimeError(
        "Normal execution failed."
    )


# ============================================================
# 19. TEST 7 - FAILURE INJECTION
# ============================================================

print(
    "TEST 7: Controlled Failure Detection"
)

print()

failure_plan = ExecutionPlan(

    plan_id=
    "failure_001",

    goal=
    "Calculate 25 multiplied by 8.",

    steps=[

        ExecutionStep(

            step_id=
            "step_01",

            tool=
            "calculator",

            arguments=
            {
                "expression":
                    "25 * 8"
            },

            depends_on=[]

        )

    ],

    verification=
    "The calculator result must be verified.",

    final_answer=
    "25 multiplied by 8 equals 200."

)

failed_execution = execute_plan(
    failure_plan,
    failure_step="step_01"
)

print(
    "Execution success:",
    failed_execution["success"]
)

print(
    "Failed step:",
    failed_execution["failed_step"]
)

print(
    "Error:",
    failed_execution["error"]
)

print()


if failed_execution[
    "success"
]:

    raise RuntimeError(
        "Failure injection did not fail."
    )


# ============================================================
# 20. TEST 8 - REPLANNING
# ============================================================

print(
    "TEST 8: Replanning After Failure"
)

print()

replanned = replan_after_failure(
    failure_plan,
    failed_execution["failed_step"],
    failed_execution["error"]
)

print(
    "New plan:",
    replanned.plan_id
)

print(
    "Steps:",
    [
        step.step_id
        for step in replanned.steps
    ]
)

print()


# ============================================================
# 21. TEST 9 - RECOVERY EXECUTION
# ============================================================

print(
    "TEST 9: Recovery Execution"
)

print()

recovery_execution = execute_plan(
    replanned
)

print(
    "Recovery success:",
    recovery_execution["success"]
)

print(
    "Completed:",
    recovery_execution["completed_steps"]
)

print(
    "Results:",
    recovery_execution["results"]
)

print()


if not recovery_execution[
    "success"
]:

    raise RuntimeError(
        "Recovery execution failed."
    )


# ============================================================
# 22. TEST 10 - CLOSED LOOP VALIDATION
# ============================================================

print(
    "TEST 10: Closed-Loop Execution Validation"
)

print()

closed_loop_valid = (

        not failed_execution[
            "success"
        ]

        and

        recovery_execution[
            "success"
        ]

        and

        len(
            recovery_execution[
                "completed_steps"
            ]
        )
        > 0

)

print(
    "Failure detected:",
    not failed_execution["success"]
)

print(
    "Recovery completed:",
    recovery_execution["success"]
)

print(
    "Closed loop valid:",
    closed_loop_valid
)

print()


if not closed_loop_valid:

    raise RuntimeError(
        "Closed-loop execution validation failed."
    )


# ============================================================
# 23. TEST 11 - BUILD TRAINING TASKS
# ============================================================

print(
    "TEST 11: Build Execution and Replanning Tasks"
)

print()

execution_tasks = [

    {
        "example_id":
            "exec_001",

        "goal":
            "Calculate 25 multiplied by 8.",

        "initial_tool":
            "unit_conversion",

        "initial_arguments":
            {
                "value":
                    25,

                "from_unit":
                    "bad_unit",

                "to_unit":
                    "minutes"
            },

        "failure_reason":
            "Unsupported conversion.",

        "recovery_tool":
            "calculator",

        "recovery_arguments":
            {
                "expression":
                    "25 * 8"
            },

        "verification":
            "The recovery calculation produces 200.",

        "answer":
            "25 multiplied by 8 equals 200."
    },

    {
        "example_id":
            "exec_002",

        "goal":
            "Convert 2 kilometers to meters.",

        "initial_tool":
            "unit_conversion",

        "initial_arguments":
            {
                "value":
                    2,

                "from_unit":
                    "kilometers",

                "to_unit":
                    "meters"
            },

        "failure_reason":
            "Execution service unavailable.",

        "recovery_tool":
            "calculator",

        "recovery_arguments":
            {
                "expression":
                    "2 * 1000"
            },

        "verification":
            "The recovery calculation produces 2000.",

        "answer":
            "2 kilometers equals 2000 meters."
    },

    {
        "example_id":
            "exec_003",

        "goal":
            "Calculate 144 divided by 12.",

        "initial_tool":
            "calculator",

        "initial_arguments":
            {
                "expression":
                    "144 / 12"
            },

        "failure_reason":
            "Tool execution failed.",

        "recovery_tool":
            "calculator",

        "recovery_arguments":
            {
                "expression":
                    "12 * 12"
            },

        "verification":
            "The recovery calculation reproduces 144.",

        "answer":
            "144 divided by 12 equals 12."
    },

    {
        "example_id":
            "exec_004",

        "goal":
            "Convert 3 hours to minutes.",

        "initial_tool":
            "unit_conversion",

        "initial_arguments":
            {
                "value":
                    3,

                "from_unit":
                    "hours",

                "to_unit":
                    "minutes"
            },

        "failure_reason":
            "Execution failure detected.",

        "recovery_tool":
            "calculator",

        "recovery_arguments":
            {
                "expression":
                    "3 * 60"
            },

        "verification":
            "The recovery calculation produces 180.",

        "answer":
            "3 hours equals 180 minutes."
    }
]

print(
    "Execution tasks:",
    len(execution_tasks)
)

print()


# ============================================================
# 24. COMPACT EXECUTION TRACE
# ============================================================

def build_execution_trace(
        task: Dict[str, Any]
) -> str:

    initial_result = execute_tool(
        task["initial_tool"],
        task["initial_arguments"],
        forced_failure=True
    )

    recovery_result = execute_tool(
        task["recovery_tool"],
        task["recovery_arguments"],
        forced_failure=False
    )

    if recovery_result.success:

        recovery_value = (
            recovery_result.result
        )

    else:

        recovery_value = {
            "error":
                recovery_result.error
        }

    lines = [

        "G:" +
        task["goal"],

        "F:" +
        task["initial_tool"],

        "E:" +
        (
                task["failure_reason"]
                +
                ";"
                +
                str(
                    initial_result.error
                )
        ),

        "R:" +
        task["recovery_tool"]
        +
        ";"
        +
        str(
            task["recovery_arguments"]
        )
        +
        ";"
        +
        str(
            recovery_value
        ),

        "V:" +
        task["verification"],

        "A:" +
        task["answer"]

    ]

    return "\n".join(
        lines
    )


# ============================================================
# 25. TEST 12 - TRACE VALIDATION
# ============================================================

print(
    "TEST 12: Validate Execution Traces"
)

print()

execution_records = []

for task in execution_tasks:

    trace = build_execution_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    execution_records.append(
        {
            "example_id":
                task["example_id"],

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
        "tokens"
    )

print()


# ============================================================
# 26. TEST 13 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 13: Execution Token Validation"
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

    for record in execution_records

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
            "Execution/replanning examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All execution/replanning examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 27. TEST 14 - FAILURE / RECOVERY SEMANTICS
# ============================================================

print(
    "TEST 14: Failure-Recovery Semantics"
)

print()

semantic_errors = []

for task in execution_tasks:

    failure = execute_tool(
        task["initial_tool"],
        task["initial_arguments"],
        forced_failure=True
    )

    recovery = execute_tool(
        task["recovery_tool"],
        task["recovery_arguments"]
    )

    if failure.success:

        semantic_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Initial failure was not detected."
            }
        )

    if not recovery.success:

        semantic_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Recovery failed."
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
        "Failure-recovery semantics failed."
    )

print(
    "Failure-recovery semantics valid:",
    len(execution_tasks)
)

print()


# ============================================================
# 28. TEST 15 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    execution_records
)

validation_count = max(
    2,
    int(
        round(
            len(execution_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(execution_records) - 1
)

execution_train_records = (
    execution_records[
        :-validation_count
    ]
)

execution_validation_records = (
    execution_records[
        -validation_count:
    ]
)

print(
    "TEST 15: Execution Train/Validation Split"
)

print(
    "Training examples:",
    len(execution_train_records)
)

print(
    "Validation examples:",
    len(execution_validation_records)
)

print()


# ============================================================
# 29. SAVE ARTIFACTS
# ============================================================

write_json(
    EXECUTION_REGISTRY_FILE,
    {
        "lesson":
            "84R",

        "capability":
            "verified_execution_and_replanning",

        "maximum_steps":
            MAX_EXECUTION_STEPS,

        "sequence_limit":
            MAX_SEQUENCE_LENGTH
    }
)

with open(
        EXECUTION_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in execution_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        EXECUTION_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in execution_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    EXECUTION_REPORT_FILE,
    {
        "lesson":
            "84R",

        "capability":
            "native_verified_execution_and_replanning",

        "training_examples":
            len(
                execution_train_records
            ),

        "validation_examples":
            len(
                execution_validation_records
            ),

        "external_llm":
            False
    }
)


# ============================================================
# 30. DATASET
# ============================================================

class ExecutionDataset(
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


def collate_execution_batch(
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


execution_train_dataset = ExecutionDataset(
    execution_train_records
)

execution_validation_dataset = ExecutionDataset(
    execution_validation_records
)

execution_train_loader = DataLoader(
    execution_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_execution_batch
)

execution_validation_loader = DataLoader(
    execution_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_execution_batch
)

print(
    "TEST 16: Execution DataLoaders"
)

print(
    "Training samples:",
    len(execution_train_dataset)
)

print(
    "Validation samples:",
    len(execution_validation_dataset)
)

print(
    "Training batches:",
    len(execution_train_loader)
)

print(
    "Validation batches:",
    len(execution_validation_loader)
)

print()


# ============================================================
# 31. EXACT SILVERWING ATTENTION
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
# 32. FEED FORWARD
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
# 33. TRANSFORMER BLOCK
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
# 34. POSITION EMBEDDING
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
# 35. SILVERWING DECODER
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

        x = self.final_norm(x)

        return self.language_model_head(x)


# ============================================================
# 36. TEST 17 - STRICT LOAD
# ============================================================

print(
    "TEST 17: Strict Load of 83R Planning Model"
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
        "83R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "83R checkpoint is missing model_state_dict."
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
                "83R checkpoint architecture mismatch. "
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
            "84R refused to load a mismatched "
            "83R Silverwing model.\n\n"
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
    "83R model is compatible with 84R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 37. BASELINE SNAPSHOT
# ============================================================

baseline_state = {

    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()

}


# ============================================================
# 38. LOSS
# ============================================================

def execution_loss(
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
# 39. EVALUATION
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

        loss = execution_loss(
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

        perplexity = float(
            "inf"
        )

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
# 40. TEST 18 - BASELINE
# ============================================================

print(
    "TEST 18: Baseline Execution Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    execution_validation_loader
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
# 41. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(execution_train_loader)
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
# 42. TEST 19 - TRAINING
# ============================================================

print(
    "TEST 19: Native Verified-Execution Fine-Tuning"
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
            execution_train_loader,
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

        loss = execution_loss(
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
            f"| Batch {batch_number}/{len(execution_train_loader)} "
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
        execution_validation_loader
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
                    "84R",

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

                "execution_task_count":
                    len(execution_tasks)
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 43. TEST 20 - FINAL EVALUATION
# ============================================================

print(
    "TEST 20: Final Execution Evaluation"
)

print()

final_metrics = evaluate(
    model,
    execution_validation_loader
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
# 44. TEST 21 - NUMERICAL HEALTH
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
# 45. TEST 22 - PARAMETER CHANGE
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
# 46. TEST 23 - CLOSED LOOP BEHAVIOR
# ============================================================

print(
    "TEST 23: Closed-Loop Behavior Validation"
)

print()

closed_loop_errors = []

for task in execution_tasks:

    initial = execute_tool(
        task["initial_tool"],
        task["initial_arguments"],
        forced_failure=True
    )

    recovery = execute_tool(
        task["recovery_tool"],
        task["recovery_arguments"]
    )

    if initial.success:

        closed_loop_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Failure was not detected."
            }
        )

    if not recovery.success:

        closed_loop_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Recovery did not succeed."
            }
        )

if closed_loop_errors:

    print(
        json.dumps(
            closed_loop_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Closed-loop behavior validation failed."
    )

print(
    "Closed-loop behavior valid:",
    len(execution_tasks)
)

print()


# ============================================================
# 47. TEST 24 - PROMOTION
# ============================================================

print(
    "TEST 24: Verified-Execution Promotion Gate"
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
        "Candidate execution loss is invalid."
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
        "Execution/replanning validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Execution/replanning validation loss did not improve."
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
# 48. TEST 25 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 25: Save Execution Candidate"
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
        "84R",

    "training_mode":
        "native_verified_execution_and_replanning",

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

    "execution_task_count":
        len(execution_tasks),

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
# 49. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "84R",

    "training_mode":
        "native_verified_execution_and_replanning",

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

    "execution_tasks":
        len(
            execution_tasks
        ),

    "training_examples":
        len(
            execution_train_records
        ),

    "validation_examples":
        len(
            execution_validation_records
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
# 50. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "84R",

    "capability":
        "native_verified_execution_and_replanning",

    "execution_tasks":
        len(
            execution_tasks
        ),

    "training_examples":
        len(
            execution_train_records
        ),

    "validation_examples":
        len(
            execution_validation_records
        ),

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

    "closed_loop":
        {
            "valid":
                len(execution_tasks)
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
# 51. CONTROL LOOP
# ============================================================

print(
    "SILVERWING VERIFIED EXECUTION LOOP"
)

print()

print("Plan")
print(" ↓")
print("Execute")
print(" ↓")
print("Observe")
print(" ↓")
print("Verify")
print(" ↓")
print("Failure?")
print(" ↓")
print("Recover")
print(" ↓")
print("Replan")
print(" ↓")
print("Execute Again")
print(" ↓")
print("Final Verification")

print()


# ============================================================
# 52. ARCHITECTURAL SIGNIFICANCE
# ============================================================

print(
    "ARCHITECTURAL SIGNIFICANCE"
)

print()

print(
    "83R gave Silverwing ordered plans."
)

print()

print(
    "84R gives Silverwing a controlled feedback loop "
    "between planning and execution."
)

print()

print(
    "This is the beginning of adaptive action rather "
    "than one-shot tool invocation."
)

print()


# ============================================================
# 53. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "84R uses controlled deterministic failures."
)

print(
    "84R does not yet perform long-horizon autonomous recovery."
)

print(
    "84R does not yet discover new tools autonomously."
)

print(
    "84R does not yet modify its own architecture."
)

print(
    "84R does not yet implement full continual learning."
)

print()


# ============================================================
# 54. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 85R: Native Mathematical Reasoning Foundation"
)

print()

print(
    "Arithmetic + Algebra + Geometry + "
    "Probability + Statistics + Formal Validation"
)

print()


# ============================================================
# 55. FOUNDATION MODEL PROGRESS
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
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 56. COMPLETE
# ============================================================

print(
    "=== LESSON 84R COMPLETE ==="
)
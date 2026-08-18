# Silverwing ML
# Phase 5 - Lesson 79R
# Native Reasoning Dataset and Reasoning Evaluation
#
# No GPT-2.
# No Qwen.
# No external reasoning model.

import hashlib
import json
import re
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
LESSON_66 = BASE_DIR.parent / "lesson66R"
LESSON_76 = BASE_DIR.parent / "lesson76R"
LESSON_77 = BASE_DIR.parent / "lesson77R"
LESSON_78 = BASE_DIR.parent / "lesson78R"

VOCABULARY_FILE = (
        LESSON_66 / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66 / "silverwing_bpe_merges.json"
)

CURRICULUM_FILE = (
        LESSON_76 / "silverwing_curriculum_config.json"
)

INSTRUCTION_CONFIG_FILE = (
        LESSON_77 / "silverwing_instruction_config.json"
)

INSTRUCTION_EVAL_FILE = (
        LESSON_78 / "silverwing_instruction_evaluation.json"
)

REASONING_CONFIG_FILE = (
        BASE_DIR / "silverwing_reasoning_config.json"
)

REASONING_DATASET_FILE = (
        BASE_DIR / "silverwing_reasoning_dataset.jsonl"
)

REASONING_TRAIN_FILE = (
        BASE_DIR / "silverwing_reasoning_train.jsonl"
)

REASONING_VALIDATION_FILE = (
        BASE_DIR / "silverwing_reasoning_validation.jsonl"
)

REASONING_REPORT_FILE = (
        BASE_DIR / "silverwing_reasoning_report.json"
)

REASONING_EVAL_FILE = (
        BASE_DIR / "silverwing_reasoning_evaluation.json"
)

VALIDATION_RATIO = 0.20

MAX_REASONING_TOKENS = 256


# ==================================================
# HELPERS
# ==================================================

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


def require_file(
        path: Path
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


def sha256_text(
        text: str
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_text(
        text: str
) -> str:

    text = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==================================================
# HEADER
# ==================================================

print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 79R")
print("Silverwing Own Foundation Model")
print("Native Reasoning Dataset and Reasoning Evaluation")
print()


# ==================================================
# VERIFY
# ==================================================

print("TEST 1: Verify Previous Artifacts")
print()


for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    CURRICULUM_FILE,
    INSTRUCTION_CONFIG_FILE,
    INSTRUCTION_EVAL_FILE,
]:

    require_file(
        path
    )

    print(
        "FOUND:",
        path
    )


print()


# ==================================================
# VOCABULARY
# ==================================================

print("TEST 2: Load Silverwing Vocabulary")
print()


vocabulary = read_json(
    VOCABULARY_FILE
)


TOKEN_TO_ID = {

    token:
        int(token_id)

    for token, token_id
    in vocabulary[
        "token_to_id"
    ].items()

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

print()


# ==================================================
# CURRICULUM
# ==================================================

print("TEST 3: Load Curriculum")
print()


curriculum = read_json(
    CURRICULUM_FILE
)


stages = curriculum.get(
    "stages",
    []
)


stage_ids = {
    stage[
        "stage_id"
    ]
    for stage in stages
}


print(
    "Curriculum stages:",
    len(stages)
)

print(
    "Stage IDs:",
    sorted(stage_ids)
)

print()


# ==================================================
# BPE
# ==================================================

print("TEST 4: Load BPE Merges")
print()


merge_data = read_json(
    MERGES_FILE
)


MERGE_RANKS = {}


for item in merge_data:

    pair = item[
        "pair"
    ]


    if (
            not isinstance(
                pair,
                list
            )
            or
            len(pair) != 2
    ):

        raise ValueError(
            f"Invalid BPE pair: {pair}"
        )


    MERGE_RANKS[
        (
            pair[0],
            pair[1]
        )
    ] = int(
        item[
            "rank"
        ]
    )


print(
    "Merge operations:",
    len(MERGE_RANKS)
)

print()


BPE_END = "</w>"


def split_words(
        text: str
) -> List[str]:

    normalized = normalize_text(
        text
    ).lower()


    return re.findall(
        r"\w+|[^\w\s]",
        normalized,
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

    token_ids = [
        BOS_ID
    ]


    for token in tokenize_text(
            text
    ):

        token_ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )


    token_ids.append(
        EOS_ID
    )


    return token_ids


# ==================================================
# SCHEMAS
# ==================================================

@dataclass
class ReasoningExample:

    example_id: str

    reasoning_type: str

    stage_id: str

    domain: str

    problem: str

    context: str

    reasoning_steps: List[str]

    final_answer: str

    difficulty: float

    source: str

    quality_score: float = 0.0

    token_count: int = 0

    content_hash: str = ""

    validated: bool = False


@dataclass
class ReasoningEvaluation:

    example_id: str

    reasoning_type: str

    expected_steps: int

    answer_present: bool

    reasoning_present: bool

    structure_valid: bool

    difficulty: float

    quality_score: float


REASONING_TYPES = [
    "deduction",
    "induction",
    "comparison",
    "diagnosis",
    "causal_reasoning",
    "multi_step",
    "constraint_reasoning",
    "numerical_reasoning",
    "error_analysis",
    "decision_reasoning",
]


# ==================================================
# REASONING DATASET
# ==================================================

raw_examples = [

    ReasoningExample(
        example_id="reason_001",
        reasoning_type="deduction",
        stage_id="stage_05",
        domain="logic",
        problem=(
            "All maintained machines have inspection "
            "records. Machine A is maintained. What follows?"
        ),
        context="",
        reasoning_steps=[
            "Machine A is maintained.",
            "Maintained machines have inspection records.",
            "Therefore Machine A has an inspection record."
        ],
        final_answer=(
            "Machine A has an inspection record."
        ),
        difficulty=0.55,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_002",
        reasoning_type="numerical_reasoning",
        stage_id="stage_05",
        domain="mathematics",
        problem=(
            "A pump moves 25 litres per minute for "
            "8 minutes. How many litres are moved?"
        ),
        context="",
        reasoning_steps=[
            "The rate is 25 litres per minute.",
            "The duration is 8 minutes.",
            "25 × 8 = 200."
        ],
        final_answer=(
            "The pump moves 200 litres."
        ),
        difficulty=0.50,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_003",
        reasoning_type="diagnosis",
        stage_id="stage_05",
        domain="machine_diagnostics",
        problem=(
            "A machine has increasing vibration, rising "
            "temperature, and unusual bearing noise. "
            "What should be investigated first?"
        ),
        context="The symptoms appeared together.",
        reasoning_steps=[
            "Bearing noise suggests possible bearing deterioration.",
            "Vibration and temperature can also result from bearing problems.",
            "Therefore inspect the bearing system first."
        ],
        final_answer=(
            "Investigate the bearing system first."
        ),
        difficulty=0.65,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_004",
        reasoning_type="comparison",
        stage_id="stage_05",
        domain="machine_learning",
        problem=(
            "Model A has training accuracy 99% and validation "
            "accuracy 70%. Model B has training accuracy 92% "
            "and validation accuracy 89%. Which generalizes better?"
        ),
        context="",
        reasoning_steps=[
            "Validation accuracy better reflects generalization.",
            "Model B has higher validation accuracy and a smaller gap.",
            "Therefore Model B generalizes better."
        ],
        final_answer=(
            "Model B generalizes better."
        ),
        difficulty=0.60,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_005",
        reasoning_type="causal_reasoning",
        stage_id="stage_05",
        domain="engineering",
        problem=(
            "A shaft starts vibrating after coupling replacement. "
            "What should be checked?"
        ),
        context="",
        reasoning_steps=[
            "The vibration began after the replacement.",
            "The new coupling may be misaligned.",
            "Inspect the coupling alignment."
        ],
        final_answer=(
            "Check the new coupling alignment."
        ),
        difficulty=0.68,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_006",
        reasoning_type="constraint_reasoning",
        stage_id="stage_06",
        domain="instruction",
        problem=(
            "A response must contain exactly two actions and no "
            "unrelated recommendations. The machine is overheating."
        ),
        context="",
        reasoning_steps=[
            "Exactly two actions are allowed.",
            "One action should address overheating.",
            "The second should investigate the cause.",
            "Unrelated advice must be excluded."
        ],
        final_answer=(
            "Stop the machine safely and inspect the cooling system."
        ),
        difficulty=0.72,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_007",
        reasoning_type="multi_step",
        stage_id="stage_05",
        domain="mathematics",
        problem=(
            "A machine operates at 2,600 rpm for 3 hours. "
            "Estimate the total revolutions."
        ),
        context="",
        reasoning_steps=[
            "Three hours is 180 minutes.",
            "Multiply 2,600 by 180.",
            "The result is 468,000 revolutions."
        ],
        final_answer=(
            "Approximately 468,000 revolutions."
        ),
        difficulty=0.62,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_008",
        reasoning_type="error_analysis",
        stage_id="stage_10",
        domain="continual_learning",
        problem=(
            "A model reduces validation loss but reasoning accuracy "
            "drops sharply. Promote it immediately?"
        ),
        context="",
        reasoning_steps=[
            "Validation loss improved.",
            "Reasoning performance regressed.",
            "Investigate the regression before promotion."
        ],
        final_answer=(
            "No. Investigate the reasoning regression first."
        ),
        difficulty=0.90,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_009",
        reasoning_type="decision_reasoning",
        stage_id="stage_10",
        domain="ai_systems",
        problem=(
            "A candidate improves new tasks but is slightly worse "
            "on established critical tasks. What should happen?"
        ),
        context="Established tasks are critical.",
        reasoning_steps=[
            "The candidate improves new tasks.",
            "It regresses on critical established tasks.",
            "Keep it isolated until the regression is resolved."
        ],
        final_answer=(
            "Keep the candidate isolated and investigate the regression."
        ),
        difficulty=0.92,
        source="synthetic_seed"
    ),

    ReasoningExample(
        example_id="reason_010",
        reasoning_type="deduction",
        stage_id="stage_05",
        domain="science",
        problem=(
            "Every validated experiment has a recorded result. "
            "Experiment X is validated. What follows?"
        ),
        context="",
        reasoning_steps=[
            "Experiment X is validated.",
            "Validated experiments have recorded results.",
            "Therefore Experiment X has a recorded result."
        ],
        final_answer=(
            "Experiment X has a recorded result."
        ),
        difficulty=0.52,
        source="synthetic_seed"
    ),
]


print(
    "TEST 5: Reasoning Schema"
)

print()

print(
    "Reasoning types:",
    len(REASONING_TYPES)
)


for reasoning_type in REASONING_TYPES:

    print(
        "-",
        reasoning_type
    )


print()


print(
    "TEST 6: Raw Reasoning Examples"
)

print()

print(
    "Examples:",
    len(raw_examples)
)

print()


# ==================================================
# VALIDATION
# ==================================================

def validate_reasoning_example(
        example: ReasoningExample
) -> Tuple[bool, List[str]]:

    errors = []


    if not example.example_id.strip():

        errors.append(
            "Missing example ID."
        )


    if (
            example.reasoning_type
            not in
            REASONING_TYPES
    ):

        errors.append(
            "Unknown reasoning type."
        )


    if (
            example.stage_id
            not in
            stage_ids
    ):

        errors.append(
            "Unknown curriculum stage."
        )


    if not example.domain.strip():

        errors.append(
            "Missing domain."
        )


    if not example.problem.strip():

        errors.append(
            "Missing problem."
        )


    if not example.reasoning_steps:

        errors.append(
            "Reasoning steps are required."
        )


    if any(
            not step.strip()
            for step
            in example.reasoning_steps
    ):

        errors.append(
            "Reasoning steps cannot be empty."
        )


    if not example.final_answer.strip():

        errors.append(
            "Missing final answer."
        )


    if not (
            0.0
            <=
            example.difficulty
            <=
            1.0
    ):

        errors.append(
            "Difficulty must be between 0 and 1."
        )


    return (
        len(errors) == 0,
        errors
    )


print(
    "TEST 7: Schema Validation"
)

print()

valid_examples = []
invalid_examples = []


for example in raw_examples:

    valid, errors = (
        validate_reasoning_example(
            example
        )
    )


    if valid:

        example.validated = True

        valid_examples.append(
            example
        )

        print(
            "VALID:",
            example.example_id
        )

    else:

        invalid_examples.append(
            {
                "example":
                    example.example_id,

                "errors":
                    errors
            }
        )


print()

print(
    "Valid:",
    len(valid_examples)
)

print(
    "Invalid:",
    len(invalid_examples)
)

print()


if invalid_examples:

    print(
        json.dumps(
            invalid_examples,
            indent=4
        )
    )

    raise RuntimeError(
        "Reasoning dataset schema validation failed."
    )


# ==================================================
# QUALITY
# ==================================================

def reasoning_quality_score(
        example: ReasoningExample
) -> float:

    score = 0.0


    if len(
            example.reasoning_steps
    ) >= 2:

        score += 0.20


    if len(
            example.reasoning_steps
    ) >= 3:

        score += 0.15


    if len(
            re.findall(
                r"\b\w+\b",
                example.problem
            )
    ) >= 8:

        score += 0.15


    if len(
            re.findall(
                r"\b\w+\b",
                example.final_answer
            )
    ) >= 3:

        score += 0.15


    if (
            example.reasoning_type
            in
            REASONING_TYPES
    ):

        score += 0.10


    if example.stage_id in stage_ids:

        score += 0.10


    if example.context.strip():

        score += 0.05


    if (
            0.0
            <=
            example.difficulty
            <=
            1.0
    ):

        score += 0.10


    return min(
        score,
        1.0
    )


print(
    "TEST 8: Reasoning Quality Scoring"
)

print()


for example in valid_examples:

    example.quality_score = (
        reasoning_quality_score(
            example
        )
    )


    print(
        example.example_id,
        "->",
        round(
            example.quality_score,
            4
        )
    )


print()


# ==================================================
# HASHES
# ==================================================

print(
    "TEST 9: Content Hashing"
)

print()


for example in valid_examples:

    canonical = json.dumps(
        {
            "reasoning_type":
                example.reasoning_type,

            "stage_id":
                example.stage_id,

            "domain":
                example.domain,

            "problem":
                example.problem,

            "context":
                example.context,

            "reasoning_steps":
                example.reasoning_steps,

            "final_answer":
                example.final_answer,

            "difficulty":
                example.difficulty

        },
        sort_keys=True
    )


    example.content_hash = (
        sha256_text(
            canonical
        )
    )


    print(
        example.example_id,
        "->",
        example.content_hash[:16]
    )


print()


# ==================================================
# DUPLICATES
# ==================================================

print(
    "TEST 10: Duplicate Detection"
)

print()


seen_hashes = set()

unique_examples = []

duplicates = []


for example in valid_examples:

    if (
            example.content_hash
            in
            seen_hashes
    ):

        duplicates.append(
            example.example_id
        )

    else:

        seen_hashes.add(
            example.content_hash
        )

        unique_examples.append(
            example
        )


print(
    "Unique examples:",
    len(unique_examples)
)

print(
    "Duplicates:",
    len(duplicates)
)

print()


# ==================================================
# FORMATTING
# ==================================================

def format_reasoning_text(
        example: ReasoningExample
) -> str:

    parts = [

        "Problem:",

        example.problem.strip()

    ]


    if example.context.strip():

        parts.extend(
            [
                "",
                "Context:",
                example.context.strip()
            ]
        )


    parts.extend(
        [
            "",
            "Reasoning:"
        ]
    )


    for number, step in enumerate(
            example.reasoning_steps,
            start=1
    ):

        parts.append(
            f"{number}. {step.strip()}"
        )


    parts.extend(
        [
            "",
            "Final Answer:",
            example.final_answer.strip()
        ]
    )


    return "\n".join(
        parts
    )


print(
    "TEST 11: Reasoning Format"
)

print()


if unique_examples:

    print(
        format_reasoning_text(
            unique_examples[0]
        )
    )


print()


# ==================================================
# TOKENIZATION
# ==================================================

print(
    "TEST 12: Reasoning Tokenization"
)

print()


for example in unique_examples:

    token_ids = encode_text(
        format_reasoning_text(
            example
        )
    )


    example.token_count = len(
        token_ids
    )


    print(
        example.example_id,
        "->",
        example.token_count,
        "tokens"
    )


print()


# ==================================================
# TOKEN LIMIT
# ==================================================

print(
    "TEST 13: Token Length Validation"
)

print()


length_errors = []


for example in unique_examples:

    if (
            example.token_count
            >
            MAX_REASONING_TOKENS
    ):

        length_errors.append(
            {
                "example_id":
                    example.example_id,

                "token_count":
                    example.token_count,

                "maximum":
                    MAX_REASONING_TOKENS
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
        "Reasoning examples exceed token limit."
    )


print(
    "All reasoning examples are within "
    "the configured token limit."
)

print()


# ==================================================
# DISTRIBUTIONS
# ==================================================

print(
    "TEST 14: Reasoning Type Distribution"
)

print()


reasoning_distribution = {}


for example in unique_examples:

    key = example.reasoning_type


    reasoning_distribution[key] = (
            reasoning_distribution.get(
                key,
                0
            )
            +
            1
    )


for key, count in sorted(
        reasoning_distribution.items()
):

    print(
        key,
        "->",
        count
    )


print()


print(
    "TEST 15: Reasoning Domain Distribution"
)

print()


domain_distribution = {}


for example in unique_examples:

    key = example.domain


    domain_distribution[key] = (
            domain_distribution.get(
                key,
                0
            )
            +
            1
    )


for key, count in sorted(
        domain_distribution.items()
):

    print(
        key,
        "->",
        count
    )


print()


print(
    "TEST 16: Difficulty Distribution"
)

print()


for example in sorted(
        unique_examples,
        key=lambda item: item.difficulty
):

    print(
        example.example_id,
        "|",
        example.difficulty,
        "|",
        example.reasoning_type
    )


print()


# ==================================================
# COMPLEXITY
# ==================================================

print(
    "TEST 17: Reasoning Complexity Metrics"
)

print()


step_counts = [

    len(
        example.reasoning_steps
    )

    for example
    in unique_examples

]


if step_counts:

    average_steps = (
            sum(step_counts)
            /
            len(step_counts)
    )


    maximum_steps = max(
        step_counts
    )


    minimum_steps = min(
        step_counts
    )


else:

    average_steps = 0.0

    maximum_steps = 0

    minimum_steps = 0


complexity_metrics = {

    "average_reasoning_steps":
        average_steps,

    "minimum_reasoning_steps":
        minimum_steps,

    "maximum_reasoning_steps":
        maximum_steps,

    "average_difficulty":
        (
            sum(
                example.difficulty
                for example
                in unique_examples
            )
            /
            len(unique_examples)
            if unique_examples
            else 0.0
        ),

    "average_quality":
        (
            sum(
                example.quality_score
                for example
                in unique_examples
            )
            /
            len(unique_examples)
            if unique_examples
            else 0.0
        )

}


print(
    json.dumps(
        complexity_metrics,
        indent=4
    )
)


print()


# ==================================================
# CONSISTENCY
# ==================================================

print(
    "TEST 18: Reasoning Consistency Check"
)

print()


consistency_errors = []


for example in unique_examples:

    if not example.reasoning_steps:

        consistency_errors.append(
            {
                "example":
                    example.example_id,

                "problem":
                    "No reasoning content."
            }
        )


    if not example.final_answer.strip():

        consistency_errors.append(
            {
                "example":
                    example.example_id,

                "problem":
                    "No final answer."
            }
        )


    for step in example.reasoning_steps:

        if len(
                step.strip()
        ) < 5:

            consistency_errors.append(
                {
                    "example":
                        example.example_id,

                    "problem":
                        "Reasoning step too short."
                }
            )


if consistency_errors:

    print(
        json.dumps(
            consistency_errors,
            indent=4
        )
    )


    raise RuntimeError(
        "Reasoning consistency validation failed."
    )


print(
    "Reasoning consistency validation passed."
)

print()


# ==================================================
# TRAIN / VALIDATION SPLIT
# ==================================================

print(
    "TEST 19: Reasoning Train/Validation Split"
)

print()


ordered_examples = sorted(
    unique_examples,
    key=lambda example: (
        example.difficulty,
        example.example_id
    )
)


if len(
        ordered_examples
) <= 1:

    train_examples = (
        ordered_examples
    )

    validation_examples = []

else:

    split_index = int(
        len(
            ordered_examples
        )
        *
        (
                1.0
                -
                VALIDATION_RATIO
        )
    )


    split_index = max(
        1,
        min(
            split_index,
            len(
                ordered_examples
            )
            -
            1
        )
    )


    train_examples = (
        ordered_examples[
            :split_index
        ]
    )


    validation_examples = (
        ordered_examples[
            split_index:
        ]
    )


print(
    "Training examples:",
    len(train_examples)
)


print(
    "Validation examples:",
    len(validation_examples)
)

print()


# ==================================================
# SERIALIZATION
# ==================================================

def serialize_example(
        example: ReasoningExample
) -> Dict[str, Any]:

    formatted_text = (
        format_reasoning_text(
            example
        )
    )


    token_ids = encode_text(
        formatted_text
    )


    data = asdict(
        example
    )


    data[
        "formatted_text"
    ] = formatted_text


    data[
        "token_ids"
    ] = token_ids


    data[
        "token_count"
    ] = len(
        token_ids
    )


    data[
        "validated"
    ] = True


    return data


serialized_examples = [

    serialize_example(
        example
    )

    for example
    in unique_examples

]


serialized_train = [

    serialize_example(
        example
    )

    for example
    in train_examples

]


serialized_validation = [

    serialize_example(
        example
    )

    for example
    in validation_examples

]


# ==================================================
# SAVE DATA
# ==================================================

print(
    "TEST 20: Save Reasoning Dataset"
)

print()


with open(
        REASONING_DATASET_FILE,
        "w",
        encoding="utf-8"
) as file:

    for example in serialized_examples:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            )
            +
            "\n"
        )


print(
    "Saved:",
    REASONING_DATASET_FILE
)

print()


print(
    "TEST 21: Save Reasoning Training Data"
)

print()


with open(
        REASONING_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for example in serialized_train:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            )
            +
            "\n"
        )


print(
    "Saved:",
    REASONING_TRAIN_FILE
)

print()


print(
    "TEST 22: Save Reasoning Validation Data"
)

print()


with open(
        REASONING_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for example in serialized_validation:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            )
            +
            "\n"
        )


print(
    "Saved:",
    REASONING_VALIDATION_FILE
)

print()


# ==================================================
# REPORT
# ==================================================

print(
    "TEST 23: Reasoning Dataset Report"
)

print()


total_reasoning_tokens = sum(

    example[
        "token_count"
    ]

    for example
    in serialized_examples

)


average_quality = (

    sum(

        example.quality_score

        for example

        in unique_examples

    )

    /

    len(
        unique_examples
    )

    if unique_examples

    else 0.0

)


report = {

    "dataset":
        "Silverwing-Reasoning-v1",

    "version":
        "79R",

    "examples":
        len(unique_examples),

    "training_examples":
        len(train_examples),

    "validation_examples":
        len(validation_examples),

    "total_reasoning_tokens":
        total_reasoning_tokens,

    "average_quality":
        average_quality,

    "reasoning_distribution":
        reasoning_distribution,

    "domain_distribution":
        domain_distribution,

    "complexity":
        complexity_metrics,

    "vocabulary_size":
        VOCABULARY_SIZE,

    "max_reasoning_tokens":
        MAX_REASONING_TOKENS,

    "created_at":
        time.time()

}


print(
    json.dumps(
        report,
        indent=4
    )
)


print()


write_json(
    REASONING_REPORT_FILE,
    report
)


# ==================================================
# CONFIGURATION
# ==================================================

print(
    "TEST 24: Save Reasoning Configuration"
)

print()


reasoning_config = {

    "dataset":
        "Silverwing-Reasoning-v1",

    "version":
        "79R",

    "format":
        "structured_reasoning",

    "fields": [

        "example_id",
        "reasoning_type",
        "stage_id",
        "domain",
        "problem",
        "context",
        "reasoning_steps",
        "final_answer",
        "difficulty",
        "source",
        "quality_score",
        "token_count",
        "content_hash",
        "validated"

    ],

    "reasoning_types":
        REASONING_TYPES,

    "validation_ratio":
        VALIDATION_RATIO,

    "max_reasoning_tokens":
        MAX_REASONING_TOKENS

}


write_json(
    REASONING_CONFIG_FILE,
    reasoning_config
)


print(
    "Saved:",
    REASONING_CONFIG_FILE
)

print()


# ==================================================
# EVALUATION
# ==================================================

print(
    "TEST 25: Reasoning Evaluation Framework"
)

print()


def evaluate_reasoning_example(
        example: ReasoningExample
) -> ReasoningEvaluation:

    structure_valid = (

            len(
                example.reasoning_steps
            )
            >=
            2

            and

            bool(
                example.final_answer.strip()
            )

            and

            bool(
                example.problem.strip()
            )

    )


    return ReasoningEvaluation(

        example_id=
        example.example_id,

        reasoning_type=
        example.reasoning_type,

        expected_steps=
        len(
            example.reasoning_steps
        ),

        answer_present=
        bool(
            example.final_answer.strip()
        ),

        reasoning_present=
        bool(
            example.reasoning_steps
        ),

        structure_valid=
        structure_valid,

        difficulty=
        example.difficulty,

        quality_score=
        example.quality_score

    )


evaluations = [

    evaluate_reasoning_example(
        example
    )

    for example
    in unique_examples

]


valid_evaluations = [

    evaluation

    for evaluation
    in evaluations

    if evaluation.structure_valid

]


evaluation_score = (

    len(
        valid_evaluations
    )

    /

    len(
        evaluations
    )

    if evaluations

    else 0.0

)


print(
    "Evaluation examples:",
    len(evaluations)
)


print(
    "Structurally valid:",
    len(valid_evaluations)
)


print(
    "Structure score:",
    evaluation_score
)


print()


# ==================================================
# DIFFICULTY BANDS
# ==================================================

print(
    "TEST 26: Reasoning Difficulty Bands"
)

print()


difficulty_bands = {

    "basic":
        0,

    "intermediate":
        0,

    "advanced":
        0

}


for example in unique_examples:

    if (
            example.difficulty
            <
            0.60
    ):

        difficulty_bands[
            "basic"
        ] += 1

    elif (
            example.difficulty
            <
            0.80
    ):

        difficulty_bands[
            "intermediate"
        ] += 1

    else:

        difficulty_bands[
            "advanced"
        ] += 1


print(
    json.dumps(
        difficulty_bands,
        indent=4
    )
)

print()


# ==================================================
# FAILURE TAXONOMY
# ==================================================

print(
    "TEST 27: Reasoning Failure Taxonomy"
)

print()


failure_categories = [

    "missing_reasoning_steps",

    "invalid_inference",

    "contradictory_conclusion",

    "unsupported_claim",

    "numerical_error",

    "constraint_violation",

    "irrelevant_reasoning",

    "premature_conclusion",

    "ignored_context",

    "unsupported_generalization"

]


for category in failure_categories:

    print(
        "-",
        category
    )


print()


# ==================================================
# SAVE EVALUATION
# ==================================================

reasoning_evaluation = {

    "lesson":
        "79R",

    "dataset":
        "Silverwing-Reasoning-v1",

    "evaluation_examples":
        len(evaluations),

    "structurally_valid":
        len(valid_evaluations),

    "structure_score":
        evaluation_score,

    "difficulty_bands":
        difficulty_bands,

    "complexity":
        complexity_metrics,

    "failure_categories":
        failure_categories,

    "created_at":
        time.time()

}


write_json(
    REASONING_EVAL_FILE,
    reasoning_evaluation
)


print(
    "TEST 28: Save Reasoning Evaluation"
)

print()


print(
    "Saved:",
    REASONING_EVAL_FILE
)

print()


# ==================================================
# TRAINING OBJECTIVE
# ==================================================

print(
    "REASONING TRAINING OBJECTIVE"
)

print()

print("Problem")
print("   ↓")
print("Interpret Conditions")
print("   ↓")
print("Identify Relevant Facts")
print("   ↓")
print("Apply Rules / Relationships")
print("   ↓")
print("Generate Intermediate Steps")
print("   ↓")
print("Check Consistency")
print("   ↓")
print("Final Answer")

print()


# ==================================================
# REASONING VS INSTRUCTION
# ==================================================

print(
    "REASONING VS INSTRUCTION TRAINING"
)

print()

print(
    "Instruction training:"
)

print(
    "Follow the requested task."
)

print()

print(
    "Reasoning training:"
)

print(
    "Determine how available information supports "
    "a conclusion or action."
)

print()


# ==================================================
# MULTI-STAGE REASONING
# ==================================================

print(
    "MULTI-STAGE REASONING"
)

print()

print("Observation")
print("    ↓")
print("Hypothesis")
print("    ↓")
print("Evidence Analysis")
print("    ↓")
print("Candidate Explanation")
print("    ↓")
print("Consistency Check")
print("    ↓")
print("Decision")

print()


# ==================================================
# BIO-INSPIRED CONNECTION
# ==================================================

print(
    "BIO-INSPIRED REASONING CONNECTION"
)

print()

print(
    "A biological agent receives signals, builds "
    "internal representations, evaluates alternatives, "
    "and acts according to learned relationships."
)

print()

print(
    "Silverwing's future reasoning loop can combine "
    "learned representations with memory, simulation, "
    "planning, tool results, and feedback."
)

print()


# ==================================================
# CONTROLLED GROWTH
# ==================================================

print(
    "CONTROLLED REASONING DATA GROWTH"
)

print()

print("New Reasoning Problem")
print("        ↓")
print("Schema Validation")
print("        ↓")
print("Reasoning Quality Check")
print("        ↓")
print("Difficulty Classification")
print("        ↓")
print("Independent Evaluation")
print("        ↓")
print("Training Candidate")
print("        ↓")
print("Reasoning Fine-Tuning")
print("        ↓")
print("Regression Evaluation")

print()


# ==================================================
# PRINCIPLE
# ==================================================

print(
    "IMPORTANT ENGINEERING PRINCIPLE"
)

print()

print(
    "Reasoning quality is not the same as language fluency."
)

print()

print(
    "A fluent response can contain incorrect reasoning."
)

print()

print(
    "Silverwing therefore requires reasoning-specific "
    "evaluation rather than relying only on language loss."
)

print()


# ==================================================
# LIMITATION
# ==================================================

print(
    "CURRENT LIMITATION"
)

print()

print(
    "The current reasoning dataset is a small engineering "
    "seed dataset."
)

print()

print(
    "It demonstrates the schema, validation, difficulty "
    "system, and evaluation framework."
)

print()

print(
    "It is not sufficient to create a broadly capable "
    "reasoning system."
)

print()


# ==================================================
# NEXT
# ==================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 80R will connect the reasoning dataset "
    "to Silverwing's native reasoning fine-tuning engine."
)

print()


# ==================================================
# PROGRESS
# ==================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

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
print("Own Decoder Language Model")
print(" ↓")
print("Own Training Data Pipeline")
print(" ↓")
print("Own Pretraining Engine")
print(" ↓")
print("Own Evaluation Framework")
print(" ↓")
print("Own Experiment / Promotion System")
print(" ↓")
print("Own Curriculum Engine")
print(" ↓")
print("Own Instruction Dataset")
print(" ↓")
print("Own Instruction Fine-Tuning Engine")
print(" ↓")
print("OWN REASONING DATASET + EVALUATION")
print(" ↓")
print("Reasoning Fine-Tuning")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Multitask Training")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()

print(
    "=== LESSON 79R COMPLETE ==="
)
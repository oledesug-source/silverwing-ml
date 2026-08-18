# Silverwing ML
# Phase 5 - Lesson 77R
# Silverwing Own Foundation Model
# Native Instruction-Training Dataset and Task Format

import hashlib
import json
import re
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LESSON_66_DIR = (
        BASE_DIR.parent / "lesson66R"
)

LESSON_72_DIR = (
        BASE_DIR.parent / "lesson72R"
)

LESSON_76_DIR = (
        BASE_DIR.parent / "lesson76R"
)

VOCABULARY_FILE = (
        LESSON_66_DIR
        / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66_DIR
        / "silverwing_bpe_merges.json"
)

DATASET_CONFIG_FILE = (
        LESSON_72_DIR
        / "silverwing_dataset_config.json"
)

CURRICULUM_CONFIG_FILE = (
        LESSON_76_DIR
        / "silverwing_curriculum_config.json"
)

INSTRUCTION_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_instruction_config.json"
)

INSTRUCTION_DATASET_FILE = (
        BASE_DIR
        / "silverwing_instruction_dataset.jsonl"
)

INSTRUCTION_TRAIN_FILE = (
        BASE_DIR
        / "silverwing_instruction_train.jsonl"
)

INSTRUCTION_VALIDATION_FILE = (
        BASE_DIR
        / "silverwing_instruction_validation.jsonl"
)

INSTRUCTION_REPORT_FILE = (
        BASE_DIR
        / "silverwing_instruction_report.json"
)

SEED = 42

VALIDATION_RATIO = 0.20

MAX_INSTRUCTION_TOKENS = 192


# ==================================================
# 2. HELPERS
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
# 3. HEADER
# ==================================================

print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 77R")
print("Silverwing Own Foundation Model")
print("Native Instruction-Training Dataset")
print()


# ==================================================
# 4. CONFIGURATION
# ==================================================

print("TEST 1: Configuration")
print()

print(
    "Vocabulary:",
    VOCABULARY_FILE
)

print(
    "Curriculum:",
    CURRICULUM_CONFIG_FILE
)

print(
    "Instruction dataset:",
    INSTRUCTION_DATASET_FILE
)

print(
    "Maximum instruction tokens:",
    MAX_INSTRUCTION_TOKENS
)

print()


# ==================================================
# 5. VERIFY ARTIFACTS
# ==================================================

print("TEST 2: Verify Previous Artifacts")
print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    DATASET_CONFIG_FILE,
    CURRICULUM_CONFIG_FILE,
]:

    require_file(path)

    print(
        "FOUND:",
        path
    )

print()


# ==================================================
# 6. LOAD VOCABULARY
# ==================================================

print("TEST 3: Load Silverwing Vocabulary")
print()

vocabulary_data = read_json(
    VOCABULARY_FILE
)

TOKEN_TO_ID = {
    token: int(token_id)
    for token, token_id
    in vocabulary_data[
        "token_to_id"
    ].items()
}

ID_TO_TOKEN = {
    token_id: token
    for token_id, token
    in TOKEN_TO_ID.items()
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

SPECIAL_TOKENS = vocabulary_data.get(
    "special_tokens",
    [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
        "<MASK>"
    ]
)

print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print()


# ==================================================
# 7. LOAD CURRICULUM
# ==================================================

print("TEST 4: Load Curriculum")
print()

curriculum_config = read_json(
    CURRICULUM_CONFIG_FILE
)

curriculum_stages = curriculum_config.get(
    "stages",
    []
)

stage_ids = {
    stage["stage_id"]
    for stage in curriculum_stages
}

print(
    "Curriculum stages:",
    len(curriculum_stages)
)

print(
    "Stage IDs:",
    sorted(stage_ids)
)

print()


# ==================================================
# 8. TOKENIZATION
# ==================================================

BPE_END = "</w>"


def load_merge_ranks() -> Dict[
    Tuple[str, str],
    int
]:

    merge_data = read_json(
        MERGES_FILE
    )

    ranks = {}

    for item in merge_data:

        pair = item["pair"]

        if (
                not isinstance(pair, list)
                or
                len(pair) != 2
        ):

            raise ValueError(
                f"Invalid BPE pair: {pair}"
            )

        ranks[
            (pair[0], pair[1])
        ] = int(
            item["rank"]
        )

    return ranks


MERGE_RANKS = load_merge_ranks()


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

    symbols[-1] = (
            symbols[-1]
            +
            BPE_END
    )

    return tuple(
        symbols
    )


def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    merged = []

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

            merged.append(
                symbols[index]
                +
                symbols[index + 1]
            )

            index += 2

        else:

            merged.append(
                symbols[index]
            )

            index += 1


    return tuple(
        merged
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

    tokens = tokenize_text(
        text
    )


    ids = [
        BOS_ID
    ]


    for token in tokens:

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


# ==================================================
# 9. INSTRUCTION SCHEMA
# ==================================================

@dataclass
class InstructionExample:

    example_id: str

    task_type: str

    stage_id: str

    domain: str

    instruction: str

    context: str

    response: str

    difficulty: float

    source: str

    quality_score: float = 0.0

    token_count: int = 0

    content_hash: str = ""

    validated: bool = False


# ==================================================
# 10. TASK TYPES
# ==================================================

TASK_TYPES = [
    "question_answering",
    "explanation",
    "classification",
    "transformation",
    "summarization",
    "constraint_following",
    "reasoning",
    "structured_output",
    "tool_planning",
    "error_analysis",
]


print("TEST 5: Instruction Schema")
print()

print(
    "Task types:",
    len(TASK_TYPES)
)

for task_type in TASK_TYPES:

    print(
        "-",
        task_type
    )

print()


# ==================================================
# 11. NATIVE INSTRUCTION DATASET
# ==================================================

raw_examples = [

    InstructionExample(
        example_id="inst_001",
        task_type="question_answering",
        stage_id="stage_01",
        domain="general_language",
        instruction=(
            "What is a sentence?"
        ),
        context="",
        response=(
            "A sentence is a group of words that "
            "expresses a complete thought."
        ),
        difficulty=0.10,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_002",
        task_type="explanation",
        stage_id="stage_02",
        domain="engineering",
        instruction=(
            "Explain the purpose of a bearing."
        ),
        context=(
            "The component is used in a rotating "
            "machine."
        ),
        response=(
            "A bearing supports a rotating shaft and "
            "reduces friction between moving components."
        ),
        difficulty=0.25,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_003",
        task_type="question_answering",
        stage_id="stage_02",
        domain="physics",
        instruction=(
            "What is kinetic energy?"
        ),
        context="",
        response=(
            "Kinetic energy is the energy associated "
            "with the motion of an object."
        ),
        difficulty=0.25,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_004",
        task_type="explanation",
        stage_id="stage_03",
        domain="machine_learning",
        instruction=(
            "Explain overfitting in machine learning."
        ),
        context="",
        response=(
            "Overfitting occurs when a model learns "
            "the training data too specifically and "
            "generalizes poorly to unseen data."
        ),
        difficulty=0.35,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_005",
        task_type="reasoning",
        stage_id="stage_05",
        domain="reasoning",
        instruction=(
            "A model has high training accuracy but "
            "low validation accuracy. What is the "
            "most likely explanation?"
        ),
        context="",
        response=(
            "The most likely explanation is overfitting "
            "because the model performs well on training "
            "examples but generalizes poorly."
        ),
        difficulty=0.60,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_006",
        task_type="constraint_following",
        stage_id="stage_06",
        domain="instruction",
        instruction=(
            "Explain a centrifugal pump in exactly "
            "three sentences."
        ),
        context="",
        response=(
            "A centrifugal pump uses a rotating impeller "
            "to transfer energy to a fluid. "
            "The impeller increases fluid velocity "
            "within the pump. "
            "The casing helps convert that velocity "
            "into useful pressure."
        ),
        difficulty=0.65,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_007",
        task_type="transformation",
        stage_id="stage_06",
        domain="language",
        instruction=(
            "Rewrite the following sentence in a "
            "more formal style."
        ),
        context=(
            "The machine is running badly."
        ),
        response=(
            "The machine is operating abnormally."
        ),
        difficulty=0.65,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_008",
        task_type="summarization",
        stage_id="stage_07",
        domain="technical",
        instruction=(
            "Summarize the supplied information in "
            "one sentence."
        ),
        context=(
            "Preventive maintenance schedules inspections "
            "and servicing before equipment failure occurs."
        ),
        response=(
            "Preventive maintenance uses planned servicing "
            "to reduce the risk of equipment failure."
        ),
        difficulty=0.72,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_009",
        task_type="structured_output",
        stage_id="stage_07",
        domain="structured_output",
        instruction=(
            "Return the machine status as structured "
            "JSON."
        ),
        context=(
            "Temperature is high and vibration is rising."
        ),
        response=(
            '{"status":"warning","reasons":["high_temperature","rising_vibration"]}'
        ),
        difficulty=0.72,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_010",
        task_type="question_answering",
        stage_id="stage_08",
        domain="memory",
        instruction=(
            "Answer using only the supplied memory."
        ),
        context=(
            "Memory: Silverwing was configured to "
            "use Python for its training experiments."
        ),
        response=(
            "Silverwing was configured to use Python "
            "for its training experiments."
        ),
        difficulty=0.80,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_011",
        task_type="tool_planning",
        stage_id="stage_09",
        domain="tools",
        instruction=(
            "Create a structured request for a calculator "
            "tool to calculate 25 * 8."
        ),
        context="",
        response=(
            '{"tool":"calculator","arguments":{"expression":"25 * 8"}}'
        ),
        difficulty=0.85,
        source="synthetic_seed"
    ),

    InstructionExample(
        example_id="inst_012",
        task_type="error_analysis",
        stage_id="stage_10",
        domain="continual_learning",
        instruction=(
            "A new training run improves validation loss "
            "but causes another benchmark to regress. "
            "What should Silverwing do?"
        ),
        context="",
        response=(
            "Silverwing should not automatically promote "
            "the candidate. It should investigate the "
            "regression, evaluate the tradeoff, and promote "
            "the candidate only if the defined release "
            "criteria are satisfied."
        ),
        difficulty=0.92,
        source="synthetic_seed"
    ),
]


print("TEST 6: Raw Instruction Examples")
print()

print(
    "Examples:",
    len(raw_examples)
)

print()


# ==================================================
# 12. TASK VALIDATION
# ==================================================

def validate_example(
        example: InstructionExample
) -> Tuple[bool, List[str]]:

    errors = []


    if not example.example_id.strip():

        errors.append(
            "Missing example ID."
        )


    if example.task_type not in TASK_TYPES:

        errors.append(
            "Unknown task type."
        )


    if example.stage_id not in stage_ids:

        errors.append(
            "Unknown curriculum stage."
        )


    if not example.domain.strip():

        errors.append(
            "Missing domain."
        )


    if not example.instruction.strip():

        errors.append(
            "Missing instruction."
        )


    if not example.response.strip():

        errors.append(
            "Missing response."
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


print("TEST 7: Schema Validation")
print()

valid_examples = []
invalid_examples = []


for example in raw_examples:

    valid, errors = validate_example(
        example
    )


    if valid:

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
        "Instruction dataset schema validation failed."
    )


# ==================================================
# 13. QUALITY SCORING
# ==================================================

def instruction_quality_score(
        example: InstructionExample
) -> float:

    score = 0.0


    instruction_words = re.findall(
        r"\b\w+\b",
        example.instruction
    )


    response_words = re.findall(
        r"\b\w+\b",
        example.response
    )


    if len(instruction_words) >= 3:

        score += 0.20


    if len(response_words) >= 5:

        score += 0.25


    if example.context.strip():

        score += 0.15


    if example.task_type in TASK_TYPES:

        score += 0.15


    if example.stage_id in stage_ids:

        score += 0.15


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


for example in valid_examples:

    example.quality_score = (
        instruction_quality_score(
            example
        )
    )


print("TEST 8: Quality Scoring")
print()

for example in valid_examples:

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
# 14. CONTENT HASHING
# ==================================================

print("TEST 9: Content Hashing")
print()


for example in valid_examples:

    canonical = json.dumps(
        {
            "task_type":
                example.task_type,

            "stage_id":
                example.stage_id,

            "domain":
                example.domain,

            "instruction":
                example.instruction,

            "context":
                example.context,

            "response":
                example.response,

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
# 15. DUPLICATE DETECTION
# ==================================================

print("TEST 10: Duplicate Detection")
print()

seen_hashes = set()

duplicate_examples = []

unique_examples = []


for example in valid_examples:

    if (
            example.content_hash
            in
            seen_hashes
    ):

        duplicate_examples.append(
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
    len(duplicate_examples)
)

print()


# ==================================================
# 16. FORMATTING
# ==================================================

def format_training_text(
        example: InstructionExample
) -> str:

    parts = [
        "Instruction:",
        example.instruction.strip()
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
            "Response:",
            example.response.strip()
        ]
    )


    return "\n".join(
        parts
    )


print("TEST 11: Instruction Formatting")
print()

if unique_examples:

    print(
        format_training_text(
            unique_examples[0]
        )
    )

print()


# ==================================================
# 17. TOKEN COUNTS
# ==================================================

print("TEST 12: Tokenization")
print()

for example in unique_examples:

    formatted = format_training_text(
        example
    )


    token_ids = encode_text(
        formatted
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
# 18. TOKEN LENGTH VALIDATION
# ==================================================

print("TEST 13: Token Length Validation")
print()

length_errors = []


for example in unique_examples:

    if (
            example.token_count
            >
            MAX_INSTRUCTION_TOKENS
    ):

        length_errors.append(
            {
                "example_id":
                    example.example_id,

                "token_count":
                    example.token_count,

                "maximum":
                    MAX_INSTRUCTION_TOKENS
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
        "Instruction examples exceed the "
        "configured token length."
    )


print(
    "All instruction examples are within "
    "the token-length limit."
)

print()


# ==================================================
# 19. TASK DISTRIBUTION
# ==================================================

print("TEST 14: Task Distribution")
print()

task_distribution = {}


for example in unique_examples:

    task = example.task_type

    task_distribution[task] = (
            task_distribution.get(
                task,
                0
            )
            +
            1
    )


for task, count in sorted(
        task_distribution.items()
):

    print(
        task,
        "->",
        count
    )

print()


# ==================================================
# 20. DOMAIN DISTRIBUTION
# ==================================================

print("TEST 15: Domain Distribution")
print()

domain_distribution = {}


for example in unique_examples:

    domain = example.domain

    domain_distribution[domain] = (
            domain_distribution.get(
                domain,
                0
            )
            +
            1
    )


for domain, count in sorted(
        domain_distribution.items()
):

    print(
        domain,
        "->",
        count
    )

print()


# ==================================================
# 21. CURRICULUM DISTRIBUTION
# ==================================================

print("TEST 16: Curriculum Stage Distribution")
print()

stage_distribution = {}


for example in unique_examples:

    stage = example.stage_id

    stage_distribution[stage] = (
            stage_distribution.get(
                stage,
                0
            )
            +
            1
    )


for stage, count in sorted(
        stage_distribution.items()
):

    print(
        stage,
        "->",
        count
    )

print()


# ==================================================
# 22. STRUCTURED OUTPUT VALIDATION
# ==================================================

print("TEST 17: Structured Output Validation")
print()


def is_valid_json_response(
        response: str
) -> bool:

    text = response.strip()


    if not (
            text.startswith("{")
            or
            text.startswith("[")
    ):

        return False


    try:

        json.loads(
            text
        )

        return True

    except json.JSONDecodeError:

        return False


structured_examples = [
    example
    for example in unique_examples
    if (
            example.task_type
            ==
            "structured_output"
            or
            example.task_type
            ==
            "tool_planning"
    )
]


for example in structured_examples:

    valid_json = (
        is_valid_json_response(
            example.response
        )
    )


    print(
        example.example_id,
        "-> valid JSON:",
        valid_json
    )


# ==================================================
# 23. CONSTRAINT VALIDATION
# ==================================================

print("TEST 18: Constraint Validation")
print()

constraint_errors = []


for example in unique_examples:

    instruction_lower = (
        example.instruction.lower()
    )


    if (
            "exactly three sentences"
            in
            instruction_lower
    ):

        sentence_count = len(
            [
                sentence
                for sentence
                in re.split(
                r"[.!?]+",
                example.response
            )
                if sentence.strip()
            ]
        )


        if sentence_count != 3:

            constraint_errors.append(
                {
                    "example":
                        example.example_id,

                    "constraint":
                        "exactly three sentences",

                    "observed":
                        sentence_count
                }
            )


if constraint_errors:

    print(
        json.dumps(
            constraint_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Constraint validation failed."
    )


print(
    "Constraint validation passed."
)

print()


# ==================================================
# 24. TRAIN / VALIDATION SPLIT
# ==================================================

print("TEST 19: Instruction Train/Validation Split")
print()

ordered_examples = sorted(
    unique_examples,
    key=lambda example: (
        example.stage_id,
        example.example_id
    )
)


if len(ordered_examples) <= 1:

    train_examples = ordered_examples

    validation_examples = []

else:

    split_index = int(
        len(ordered_examples)
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
            len(ordered_examples) - 1
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
# 25. SERIALIZATION
# ==================================================

def serialize_example(
        example: InstructionExample
) -> Dict[str, Any]:

    data = asdict(
        example
    )


    formatted_text = (
        format_training_text(
            example
        )
    )


    data[
        "formatted_text"
    ] = formatted_text


    input_token_ids = encode_text(
        formatted_text
    )


    data[
        "input_token_ids"
    ] = input_token_ids


    if len(input_token_ids) > 1:

        data[
            "target_token_ids"
        ] = input_token_ids[1:]

    else:

        data[
            "target_token_ids"
        ] = []


    data[
        "input_token_count"
    ] = len(
        input_token_ids
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
# 26. SAVE COMPLETE DATASET
# ==================================================

print("TEST 20: Save Instruction Dataset")
print()

with open(
        INSTRUCTION_DATASET_FILE,
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
    INSTRUCTION_DATASET_FILE
)

print()


# ==================================================
# 27. SAVE TRAIN DATA
# ==================================================

print("TEST 21: Save Instruction Training Data")
print()

with open(
        INSTRUCTION_TRAIN_FILE,
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
    INSTRUCTION_TRAIN_FILE
)

print()


# ==================================================
# 28. SAVE VALIDATION DATA
# ==================================================

print("TEST 22: Save Instruction Validation Data")
print()

with open(
        INSTRUCTION_VALIDATION_FILE,
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
    INSTRUCTION_VALIDATION_FILE
)

print()


# ==================================================
# 29. DATASET REPORT
# ==================================================

print("TEST 23: Instruction Dataset Report")
print()

total_instruction_tokens = sum(
    example[
        "input_token_count"
    ]
    for example in serialized_examples
)


average_quality = (
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


report = {
    "dataset":
        "Silverwing-Instruction-v1",

    "version":
        "77R",

    "examples":
        len(unique_examples),

    "training_examples":
        len(train_examples),

    "validation_examples":
        len(validation_examples),

    "total_instruction_tokens":
        total_instruction_tokens,

    "average_quality":
        average_quality,

    "task_distribution":
        task_distribution,

    "domain_distribution":
        domain_distribution,

    "stage_distribution":
        stage_distribution,

    "vocabulary_size":
        VOCABULARY_SIZE,

    "max_instruction_tokens":
        MAX_INSTRUCTION_TOKENS,

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


# ==================================================
# 30. SAVE REPORT
# ==================================================

write_json(
    INSTRUCTION_REPORT_FILE,
    report
)


# ==================================================
# 31. SAVE CONFIGURATION
# ==================================================

print("TEST 24: Save Instruction Configuration")
print()

instruction_config = {
    "dataset":
        "Silverwing-Instruction-v1",

    "version":
        "77R",

    "format":
        "supervised_instruction",

    "fields": [
        "example_id",
        "task_type",
        "stage_id",
        "domain",
        "instruction",
        "context",
        "response",
        "difficulty",
        "source",
        "quality_score",
        "token_count",
        "content_hash",
        "validated"
    ],

    "task_types":
        TASK_TYPES,

    "validation_ratio":
        VALIDATION_RATIO,

    "max_instruction_tokens":
        MAX_INSTRUCTION_TOKENS,

    "special_tokens": {
        "pad":
            PAD_ID,

        "unknown":
            UNK_ID,

        "bos":
            BOS_ID,

        "eos":
            EOS_ID
    }
}


write_json(
    INSTRUCTION_CONFIG_FILE,
    instruction_config
)


print(
    "Saved:",
    INSTRUCTION_CONFIG_FILE
)

print()


# ==================================================
# 32. NATIVE INSTRUCTION TEMPLATE
# ==================================================

print("TEST 25: Native Instruction Template")
print()

print(
    "Instruction:"
)

print(
    "What is the purpose of a bearing?"
)

print()

print(
    "Context:"
)

print(
    "The component is used in a rotating machine."
)

print()

print(
    "Response:"
)

print(
    "A bearing supports a rotating shaft and "
    "reduces friction between moving components."
)

print()


# ==================================================
# 33. PRETRAINING VS INSTRUCTION TRAINING
# ==================================================

print(
    "WHY INSTRUCTION TRAINING IS DIFFERENT"
)

print()

print(
    "Pretraining teaches Silverwing statistical "
    "language patterns from large text corpora."
)

print()

print(
    "Instruction training teaches Silverwing how "
    "to transform instructions and context into "
    "task-oriented responses."
)

print()

print(
    "These are different training objectives."
)

print()


# ==================================================
# 34. TASK SPECIALIZATION
# ==================================================

print("TASK SPECIALIZATION")
print()

for task_type in TASK_TYPES:

    print(
        task_type
    )

print()


# ==================================================
# 35. FUTURE INSTRUCTION CURRICULUM
# ==================================================

print("FUTURE INSTRUCTION CURRICULUM")
print()

print("Basic Questions")
print("      ↓")
print("Explanations")
print("      ↓")
print("Transformations")
print("      ↓")
print("Constraints")
print("      ↓")
print("Reasoning")
print("      ↓")
print("Structured Outputs")
print("      ↓")
print("Tool Planning")
print("      ↓")
print("Multi-Step Tasks")
print("      ↓")
print("Memory-Aware Tasks")
print("      ↓")
print("Adaptive Tasks")

print()


# ==================================================
# 36. BIO-INSPIRED CONNECTION
# ==================================================

print("BIO-INSPIRED CONNECTION")
print()

print(
    "Instruction training teaches the model to map "
    "goals and contextual signals to appropriate "
    "responses or actions."
)

print()

print(
    "Later Silverwing systems can combine these "
    "learned behaviors with memory, tools, planning, "
    "evaluation, and continual adaptation."
)

print()


# ==================================================
# 37. CONTROLLED DATA GROWTH
# ==================================================

print("CONTROLLED INSTRUCTION DATA GROWTH")
print()

print("New Task")
print("   ↓")
print("Schema Validation")
print("   ↓")
print("Quality Assessment")
print("   ↓")
print("Deduplication")
print("   ↓")
print("Difficulty Classification")
print("   ↓")
print("Curriculum Assignment")
print("   ↓")
print("Training Candidate")
print("   ↓")
print("Independent Evaluation")

print()


# ==================================================
# 38. ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Silverwing should not treat every generated "
    "instruction example as automatically correct."
)

print()

print(
    "Instruction datasets require validation of "
    "content, constraints, provenance, formatting, "
    "and task quality."
)

print()


# ==================================================
# 39. CURRENT LIMITATION
# ==================================================

print("CURRENT LIMITATION")
print()

print(
    "The current instruction dataset is a small "
    "engineering seed dataset."
)

print()

print(
    "It is not sufficient to make Silverwing a "
    "capable instruction-following model."
)

print()

print(
    "A serious training corpus will require many "
    "more examples across domains, difficulty levels, "
    "reasoning types, constraints, and failure cases."
)

print()


# ==================================================
# 40. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "The native instruction dataset format now exists."
)

print()

print(
    "The next lesson will connect this dataset to "
    "a supervised fine-tuning engine for Silverwing's "
    "own decoder."
)

print()

print(
    "Lesson 78R: Silverwing Native Instruction "
    "Fine-Tuning Engine."
)

print()


# ==================================================
# 41. FOUNDATION PROGRESS
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
print("OWN INSTRUCTION DATASET")
print(" ↓")
print("Instruction Fine-Tuning")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Multitask Training")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 77R COMPLETE ===")
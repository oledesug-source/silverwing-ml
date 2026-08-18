# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 81R
# Native Memory-Aware Training Engine
#
# 79R -> Native reasoning dataset + evaluation
# 80R -> Native reasoning fine-tuning
# 81R -> Native memory-aware training
#
# No GPT-2
# No Qwen
# No external LLM
#
# Exact established Silverwing decoder hierarchy:
#
# token_embedding
# position_embedding.embedding
# layers.N.attention.query_projection
# layers.N.attention.key_projection
# layers.N.attention.value_projection
# layers.N.attention.output_projection
# layers.N.feed_forward.input_projection
# layers.N.feed_forward.output_projection
# layers.N.norm_attention
# layers.N.norm_feed_forward
# final_norm
# language_model_head
#
# IMPORTANT MEMORY DESIGN:
#   MAX_MEMORY_ITEMS = 1
#
# This preserves the established 256-token model limit while
# still teaching the model to use retrieved persistent memory.
# ============================================================

import hashlib
import json
import math
import random
import re
import time

from dataclasses import asdict, dataclass
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
LESSON_80R = PHASE5_DIR / "lesson80R"

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
        LESSON_80R
        / "checkpoints"
        / "silverwing_reasoning_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_80R
        / "checkpoints"
        / "silverwing_reasoning_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MEMORY_BANK_FILE = (
        BASE_DIR
        / "silverwing_memory_bank.jsonl"
)

MEMORY_CONFIG_FILE = (
        BASE_DIR
        / "silverwing_memory_config.json"
)

MEMORY_TRAIN_FILE = (
        BASE_DIR
        / "silverwing_memory_train.jsonl"
)

MEMORY_VALIDATION_FILE = (
        BASE_DIR
        / "silverwing_memory_validation.jsonl"
)

MEMORY_REPORT_FILE = (
        BASE_DIR
        / "silverwing_memory_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_memory_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_memory_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_memory_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR
        / "silverwing_memory_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1.5e-5

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

# CRITICAL:
# One memory item only.
# This prevents retrieved context from overflowing
# Silverwing's established sequence length.
MAX_MEMORY_ITEMS = 1

MIN_MEMORY_SCORE = 0.18

MAX_MEMORY_TEXT_LENGTH = 150

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


def sha256_text(
        text: str
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():

        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():

        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 80R reasoning checkpoint found.\n"
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
    "PHASE 5 - LESSON 81R"
)

print(
    "Native Memory-Aware Training Engine"
)

print()

print(
    "79R -> Native Reasoning Dataset"
)

print(
    "80R -> Native Reasoning Fine-Tuning"
)

print(
    "81R -> Native Memory-Aware Training"
)

print()

print(
    "External LLM: NONE"
)

print(
    "Retrieved memory per example:",
    MAX_MEMORY_ITEMS
)

print()


# ============================================================
# 5. TEST 1 - VERIFY INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 80R and Silverwing Inputs"
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
            "Model dimension must be "
            "divisible by attention heads."
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


required_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
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
        item[
            "rank"
        ]
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
                index
                <
                len(symbols) - 1

                and

                (
                        symbols[index],
                        symbols[index + 1]
                )
                ==
                pair

        ):

            output.append(
                (
                        symbols[index]
                        +
                        symbols[index + 1]
                )
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
# 10. MEMORY SCHEMA
# ============================================================

@dataclass
class MemoryRecord:

    memory_id: str

    memory_type: str

    domain: str

    content: str

    importance: float

    tags: List[str]

    source: str

    created_at: float

    access_count: int = 0

    validated: bool = True


# ============================================================
# 11. TEST 5 - MEMORY BANK
# ============================================================

print(
    "TEST 5: Create Native Memory Bank"
)

print()


created_at = time.time()


memory_records = [

    MemoryRecord(
        memory_id="mem_001",
        memory_type="system_principle",
        domain="engineering",
        content=(
            "Silverwing should preserve established architecture "
            "when extending its capabilities."
        ),
        importance=0.98,
        tags=[
            "architecture",
            "compatibility",
            "engineering"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_002",
        memory_type="training_principle",
        domain="machine_learning",
        content=(
            "A candidate model must be evaluated against its "
            "baseline before promotion."
        ),
        importance=0.98,
        tags=[
            "baseline",
            "evaluation",
            "promotion"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_003",
        memory_type="reasoning_principle",
        domain="reasoning",
        content=(
            "A conclusion should be supported by relevant "
            "evidence rather than fluent wording alone."
        ),
        importance=0.96,
        tags=[
            "reasoning",
            "evidence",
            "validation"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_004",
        memory_type="workflow",
        domain="software_engineering",
        content=(
            "Validate syntax before running a changed training "
            "or deployment program."
        ),
        importance=0.91,
        tags=[
            "syntax",
            "validation",
            "software"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_005",
        memory_type="diagnostic",
        domain="engineering",
        content=(
            "Machine diagnosis should separate observed "
            "symptoms from possible causes."
        ),
        importance=0.90,
        tags=[
            "machine",
            "diagnosis",
            "symptoms",
            "causes"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_006",
        memory_type="continual_learning",
        domain="ai_systems",
        content=(
            "A new capability should be added through a "
            "validated candidate and a controlled promotion gate."
        ),
        importance=0.97,
        tags=[
            "capability",
            "candidate",
            "promotion"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_007",
        memory_type="data_quality",
        domain="data",
        content=(
            "Training data should be checked for duplicates, "
            "schema errors, token limits and consistency."
        ),
        importance=0.93,
        tags=[
            "dataset",
            "duplicates",
            "schema",
            "tokens"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    ),


    MemoryRecord(
        memory_id="mem_008",
        memory_type="architecture",
        domain="memory",
        content=(
            "Persistent memory should provide retrieved "
            "context without replacing the model parameters."
        ),
        importance=0.97,
        tags=[
            "memory",
            "context",
            "parameters"
        ],
        source="silverwing_native_seed",
        created_at=created_at
    )

]


# ============================================================
# 12. TEST 6 - MEMORY VALIDATION
# ============================================================

print(
    "TEST 6: Validate Memory Bank"
)

print()


memory_errors = []

seen_ids = set()


for record in memory_records:

    if record.memory_id in seen_ids:

        memory_errors.append(
            f"Duplicate memory ID: {record.memory_id}"
        )


    seen_ids.add(
        record.memory_id
    )


    if not record.content.strip():

        memory_errors.append(
            f"Empty memory content: {record.memory_id}"
        )


    if not (
            0.0
            <=
            record.importance
            <=
            1.0
    ):

        memory_errors.append(
            f"Invalid importance: {record.memory_id}"
        )


    if (
            len(record.content)
            >
            MAX_MEMORY_TEXT_LENGTH
    ):

        memory_errors.append(
            f"Memory too long: {record.memory_id}"
        )


if memory_errors:

    print(
        json.dumps(
            memory_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Memory bank validation failed."
    )


print(
    "Valid memories:",
    len(memory_records)
)

print()


# ============================================================
# 13. TEST 7 - MEMORY HASHES
# ============================================================

print(
    "TEST 7: Memory Identity Hashes"
)

print()


for record in memory_records:

    canonical = "|".join(
        [
            record.memory_id,
            record.memory_type,
            record.domain,
            record.content
        ]
    )


    digest = sha256_text(
        canonical
    )


    print(
        record.memory_id,
        "->",
        digest[:16]
    )


print()


# ============================================================
# 14. MEMORY RETRIEVAL
# ============================================================

def query_terms(
        text: str
) -> set:

    return {

        token

        for token
        in split_words(
            text
        )

        if len(token) >= 3

    }


def memory_score(
        query: str,
        memory: MemoryRecord
) -> float:

    query_set = query_terms(
        query
    )


    content_set = query_terms(
        memory.content
    )


    if query_set:

        overlap_score = (

                len(
                    query_set
                    &
                    content_set
                )

                /

                len(
                    query_set
                )

        )

    else:

        overlap_score = 0.0


    query_lower = query.lower()


    tag_hits = sum(

        1

        for tag
        in memory.tags

        if tag.lower()
        in
        query_lower

    )


    tag_score = min(
        0.30,
        tag_hits * 0.10
    )


    domain_score = (

        0.20

        if memory.domain.lower()
           in
           query_lower

        else

        0.0

    )


    score = (

            0.50
            *
            overlap_score

            +

            tag_score

            +

            domain_score

            +

            0.30
            *
            memory.importance

    )


    return min(
        1.0,
        score
    )


def retrieve_memories(
        query: str,
        limit: int = MAX_MEMORY_ITEMS
) -> List[
    Tuple[
        MemoryRecord,
        float
    ]
]:

    scored = []


    for memory in memory_records:

        score = memory_score(
            query,
            memory
        )


        if (
                score
                >=
                MIN_MEMORY_SCORE
        ):

            scored.append(
                (
                    memory,
                    score
                )
            )


    scored.sort(

        key=lambda item: (

            item[1],

            item[0].importance

        ),

        reverse=True

    )


    selected = scored[
        :limit
    ]


    for memory, _ in selected:

        memory.access_count += 1


    return selected


# ============================================================
# 15. TEST 8 - MEMORY RETRIEVAL
# ============================================================

print(
    "TEST 8: Native Memory Retrieval"
)

print()


retrieval_queries = [

    "How should a candidate model be promoted?",

    "How should a training dataset be checked?",

    "How should a machine diagnosis reason about symptoms?",

    "What should persistent memory do?"

]


for query in retrieval_queries:

    print(
        "QUERY:",
        query
    )


    results = retrieve_memories(
        query,
        limit=MAX_MEMORY_ITEMS
    )


    for memory, score in results:

        print(
            " ",
            memory.memory_id,
            "score=",
            round(
                score,
                4
            )
        )


    print()


# ============================================================
# 16. MEMORY CONTEXT
# ============================================================

def build_memory_context(
        query: str
) -> str:

    retrieved = retrieve_memories(
        query,
        limit=1
    )


    if not retrieved:

        return (
            "Relevant Memory:\n"
            "- None."
        )


    memory, _ = retrieved[0]


    return (
        "Relevant Memory:\n"
        f"- [{memory.memory_id}] "
        f"{memory.content}"
    )


# ============================================================
# 17. TEST 9 - MEMORY TASKS
# ============================================================

print(
    "TEST 9: Build Memory-Aware Training Tasks"
)

print()


memory_tasks = [

    {
        "example_id":
            "memory_001",

        "query":
            (
                "What should happen before a new model "
                "is promoted?"
            ),

        "context":
            (
                "A candidate model has completed training."
            ),

        "response":
            (
                "Evaluate the candidate against its "
                "baseline and pass the promotion gate."
            )
    },


    {
        "example_id":
            "memory_002",

        "query":
            (
                "A new training script was changed. "
                "What should happen before execution?"
            ),

        "context":
            (
                "The code has not yet been executed."
            ),

        "response":
            (
                "Validate the program syntax before running it."
            )
    },


    {
        "example_id":
            "memory_003",

        "query":
            (
                "A model gives fluent but unsupported "
                "conclusions. How should it be evaluated?"
            ),

        "context":
            (
                "The response sounds convincing."
            ),

        "response":
            (
                "Check whether the conclusions are supported "
                "by relevant evidence."
            )
    },


    {
        "example_id":
            "memory_004",

        "query":
            (
                "A machine is vibrating and overheating. "
                "How should diagnosis begin?"
            ),

        "context":
            (
                "The root cause has not been established."
            ),

        "response":
            (
                "Separate observed symptoms from possible causes "
                "before selecting a diagnosis."
            )
    },


    {
        "example_id":
            "memory_005",

        "query":
            (
                "How should a newly created training dataset "
                "be validated?"
            ),

        "context":
            (
                "The dataset is ready for ingestion."
            ),

        "response":
            (
                "Check schema, duplicates, token limits and "
                "consistency."
            )
    },


    {
        "example_id":
            "memory_006",

        "query":
            (
                "What is the role of persistent memory "
                "in Silverwing?"
            ),

        "context":
            (
                "Silverwing now has a memory subsystem."
            ),

        "response":
            (
                "Persistent memory should provide relevant "
                "context without replacing model parameters."
            )
    },


    {
        "example_id":
            "memory_007",

        "query":
            (
                "Should a new capability be installed directly "
                "into the production model?"
            ),

        "context":
            (
                "The capability has not completed evaluation."
            ),

        "response":
            (
                "No. Validate the candidate first and promote "
                "it through the controlled promotion gate."
            )
    },


    {
        "example_id":
            "memory_008",

        "query":
            (
                "What architectural rule should Silverwing "
                "follow when extending itself?"
            ),

        "context":
            (
                "A new training capability is being added."
            ),

        "response":
            (
                "Preserve the established architecture and "
                "extend it through validated compatible stages."
            )
    }

]


print(
    "Training tasks:",
    len(memory_tasks)
)

print()


# ============================================================
# 18. MEMORY TASK FORMAT
# ============================================================

def format_memory_task(
        task: Dict[str, Any]
) -> Tuple[
    str,
    str,
    List[str]
]:

    query = task[
        "query"
    ].strip()


    context = task[
        "context"
    ].strip()


    response = task[
        "response"
    ].strip()


    retrieval_query = (

            query
            +
            " "
            +
            context

    )


    retrieved = retrieve_memories(

        retrieval_query,

        limit=1

    )


    if not retrieved:

        raise RuntimeError(

            (
                "No relevant memory retrieved for "
                f"{task['example_id']}."
            )

        )


    memory, score = retrieved[0]


    memory_context = (

        "Relevant Memory:\n"

        f"- [{memory.memory_id}] "

        f"{memory.content}"

    )


    prompt = "\n".join(

        [

            "Task:",

            query,

            "",

            "Context:",

            context,

            "",

            memory_context,

            "",

            "Response:"

        ]

    )


    full_text = (

            prompt

            +

            "\n"

            +

            response

    )


    return (

        prompt,

        full_text,

        [

            memory.memory_id

        ]

    )


# ============================================================
# 19. TEST 10 - MEMORY TASK VALIDATION
# ============================================================

print(
    "TEST 10: Validate Memory Retrieval Contract"
)

print()


memory_task_records = []


for task in memory_tasks:

    prompt, full_text, memory_ids = (

        format_memory_task(
            task
        )

    )


    token_count = len(

        encode_text(
            full_text
        )

    )


    memory_task_records.append(

        {

            "example_id":
                task[
                    "example_id"
                ],

            "query":
                task[
                    "query"
                ],

            "context":
                task[
                    "context"
                ],

            "response":
                task[
                    "response"
                ],

            "prompt":
                prompt,

            "formatted_text":
                full_text,

            "memory_ids":
                memory_ids,

            "token_count":
                token_count

        }

    )


    print(

        task[
            "example_id"
        ],

        "->",

        token_count,

        "tokens",

        "| memory:",

        memory_ids

    )


print()


# ============================================================
# 20. TEST 11 - TOKEN LIMIT
# ============================================================

print(
    "TEST 11: Memory-Aware Token Validation"
)

print()


memory_length_errors = [

    {

        "example_id":
            record[
                "example_id"
            ],

        "token_count":
            record[
                "token_count"
            ],

        "maximum":
            MAX_SEQUENCE_LENGTH

    }

    for record
    in memory_task_records

    if record[
           "token_count"
       ]
       >
       MAX_SEQUENCE_LENGTH

]


if memory_length_errors:

    print(

        json.dumps(

            memory_length_errors,

            indent=4

        )

    )


    raise RuntimeError(

        (
            "Memory-aware examples exceed "
            "the Silverwing sequence limit."
        )

    )


print(
    "All memory-aware examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 21. TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    memory_task_records
)


if (
        len(memory_task_records)
        <=
        2
):

    memory_train_records = list(
        memory_task_records
    )

    memory_validation_records = list(
        memory_task_records
    )

else:

    validation_count = max(

        2,

        int(

            round(

                len(memory_task_records)
                *
                0.25

            )

        )

    )


    validation_count = min(

        validation_count,

        len(memory_task_records)
        -
        1

    )


    memory_train_records = (

        memory_task_records[
            :-validation_count
        ]

    )


    memory_validation_records = (

        memory_task_records[
            -validation_count:
        ]

    )


print(
    "TEST 12: Memory Train/Validation Split"
)

print(
    "Training examples:",
    len(memory_train_records)
)

print(
    "Validation examples:",
    len(memory_validation_records)
)

print()


# ============================================================
# 22. TEST 13 - PERSIST MEMORY ARTIFACTS
# ============================================================

print(
    "TEST 13: Persist Memory Artifacts"
)

print()


with open(

        MEMORY_BANK_FILE,

        "w",

        encoding="utf-8"

) as file:

    for memory in memory_records:

        file.write(

            json.dumps(

                asdict(
                    memory
                ),

                ensure_ascii=False

            )

            +

            "\n"

        )


with open(

        MEMORY_TRAIN_FILE,

        "w",

        encoding="utf-8"

) as file:

    for record in memory_train_records:

        file.write(

            json.dumps(

                record,

                ensure_ascii=False

            )

            +

            "\n"

        )


with open(

        MEMORY_VALIDATION_FILE,

        "w",

        encoding="utf-8"

) as file:

    for record in memory_validation_records:

        file.write(

            json.dumps(

                record,

                ensure_ascii=False

            )

            +

            "\n"

        )


memory_config = {

    "lesson":
        "81R",

    "system":
        "native_memory_aware_training",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "maximum_memory_items":
        MAX_MEMORY_ITEMS,

    "minimum_memory_score":
        MIN_MEMORY_SCORE,

    "memory_count":
        len(
            memory_records
        ),

    "training_examples":
        len(
            memory_train_records
        ),

    "validation_examples":
        len(
            memory_validation_records
        )

}


write_json(

    MEMORY_CONFIG_FILE,

    memory_config

)


print(
    "Memory bank:",
    MEMORY_BANK_FILE
)

print(
    "Memory training:",
    MEMORY_TRAIN_FILE
)

print(
    "Memory validation:",
    MEMORY_VALIDATION_FILE
)

print()


# ============================================================
# 23. MEMORY DATASET
# ============================================================

class MemoryDataset(
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
                        token_ids[
                            :-1
                        ],

                    "labels":
                        token_ids[
                            1:
                        ]

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


def collate_memory_batch(

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


memory_train_dataset = MemoryDataset(

    memory_train_records

)


memory_validation_dataset = MemoryDataset(

    memory_validation_records

)


memory_train_loader = DataLoader(

    memory_train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_memory_batch

)


memory_validation_loader = DataLoader(

    memory_validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_memory_batch

)


print(
    "TEST 14: Memory DataLoaders"
)

print(
    "Training samples:",
    len(memory_train_dataset)
)

print(
    "Validation samples:",
    len(memory_validation_dataset)
)

print(
    "Training batches:",
    len(memory_train_loader)
)

print(
    "Validation batches:",
    len(memory_validation_loader)
)

print()


# ============================================================
# 24. EXACT SILVERWING ATTENTION
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

        batch_size = x.shape[
            0
        ]

        sequence_length = x.shape[
            1
        ]


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
# 25. EXACT SILVERWING FEED FORWARD
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

        x = self.input_projection(
            x
        )


        x = F.gelu(
            x
        )


        return self.output_projection(
            x
        )


# ============================================================
# 26. EXACT SILVERWING BLOCK
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
# 27. EXACT POSITION EMBEDDING
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
# 28. EXACT SILVERWING DECODER
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

        sequence_length = input_ids.shape[
            1
        ]


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

                ).unsqueeze(
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


# ============================================================
# 29. TEST 15 - STRICT LOAD
# ============================================================

print(
    "TEST 15: Strict Load of 80R Reasoning Model"
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
        "80R checkpoint is not a dictionary."
    )


if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "80R checkpoint is missing model_state_dict."
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

            for key
            in state_dict.keys()

    ):

        raise RuntimeError(

            (

                "80R checkpoint architecture mismatch. "

                f"Missing prefix: {prefix}"

            )

        )


model = (

    SilverwingDecoder()

    .to(
        DEVICE
    )

)


try:

    model.load_state_dict(

        state_dict,

        strict=True

    )


except RuntimeError as exc:

    raise RuntimeError(

        (

            "81R refused to load a mismatched "
            "80R Silverwing model.\n\n"

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
    "80R model is compatible with 81R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 30. BASELINE SNAPSHOT
# ============================================================

baseline_state = {

    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()

}


# ============================================================
# 31. LOSS
# ============================================================

def memory_aware_loss(

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
# 32. EVALUATION
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


        loss = memory_aware_loss(

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

                    predictions[
                        mask
                    ]

                    ==

                    labels[
                        mask
                    ]

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

            loss_value
            <
            50

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
# 33. TEST 16 - BASELINE
# ============================================================

print(
    "TEST 16: Baseline Memory-Aware Evaluation"
)

print()


baseline_metrics = evaluate(

    model,

    memory_validation_loader

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
# 34. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


total_steps = max(

    1,

    len(
        memory_train_loader
    )
    *
    EPOCHS

)


scheduler = (

    torch.optim.lr_scheduler

    .CosineAnnealingLR(

        optimizer,

        T_max=total_steps

    )

)


# ============================================================
# 35. TEST 17 - MEMORY-AWARE TRAINING
# ============================================================

print(
    "TEST 17: Native Memory-Aware Fine-Tuning"
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

            memory_train_loader,

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


        loss = memory_aware_loss(

            logits,

            labels

        )


        loss.backward()


        gradient_norm = (

            torch.nn.utils

            .clip_grad_norm_(

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
            f"| Batch {batch_number}/{len(memory_train_loader)} "
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

        memory_validation_loader

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
                    "81R",

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

                "memory_records":
                    len(
                        memory_records
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
# 36. TEST 18 - FINAL
# ============================================================

print(
    "TEST 18: Final Memory-Aware Evaluation"
)

print()


final_metrics = evaluate(

    model,

    memory_validation_loader

)


print(
    "Final loss:",
    final_metrics[
        "loss"
    ]
)

print(
    "Final perplexity:",
    final_metrics[
        "perplexity"
    ]
)

print(
    "Final accuracy:",
    final_metrics[
        "accuracy"
    ]
)

print()


# ============================================================
# 37. TEST 19 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 19: Numerical Health"
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
# 38. TEST 20 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 20: Parameter Change"
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
# 39. TEST 21 - PROMOTION
# ============================================================

print(
    "TEST 21: Memory Model Promotion Gate"
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
        "Candidate memory loss is invalid."
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
        "Memory-aware validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Memory-aware validation loss did not improve."
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
# 40. TEST 22 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 22: Save Memory Candidate"
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
        "81R",

    "training_mode":
        "native_memory_aware_fine_tuning",

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

    "memory_configuration":
        {

            "maximum_memory_items":
                MAX_MEMORY_ITEMS,

            "minimum_memory_score":
                MIN_MEMORY_SCORE,

            "memory_count":
                len(
                    memory_records
                )

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
# 41. MEMORY REPORT
# ============================================================

memory_report = {

    "lesson":
        "81R",

    "memory_system":
        "native_retrieval_context",

    "memory_count":
        len(
            memory_records
        ),

    "training_examples":
        len(
            memory_train_records
        ),

    "validation_examples":
        len(
            memory_validation_records
        ),

    "retrieval_limit":
        MAX_MEMORY_ITEMS,

    "minimum_score":
        MIN_MEMORY_SCORE,

    "memory_types":
        sorted(

            {

                record.memory_type

                for record
                in memory_records

            }

        ),

    "domains":
        sorted(

            {

                record.domain

                for record
                in memory_records

            }

        )

}


write_json(

    MEMORY_REPORT_FILE,

    memory_report

)


# ============================================================
# 42. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "81R",

    "training_mode":
        "memory_aware_fine_tuning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "external_llm":
        False,

    "external_memory_model":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_count":
        len(
            memory_records
        ),

    "training_examples":
        len(
            memory_train_records
        ),

    "validation_examples":
        len(
            memory_validation_records
        ),

    "epochs":
        EPOCHS,

    "global_steps":
        global_step,

    "duration_seconds":
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
# 43. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "81R",

    "model":
        "Silverwing native decoder",

    "capability":
        "memory_aware_learning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "baseline":
        baseline_metrics,

    "candidate":
        final_metrics,

    "memory":
        {

            "records":
                len(
                    memory_records
                ),

            "training_examples":
                len(
                    memory_train_records
                ),

            "validation_examples":
                len(
                    memory_validation_records
                ),

            "retrieval_limit":
                MAX_MEMORY_ITEMS,

            "minimum_score":
                MIN_MEMORY_SCORE

        },

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
# 44. MEMORY ARCHITECTURE
# ============================================================

print(
    "SILVERWING MEMORY ARCHITECTURE"
)

print()

print(
    "Persistent Memory"
)

print(
    "        ↓"
)

print(
    "Memory Validation"
)

print(
    "        ↓"
)

print(
    "Top-1 Relevant Memory"
)

print(
    "        ↓"
)

print(
    "Retrieved Context"
)

print(
    "        ↓"
)

print(
    "Silverwing Native Decoder"
)

print(
    "        ↓"
)

print(
    "Memory-Aware Learning"
)

print(
    "        ↓"
)

print(
    "Evaluation"
)

print(
    "        ↓"
)

print(
    "Promotion Gate"
)

print()


# ============================================================
# 45. MODEL VS MEMORY
# ============================================================

print(
    "MODEL PARAMETERS VS MEMORY"
)

print()

print(
    "Model parameters store learned representations."
)

print()

print(
    "Persistent memory stores retrievable contextual state."
)

print()

print(
    "81R connects retrieved memory with the Silverwing "
    "decoder without replacing its architecture."
)

print()


# ============================================================
# 46. CONTROLLED MEMORY GROWTH
# ============================================================

print(
    "CONTROLLED MEMORY GROWTH"
)

print()

print(
    "Experience"
)

print(
    "    ↓"
)

print(
    "Memory Candidate"
)

print(
    "    ↓"
)

print(
    "Validation"
)

print(
    "    ↓"
)

print(
    "Persistent Storage"
)

print(
    "    ↓"
)

print(
    "Top-1 Retrieval"
)

print(
    "    ↓"
)

print(
    "Context"
)

print(
    "    ↓"
)

print(
    "Reasoning"
)

print(
    "    ↓"
)

print(
    "Feedback"
)

print(
    "    ↓"
)

print(
    "Memory Revision"
)

print()


# ============================================================
# 47. CURRENT LIMITATION
# ============================================================

print(
    "CURRENT LIMITATION"
)

print()

print(
    "81R currently uses a small native memory bank "
    "and lexical retrieval."
)

print()

print(
    "Only the highest-scoring memory is injected into "
    "each training example to preserve the 256-token limit."
)

print()

print(
    "This establishes the memory-learning foundation. "
    "It is not yet the final autonomous memory architecture."
)

print()


# ============================================================
# 48. NEXT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 82R: Native Tool-Aware Learning"
)

print()

print(
    "Memory + Reasoning + Tool Selection + "
    "Tool Results + Verification"
)

print()


# ============================================================
# 49. FOUNDATION MODEL PROGRESS
# ============================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()

print(
    "Own Tokenizer"
)

print(
    " ↓"
)

print(
    "Own Vocabulary"
)

print(
    " ↓"
)

print(
    "Own Decoder"
)

print(
    " ↓"
)

print(
    "Own Training"
)

print(
    " ↓"
)

print(
    "Own Evaluation"
)

print(
    " ↓"
)

print(
    "Instruction Learning"
)

print(
    " ↓"
)

print(
    "79R Native Reasoning Dataset"
)

print(
    " ↓"
)

print(
    "80R Native Reasoning Fine-Tuning"
)

print(
    " ↓"
)

print(
    "81R Native Memory-Aware Training"
)

print(
    " ↓"
)

print(
    "82R Native Tool-Aware Learning"
)

print(
    " ↓"
)

print(
    "Planning"
)

print(
    " ↓"
)

print(
    "Continual Learning"
)

print(
    " ↓"
)

print(
    "Controlled Autonomous Improvement"
)

print()


# ============================================================
# 50. COMPLETE
# ============================================================

print(
    "=== LESSON 81R COMPLETE ==="
)
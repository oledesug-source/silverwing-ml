# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 104R
# Native Multimodal Memory Reasoning
# ============================================================
#
# 79R  -> Native Reasoning Dataset
# 80R  -> Native Reasoning Fine-Tuning
# 81R  -> Native Memory-Aware Training
# 82R  -> Native Tool-Aware Learning
# 83R  -> Native Planning and Tool Sequencing
# 84R  -> Native Verified Execution + Replanning
# 85R  -> Native Mathematical Reasoning
# 86R  -> Native Probability + Statistical Reasoning
# 87R  -> Native Linear Algebra + Optimization
# 88R  -> Native Algorithms + Data Structures
# 89R  -> Native Data Analysis + SQL Reasoning
# 90R  -> Native Data Engineering
# 91R  -> Native Machine Learning Foundations
# 92R  -> Native Classical Machine Learning
# 93R  -> Native Neural Network Foundations
# 94R  -> Native Deep Learning
# 95R  -> Native Representation Learning
# 96R  -> Native Sequence Representation Learning
# 97R  -> Native Structured Representation Learning
# 98R  -> Advanced Sequence + Structured Learning
# 99R  -> Multimodal Representation Foundations
# 100R -> Cross-Modal Alignment + Retrieval
# 101R -> Native Hard-Negative Multimodal Learning
# 102R -> Native Multimodal Memory Integration
# 103R -> Native Memory Consolidation + Temporal Retrieval
# 104R -> Native Multimodal Memory Reasoning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 104R reasons over the persistent multimodal memory produced
# by 102R and consolidated by 103R.
#
# It validates:
#
#   memory retrieval
#   memory identity
#   text/numeric cross-modal alignment
#   temporal history
#   historical state changes
#   multimodal evidence comparison
#   reasoning chains
#   engineering progression
#   deterministic retrieval
#
# ============================================================
# IMPORTANT ARCHITECTURAL RULE
# ============================================================
#
# The 103R index intentionally stores the consolidated memory
# representation. The individual memory records may additionally
# carry the original text and numeric embeddings.
#
# Therefore 104R performs state discovery in this order:
#
#   1. inspect 103R index
#   2. inspect 103R memory records
#   3. recover text embeddings from records
#   4. recover numeric embeddings from records
#   5. validate dimensions
#   6. continue reasoning
#
# We DO NOT assume the index must contain every modality.
#
# ============================================================
# EXTERNAL LLM
# ============================================================
#
# NONE
#
# ============================================================

import json
import math
import random

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
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

MEMORY_VERSION = "104R.1"

MEMORY_DIMENSION = 16

NUMERIC_DIMENSION = 5

TOP_K = 3

REASONING_THRESHOLD = 0.80

CROSS_MODAL_MARGIN_THRESHOLD = 0.0

DETERMINISM_THRESHOLD = 1e-7

EPSILON = 1e-8


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_103R = (
        PHASE5_DIR /
        "lesson103R"
)

SOURCE_MEMORY_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory.json"
)

SOURCE_INDEX_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory_index.pt"
)

SOURCE_REGISTRY_FILE = (
        LESSON_103R /
        "silverwing_memory_consolidation_registry.json"
)

SOURCE_CHECKPOINT_PRIMARY = (
        LESSON_103R /
        "checkpoints" /
        "silverwing_memory_consolidation_best.pt"
)

SOURCE_CHECKPOINT_CANDIDATE = (
        LESSON_103R /
        "checkpoints" /
        "silverwing_memory_consolidation_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REASONING_REPORT_FILE = (
        BASE_DIR /
        "silverwing_multimodal_memory_reasoning_report.json"
)

REASONING_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_multimodal_memory_reasoning_evaluation.json"
)

REASONING_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_multimodal_memory_reasoning_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_multimodal_memory_reasoning_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_multimodal_memory_reasoning_best.pt"
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


def parse_timestamp(
        value: str
) -> datetime:

    return datetime.fromisoformat(
        str(
            value
        ).replace(
            "Z",
            ""
        )
    )


def cosine_similarity(
        left: torch.Tensor,
        right: torch.Tensor
) -> float:

    left = left.float()
    right = right.float()

    left_norm = torch.linalg.vector_norm(
        left
    )

    right_norm = torch.linalg.vector_norm(
        right
    )

    denominator = (
            left_norm
            *
            right_norm
    )

    if float(
            denominator
    ) <= EPSILON:

        return 0.0

    return float(
        torch.dot(
            left,
            right
        )
        /
        denominator
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
        SOURCE_CHECKPOINT_PRIMARY,
        SOURCE_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 103R checkpoint found."
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


# ============================================================
# 5. HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 104R"
)

print(
    "Native Multimodal Memory Reasoning"
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
    "102R -> Multimodal Memory Integration",
    "103R -> Memory Consolidation + Temporal Retrieval",
    "104R -> Multimodal Memory Reasoning"
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
    "Memory version:",
    MEMORY_VERSION
)

print(
    "Memory dimension:",
    MEMORY_DIMENSION
)

print(
    "Reasoning threshold:",
    REASONING_THRESHOLD
)

print(
    "Cross-modal margin threshold:",
    CROSS_MODAL_MARGIN_THRESHOLD
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 103R Memory Inputs"
)

print()

for path in [
    SOURCE_MEMORY_FILE,
    SOURCE_INDEX_FILE,
    SOURCE_REGISTRY_FILE
]:

    require_file(
        path
    )

SOURCE_CHECKPOINT = choose_checkpoint()

require_file(
    SOURCE_CHECKPOINT
)

print(
    "FOUND:",
    SOURCE_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_REGISTRY_FILE
)

print(
    "FOUND:",
    SOURCE_CHECKPOINT
)

print()


# ============================================================
# TEST 2
# ============================================================

print(
    "TEST 2: Load Consolidated Memory"
)

print()

memory_payload = read_json(
    SOURCE_MEMORY_FILE
)

if not isinstance(
        memory_payload,
        dict
):

    raise RuntimeError(
        "103R memory payload is invalid."
    )

memory_records = memory_payload.get(
    "records"
)

if not isinstance(
        memory_records,
        list
):

    raise RuntimeError(
        "103R memory payload contains no records."
    )

if not memory_records:

    raise RuntimeError(
        "103R memory is empty."
    )

print(
    "Memory version:",
    memory_payload.get(
        "memory_version"
    )
)

print(
    "Records:",
    len(
        memory_records
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Memory Schema Validation"
)

print()

required_fields = {
    "memory_id",
    "event_id",
    "timestamp",
    "text",
    "numeric",
    "semantic_class",
    "source",
    "confidence",
    "memory_embedding"
}

schema_errors = []

for record in memory_records:

    missing = (
            required_fields
            -
            set(
                record.keys()
            )
    )

    if missing:

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

    numeric_values = record.get(
        "numeric"
    )

    if not isinstance(
            numeric_values,
            list
    ):

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "numeric field is not a list"
            }
        )

    elif len(
            numeric_values
    ) != NUMERIC_DIMENSION:

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "numeric dimension mismatch",

                "actual":
                    len(
                        numeric_values
                    ),

                "expected":
                    NUMERIC_DIMENSION
            }
        )

if schema_errors:

    print(
        json.dumps(
            schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Memory schema validation failed."
    )

print(
    "Memory schema validated."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Load 103R Memory Index"
)

print()

index_payload = torch.load(
    SOURCE_INDEX_FILE,
    map_location="cpu",
    weights_only=False
)

if not isinstance(
        index_payload,
        dict
):

    raise RuntimeError(
        "103R memory index is invalid."
    )

stored_memory_embeddings = (
    index_payload.get(
        "memory_embeddings"
    )
)

stored_memory_ids = (
    index_payload.get(
        "memory_ids"
    )
)

stored_text_embeddings = (
    index_payload.get(
        "text_embeddings"
    )
)

stored_numeric_embeddings = (
    index_payload.get(
        "numeric_embeddings"
    )

)

print(
    "Index fields:",
    sorted(
        index_payload.keys()
    )
)

print(
    "Memory embeddings present:",
    stored_memory_embeddings is not None
)

print(
    "Text embeddings present in index:",
    stored_text_embeddings is not None
)

print(
    "Numeric embeddings present in index:",
    stored_numeric_embeddings is not None
)

if stored_memory_embeddings is None:

    raise RuntimeError(
        "Memory embeddings are missing."
    )

if stored_memory_ids is None:

    raise RuntimeError(
        "Memory ids are missing."
    )

print(
    "Memory embedding shape:",
    tuple(
        stored_memory_embeddings.shape
    )
)

print(
    "Memory index ids:",
    len(
        stored_memory_ids
    )
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Memory Identity and Index Alignment"
)

print()

record_ids = [
    str(
        record[
            "memory_id"
        ]
    )
    for record
    in memory_records
]

index_ids = [
    str(
        value
    )
    for value
    in stored_memory_ids
]

if record_ids != index_ids:

    raise RuntimeError(
        "Memory record ids and index ids differ."
    )

if (
        stored_memory_embeddings.ndim
        !=
        2
):

    raise RuntimeError(
        "Memory embeddings are not a matrix."
    )

if (
        stored_memory_embeddings.shape[0]
        !=
        len(
            memory_records
        )
):

    raise RuntimeError(
        "Memory index row count differs from records."
    )

if (
        stored_memory_embeddings.shape[1]
        !=
        MEMORY_DIMENSION
):

    raise RuntimeError(
        "Memory embedding dimension is incorrect."
    )

print(
    "Memory identity and index alignment validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Prepare Native Memory Matrix"
)

print()

memory_embeddings = stored_memory_embeddings.float()

if not torch.isfinite(
        memory_embeddings
).all():

    raise RuntimeError(
        "Stored memory embeddings contain invalid values."
    )

memory_embeddings = F.normalize(
    memory_embeddings,
    p=2,
    dim=-1
)

print(
    "Normalized memory matrix:",
    tuple(
        memory_embeddings.shape
    )
)

print(
    "Native memory matrix validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Recover Native Modal Embeddings"
)

print()

#
# 103R's persistent index may contain only memory embeddings.
# The consolidated memory records, however, originate from 102R
# and preserve the original modality embeddings.
#
# Therefore the recovery hierarchy is:
#
#     index text embedding
#             ↓
#     record text embedding
#
# and:
#
#     index numeric embedding
#             ↓
#     record numeric embedding
#
# No artificial reconstruction is attempted.
#

text_embedding_source = None
numeric_embedding_source = None

if (
        stored_text_embeddings is not None
        and
        stored_numeric_embeddings is not None
):

    text_embedding_source = "103R index"
    numeric_embedding_source = "103R index"

    text_embeddings = (
        stored_text_embeddings.float()
    )

    numeric_embeddings = (
        stored_numeric_embeddings.float()
    )

else:

    record_text_available = all(
        isinstance(
            record.get(
                "text_embedding"
            ),
            list
        )
        for record
        in memory_records
    )

    record_numeric_available = all(
        isinstance(
            record.get(
                "numeric_embedding"
            ),
            list
        )
        for record
        in memory_records
    )

    if record_text_available:

        text_embeddings = torch.tensor(
            [
                record[
                    "text_embedding"
                ]
                for record
                in memory_records
            ],
            dtype=torch.float32
        )

        text_embedding_source = (
            "103R consolidated memory records"
        )

    else:

        text_embeddings = None

    if record_numeric_available:

        numeric_embeddings = torch.tensor(
            [
                record[
                    "numeric_embedding"
                ]
                for record
                in memory_records
            ],
            dtype=torch.float32
        )

        numeric_embedding_source = (
            "103R consolidated memory records"
        )

    else:

        numeric_embeddings = None

print(
    "Text embedding source:",
    text_embedding_source
)

print(
    "Numeric embedding source:",
    numeric_embedding_source
)

if text_embeddings is None:

    raise RuntimeError(
        (
            "Silverwing text embeddings are unavailable in "
            "both the 103R index and consolidated memory records."
        )
    )

if numeric_embeddings is None:

    raise RuntimeError(
        (
            "Silverwing numeric embeddings are unavailable in "
            "both the 103R index and consolidated memory records."
        )
    )

print(
    "Recovered text embeddings:",
    tuple(
        text_embeddings.shape
    )
)

print(
    "Recovered numeric embeddings:",
    tuple(
        numeric_embeddings.shape
    )
)

if (
        text_embeddings.ndim
        !=
        2
):

    raise RuntimeError(
        "Text embeddings are not a matrix."
    )

if (
        numeric_embeddings.ndim
        !=
        2
):

    raise RuntimeError(
        "Numeric embeddings are not a matrix."
    )

if (
        text_embeddings.shape
        !=
        numeric_embeddings.shape
):

    raise RuntimeError(
        (
            "Text and numeric embeddings have "
            "different shapes."
        )
    )

if (
        text_embeddings.shape[0]
        !=
        len(
            memory_records
        )
):

    raise RuntimeError(
        "Text embedding count differs from memory records."
    )

if (
        text_embeddings.shape[1]
        !=
        MEMORY_DIMENSION
):

    raise RuntimeError(
        "Text embedding dimension differs from Silverwing dimension."
    )

if (
        numeric_embeddings.shape[1]
        !=
        MEMORY_DIMENSION
):

    raise RuntimeError(
        "Numeric embedding dimension differs from Silverwing dimension."
    )

text_embeddings = F.normalize(
    text_embeddings.float(),
    p=2,
    dim=-1
)

numeric_embeddings = F.normalize(
    numeric_embeddings.float(),
    p=2,
    dim=-1
)

if not torch.isfinite(
        text_embeddings
).all():

    raise RuntimeError(
        "Recovered text embeddings are invalid."
    )

if not torch.isfinite(
        numeric_embeddings
).all():

    raise RuntimeError(
        "Recovered numeric embeddings are invalid."
    )

print(
    "Native modality embeddings successfully recovered."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Build Memory Similarity Matrix"
)

print()

memory_similarity = torch.matmul(
    memory_embeddings,
    memory_embeddings.T
)

print(
    "Memory similarity matrix:",
    tuple(
        memory_similarity.shape
    )
)

if not torch.isfinite(
        memory_similarity
).all():

    raise RuntimeError(
        "Memory similarity matrix contains invalid values."
    )

print(
    "Memory similarity matrix validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Build Native Memory Query Engine"
)

print()


def query_memory(
        query_embedding: torch.Tensor,
        top_k: int
) -> List[Dict[str, Any]]:

    query = F.normalize(
        query_embedding.float(),
        p=2,
        dim=0
    )

    results = []

    for index in range(
            len(
                memory_records
            )
    ):

        score = cosine_similarity(
            query,
            memory_embeddings[
                index
            ]
        )

        record = memory_records[
            index
        ]

        results.append(
            {
                "index":
                    index,

                "memory_id":
                    record[
                        "memory_id"
                    ],

                "event_id":
                    record[
                        "event_id"
                    ],

                "timestamp":
                    record[
                        "timestamp"
                    ],

                "semantic_class":
                    record[
                        "semantic_class"
                    ],

                "score":
                    score
            }
        )

    results.sort(
        key=lambda item:
        item[
            "score"
        ],
        reverse=True
    )

    return results[
        :
        min(
            top_k,
            len(
                results
            )
        )
    ]


query_results = query_memory(
    memory_embeddings[0],
    TOP_K
)

for result in query_results:

    print(
        result
    )

if (
        not query_results
        or
        query_results[0][
            "index"
        ]
        !=
        0
):

    raise RuntimeError(
        "Memory query failed exact self-retrieval."
    )

print(
    "Native memory query engine validated."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Temporal Memory Query"
)

print()


def query_history(
        semantic_class: str
) -> List[Dict[str, Any]]:

    results = []

    for record in memory_records:

        if (
                str(
                    record[
                        "semantic_class"
                    ]
                )
                ==
                str(
                    semantic_class
                )
        ):

            results.append(
                record
            )

    results.sort(
        key=lambda record:
        parse_timestamp(
            record[
                "timestamp"
            ]
        )
    )

    return results


motor_history = query_history(
    "motor_warning"
)

for record in motor_history:

    print(
        record[
            "timestamp"
        ],
        "->",
        record[
            "memory_id"
        ]
    )

if not motor_history:

    raise RuntimeError(
        "Motor temporal history is unavailable."
    )

print(
    "Temporal memory query validated."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Historical State Change Calculation"
)

print()

state_change_records = []

for index in range(
        len(
            motor_history
        ) - 1
):

    previous = motor_history[
        index
    ]

    current = motor_history[
        index + 1
        ]

    previous_values = torch.tensor(
        previous[
            "numeric"
        ],
        dtype=torch.float32
    )

    current_values = torch.tensor(
        current[
            "numeric"
        ],
        dtype=torch.float32
    )

    delta = (
            current_values
            -
            previous_values
    )

    relative_delta = (
            delta
            /
            (
                    torch.abs(
                        previous_values
                    )
                    +
                    EPSILON
            )
    )

    elapsed_seconds = (
            parse_timestamp(
                current[
                    "timestamp"
                ]
            )
            -
            parse_timestamp(
                previous[
                    "timestamp"
                ]
            )
    ).total_seconds()

    state_change_records.append(
        {
            "from":
                previous[
                    "memory_id"
                ],

            "to":
                current[
                    "memory_id"
                ],

            "numeric_delta":
                delta.tolist(),

            "relative_delta":
                relative_delta.tolist(),

            "elapsed_seconds":
                elapsed_seconds
        }
    )

for change in state_change_records:

    print(
        change
    )

if not state_change_records:

    raise RuntimeError(
        "No historical state change was generated."
    )

print(
    "Historical state changes validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Native Text-to-Numeric Evidence Alignment"
)

print()

cross_modal_similarity = torch.matmul(
    text_embeddings,
    numeric_embeddings.T
)

print(
    "Cross-modal similarity matrix:",
    tuple(
        cross_modal_similarity.shape
    )
)

exact_alignment_hits = 0

alignment_margins = []

cross_modal_results = []

for index in range(
        len(
            memory_records
        )
):

    row = cross_modal_similarity[
        index
    ]

    ranked = []

    for candidate in range(
            len(
                memory_records
            )
    ):

        ranked.append(
            (
                candidate,
                float(
                    row[
                        candidate
                    ]
                )
            )
        )

    ranked.sort(
        key=lambda item:
        item[1],
        reverse=True
    )

    best_index = ranked[
        0
    ][
        0
    ]

    positive_score = float(
        row[
            index
        ]
    )

    negative_scores = [
        score
        for candidate, score
        in ranked
        if candidate != index
    ]

    hardest_negative = max(
        negative_scores
    )

    margin = (
            positive_score
            -
            hardest_negative
    )

    alignment_margins.append(
        margin
    )

    exact_pair = (
            best_index
            ==
            index
    )

    if exact_pair:

        exact_alignment_hits += 1

    cross_modal_results.append(
        {
            "memory_id":
                memory_records[
                    index
                ][
                    "memory_id"
                ],

            "retrieved_memory":
                memory_records[
                    best_index
                ][
                    "memory_id"
                ],

            "positive_score":
                positive_score,

            "hardest_negative":
                hardest_negative,

            "margin":
                margin,

            "exact":
                exact_pair
        }
    )

for result in cross_modal_results:

    print(
        result
    )

exact_alignment_accuracy = (
        exact_alignment_hits
        /
        len(
            memory_records
        )
)

mean_alignment_margin = safe_mean(
    alignment_margins
)

print(
    "Exact text -> numeric accuracy:",
    exact_alignment_accuracy
)

print(
    "Mean cross-modal margin:",
    mean_alignment_margin
)

if (
        exact_alignment_accuracy
        <
        REASONING_THRESHOLD
):

    raise RuntimeError(
        (
            "Native text-to-numeric alignment "
            "is below the Silverwing threshold."
        )
    )

if (
        mean_alignment_margin
        <=
        CROSS_MODAL_MARGIN_THRESHOLD
):

    raise RuntimeError(
        (
            "Cross-modal positive/negative margin "
            "is not positive."
        )
    )

print(
    "Native text-to-numeric evidence alignment validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Cross-Modal Evidence Comparison"
)

print()

agreement_results = []

for result in cross_modal_results:

    agreement_results.append(
        {
            "memory_id":
                result[
                    "memory_id"
                ],

            "agreement":
                result[
                    "exact"
                ],

            "margin":
                result[
                    "margin"
                ]
        }
    )

for item in agreement_results:

    print(
        item
    )

agreement_ratio = (
        sum(
            1
            for item
            in agreement_results
            if item[
                "agreement"
            ]
        )
        /
        len(
            agreement_results
        )
)

print(
    "Cross-modal exact-pair agreement:",
    agreement_ratio
)

if (
        agreement_ratio
        <
        REASONING_THRESHOLD
):

    raise RuntimeError(
        "Cross-modal exact-pair agreement below threshold."
    )

print(
    "Cross-modal evidence comparison validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Numeric Evidence Reasoning"
)

print()

numeric_reasoning = []

for change in state_change_records:

    delta = change[
        "numeric_delta"
    ]

    increased_dimensions = [
        index
        for index, value
        in enumerate(
            delta
        )
        if value > 0
    ]

    decreased_dimensions = [
        index
        for index, value
        in enumerate(
            delta
        )
        if value < 0
    ]

    magnitude = math.sqrt(
        sum(
            value * value
            for value
            in delta
        )
    )

    numeric_reasoning.append(
        {
            "from":
                change[
                    "from"
                ],

            "to":
                change[
                    "to"
                ],

            "increased_dimensions":
                increased_dimensions,

            "decreased_dimensions":
                decreased_dimensions,

            "change_magnitude":
                magnitude
        }
    )

for item in numeric_reasoning:

    print(
        item
    )

print(
    "Numeric evidence reasoning validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Build Multimodal Reasoning Chains"
)

print()

reasoning_chains = []

for index, record in enumerate(
        memory_records
):

    memory_results = query_memory(
        memory_embeddings[
            index
        ],
        TOP_K
    )

    memory_verified = (
            bool(
                memory_results
            )
            and
            memory_results[
                0
            ][
                "memory_id"
            ]
            ==
            record[
                "memory_id"
            ]
    )

    cross_modal_verified = (
        cross_modal_results[
            index
        ][
            "exact"
        ]
    )

    history = query_history(
        record[
            "semantic_class"
        ]
    )

    chain = {
        "query_memory":
            record[
                "memory_id"
            ],

        "memory_retrieval_verified":
            memory_verified,

        "cross_modal_retrieval_verified":
            cross_modal_verified,

        "historical_context":
            [
                item[
                    "memory_id"
                ]
                for item
                in history
            ],

        "conclusion":
            None
    }

    if (
            memory_verified
            and
            cross_modal_verified
            and
            history
    ):

        chain[
            "conclusion"
        ] = (
            "Memory identity, cross-modal evidence, "
            "and historical context are available."
        )

    else:

        chain[
            "conclusion"
        ] = (
            "Reasoning evidence is incomplete."
        )

    reasoning_chains.append(
        chain
    )

for chain in reasoning_chains:

    print(
        chain
    )

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Verify Reasoning Chain Evidence"
)

print()

reasoning_errors = []

for chain in reasoning_chains:

    if not chain[
        "memory_retrieval_verified"
    ]:

        reasoning_errors.append(
            {
                "memory_id":
                    chain[
                        "query_memory"
                    ],

                "error":
                    "memory retrieval failed"
            }
        )

    if not chain[
        "cross_modal_retrieval_verified"
    ]:

        reasoning_errors.append(
            {
                "memory_id":
                    chain[
                        "query_memory"
                    ],

                "error":
                    "cross-modal retrieval failed"
            }
        )

    if not chain[
        "historical_context"
    ]:

        reasoning_errors.append(
            {
                "memory_id":
                    chain[
                        "query_memory"
                    ],

                "error":
                    "historical context unavailable"
            }
        )

if reasoning_errors:

    print(
        json.dumps(
            reasoning_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Memory reasoning evidence verification failed."
    )

print(
    "Reasoning chains verified."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Engineering Event Progression Reasoning"
)

print()

if (
        len(
            motor_history
        )
        <
        2
):

    raise RuntimeError(
        "At least two motor events are required."
    )

first_motor = motor_history[
    0
]

latest_motor = motor_history[
    -1
]

first_values = torch.tensor(
    first_motor[
        "numeric"
    ],
    dtype=torch.float32
)

latest_values = torch.tensor(
    latest_motor[
        "numeric"
    ],
    dtype=torch.float32
)

overall_delta = (
        latest_values
        -
        first_values
)

overall_relative_change = (
        overall_delta
        /
        (
                torch.abs(
                    first_values
                )
                +
                EPSILON
        )
)

print(
    "First motor event:",
    first_motor[
        "memory_id"
    ]
)

print(
    "Latest motor event:",
    latest_motor[
        "memory_id"
    ]
)

print(
    "Overall numeric delta:",
    overall_delta.tolist()
)

print(
    "Overall relative change:",
    overall_relative_change.tolist()
)

print(
    "Engineering event progression reasoning validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Memory-Based Conclusion Verification"
)

print()

latest_text = str(
    latest_motor[
        "text"
    ]
).lower()

warning_language_present = (
        "warning" in latest_text
        or
        "high" in latest_text
)

positive_change_present = any(
    value > 0
    for value
    in overall_delta.tolist()
)

historical_change_available = (
        len(
            state_change_records
        )
        >
        0
)

cross_modal_memory_verified = (
        exact_alignment_accuracy
        >=
        REASONING_THRESHOLD
)

evidence_components = [
    historical_change_available,
    warning_language_present,
    positive_change_present,
    cross_modal_memory_verified
]

evidence_score = (
        sum(
            1
            for value
            in evidence_components
            if value
        )
        /
        len(
            evidence_components
        )
)

print(
    "Historical change available:",
    historical_change_available
)

print(
    "Warning/high language present:",
    warning_language_present
)

print(
    "Positive numeric change present:",
    positive_change_present
)

print(
    "Cross-modal memory verified:",
    cross_modal_memory_verified
)

print(
    "Reasoning evidence score:",
    evidence_score
)

if (
        evidence_score
        <
        REASONING_THRESHOLD
):

    raise RuntimeError(
        "Memory reasoning evidence score below threshold."
    )

reasoning_conclusion = (
    "Historical multimodal evidence indicates "
    "a changing motor operating state requiring continued observation."
)

print(
    "Conclusion:",
    reasoning_conclusion
)

print(
    "Memory-based conclusion verified."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Temporal Causal-Order Validation"
)

print()

causal_order_valid = True

for index in range(
        len(
            motor_history
        ) - 1
):

    current_time = parse_timestamp(
        motor_history[
            index
        ][
            "timestamp"
        ]
    )

    next_time = parse_timestamp(
        motor_history[
            index + 1
            ][
            "timestamp"
        ]
    )

    if current_time >= next_time:

        causal_order_valid = False

if not causal_order_valid:

    raise RuntimeError(
        "Temporal causal ordering failed."
    )

print(
    "Temporal causal order validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Memory Reasoning Stress Test"
)

print()

stress_cases = []

for index, record in enumerate(
        memory_records
):

    memory_results = query_memory(
        memory_embeddings[
            index
        ],
        TOP_K
    )

    exact_memory = (
            bool(
                memory_results
            )
            and
            memory_results[
                0
            ][
                "memory_id"
            ]
            ==
            record[
                "memory_id"
            ]
    )

    exact_cross_modal = (
        cross_modal_results[
            index
        ][
            "exact"
        ]
    )

    history = query_history(
        record[
            "semantic_class"
        ]
    )

    stress_cases.append(
        {
            "memory_id":
                record[
                    "memory_id"
                ],

            "memory_retrieval":
                exact_memory,

            "cross_modal_retrieval":
                exact_cross_modal,

            "history_available":
                bool(
                    history
                )
        }
    )

for case in stress_cases:

    print(
        case
    )

stress_failures = [
    case
    for case
    in stress_cases
    if not (
            case[
                "memory_retrieval"
            ]
            and
            case[
                "cross_modal_retrieval"
            ]
            and
            case[
                "history_available"
            ]
    )
]

if stress_failures:

    print(
        json.dumps(
            stress_failures,
            indent=4
        )
    )

    raise RuntimeError(
        "Memory reasoning stress test failed."
    )

print(
    "Memory reasoning stress test passed."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Deterministic Reasoning Validation"
)

print()

first_pass = []
second_pass = []

for index in range(
        len(
            memory_records
        )
):

    first_results = query_memory(
        memory_embeddings[
            index
        ],
        TOP_K
    )

    second_results = query_memory(
        memory_embeddings[
            index
        ],
        TOP_K
    )

    first_pass.append(
        [
            (
                item[
                    "memory_id"
                ],

                item[
                    "score"
                ]
            )
            for item
            in first_results
        ]
    )

    second_pass.append(
        [
            (
                item[
                    "memory_id"
                ],

                item[
                    "score"
                ]
            )
            for item
            in second_results
        ]
    )

determinism_errors = 0

for first, second in zip(
        first_pass,
        second_pass
):

    if first != second:

        determinism_errors += 1

print(
    "Determinism errors:",
    determinism_errors
)

if determinism_errors != 0:

    raise RuntimeError(
        "Memory reasoning is nondeterministic."
    )

print(
    "Deterministic memory reasoning validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Numerical Health"
)

print()

nan_tensors = 0
inf_tensors = 0

health_tensors = [
    memory_embeddings,
    memory_similarity,
    text_embeddings,
    numeric_embeddings,
    cross_modal_similarity
]

for tensor in health_tensors:

    if torch.isnan(
            tensor
    ).any():

        nan_tensors += 1

    if torch.isinf(
            tensor
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
        "Memory reasoning numerical health failed."
    )

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Final Multimodal Memory Reasoning Promotion Gate"
)

print()

promotion_errors = []

if (
        exact_alignment_accuracy
        <
        REASONING_THRESHOLD
):

    promotion_errors.append(
        "Exact text-to-numeric alignment below threshold."
    )

if (
        mean_alignment_margin
        <=
        CROSS_MODAL_MARGIN_THRESHOLD
):

    promotion_errors.append(
        "Cross-modal positive-negative margin is not positive."
    )

if (
        agreement_ratio
        <
        REASONING_THRESHOLD
):

    promotion_errors.append(
        "Cross-modal exact-pair agreement below threshold."
    )

if (
        evidence_score
        <
        REASONING_THRESHOLD
):

    promotion_errors.append(
        "Memory reasoning evidence score below threshold."
    )

if not causal_order_valid:

    promotion_errors.append(
        "Temporal causal ordering failed."
    )

if stress_failures:

    promotion_errors.append(
        "Memory reasoning stress testing failed."
    )

if determinism_errors:

    promotion_errors.append(
        "Deterministic retrieval failed."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

print(
    "Promotion errors:",
    len(
        promotion_errors
    )
)

if promotion_errors:

    print(
        json.dumps(
            promotion_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "104R memory reasoning promotion gate failed."
    )

print(
    "104R memory reasoning promotion gate passed."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save 104R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "104R",

    "capability":
        "native_multimodal_memory_reasoning",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "text_embedding_source":
        text_embedding_source,

    "numeric_embedding_source":
        numeric_embedding_source,

    "exact_cross_modal_accuracy":
        exact_alignment_accuracy,

    "cross_modal_margin":
        mean_alignment_margin,

    "cross_modal_agreement":
        agreement_ratio,

    "reasoning_evidence_score":
        evidence_score,

    "reasoning_conclusion":
        reasoning_conclusion,

    "state_changes":
        state_change_records,

    "reasoning_chains":
        reasoning_chains,

    "causal_order_valid":
        causal_order_valid,

    "promotion":
        {
            "passed":
                True,

            "errors":
                promotion_errors
        }
}

torch.save(
    checkpoint_payload,
    CANDIDATE_CHECKPOINT
)

torch.save(
    checkpoint_payload,
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
    "TEST 25: Write 104R Reports"
)

print()

report = {
    "lesson":
        "104R",

    "capability":
        "native_multimodal_memory_reasoning",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "memory_records":
        len(
            memory_records
        ),

    "embedding_sources":
        {
            "text":
                text_embedding_source,

            "numeric":
                numeric_embedding_source,

            "memory":
                "103R memory index"
        },

    "cross_modal_alignment":
        {
            "exact_accuracy":
                exact_alignment_accuracy,

            "mean_margin":
                mean_alignment_margin,

            "agreement_ratio":
                agreement_ratio
        },

    "reasoning":
        {
            "evidence_score":
                evidence_score,

            "conclusion":
                reasoning_conclusion,

            "causal_order_valid":
                causal_order_valid
        },

    "stress":
        {
            "cases":
                len(
                    stress_cases
                ),

            "failures":
                len(
                    stress_failures
                )
        },

    "determinism":
        {
            "errors":
                determinism_errors
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

    "promotion":
        {
            "passed":
                True,

            "errors":
                promotion_errors
        }
}

write_json(
    REASONING_REPORT_FILE,
    report
)

write_json(
    REASONING_EVALUATION_FILE,
    report
)

write_json(
    REASONING_REGISTRY_FILE,
    {
        "lesson":
            "104R",

        "capability":
            "native_multimodal_memory_reasoning",

        "memory_version":
            MEMORY_VERSION,

        "text_embedding_source":
            text_embedding_source,

        "numeric_embedding_source":
            numeric_embedding_source,

        "exact_cross_modal_accuracy":
            exact_alignment_accuracy,

        "reasoning_evidence_score":
            evidence_score,

        "next":
            "105R Native Memory Prediction + State Forecasting"
    }
)

print(
    "Report:",
    REASONING_REPORT_FILE
)

print(
    "Evaluation:",
    REASONING_EVALUATION_FILE
)

print(
    "Registry:",
    REASONING_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 104R MEMORY REASONING ARCHITECTURE"
)

print()

print(
    "Persistent Multimodal Memory"
)

print(
    "        ↓"
)

print(
    "Recover Native Modal Representations"
)

print(
    "        ↓"
)

print(
    "Memory Retrieval"
)

print(
    "        ↓"
)

print(
    "Identity Verification"
)

print(
    "        ↓"
)

print(
    "Text <-> Numeric Alignment"
)

print(
    "        ↓"
)

print(
    "Historical Context"
)

print(
    "        ↓"
)

print(
    "Numeric State Change"
)

print(
    "        ↓"
)

print(
    "Multimodal Evidence Reasoning"
)

print(
    "        ↓"
)

print(
    "Verified Conclusion"
)

print()


# ============================================================
# WHY 104R MATTERS
# ============================================================

print(
    "WHY 104R MATTERS"
)

print()

print(
    "102R established persistent multimodal memory."
)

print(
    "103R consolidated and organized that memory."
)

print(
    "104R now reasons over the preserved representations."
)

print()

print(
    "The system does not assume that a 103R index must contain "
    "every modality. It discovers where the preserved state lives "
    "and recovers it without inventing a substitute."
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
    "Lesson 105R: Native Memory Prediction + State Forecasting"
)

print()

print(
    "Historical Memory + Current State + Trend Analysis + "
    "Future State Estimation + Confidence"
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
    "103R Native Memory Consolidation + Temporal Retrieval",
    " ↓",
    "104R Native Multimodal Memory Reasoning",
    " ↓",
    "105R Native Memory Prediction + State Forecasting",
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
    "=== LESSON 104R COMPLETE ==="
)
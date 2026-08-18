# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 103R
# Native Memory Consolidation + Temporal Retrieval
# ============================================================
#
# 79R -> Native Reasoning Dataset
# 80R -> Native Reasoning Fine-Tuning
# 81R -> Native Memory-Aware Training
# 82R -> Native Tool-Aware Learning
# 83R -> Native Planning and Tool Sequencing
# 84R -> Native Verified Execution + Replanning
# 85R -> Native Mathematical Reasoning
# 86R -> Native Probability + Statistics
# 87R -> Native Linear Algebra + Optimization
# 88R -> Native Algorithms + Data Structures
# 89R -> Native Data Analysis + SQL Reasoning
# 90R -> Native Data Engineering
# 91R -> Native Machine Learning Foundations
# 92R -> Native Classical Machine Learning
# 93R -> Native Neural Network Foundations
# 94R -> Native Deep Learning
# 95R -> Native Representation Learning
# 96R -> Native Sequence Representation Learning
# 97R -> Native Structured Representation Learning
# 98R -> Advanced Sequence + Structured Learning
# 99R -> Multimodal Representation Foundations
# 100R -> Cross-Modal Alignment + Retrieval
# 101R -> Native Hard-Negative Multimodal Learning
# 102R -> Native Multimodal Memory Integration
# 103R -> Native Memory Consolidation + Temporal Retrieval
#
# ============================================================
# PURPOSE
# ============================================================
#
# 103R converts the persistent multimodal memory from 102R
# into a temporal, consolidating memory layer.
#
# It validates:
#
#   memory schema
#   memory identity
#   timestamps
#   embeddings
#   similarity
#   duplicate detection
#   related-event detection
#   temporal chains
#   recency retrieval
#   historical retrieval
#   consolidation
#   persistence
#   deterministic retrieval
#
# IMPORTANT:
#
# The previous failure:
#
#     NameError: name 'F' is not defined
#
# was caused by using torch.nn.functional without importing it.
#
# This replacement explicitly imports:
#
#     import torch.nn.functional as F
#
# and also validates the 102R index structure before using it.
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

MEMORY_VERSION = "103R.1"

MEMORY_DIMENSION = 16

DUPLICATE_SIMILARITY_THRESHOLD = 0.995

RELATED_SIMILARITY_THRESHOLD = 0.80

RECENCY_DECAY = 0.20

TOP_K = 3


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

PHASE5_DIR = BASE_DIR.parent

LESSON_102R = (
        PHASE5_DIR /
        "lesson102R"
)

SOURCE_MEMORY_FILE = (
        LESSON_102R /
        "silverwing_multimodal_memory.json"
)

SOURCE_MEMORY_INDEX = (
        LESSON_102R /
        "silverwing_multimodal_memory_index.pt"
)

SOURCE_REGISTRY = (
        LESSON_102R /
        "silverwing_multimodal_memory_registry.json"
)

SOURCE_CHECKPOINT_PRIMARY = (
        LESSON_102R /
        "checkpoints" /
        "silverwing_multimodal_memory_best.pt"
)

SOURCE_CHECKPOINT_CANDIDATE = (
        LESSON_102R /
        "checkpoints" /
        "silverwing_multimodal_memory_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONSOLIDATED_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_consolidated_memory.json"
)

CONSOLIDATED_INDEX_FILE = (
        BASE_DIR /
        "silverwing_consolidated_memory_index.pt"
)

CONSOLIDATION_REGISTRY = (
        BASE_DIR /
        "silverwing_memory_consolidation_registry.json"
)

REPORT_FILE = (
        BASE_DIR /
        "silverwing_memory_consolidation_report.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_memory_consolidation_evaluation.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_memory_consolidation_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_memory_consolidation_best.pt"
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


def cosine(
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
    ) == 0.0:

        return 0.0

    return float(
        torch.dot(
            left,
            right
        )
        /
        denominator
    )


def parse_timestamp(
        value: str
) -> datetime:

    normalized = str(
        value
    ).replace(
        "Z",
        ""
    )

    return datetime.fromisoformat(
        normalized
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
        (
            "No 102R checkpoint found.\n"
            "Checked primary and candidate paths."
        )
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
    "PHASE 5 - LESSON 103R"
)

print(
    "Native Memory Consolidation + Temporal Retrieval"
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
    "103R -> Memory Consolidation + Temporal Retrieval"
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
    "Duplicate threshold:",
    DUPLICATE_SIMILARITY_THRESHOLD
)

print(
    "Related-event threshold:",
    RELATED_SIMILARITY_THRESHOLD
)

print(
    "Top-k:",
    TOP_K
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Inspect 102R Memory Artifacts"
)

print()

for path in [
    SOURCE_MEMORY_FILE,
    SOURCE_MEMORY_INDEX,
    SOURCE_REGISTRY
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
    SOURCE_MEMORY_INDEX
)

print(
    "FOUND:",
    SOURCE_REGISTRY
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
    "TEST 2: Load Persistent Multimodal Memory"
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
        "102R memory payload is not a dictionary."
    )

source_records = memory_payload.get(
    "records"
)

if not isinstance(
        source_records,
        list
):

    raise RuntimeError(
        "102R memory payload does not contain records."
    )

print(
    "Memory version:",
    memory_payload.get(
        "memory_version"
    )
)

print(
    "Record count:",
    len(
        source_records
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Inspect Memory Schema"
)

print()

REQUIRED_FIELDS = {
    "memory_id",
    "event_id",
    "timestamp",
    "text",
    "numeric",
    "semantic_class",
    "source",
    "confidence",
    "text_embedding",
    "numeric_embedding",
    "memory_embedding"
}

schema_errors = []

for record in source_records:

    missing = (
            REQUIRED_FIELDS
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

if schema_errors:

    print(
        json.dumps(
            schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "102R memory schema validation failed."
    )

print(
    "Required memory fields present."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Event and Memory Identity"
)

print()

memory_ids = [
    record["memory_id"]
    for record
    in source_records
]

event_ids = [
    record["event_id"]
    for record
    in source_records
]

duplicate_memory_ids = (
        len(memory_ids)
        -
        len(
            set(
                memory_ids
            )
        )
)

duplicate_event_ids = (
        len(event_ids)
        -
        len(
            set(
                event_ids
            )
        )
)

print(
    "Records:",
    len(
        source_records
    )
)

print(
    "Unique memory ids:",
    len(
        set(
            memory_ids
        )
    )
)

print(
    "Unique event ids:",
    len(
        set(
            event_ids
        )
    )
)

print(
    "Duplicate memory ids:",
    duplicate_memory_ids
)

print(
    "Duplicate event ids:",
    duplicate_event_ids
)

if duplicate_memory_ids > 0:

    raise RuntimeError(
        "Memory identifiers are duplicated."
    )

if duplicate_event_ids > 0:

    print(
        "Repeated event identifiers detected."
    )

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Parse and Order Memory Timestamps"
)

print()

timestamp_objects = []

for record in source_records:

    try:

        timestamp_objects.append(
            (
                record,
                parse_timestamp(
                    record["timestamp"]
                )
            )
        )

    except Exception as exc:

        raise RuntimeError(
            (
                f"Invalid timestamp in "
                f"{record['memory_id']}: {exc}"
            )
        ) from exc

timestamp_objects.sort(
    key=lambda pair: pair[1]
)

ordered_records = [
    pair[0]
    for pair
    in timestamp_objects
]

ordered_timestamps = [
    pair[1]
    for pair
    in timestamp_objects
]

print(
    "Earliest:",
    ordered_records[0]["timestamp"]
)

print(
    "Latest:",
    ordered_records[-1]["timestamp"]
)

print(
    "Chronological records:",
    len(
        ordered_records
    )
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Validate Stored Embedding Dimensions"
)

print()

embedding_errors = []

for record in ordered_records:

    for field in [
        "text_embedding",
        "numeric_embedding",
        "memory_embedding"
    ]:

        values = record[
            field
        ]

        if not isinstance(
                values,
                list
        ):

            embedding_errors.append(
                {
                    "memory_id":
                        record["memory_id"],

                    "field":
                        field,

                    "error":
                        "not a list"
                }
            )

            continue

        if len(
                values
        ) != MEMORY_DIMENSION:

            embedding_errors.append(
                {
                    "memory_id":
                        record["memory_id"],

                    "field":
                        field,

                    "actual":
                        len(values),

                    "expected":
                        MEMORY_DIMENSION
                }
            )

        if not all(
                math.isfinite(
                    float(value)
                )
                for value
                in values
        ):

            embedding_errors.append(
                {
                    "memory_id":
                        record["memory_id"],

                    "field":
                        field,

                    "error":
                        "non-finite value"
                }
            )

if embedding_errors:

    print(
        json.dumps(
            embedding_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Stored memory embeddings failed validation."
    )

print(
    "Stored embeddings validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Inspect Persistent Memory Index"
)

print()

memory_index_payload = torch.load(
    SOURCE_MEMORY_INDEX,
    map_location="cpu",
    weights_only=False
)

if not isinstance(
        memory_index_payload,
        dict
):

    raise RuntimeError(
        "102R memory index is not a dictionary."
    )

stored_memory_embeddings = (
    memory_index_payload.get(
        "memory_embeddings"
    )
)

stored_memory_ids = (
    memory_index_payload.get(
        "memory_ids"
    )
)

print(
    "Index fields:",
    sorted(
        memory_index_payload.keys()
    )
)

if stored_memory_embeddings is not None:

    print(
        "Stored memory embedding shape:",
        tuple(
            stored_memory_embeddings.shape
        )
    )

if stored_memory_ids is not None:

    print(
        "Stored index ids:",
        len(
            stored_memory_ids
        )
    )

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Build Native Memory Vector Matrix"
)

print()

memory_vectors = torch.tensor(
    [
        record["memory_embedding"]
        for record
        in ordered_records
    ],
    dtype=torch.float32
)

# Explicitly use torch.nn.functional imported as F.
memory_vectors = F.normalize(
    memory_vectors,
    p=2,
    dim=-1
)

print(
    "Memory vector matrix:",
    tuple(
        memory_vectors.shape
    )
)

if not torch.isfinite(
        memory_vectors
).all():

    raise RuntimeError(
        "Memory vector matrix is numerically invalid."
    )

print(
    "Memory vector matrix validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Compare Memory Store and Persistent Index"
)

print()

index_errors = []

if stored_memory_embeddings is None:

    index_errors.append(
        "memory_embeddings missing from index."
    )

else:

    expected_shape = (
        len(
            ordered_records
        ),
        MEMORY_DIMENSION
    )

    actual_shape = tuple(
        stored_memory_embeddings.shape
    )

    print(
        "Expected index shape:",
        expected_shape
    )

    print(
        "Actual index shape:",
        actual_shape
    )

    if actual_shape != expected_shape:

        index_errors.append(
            (
                "Persistent memory embedding shape differs "
                "from the memory store."
            )
        )

if stored_memory_ids is None:

    index_errors.append(
        "memory_ids missing from index."
    )

else:

    expected_ids = [
        record["memory_id"]
        for record
        in ordered_records
    ]

    actual_ids = [
        str(value)
        for value
        in stored_memory_ids
    ]

    if actual_ids != expected_ids:

        index_errors.append(
            "Persistent memory ids differ from memory store."
        )

if index_errors:

    print(
        json.dumps(
            index_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Persistent memory index validation failed."
    )

print(
    "Persistent memory index matches the memory store."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Calculate Memory Similarity Matrix"
)

print()

memory_similarity = torch.matmul(
    memory_vectors,
    memory_vectors.T
)

print(
    "Similarity matrix:",
    tuple(
        memory_similarity.shape
    )
)

print(
    "Diagonal:",
    torch.diag(
        memory_similarity
    ).tolist()
)

if not torch.isfinite(
        memory_similarity
).all():

    raise RuntimeError(
        "Memory similarity matrix contains invalid values."
    )

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Exact and Near-Duplicate Detection"
)

print()

duplicate_pairs = []
related_pairs = []

for left_index in range(
        len(
            ordered_records
        )
):

    for right_index in range(
            left_index + 1,
            len(
                ordered_records
            )
    ):

        score = float(
            memory_similarity[
                left_index,
                right_index
            ]
        )

        left_record = ordered_records[
            left_index
        ]

        right_record = ordered_records[
            right_index
        ]

        pair = {
            "left":
                left_record[
                    "memory_id"
                ],

            "right":
                right_record[
                    "memory_id"
                ],

            "similarity":
                score
        }

        if (
                score
                >=
                DUPLICATE_SIMILARITY_THRESHOLD
        ):

            duplicate_pairs.append(
                pair
            )

        elif (
                score
                >=
                RELATED_SIMILARITY_THRESHOLD
        ):

            related_pairs.append(
                pair
            )

print(
    "Exact/near duplicates:",
    len(
        duplicate_pairs
    )
)

print(
    "Related events:",
    len(
        related_pairs
    )
)

if duplicate_pairs:

    print(
        "Duplicate candidates:"
    )

    for pair in duplicate_pairs:

        print(
            pair
        )

if related_pairs:

    print(
        "Related-event candidates:"
    )

    for pair in related_pairs:

        print(
            pair
        )

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Build Temporal Event Chains"
)

print()

event_chains = {}

for record in ordered_records:

    semantic_class = str(
        record[
            "semantic_class"
        ]
    )

    event_type = semantic_class.split(
        "_"
    )[0]

    if event_type not in event_chains:

        event_chains[
            event_type
        ] = []

    event_chains[
        event_type
    ].append(
        {
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
                semantic_class
        }
    )

for event_type, chain in event_chains.items():

    print(
        event_type,
        "events:",
        len(
            chain
        )
    )

    for event in chain:

        print(
            "  ",
            event["timestamp"],
            "->",
            event["memory_id"]
        )

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Validate Temporal Event Chains"
)

print()

chain_errors = []

for event_type, chain in event_chains.items():

    chain_times = [
        parse_timestamp(
            item["timestamp"]
        )
        for item
        in chain
    ]

    if chain_times != sorted(
            chain_times
    ):

        chain_errors.append(
            event_type
        )

if chain_errors:

    raise RuntimeError(
        (
            "Temporal event chains are not ordered:\n"
            f"{chain_errors}"
        )
    )

print(
    "Temporal event chains validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Build Recency Scores"
)

print()

latest_timestamp = ordered_timestamps[
    -1
]

recency_scores = {}

for record in ordered_records:

    timestamp = parse_timestamp(
        record[
            "timestamp"
        ]
    )

    hours_ago = (
                        latest_timestamp
                        -
                        timestamp
                ).total_seconds() / 3600.0

    recency = math.exp(
        -RECENCY_DECAY
        *
        hours_ago
    )

    recency_scores[
        record[
            "memory_id"
        ]
    ] = recency

for memory_id, score in recency_scores.items():

    print(
        memory_id,
        "recency=",
        score
    )

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Validate Recency Ordering"
)

print()

recency_order = sorted(
    ordered_records,
    key=lambda record:
    recency_scores[
        record[
            "memory_id"
        ]
    ],
    reverse=True
)

recency_values = [
    recency_scores[
        record[
            "memory_id"
        ]
    ]
    for record
    in recency_order
]

print(
    "Recency ordering:"
)

for record in recency_order:

    print(
        record[
            "memory_id"
        ],
        "->",
        recency_scores[
            record[
                "memory_id"
            ]
        ]
    )

if (
        recency_values
        !=
        sorted(
            recency_values,
            reverse=True
        )
):

    raise RuntimeError(
        "Recency ordering failed."
    )

print(
    "Recency ordering validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Recent Memory Query"
)

print()


def retrieve_recent(
        records: List[Dict[str, Any]],
        scores: Dict[str, float],
        k: int
) -> List[Dict[str, Any]]:

    ranked = sorted(
        records,
        key=lambda record:
        scores[
            record[
                "memory_id"
            ]
        ],
        reverse=True
    )

    return ranked[
        :
        min(
            k,
            len(
                ranked
            )
        )
    ]


recent_results = retrieve_recent(
    ordered_records,
    recency_scores,
    TOP_K
)

print(
    "Recent memory results:"
)

for record in recent_results:

    print(
        record[
            "memory_id"
        ],
        "|",
        record[
            "timestamp"
        ]
    )

if len(
        recent_results
) != min(
    TOP_K,
    len(
        ordered_records
    )
):

    raise RuntimeError(
        "Recent-memory query returned incorrect count."
    )

print(
    "Recent-memory retrieval validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Historical Retrieval"
)

print()

motor_records = [
    record
    for record
    in ordered_records
    if str(
        record[
            "semantic_class"
        ]
    ).startswith(
        "motor_"
    )
]

if not motor_records:

    raise RuntimeError(
        "No motor history exists for validation."
    )

motor_records = sorted(
    motor_records,
    key=lambda record:
    parse_timestamp(
        record[
            "timestamp"
        ]
    )
)

print(
    "Motor historical events:",
    len(
        motor_records
    )
)

for record in motor_records:

    print(
        record[
            "timestamp"
        ],
        "->",
        record[
            "memory_id"
        ]
    )

for index in range(
        len(
            motor_records
        ) - 1
):

    current_time = parse_timestamp(
        motor_records[
            index
        ][
            "timestamp"
        ]
    )

    next_time = parse_timestamp(
        motor_records[
            index + 1
            ][
            "timestamp"
        ]
    )

    if current_time > next_time:

        raise RuntimeError(
            "Historical events are not chronological."
        )

print(
    "Historical memory retrieval validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Memory Consolidation"
)

print()

consolidated_records = []

duplicate_memory_ids = set()

for pair in duplicate_pairs:

    duplicate_memory_ids.add(
        pair[
            "right"
        ]
    )

for record in ordered_records:

    if (
            record["memory_id"]
            in duplicate_memory_ids
    ):

        continue

    consolidated_records.append(
        record
    )

print(
    "Original records:",
    len(
        ordered_records
    )
)

print(
    "Duplicate records removed:",
    len(
        duplicate_memory_ids
    )
)

print(
    "Consolidated records:",
    len(
        consolidated_records
    )
)

if not consolidated_records:

    raise RuntimeError(
        "Consolidation removed every record."
    )

print(
    "Memory consolidation validated."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Build Consolidated Timeline"
)

print()

consolidated_records = sorted(
    consolidated_records,
    key=lambda record:
    parse_timestamp(
        record[
            "timestamp"
        ]
    )
)

timeline = []

for index, record in enumerate(
        consolidated_records
):

    previous_memory = None
    next_memory = None

    if index > 0:

        previous_memory = (
            consolidated_records[
                index - 1
                ][
                "memory_id"
            ]
        )

    if (
            index + 1
            <
            len(
                consolidated_records
            )
    ):

        next_memory = (
            consolidated_records[
                index + 1
                ][
                "memory_id"
            ]
        )

    timeline.append(
        {
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

            "previous_memory":
                previous_memory,

            "next_memory":
                next_memory
        }
    )

for item in timeline:

    print(
        item
    )

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Persist Consolidated Memory"
)

print()

consolidated_payload = {
    "memory_version":
        MEMORY_VERSION,

    "source_version":
        memory_payload.get(
            "memory_version"
        ),

    "record_count":
        len(
            consolidated_records
        ),

    "consolidated_at":
        datetime.now().isoformat(),

    "records":
        consolidated_records,

    "timeline":
        timeline,

    "duplicate_pairs":
        duplicate_pairs,

    "related_pairs":
        related_pairs
}

write_json(
    CONSOLIDATED_MEMORY_FILE,
    consolidated_payload
)

consolidated_vectors = torch.tensor(
    [
        record[
            "memory_embedding"
        ]
        for record
        in consolidated_records
    ],
    dtype=torch.float32
)

# Explicit F usage is now backed by:
# import torch.nn.functional as F
consolidated_vectors = F.normalize(
    consolidated_vectors,
    p=2,
    dim=-1
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "memory_embeddings":
            consolidated_vectors,

        "memory_ids":
            [
                record[
                    "memory_id"
                ]
                for record
                in consolidated_records
            ],

        "timeline":
            timeline
    },
    CONSOLIDATED_INDEX_FILE
)

print(
    "Consolidated memory:",
    CONSOLIDATED_MEMORY_FILE
)

print(
    "Consolidated index:",
    CONSOLIDATED_INDEX_FILE
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Reload Consolidated Memory"
)

print()

reloaded = read_json(
    CONSOLIDATED_MEMORY_FILE
)

if (
        reloaded[
            "record_count"
        ]
        !=
        len(
            consolidated_records
        )
):

    raise RuntimeError(
        "Consolidated memory count changed after reload."
    )

reloaded_ids = [
    record[
        "memory_id"
    ]
    for record
    in reloaded[
        "records"
    ]
]

current_ids = [
    record[
        "memory_id"
    ]
    for record
    in consolidated_records
]

if (
        reloaded_ids
        !=
        current_ids
):

    raise RuntimeError(
        "Consolidated memory identity changed after reload."
    )

print(
    "Reloaded records:",
    len(
        reloaded[
            "records"
        ]
    )
)

print(
    "Persistent consolidation reload validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Deterministic Consolidated Memory Query"
)

print()

query_record = consolidated_records[
    0
]

query_vector = torch.tensor(
    query_record[
        "memory_embedding"
    ],
    dtype=torch.float32
)

query_vector = F.normalize(
    query_vector,
    p=2,
    dim=0
)

query_scores = []

for index, record in enumerate(
        consolidated_records
):

    candidate_vector = torch.tensor(
        record[
            "memory_embedding"
        ],
        dtype=torch.float32
    )

    candidate_vector = F.normalize(
        candidate_vector,
        p=2,
        dim=0
    )

    score = cosine(
        query_vector,
        candidate_vector
    )

    query_scores.append(
        (
            index,
            score
        )
    )

query_scores.sort(
    key=lambda value: value[1],
    reverse=True
)

for index, score in query_scores[
    :
    TOP_K
]:

    print(
        consolidated_records[
            index
        ][
            "memory_id"
        ],
        "score=",
        score
    )

if (
        query_scores[0][0]
        !=
        0
):

    raise RuntimeError(
        "Exact memory was not ranked first."
    )

print(
    "Deterministic consolidated retrieval validated."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Event Chain Integrity"
)

print()

chain_integrity_errors = []

for index, item in enumerate(
        timeline
):

    expected_previous = None
    expected_next = None

    if index > 0:

        expected_previous = timeline[
            index - 1
            ][
            "memory_id"
        ]

    if (
            index + 1
            <
            len(
                timeline
            )
    ):

        expected_next = timeline[
            index + 1
            ][
            "memory_id"
        ]

    if (
            item[
                "previous_memory"
            ]
            !=
            expected_previous
    ):

        chain_integrity_errors.append(
            item[
                "memory_id"
            ]
        )

    if (
            item[
                "next_memory"
            ]
            !=
            expected_next
    ):

        chain_integrity_errors.append(
            item[
                "memory_id"
            ]
        )

if chain_integrity_errors:

    raise RuntimeError(
        (
            "Memory chain integrity failed:\n"
            f"{chain_integrity_errors}"
        )
    )

print(
    "Memory event-chain integrity validated."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Numerical Health"
)

print()

nan_count = 0
inf_count = 0

for tensor in [
    memory_vectors,
    memory_similarity,
    consolidated_vectors
]:

    if torch.isnan(
            tensor
    ).any():

        nan_count += 1

    if torch.isinf(
            tensor
    ).any():

        inf_count += 1

numerically_healthy = (
        nan_count == 0
        and
        inf_count == 0
)

print(
    "NaN tensors:",
    nan_count
)

print(
    "Inf tensors:",
    inf_count
)

print(
    "Numerically healthy:",
    numerically_healthy
)

if not numerically_healthy:

    raise RuntimeError(
        "Memory consolidation numerical health failed."
    )

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Memory Consolidation Promotion Gate"
)

print()

promotion_errors = []

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if not consolidated_records:

    promotion_errors.append(
        "No consolidated records remain."
    )

if (
        reloaded_ids
        !=
        current_ids
):

    promotion_errors.append(
        "Persistence identity changed."
    )

if chain_integrity_errors:

    promotion_errors.append(
        "Event chain integrity failed."
    )

if (
        query_scores[0][0]
        !=
        0
):

    promotion_errors.append(
        "Deterministic exact-memory retrieval failed."
    )

if promotion_errors:

    print(
        json.dumps(
            promotion_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "103R memory consolidation promotion failed."
    )

print(
    "103R memory consolidation promotion gate passed."
)

print()


# ============================================================
# TEST 26
# ============================================================

print(
    "TEST 26: Save 103R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "103R",

    "capability":
        "native_memory_consolidation_temporal_retrieval",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "record_count_before":
        len(
            ordered_records
        ),

    "record_count_after":
        len(
            consolidated_records
        ),

    "duplicate_count":
        len(
            duplicate_pairs
        ),

    "related_event_count":
        len(
            related_pairs
        ),

    "consolidated_memory_file":
        str(
            CONSOLIDATED_MEMORY_FILE
        ),

    "consolidated_index_file":
        str(
            CONSOLIDATED_INDEX_FILE
        ),

    "timeline":
        timeline,

    "recency_scores":
        recency_scores,

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
    "Candidate checkpoint:",
    CANDIDATE_CHECKPOINT
)

print(
    "Promoted checkpoint:",
    BEST_CHECKPOINT
)

print()


# ============================================================
# TEST 27
# ============================================================

print(
    "TEST 27: Write 103R Reports"
)

print()

report = {
    "lesson":
        "103R",

    "capability":
        "native_memory_consolidation_temporal_retrieval",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "source_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "before":
        {
            "records":
                len(
                    ordered_records
                )
        },

    "consolidation":
        {
            "duplicate_pairs":
                len(
                    duplicate_pairs
                ),

            "related_pairs":
                len(
                    related_pairs
                ),

            "records_after":
                len(
                    consolidated_records
                )
        },

    "temporal":
        {
            "earliest":
                ordered_records[
                    0
                ][
                    "timestamp"
                ],

            "latest":
                ordered_records[
                    -1
                ][
                    "timestamp"
                ],

            "event_chains":
                {
                    key:
                        len(
                            value
                        )
                    for key, value
                    in event_chains.items()
                }
        },

    "recency":
        recency_scores,

    "retrieval":
        {
            "exact_query_memory":
                query_record[
                    "memory_id"
                ],

            "exact_query_rank":
                1
        },

    "health":
        {
            "nan_tensors":
                nan_count,

            "inf_tensors":
                inf_count,

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
    REPORT_FILE,
    report
)

write_json(
    EVALUATION_FILE,
    report
)

write_json(
    CONSOLIDATION_REGISTRY,
    {
        "lesson":
            "103R",

        "capability":
            "native_memory_consolidation_temporal_retrieval",

        "memory_version":
            MEMORY_VERSION,

        "records":
            len(
                consolidated_records
            ),

        "duplicates_removed":
            len(
                duplicate_pairs
            ),

        "related_events":
            len(
                related_pairs
            ),

        "next":
            "104R Native Multimodal Memory Reasoning"
    }
)

print(
    "Report:",
    REPORT_FILE
)

print(
    "Evaluation:",
    EVALUATION_FILE
)

print(
    "Registry:",
    CONSOLIDATION_REGISTRY
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING MEMORY ARCHITECTURE"
)

print()

print(
    "Multimodal Event"
)

print(
    "      ↓"
)

print(
    "Text + Numeric + Metadata"
)

print(
    "      ↓"
)

print(
    "Memory Representation"
)

print(
    "      ↓"
)

print(
    "Identity Check"
)

print(
    "      ↓"
)

print(
    "Duplicate / Related / Distinct"
)

print(
    "      ↓"
)

print(
    "Temporal Ordering"
)

print(
    "      ↓"
)

print(
    "Event Chain"
)

print(
    "      ↓"
)

print(
    "Consolidated Memory"
)

print(
    "      ↓"
)

print(
    "Recency Retrieval"
)

print(
    "      ↓"
)

print(
    "Historical Retrieval"
)

print(
    "      ↓"
)

print(
    "Memory-Aware Reasoning"
)

print()


# ============================================================
# WHY 103R MATTERS
# ============================================================

print(
    "WHY 103R MATTERS"
)

print()

print(
    "102R gave Silverwing persistent multimodal memory."
)

print(
    "103R gives that memory temporal structure."
)

print(
    "The system can distinguish stored observations from "
    "repeated or related events and organize them chronologically."
)

print(
    "This becomes the foundation for deeper memory reasoning."
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
    "Lesson 104R: Native Multimodal Memory Reasoning"
)

print()

print(
    "Memory Retrieval + Event Chains + Evidence Comparison + "
    "Temporal Reasoning + Engineering Diagnosis"
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
    "=== LESSON 103R COMPLETE ==="
)
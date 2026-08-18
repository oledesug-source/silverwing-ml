# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 109R
# Native Failure Pattern Clustering + Prototype Memory
# ============================================================
#
# 79R  -> Native Reasoning Dataset
# 80R  -> Native Reasoning Fine-Tuning
# 81R  -> Native Memory-Aware Training
# 82R  -> Native Tool-Aware Learning
# 83R  -> Native Planning + Tool Sequencing
# 84R  -> Native Verified Execution + Replanning
# 85R  -> Native Mathematical Reasoning
# 86R  -> Native Probability + Statistics
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
# 105R -> Native Memory Prediction + State Forecasting
# 106R -> Native Predictive Memory + Anomaly Detection
# 107R -> Native Predictive Risk + Failure Reasoning
# 108R -> Native Failure Pattern Memory + Retrieval
# 109R -> Native Failure Pattern Clustering + Prototype Memory
#
# ============================================================
# PURPOSE
# ============================================================
#
# 109R transforms the individual failure patterns created by
# 108R into reusable prototype memory.
#
# The pipeline is:
#
#   failure pattern memory
#          ↓
#   discover actual vector schema
#          ↓
#   validate pattern vectors
#          ↓
#   normalize representations
#          ↓
#   compute similarities
#          ↓
#   deterministic clustering
#          ↓
#   prototype construction
#          ↓
#   prototype retrieval
#          ↓
#   generalized risk reasoning
#          ↓
#   prototype memory persistence
#
# ============================================================
# CRITICAL COMPATIBILITY RULE
# ============================================================
#
# 108R is the source of truth for the pattern vector schema.
#
# DO NOT hard-code the dimensionality of the stored vectors.
#
# The previous 109R declared 13 dimensions while the actual
# 108R vectors contained:
#
#   normalized delta       5
#   normalized trend       5
#   magnitude signal       1
#   warning signal         1
#   historical risk        1
#   persistence signal     1
#
# Total:
#
#   14 dimensions
#
# This script discovers the actual dimension from the persisted
# 108R artifact and uses that dimension consistently.
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

MEMORY_VERSION = "109R.2"

CLUSTER_SIMILARITY_THRESHOLD = 0.70

TOP_K = 3

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8

MIN_CLUSTER_CONFIDENCE = 0.50


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_108R = (
        PHASE5_DIR /
        "lesson108R"
)

SOURCE_PATTERN_MEMORY_FILE = (
        LESSON_108R /
        "silverwing_failure_pattern_memory.json"
)

SOURCE_PATTERN_INDEX_FILE = (
        LESSON_108R /
        "silverwing_failure_pattern_index.pt"
)

SOURCE_PATTERN_DATASET_FILE = (
        LESSON_108R /
        "silverwing_failure_pattern_dataset.json"
)

SOURCE_PATTERN_REPORT_FILE = (
        LESSON_108R /
        "silverwing_failure_pattern_report.json"
)

SOURCE_PATTERN_REGISTRY_FILE = (
        LESSON_108R /
        "silverwing_failure_pattern_registry.json"
)

SOURCE_PATTERN_CHECKPOINT_PRIMARY = (
        LESSON_108R /
        "checkpoints" /
        "silverwing_failure_pattern_best.pt"
)

SOURCE_PATTERN_CHECKPOINT_CANDIDATE = (
        LESSON_108R /
        "checkpoints" /
        "silverwing_failure_pattern_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROTOTYPE_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_memory.json"
)

PROTOTYPE_INDEX_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_index.pt"
)

PROTOTYPE_DATASET_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_dataset.json"
)

PROTOTYPE_REPORT_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_report.json"
)

PROTOTYPE_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evaluation.json"
)

PROTOTYPE_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_failure_prototype_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_failure_prototype_best.pt"
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


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_PATTERN_CHECKPOINT_PRIMARY,
        SOURCE_PATTERN_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 108R checkpoint found."
    )


def clamp(
        value: float,
        lower: float = 0.0,
        upper: float = 1.0
) -> float:

    return max(
        lower,
        min(
            upper,
            float(value)
        )
    )


def construct_clusters(
        similarity_matrix: torch.Tensor,
        threshold: float
) -> List[int]:

    count = (
        similarity_matrix
        .shape[0]
    )

    assignments = [
        -1
        for _ in range(
            count
        )
    ]

    cluster_id = 0

    for start in range(
            count
    ):

        if assignments[
            start
        ] != -1:

            continue

        assignments[
            start
        ] = cluster_id

        queue = [
            start
        ]

        while queue:

            current = queue.pop(
                0
            )

            for candidate in range(
                    count
            ):

                if assignments[
                    candidate
                ] != -1:

                    continue

                score = float(
                    similarity_matrix[
                        current,
                        candidate
                    ]
                )

                if (
                        score
                        >=
                        threshold
                ):

                    assignments[
                        candidate
                    ] = cluster_id

                    queue.append(
                        candidate
                    )

        cluster_id += 1

    return assignments


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
    "PHASE 5 - LESSON 109R"
)

print(
    "Native Failure Pattern Clustering + Prototype Memory"
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
    "104R -> Multimodal Memory Reasoning",
    "105R -> Memory Prediction + State Forecasting",
    "106R -> Predictive Memory + Anomaly Detection",
    "107R -> Predictive Risk + Failure Reasoning",
    "108R -> Failure Pattern Memory + Retrieval",
    "109R -> Failure Pattern Clustering + Prototype Memory"
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
    "Cluster similarity threshold:",
    CLUSTER_SIMILARITY_THRESHOLD
)

print(
    "Top-K:",
    TOP_K
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 108R Pattern-Memory Inputs"
)

print()

for path in [
    SOURCE_PATTERN_MEMORY_FILE,
    SOURCE_PATTERN_INDEX_FILE,
    SOURCE_PATTERN_DATASET_FILE,
    SOURCE_PATTERN_REPORT_FILE,
    SOURCE_PATTERN_REGISTRY_FILE
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
    SOURCE_PATTERN_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_PATTERN_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_PATTERN_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_PATTERN_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_PATTERN_REGISTRY_FILE
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
    "TEST 2: Load Failure Pattern Memory"
)

print()

pattern_memory = read_json(
    SOURCE_PATTERN_MEMORY_FILE
)

pattern_dataset = read_json(
    SOURCE_PATTERN_DATASET_FILE
)

pattern_report = read_json(
    SOURCE_PATTERN_REPORT_FILE
)

if not isinstance(
        pattern_memory,
        dict
):

    raise RuntimeError(
        "108R pattern memory is invalid."
    )

patterns = pattern_memory.get(
    "patterns"
)

if not isinstance(
        patterns,
        list
):

    raise RuntimeError(
        "108R pattern memory contains no patterns."
    )

if not patterns:

    raise RuntimeError(
        "108R pattern memory is empty."
    )

print(
    "Memory version:",
    pattern_memory.get(
        "memory_version"
    )
)

print(
    "Pattern count:",
    len(
        patterns
    )
)

print(
    "Pattern dataset loaded:",
    bool(
        pattern_dataset
    )
)

print(
    "Pattern report loaded:",
    bool(
        pattern_report
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Discover Persisted Pattern Vector Schema"
)

print()

vector_lengths = []

for pattern in patterns:

    vector = pattern.get(
        "pattern_vector"
    )

    if not isinstance(
            vector,
            list
    ):

        raise RuntimeError(
            (
                "Pattern "
                f"{pattern.get('pattern_id', 'unknown')} "
                "does not contain a list pattern_vector."
            )
        )

    if not vector:

        raise RuntimeError(
            (
                "Pattern "
                f"{pattern.get('pattern_id', 'unknown')} "
                "contains an empty pattern_vector."
            )
        )

    vector_lengths.append(
        len(
            vector
        )
    )

actual_dimensions = set(
    vector_lengths
)

print(
    "Persisted vector lengths:",
    vector_lengths
)

print(
    "Unique vector dimensions:",
    sorted(
        actual_dimensions
    )
)

if len(
        actual_dimensions
) != 1:

    raise RuntimeError(
        (
            "108R contains inconsistent pattern-vector "
            "dimensions: "
            f"{sorted(actual_dimensions)}"
        )
    )

PATTERN_VECTOR_DIMENSION = (
    vector_lengths[
        0
    ]
)

print(
    "Discovered pattern vector dimension:",
    PATTERN_VECTOR_DIMENSION
)

print(
    "108R persisted schema accepted as source of truth."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Failure Pattern Schema"
)

print()

required_fields = {
    "pattern_id",
    "semantic_class",
    "from_memory",
    "to_memory",
    "delta",
    "trend",
    "change_magnitude",
    "pattern_vector"
}

schema_errors = []

for pattern in patterns:

    missing = (
            required_fields
            -
            set(
                pattern.keys()
            )
    )

    if missing:

        schema_errors.append(
            {
                "pattern_id":
                    pattern.get(
                        "pattern_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

        continue

    vector = pattern[
        "pattern_vector"
    ]

    if (
            not isinstance(
                vector,
                list
            )
            or
            len(vector)
            !=
            PATTERN_VECTOR_DIMENSION
    ):

        schema_errors.append(
            {
                "pattern_id":
                    pattern[
                        "pattern_id"
                    ],

                "error":
                    "pattern vector dimension mismatch",

                "actual":
                    len(
                        vector
                    )
                    if isinstance(
                        vector,
                        list
                    )
                    else
                    None,

                "expected":
                    PATTERN_VECTOR_DIMENSION
            }
        )

        continue

    if not all(
            math.isfinite(
                float(value)
            )
            for value
            in vector
    ):

        schema_errors.append(
            {
                "pattern_id":
                    pattern[
                        "pattern_id"
                    ],

                "error":
                    "pattern vector contains non-finite values"
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
        "109R failure pattern schema validation failed."
    )

print(
    "Failure pattern schema validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Native Pattern Matrix"
)

print()

pattern_vectors = torch.tensor(
    [
        pattern[
            "pattern_vector"
        ]
        for pattern
        in patterns
    ],
    dtype=torch.float32
)

print(
    "Pattern matrix:",
    tuple(
        pattern_vectors.shape
    )
)

if (
        pattern_vectors.ndim
        !=
        2
):

    raise RuntimeError(
        "Pattern matrix is not two-dimensional."
    )

if (
        pattern_vectors.shape[1]
        !=
        PATTERN_VECTOR_DIMENSION
):

    raise RuntimeError(
        "Pattern matrix dimension does not match discovered schema."
    )

if not finite_tensor(
        pattern_vectors
):

    raise RuntimeError(
        "Pattern matrix contains invalid values."
    )

normalized_patterns = F.normalize(
    pattern_vectors,
    p=2,
    dim=-1
)

if not finite_tensor(
        normalized_patterns
):

    raise RuntimeError(
        "Normalized pattern matrix contains invalid values."
    )

print(
    "Normalized pattern matrix:",
    tuple(
        normalized_patterns.shape
    )
)

print(
    "Native pattern matrix validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Pattern Similarity Matrix"
)

print()

pattern_similarity = torch.matmul(
    normalized_patterns,
    normalized_patterns.T
)

print(
    "Similarity matrix:",
    tuple(
        pattern_similarity.shape
    )
)

if not finite_tensor(
        pattern_similarity
):

    raise RuntimeError(
        "Pattern similarity matrix is invalid."
    )

print(
    "Pattern similarity matrix validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Construct Similarity-Connected Clusters"
)

print()

cluster_ids = construct_clusters(
    pattern_similarity,
    CLUSTER_SIMILARITY_THRESHOLD
)

cluster_count = (
        max(
            cluster_ids
        )
        +
        1
)

print(
    "Clusters discovered:",
    cluster_count
)

for cluster_id in range(
        cluster_count
):

    members = [
        patterns[
            index
        ][
            "pattern_id"
        ]
        for index in range(
            len(
                patterns
            )
        )
        if cluster_ids[
               index
           ]
           ==
           cluster_id
    ]

    print(
        f"cluster_{cluster_id + 1:03d}:",
        members
    )

if any(
        cluster_id < 0
        for cluster_id
        in cluster_ids
):

    raise RuntimeError(
        "Some failure patterns were not assigned."
    )

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Validate Cluster Assignment"
)

print()

if (
        len(
            cluster_ids
        )
        !=
        len(
            patterns
        )
):

    raise RuntimeError(
        "Cluster assignment count mismatch."
    )

for index, cluster_id in enumerate(
        cluster_ids
):

    print(
        patterns[
            index
        ][
            "pattern_id"
        ],
        "-> cluster",
        cluster_id
    )

print(
    "Every failure pattern has a valid cluster."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Build Cluster Prototypes"
)

print()

prototype_records = []

for cluster_id in range(
        cluster_count
):

    member_indices = [
        index
        for index in range(
            len(
                patterns
            )
        )
        if cluster_ids[
               index
           ]
           ==
           cluster_id
    ]

    member_matrix = normalized_patterns[
        member_indices
    ]

    prototype_vector = member_matrix.mean(
        dim=0
    )

    prototype_vector = F.normalize(
        prototype_vector,
        p=2,
        dim=0
    )

    member_similarity_scores = []

    for member_index in member_indices:

        member_similarity_scores.append(
            cosine_similarity(
                normalized_patterns[
                    member_index
                ],
                prototype_vector
            )
        )

    prototype_confidence = clamp(
        safe_mean(
            [
                (
                        score
                        +
                        1.0
                )
                /
                2.0
                for score
                in member_similarity_scores
            ]
        )
    )

    semantic_classes = [
        patterns[
            index
        ][
            "semantic_class"
        ]
        for index
        in member_indices
    ]

    prototype_records.append(
        {
            "prototype_id":
                f"prototype_{cluster_id + 1:03d}",

            "cluster_id":
                cluster_id,

            "member_pattern_ids":
                [
                    patterns[
                        index
                    ][
                        "pattern_id"
                    ]
                    for index
                    in member_indices
                ],

            "member_count":
                len(
                    member_indices
                ),

            "semantic_classes":
                semantic_classes,

            "prototype_vector":
                prototype_vector.tolist(),

            "prototype_dimension":
                PATTERN_VECTOR_DIMENSION,

            "prototype_confidence":
                prototype_confidence,

            "member_similarity_scores":
                member_similarity_scores
        }
    )

for prototype in prototype_records:

    print(
        prototype[
            "prototype_id"
        ],
        "| members=",
        prototype[
            "member_pattern_ids"
        ],
        "| confidence=",
        prototype[
            "prototype_confidence"
        ]
    )

if not prototype_records:

    raise RuntimeError(
        "No failure prototypes were created."
    )

print(
    "Failure prototypes constructed."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Validate Prototype Dimensions"
)

print()

prototype_errors = []

for prototype in prototype_records:

    vector = prototype[
        "prototype_vector"
    ]

    if len(
            vector
    ) != PATTERN_VECTOR_DIMENSION:

        prototype_errors.append(
            {
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "actual":
                    len(
                        vector
                    ),

                "expected":
                    PATTERN_VECTOR_DIMENSION
            }
        )

if prototype_errors:

    print(
        json.dumps(
            prototype_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Prototype dimension validation failed."
    )

print(
    "All prototype dimensions match persisted pattern dimensions."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Prototype Confidence Validation"
)

print()

confidence_errors = []

for prototype in prototype_records:

    confidence = float(
        prototype[
            "prototype_confidence"
        ]
    )

    if not (
            0.0
            <=
            confidence
            <=
            1.0
    ):

        confidence_errors.append(
            prototype[
                "prototype_id"
            ]
        )

if confidence_errors:

    print(
        confidence_errors
    )

    raise RuntimeError(
        "Prototype confidence validation failed."
    )

print(
    "Prototype confidences validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Identify Active Failure Pattern"
)

print()

active_pattern_id = (
    pattern_memory.get(
        "active_pattern"
    )
)

active_pattern_index = None

for index, pattern in enumerate(
        patterns
):

    if (
            pattern[
                "pattern_id"
            ]
            ==
            active_pattern_id
    ):

        active_pattern_index = index
        break

if active_pattern_index is None:

    active_pattern_index = (
            len(
                patterns
            )
            -
            1
    )

    active_pattern_id = patterns[
        active_pattern_index
    ][
        "pattern_id"
    ]

print(
    "Active pattern:",
    active_pattern_id
)

print(
    "Active pattern index:",
    active_pattern_index
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Retrieve Nearest Failure Prototypes"
)

print()

active_pattern_vector = normalized_patterns[
    active_pattern_index
]

prototype_results = []

for prototype in prototype_records:

    prototype_vector = torch.tensor(
        prototype[
            "prototype_vector"
        ],
        dtype=torch.float32
    )

    if (
            prototype_vector.shape[0]
            !=
            PATTERN_VECTOR_DIMENSION
    ):

        raise RuntimeError(
            "Prototype vector dimension mismatch during retrieval."
        )

    score = cosine_similarity(
        active_pattern_vector,
        prototype_vector
    )

    prototype_results.append(
        {
            "prototype_id":
                prototype[
                    "prototype_id"
                ],

            "cluster_id":
                prototype[
                    "cluster_id"
                ],

            "member_count":
                prototype[
                    "member_count"
                ],

            "confidence":
                prototype[
                    "prototype_confidence"
                ],

            "score":
                score
        }
    )

prototype_results.sort(
    key=lambda item:
    (
        item[
            "score"
        ],
        item[
            "prototype_id"
        ]
    ),
    reverse=True
)

for result in prototype_results[
    :
    min(
        TOP_K,
        len(
            prototype_results
        )
    )
]:

    print(
        result
    )

if not prototype_results:

    raise RuntimeError(
        "Prototype retrieval returned no result."
    )

nearest_prototype = prototype_results[
    0
]

print(
    "Nearest prototype:",
    nearest_prototype[
        "prototype_id"
    ]
)

print(
    "Prototype similarity:",
    nearest_prototype[
        "score"
    ]
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Prototype Generalization Score"
)

print()

retrieved_prototype = None

for prototype in prototype_records:

    if (
            prototype[
                "prototype_id"
            ]
            ==
            nearest_prototype[
                "prototype_id"
            ]
    ):

        retrieved_prototype = prototype
        break

if retrieved_prototype is None:

    raise RuntimeError(
        "Nearest prototype could not be resolved."
    )

member_scores = []

for pattern_id in retrieved_prototype[
    "member_pattern_ids"
]:

    for index, pattern in enumerate(
            patterns
    ):

        if (
                pattern[
                    "pattern_id"
                ]
                ==
                pattern_id
        ):

            member_scores.append(
                cosine_similarity(
                    active_pattern_vector,
                    normalized_patterns[
                        index
                    ]
                )
            )

generalization_score = clamp(
    safe_mean(
        [
            (
                    score
                    +
                    1.0
            )
            /
            2.0
            for score
            in member_scores
        ]
    )
)

print(
    "Retrieved prototype members:",
    retrieved_prototype[
        "member_pattern_ids"
    ]
)

print(
    "Active-to-member scores:",
    member_scores
)

print(
    "Generalization score:",
    generalization_score
)

if not math.isfinite(
        generalization_score
):

    raise RuntimeError(
        "Prototype generalization score is invalid."
    )

print(
    "Prototype generalization validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Cluster-Level Risk Evidence"
)

print()

cluster_risk_scores = []

for index, pattern in enumerate(
        patterns
):

    if (
            cluster_ids[
                index
            ]
            !=
            nearest_prototype[
                "cluster_id"
            ]
    ):

        continue

    warning_signal = (
        1.0
        if pattern.get(
            "warning_language",
            False
        )
        else
        0.0
    )

    magnitude_signal = clamp(
        float(
            pattern.get(
                "change_magnitude",
                0.0
            )
        )
        /
        10.0
    )

    cluster_risk_scores.append(
        safe_mean(
            [
                warning_signal,
                magnitude_signal
            ]
        )
    )

cluster_risk_score = clamp(
    safe_mean(
        cluster_risk_scores
    )
)

print(
    "Cluster risk observations:",
    cluster_risk_scores
)

print(
    "Cluster risk score:",
    cluster_risk_score
)

print(
    "Cluster-level risk evidence validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Prototype-Aware Case Reasoning"
)

print()

source_case_score = clamp(
    float(
        pattern_memory.get(
            "case_reasoning_score",
            0.0
        )
    )
)

prototype_similarity_score = clamp(
    (
            nearest_prototype[
                "score"
            ]
            +
            1.0
    )
    /
    2.0
)

prototype_confidence = clamp(
    float(
        nearest_prototype[
            "confidence"
        ]
    )
)

prototype_reasoning_score = safe_mean(
    [
        source_case_score,
        prototype_similarity_score,
        prototype_confidence,
        generalization_score,
        cluster_risk_score
    ]
)

print(
    "108R case reasoning score:",
    source_case_score
)

print(
    "Prototype similarity score:",
    prototype_similarity_score
)

print(
    "Prototype confidence:",
    prototype_confidence
)

print(
    "Generalization score:",
    generalization_score
)

print(
    "Cluster risk score:",
    cluster_risk_score
)

print(
    "Prototype-aware reasoning score:",
    prototype_reasoning_score
)

if not math.isfinite(
        prototype_reasoning_score
):

    raise RuntimeError(
        "Prototype-aware reasoning score is invalid."
    )

print(
    "Prototype-aware case reasoning validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Generate Prototype-Grounded Explanation"
)

print()

prototype_explanation = (
    "The active Silverwing failure pattern was compared "
    "with clustered historical failure prototypes. "
    "The nearest prototype summarizes related historical "
    "patterns and provides generalized evidence rather than "
    "claiming an identical physical failure."
)

if (
        nearest_prototype[
            "score"
        ]
        >=
        CLUSTER_SIMILARITY_THRESHOLD
):

    prototype_explanation += (
        " Prototype similarity exceeds the configured "
        "cluster threshold."
    )

else:

    prototype_explanation += (
        " Prototype similarity is below the configured "
        "cluster threshold, so historical generalization "
        "should be treated cautiously."
    )

print(
    prototype_explanation
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Deterministic Cluster Assignment"
)

print()

first_assignments = construct_clusters(
    pattern_similarity,
    CLUSTER_SIMILARITY_THRESHOLD
)

second_assignments = construct_clusters(
    pattern_similarity,
    CLUSTER_SIMILARITY_THRESHOLD
)

cluster_deterministic = (
        first_assignments
        ==
        second_assignments
)

print(
    "First assignments:",
    first_assignments
)

print(
    "Second assignments:",
    second_assignments
)

print(
    "Deterministic clustering:",
    cluster_deterministic
)

if not cluster_deterministic:

    raise RuntimeError(
        "Clustering is nondeterministic."
    )

print(
    "Deterministic clustering validated."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Deterministic Prototype Retrieval"
)

print()


def retrieve_prototypes(
        query_vector: torch.Tensor
) -> List[Dict[str, Any]]:

    results = []

    for prototype in prototype_records:

        prototype_vector = torch.tensor(
            prototype[
                "prototype_vector"
            ],
            dtype=torch.float32
        )

        results.append(
            {
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "score":
                    cosine_similarity(
                        query_vector,
                        prototype_vector
                    )
            }
        )

    results.sort(
        key=lambda item:
        (
            item[
                "score"
            ],
            item[
                "prototype_id"
            ]
        ),
        reverse=True
    )

    return results


first_retrieval = retrieve_prototypes(
    active_pattern_vector
)

second_retrieval = retrieve_prototypes(
    active_pattern_vector
)

retrieval_deterministic = (
        first_retrieval
        ==
        second_retrieval
)

print(
    "First retrieval:",
    first_retrieval
)

print(
    "Second retrieval:",
    second_retrieval
)

print(
    "Deterministic prototype retrieval:",
    retrieval_deterministic
)

if not retrieval_deterministic:

    raise RuntimeError(
        "Prototype retrieval is nondeterministic."
    )

print(
    "Deterministic prototype retrieval validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Prototype Curriculum"
)

print()

prototype_tasks = [
    {
        "example_id":
            "prototype_001",

        "domain":
            "pattern_clustering",

        "question":
            "Why cluster failure patterns?",

        "answer":
            "To organize related incidents into reusable behavioral groups."
    },

    {
        "example_id":
            "prototype_002",

        "domain":
            "prototype_memory",

        "question":
            "What is a failure prototype?",

        "answer":
            "A representative pattern constructed from related historical observations."
    },

    {
        "example_id":
            "prototype_003",

        "domain":
            "cluster_retrieval",

        "question":
            "Why retrieve prototypes?",

        "answer":
            "A prototype summarizes multiple related historical cases."
    },

    {
        "example_id":
            "prototype_004",

        "domain":
            "generalization",

        "question":
            "What does prototype generalization provide?",

        "answer":
            "It allows evidence from several related patterns to influence a new case."
    },

    {
        "example_id":
            "prototype_005",

        "domain":
            "risk_reasoning",

        "question":
            "How should prototype similarity affect risk reasoning?",

        "answer":
            "It should modify confidence according to evidence strength."
    },

    {
        "example_id":
            "prototype_006",

        "domain":
            "engineering_intelligence",

        "question":
            "Why are failure prototypes useful?",

        "answer":
            "They provide reusable representations of recurring engineering patterns."
    }
]

for task in prototype_tasks:

    print(
        task[
            "example_id"
        ],
        "->",
        task[
            "domain"
        ]
    )

print(
    "Prototype tasks:",
    len(
        prototype_tasks
    )
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Prototype Curriculum Coverage"
)

print()

expected_domains = {
    "pattern_clustering",
    "prototype_memory",
    "cluster_retrieval",
    "generalization",
    "risk_reasoning",
    "engineering_intelligence"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in prototype_tasks
}

print(
    "Domains:",
    sorted(
        actual_domains
    )
)

if (
        actual_domains
        !=
        expected_domains
):

    raise RuntimeError(
        "Prototype curriculum coverage is incomplete."
    )

print(
    "Prototype curriculum validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Numerical Health"
)

print()

health_tensors = [
    pattern_vectors,
    normalized_patterns,
    pattern_similarity
]

for prototype in prototype_records:

    health_tensors.append(
        torch.tensor(
            prototype[
                "prototype_vector"
            ],
            dtype=torch.float32
        )
    )

nan_count = 0
inf_count = 0

for tensor in health_tensors:

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
        "Prototype memory numerical health failed."
    )

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Final Prototype Memory Promotion Gate"
)

print()

promotion_errors = []

if not cluster_deterministic:

    promotion_errors.append(
        "Cluster assignment is nondeterministic."
    )

if not retrieval_deterministic:

    promotion_errors.append(
        "Prototype retrieval is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if not prototype_records:

    promotion_errors.append(
        "No prototypes were created."
    )

if len(
        prototype_tasks
) < 6:

    promotion_errors.append(
        "Prototype curriculum is incomplete."
    )

if not math.isfinite(
        prototype_reasoning_score
):

    promotion_errors.append(
        "Prototype reasoning score is invalid."
    )

if len(
        patterns
) < 2:

    promotion_errors.append(
        "Insufficient patterns for clustering."
    )

print(
    "Discovered pattern dimension:",
    PATTERN_VECTOR_DIMENSION
)

print(
    "Prototype dimension:",
    len(
        prototype_records[
            0
        ][
            "prototype_vector"
        ]
    )
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
        "109R prototype-memory promotion gate failed."
    )

print(
    "109R prototype-memory promotion gate passed."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Persist Prototype Memory"
)

print()

prototype_memory_payload = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_failure_pattern_clustering_prototype_memory",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        pattern_memory.get(
            "memory_version"
        ),

    "source_pattern_count":
        len(
            patterns
        ),

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "prototype_count":
        len(
            prototype_records
        ),

    "cluster_similarity_threshold":
        CLUSTER_SIMILARITY_THRESHOLD,

    "prototypes":
        prototype_records,

    "active_pattern":
        active_pattern_id,

    "nearest_prototype":
        nearest_prototype,

    "prototype_reasoning_score":
        prototype_reasoning_score,

    "prototype_explanation":
        prototype_explanation
}

write_json(
    PROTOTYPE_MEMORY_FILE,
    prototype_memory_payload
)

prototype_matrix = torch.stack(
    [
        torch.tensor(
            prototype[
                "prototype_vector"
            ],
            dtype=torch.float32
        )
        for prototype
        in prototype_records
    ],
    dim=0
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "pattern_vector_dimension":
            PATTERN_VECTOR_DIMENSION,

        "prototype_ids":
            [
                prototype[
                    "prototype_id"
                ]
                for prototype
                in prototype_records
            ],

        "prototype_matrix":
            prototype_matrix,

        "cluster_ids":
            cluster_ids
    },
    PROTOTYPE_INDEX_FILE
)

print(
    "Prototype memory:",
    PROTOTYPE_MEMORY_FILE
)

print(
    "Prototype index:",
    PROTOTYPE_INDEX_FILE
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Reload Prototype Memory"
)

print()

reloaded = read_json(
    PROTOTYPE_MEMORY_FILE
)

if (
        reloaded[
            "pattern_vector_dimension"
        ]
        !=
        PATTERN_VECTOR_DIMENSION
):

    raise RuntimeError(
        "Persisted prototype dimension changed."
    )

if (
        reloaded[
            "prototype_count"
        ]
        !=
        len(
            prototype_records
        )
):

    raise RuntimeError(
        "Prototype count changed after persistence."
    )

reloaded_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in reloaded[
        "prototypes"
    ]
]

current_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in prototype_records
]

if reloaded_ids != current_ids:

    raise RuntimeError(
        "Prototype identity changed after persistence."
    )

print(
    "Reloaded prototype dimension:",
    reloaded[
        "pattern_vector_dimension"
    ]
)

print(
    "Reloaded prototypes:",
    len(
        reloaded[
            "prototypes"
        ]
    )
)

print(
    "Persistent prototype memory validated."
)

print()


# ============================================================
# TEST 26
# ============================================================

print(
    "TEST 26: Save Prototype Dataset"
)

print()

prototype_dataset = {
    "lesson":
        "109R",

    "capability":
        "native_failure_pattern_clustering_prototype_memory",

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "pattern_count":
        len(
            patterns
        ),

    "cluster_count":
        cluster_count,

    "cluster_assignments":
        [
            {
                "pattern_id":
                    patterns[
                        index
                    ][
                        "pattern_id"
                    ],

                "cluster_id":
                    cluster_ids[
                        index
                    ]
            }
            for index in range(
            len(
                patterns
            )
        )
        ],

    "prototypes":
        prototype_records,

    "active_pattern":
        active_pattern_id,

    "nearest_prototype":
        nearest_prototype,

    "prototype_reasoning_score":
        prototype_reasoning_score,

    "prototype_explanation":
        prototype_explanation
}

write_json(
    PROTOTYPE_DATASET_FILE,
    prototype_dataset
)

print(
    "Prototype dataset:",
    PROTOTYPE_DATASET_FILE
)

print()


# ============================================================
# TEST 27
# ============================================================

print(
    "TEST 27: Save 109R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "109R",

    "capability":
        "native_failure_pattern_clustering_prototype_memory",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "pattern_count":
        len(
            patterns
        ),

    "prototype_count":
        len(
            prototype_records
        ),

    "cluster_assignments":
        cluster_ids,

    "prototypes":
        prototype_records,

    "active_pattern":
        active_pattern_id,

    "nearest_prototype":
        nearest_prototype,

    "prototype_reasoning_score":
        prototype_reasoning_score,

    "prototype_explanation":
        prototype_explanation,

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
# TEST 28
# ============================================================

print(
    "TEST 28: Write 109R Reports"
)

print()

report = {
    "lesson":
        "109R",

    "capability":
        "native_failure_pattern_clustering_prototype_memory",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "pattern_count":
        len(
            patterns
        ),

    "prototype_count":
        len(
            prototype_records
        ),

    "cluster_similarity_threshold":
        CLUSTER_SIMILARITY_THRESHOLD,

    "clusters":
        [
            {
                "cluster_id":
                    cluster_id,

                "pattern_ids":
                    [
                        patterns[
                            index
                        ][
                            "pattern_id"
                        ]
                        for index in range(
                        len(
                            patterns
                        )
                    )
                        if cluster_ids[
                               index
                           ]
                           ==
                           cluster_id
                    ]
            }
            for cluster_id
            in range(
            cluster_count
        )
        ],

    "prototypes":
        prototype_records,

    "retrieval":
        {
            "active_pattern":
                active_pattern_id,

            "nearest":
                prototype_results[
                    :
                    TOP_K
                ]
        },

    "reasoning":
        {
            "generalization_score":
                generalization_score,

            "cluster_risk_score":
                cluster_risk_score,

            "prototype_reasoning_score":
                prototype_reasoning_score
        },

    "verification":
        {
            "cluster_deterministic":
                cluster_deterministic,

            "retrieval_deterministic":
                retrieval_deterministic,

            "prototype_dimension":
                PATTERN_VECTOR_DIMENSION
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
    PROTOTYPE_REPORT_FILE,
    report
)

write_json(
    PROTOTYPE_EVALUATION_FILE,
    report
)

write_json(
    PROTOTYPE_REGISTRY_FILE,
    {
        "lesson":
            "109R",

        "capability":
            "native_failure_pattern_clustering_prototype_memory",

        "memory_version":
            MEMORY_VERSION,

        "pattern_vector_dimension":
            PATTERN_VECTOR_DIMENSION,

        "pattern_count":
            len(
                patterns
            ),

        "prototype_count":
            len(
                prototype_records
            ),

        "prototype_reasoning_score":
            prototype_reasoning_score,

        "next":
            "110R Native Failure Prototype Evolution + Continual Memory"
    }
)

print(
    "Prototype report:",
    PROTOTYPE_REPORT_FILE
)

print(
    "Prototype evaluation:",
    PROTOTYPE_EVALUATION_FILE
)

print(
    "Prototype registry:",
    PROTOTYPE_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 109R FAILURE PROTOTYPE ARCHITECTURE"
)

print()

print(
    "Failure Pattern Memory"
)

print(
    "        ↓"
)

print(
    "Discover Persisted Pattern Schema"
)

print(
    "        ↓"
)

print(
    "Pattern Similarity"
)

print(
    "        ↓"
)

print(
    "Similarity-Connected Clusters"
)

print(
    "        ↓"
)

print(
    "Prototype Construction"
)

print(
    "        ↓"
)

print(
    "Prototype Memory"
)

print(
    "        ↓"
)

print(
    "Cluster-Aware Retrieval"
)

print(
    "        ↓"
)

print(
    "Prototype Evidence"
)

print(
    "        ↓"
)

print(
    "Generalized Risk Reasoning"
)

print()


# ============================================================
# WHY 109R MATTERS
# ============================================================

print(
    "WHY 109R MATTERS"
)

print()

print(
    "108R stored individual failure patterns."
)

print(
    "109R organizes those patterns into reusable prototypes."
)

print()

print(
    "The prototype layer provides an abstraction above individual "
    "cases while preserving source pattern identity."
)

print()

print(
    "This is the foundation for continual prototype evolution."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "109R LIMITATION"
)

print()

print(
    "The current pattern memory contains only a small number "
    "of controlled examples."
)

print(
    "Therefore the clustering system establishes the architecture "
    "and validation contract rather than proving production-grade "
    "failure-mode discovery."
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
    "Lesson 110R: Native Failure Prototype Evolution + Continual Memory"
)

print()

print(
    "New Incident + Existing Prototype + Prototype Update + "
    "Novelty Detection + Memory Evolution"
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
    "83R Native Planning + Tool Sequencing",
    " ↓",
    "84R Native Verified Execution + Replanning",
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
    "106R Native Predictive Memory + Anomaly Detection",
    " ↓",
    "107R Native Predictive Risk + Failure Reasoning",
    " ↓",
    "108R Native Failure Pattern Memory + Retrieval",
    " ↓",
    "109R Native Failure Pattern Clustering + Prototype Memory",
    " ↓",
    "110R Native Failure Prototype Evolution + Continual Memory",
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
    "=== LESSON 109R COMPLETE ==="
)
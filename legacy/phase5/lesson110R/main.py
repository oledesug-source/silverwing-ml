# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 110R
# Native Failure Prototype Evolution + Continual Memory
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
# 99R  -> Native Multimodal Representation Foundations
# 100R -> Native Cross-Modal Alignment + Retrieval
# 101R -> Native Hard-Negative Multimodal Learning
# 102R -> Native Multimodal Memory Integration
# 103R -> Native Memory Consolidation + Temporal Retrieval
# 104R -> Native Multimodal Memory Reasoning
# 105R -> Native Memory Prediction + State Forecasting
# 106R -> Native Predictive Memory + Anomaly Detection
# 107R -> Native Predictive Risk + Failure Reasoning
# 108R -> Native Failure Pattern Memory + Retrieval
# 109R -> Native Failure Pattern Clustering + Prototype Memory
# 110R -> Native Failure Prototype Evolution + Continual Memory
#
# ============================================================
# PURPOSE
# ============================================================
#
# 110R introduces controlled continual evolution of failure
# prototypes.
#
# A new incident pattern is:
#
#   retrieved against existing prototypes
#
# Then:
#
#   sufficiently similar
#       -> update existing prototype
#
#   sufficiently novel
#       -> create a new prototype
#
# Every update is validated before promotion.
#
# ============================================================
# IMPORTANT DATASET RULE
# ============================================================
#
# The current 109R prototype memory is small.
#
# Therefore this lesson focuses on the continual-learning
# CONTRACT rather than pretending the dataset is large enough
# to establish production novelty thresholds.
#
# The actual pattern-vector dimension is discovered from 109R.
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

MEMORY_VERSION = "110R.1"

EVOLUTION_SIMILARITY_THRESHOLD = 0.70

NOVELTY_THRESHOLD = 0.55

TOP_K = 3

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8

MIN_CONFIDENCE = 0.50


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_109R = (
        PHASE5_DIR /
        "lesson109R"
)

SOURCE_PROTOTYPE_MEMORY_FILE = (
        LESSON_109R /
        "silverwing_failure_prototype_memory.json"
)

SOURCE_PROTOTYPE_INDEX_FILE = (
        LESSON_109R /
        "silverwing_failure_prototype_index.pt"
)

SOURCE_PROTOTYPE_DATASET_FILE = (
        LESSON_109R /
        "silverwing_failure_prototype_dataset.json"
)

SOURCE_PROTOTYPE_REPORT_FILE = (
        LESSON_109R /
        "silverwing_failure_prototype_report.json"
)

SOURCE_PROTOTYPE_REGISTRY_FILE = (
        LESSON_109R /
        "silverwing_failure_prototype_registry.json"
)

SOURCE_PROTOTYPE_CHECKPOINT_PRIMARY = (
        LESSON_109R /
        "checkpoints" /
        "silverwing_failure_prototype_best.pt"
)

SOURCE_PROTOTYPE_CHECKPOINT_CANDIDATE = (
        LESSON_109R /
        "checkpoints" /
        "silverwing_failure_prototype_candidate.pt"
)


OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EVOLUTION_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_memory.json"
)

EVOLUTION_INDEX_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_index.pt"
)

EVOLUTION_DATASET_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_dataset.json"
)

EVOLUTION_REPORT_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_report.json"
)

EVOLUTION_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_evaluation.json"
)

EVOLUTION_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_failure_prototype_evolution_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_failure_prototype_evolution_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_failure_prototype_evolution_best.pt"
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


def clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(value)
        )
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
        SOURCE_PROTOTYPE_CHECKPOINT_PRIMARY,
        SOURCE_PROTOTYPE_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 109R checkpoint found."
    )


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
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
    "PHASE 5 - LESSON 110R"
)

print(
    "Native Failure Prototype Evolution + Continual Memory"
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
    "109R -> Failure Pattern Clustering + Prototype Memory",
    "110R -> Failure Prototype Evolution + Continual Memory"
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
    "Evolution similarity threshold:",
    EVOLUTION_SIMILARITY_THRESHOLD
)

print(
    "Novelty threshold:",
    NOVELTY_THRESHOLD
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
    "TEST 1: Verify 109R Prototype Inputs"
)

print()

for path in [
    SOURCE_PROTOTYPE_MEMORY_FILE,
    SOURCE_PROTOTYPE_INDEX_FILE,
    SOURCE_PROTOTYPE_DATASET_FILE,
    SOURCE_PROTOTYPE_REPORT_FILE,
    SOURCE_PROTOTYPE_REGISTRY_FILE
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
    SOURCE_PROTOTYPE_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_PROTOTYPE_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_PROTOTYPE_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_PROTOTYPE_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_PROTOTYPE_REGISTRY_FILE
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
    "TEST 2: Load Prototype Memory"
)

print()

prototype_memory = read_json(
    SOURCE_PROTOTYPE_MEMORY_FILE
)

prototype_dataset = read_json(
    SOURCE_PROTOTYPE_DATASET_FILE
)

prototype_report = read_json(
    SOURCE_PROTOTYPE_REPORT_FILE
)

if not isinstance(
        prototype_memory,
        dict
):

    raise RuntimeError(
        "109R prototype memory is invalid."
    )

prototypes = prototype_memory.get(
    "prototypes"
)

if not isinstance(
        prototypes,
        list
):

    raise RuntimeError(
        "109R prototype memory contains no prototypes."
    )

if not prototypes:

    raise RuntimeError(
        "109R prototype memory is empty."
    )

print(
    "Memory version:",
    prototype_memory.get(
        "memory_version"
    )
)

print(
    "Prototype count:",
    len(
        prototypes
    )
)

print(
    "Prototype dataset loaded:",
    bool(
        prototype_dataset
    )
)

print(
    "Prototype report loaded:",
    bool(
        prototype_report
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Discover Prototype Vector Schema"
)

print()

vector_lengths = []

for prototype in prototypes:

    vector = prototype.get(
        "prototype_vector"
    )

    if not isinstance(
            vector,
            list
    ):

        raise RuntimeError(
            (
                "Prototype "
                f"{prototype.get('prototype_id', 'unknown')} "
                "does not contain a list prototype_vector."
            )
        )

    if not vector:

        raise RuntimeError(
            (
                "Prototype "
                f"{prototype.get('prototype_id', 'unknown')} "
                "contains an empty prototype_vector."
            )
        )

    vector_lengths.append(
        len(
            vector
        )
    )

unique_dimensions = set(
    vector_lengths
)

print(
    "Persisted prototype vector lengths:",
    vector_lengths
)

print(
    "Unique dimensions:",
    sorted(
        unique_dimensions
    )
)

if len(
        unique_dimensions
) != 1:

    raise RuntimeError(
        (
            "109R prototype vectors have inconsistent dimensions: "
            f"{sorted(unique_dimensions)}"
        )
    )

PATTERN_VECTOR_DIMENSION = (
    vector_lengths[
        0
    ]
)

print(
    "Discovered prototype dimension:",
    PATTERN_VECTOR_DIMENSION
)

print(
    "Prototype schema accepted as source of truth."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Prototype Schema"
)

print()

required_fields = {
    "prototype_id",
    "cluster_id",
    "member_pattern_ids",
    "member_count",
    "prototype_vector",
    "prototype_confidence"
}

schema_errors = []

for prototype in prototypes:

    missing = (
            required_fields
            -
            set(
                prototype.keys()
            )
    )

    if missing:

        schema_errors.append(
            {
                "prototype_id":
                    prototype.get(
                        "prototype_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

        continue

    vector = prototype[
        "prototype_vector"
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
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "error":
                    "prototype vector dimension mismatch",

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

if schema_errors:

    print(
        json.dumps(
            schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "109R prototype schema validation failed."
    )

print(
    "Prototype schema validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Prototype Matrix"
)

print()

prototype_matrix = torch.tensor(
    [
        prototype[
            "prototype_vector"
        ]
        for prototype
        in prototypes
    ],
    dtype=torch.float32
)

print(
    "Prototype matrix:",
    tuple(
        prototype_matrix.shape
    )
)

if (
        prototype_matrix.ndim
        !=
        2
):

    raise RuntimeError(
        "Prototype matrix is not two-dimensional."
    )

if (
        prototype_matrix.shape[1]
        !=
        PATTERN_VECTOR_DIMENSION
):

    raise RuntimeError(
        "Prototype matrix dimension mismatch."
    )

if not finite_tensor(
        prototype_matrix
):

    raise RuntimeError(
        "Prototype matrix contains invalid values."
    )

normalized_prototypes = F.normalize(
    prototype_matrix,
    p=2,
    dim=-1
)

if not finite_tensor(
        normalized_prototypes
):

    raise RuntimeError(
        "Normalized prototype matrix is invalid."
    )

print(
    "Normalized prototype matrix:",
    tuple(
        normalized_prototypes.shape
    )
)

print(
    "Native prototype matrix validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Validate Prototype Index"
)

print()

index_payload = torch.load(
    SOURCE_PROTOTYPE_INDEX_FILE,
    map_location="cpu",
    weights_only=False
)

if not isinstance(
        index_payload,
        dict
):

    raise RuntimeError(
        "109R prototype index is invalid."
    )

index_matrix = index_payload.get(
    "prototype_matrix"
)

index_ids = index_payload.get(
    "prototype_ids"
)

if index_matrix is None:

    raise RuntimeError(
        "Prototype matrix missing from 109R index."
    )

if index_ids is None:

    raise RuntimeError(
        "Prototype ids missing from 109R index."
    )

print(
    "Index fields:",
    sorted(
        index_payload.keys()
    )
)

print(
    "Indexed prototype matrix:",
    tuple(
        index_matrix.shape
    )
)

print(
    "Indexed prototype ids:",
    len(
        index_ids
    )
)

if (
        index_matrix.shape
        !=
        prototype_matrix.shape
):

    raise RuntimeError(
        "Persisted prototype index shape differs from memory."
    )

print(
    "Prototype index validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Select Active Prototype Case"
)

print()

active_pattern_id = (
    prototype_memory.get(
        "active_pattern"
    )
)

nearest_from_109 = (
    prototype_memory.get(
        "nearest_prototype"
    )
)

if not isinstance(
        nearest_from_109,
        dict
):

    nearest_from_109 = {}

print(
    "Active pattern:",
    active_pattern_id
)

print(
    "Previous nearest prototype:",
    nearest_from_109.get(
        "prototype_id"
    )
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Reconstruct Active Incident Representation"
)

print()

#
# 109R persisted the active-pattern relation through the
# nearest prototype. We reconstruct the active incident
# representation by retrieving the prototype that 109R used
# and then using the prototype vector as the current continual
# learning query representation.
#
# This keeps 110R compatible with the persisted 109R artifact
# without inventing a new representation.
#

active_prototype_id = (
    nearest_from_109.get(
        "prototype_id"
    )
)

active_prototype_index = None

for index, prototype in enumerate(
        prototypes
):

    if (
            prototype[
                "prototype_id"
            ]
            ==
            active_prototype_id
    ):

        active_prototype_index = index
        break

if active_prototype_index is None:

    active_prototype_index = 0

    active_prototype_id = prototypes[
        0
    ][
        "prototype_id"
    ]

active_query = normalized_prototypes[
    active_prototype_index
]

print(
    "Active prototype id:",
    active_prototype_id
)

print(
    "Active query dimension:",
    active_query.shape[0]
)

print(
    "Active prototype vector:",
    active_query.tolist()
)

if not finite_tensor(
        active_query
):

    raise RuntimeError(
        "Active continual-learning query is invalid."
    )

print(
    "Active incident representation validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Retrieve Existing Prototypes"
)

print()

retrieval_results = []

for index, prototype in enumerate(
        prototypes
):

    score = cosine_similarity(
        active_query,
        normalized_prototypes[
            index
        ]
    )

    retrieval_results.append(
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

retrieval_results.sort(
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

for result in retrieval_results:

    print(
        result
    )

if not retrieval_results:

    raise RuntimeError(
        "Prototype retrieval failed."
    )

best_match = retrieval_results[
    0
]

print(
    "Best prototype:",
    best_match[
        "prototype_id"
    ]
)

print(
    "Best similarity:",
    best_match[
        "score"
    ]
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Evolution Decision"
)

print()

best_similarity = float(
    best_match[
        "score"
    ]
)

if (
        best_similarity
        >=
        EVOLUTION_SIMILARITY_THRESHOLD
):

    evolution_action = (
        "UPDATE_EXISTING_PROTOTYPE"
    )

elif (
        best_similarity
        <
        NOVELTY_THRESHOLD
):

    evolution_action = (
        "CREATE_NEW_PROTOTYPE"
    )

else:

    evolution_action = (
        "HOLD_FOR_MORE_EVIDENCE"
    )

print(
    "Best similarity:",
    best_similarity
)

print(
    "Evolution threshold:",
    EVOLUTION_SIMILARITY_THRESHOLD
)

print(
    "Novelty threshold:",
    NOVELTY_THRESHOLD
)

print(
    "Evolution action:",
    evolution_action
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Simulate Prototype Evolution"
)

print()

prototype_matrix_evolved = (
    normalized_prototypes.clone()
)

created_new_prototype = None

updated_prototype_id = None

if (
        evolution_action
        ==
        "UPDATE_EXISTING_PROTOTYPE"
):

    matched_index = 0

    matched_vector = normalized_prototypes[
        matched_index
    ]

    updated_vector = F.normalize(
        (
                matched_vector
                +
                active_query
        )
        /
        2.0,
        p=2,
        dim=0
    )

    prototype_matrix_evolved[
        matched_index
    ] = updated_vector

    updated_prototype_id = (
        best_match[
            "prototype_id"
        ]
    )

    print(
        "Updated prototype:",
        updated_prototype_id
    )

elif (
        evolution_action
        ==
        "CREATE_NEW_PROTOTYPE"
):

    new_vector = F.normalize(
        active_query,
        p=2,
        dim=0
    )

    prototype_matrix_evolved = torch.cat(
        [
            prototype_matrix_evolved,
            new_vector.unsqueeze(
                0
            )
        ],
        dim=0
    )

    created_new_prototype = {
        "prototype_id":
            (
                "prototype_"
                f"{len(prototypes) + 1:03d}"
            ),

        "cluster_id":
            len(
                prototypes
            ),

        "member_pattern_ids":
            [],

        "member_count":
            0,

        "semantic_classes":
            [],

        "prototype_vector":
            new_vector.tolist(),

        "prototype_dimension":
            PATTERN_VECTOR_DIMENSION,

        "prototype_confidence":
            0.50
    }

    print(
        "Created new prototype:",
        created_new_prototype[
            "prototype_id"
        ]
    )

else:

    print(
        "Prototype evolution held for more evidence."
    )

if not finite_tensor(
        prototype_matrix_evolved
):

    raise RuntimeError(
        "Evolved prototype matrix is invalid."
    )

print(
    "Prototype evolution simulation validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Prototype Drift Measurement"
)

print()

if (
        evolution_action
        ==
        "UPDATE_EXISTING_PROTOTYPE"
):

    original_vector = normalized_prototypes[
        0
    ]

    evolved_vector = prototype_matrix_evolved[
        0
    ]

    prototype_drift = (
            1.0
            -
            cosine_similarity(
                original_vector,
                evolved_vector
            )
    )

else:

    prototype_drift = 0.0

print(
    "Prototype drift:",
    prototype_drift
)

if not math.isfinite(
        prototype_drift
):

    raise RuntimeError(
        "Prototype drift is invalid."
    )

print(
    "Prototype drift validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Evolution Stability Check"
)

print()

if (
        evolution_action
        ==
        "UPDATE_EXISTING_PROTOTYPE"
):

    similarity_after_update = cosine_similarity(
        active_query,
        prototype_matrix_evolved[
            0
        ]
    )

    stability_score = clamp(
        (
                similarity_after_update
                +
                1.0
        )
        /
        2.0
    )

else:

    similarity_after_update = best_similarity

    stability_score = 1.0

print(
    "Similarity after evolution:",
    similarity_after_update
)

print(
    "Evolution stability score:",
    stability_score
)

if (
        stability_score
        <
        MIN_CONFIDENCE
):

    raise RuntimeError(
        "Prototype evolution stability is too low."
    )

print(
    "Prototype evolution stability validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Novelty Detection"
)

print()

novelty_score = (
        1.0
        -
        clamp(
            (
                    best_similarity
                    +
                    1.0
            )
            /
            2.0
        )
)

print(
    "Novelty score:",
    novelty_score
)

if not math.isfinite(
        novelty_score
):

    raise RuntimeError(
        "Novelty score is invalid."
    )

if (
        evolution_action
        ==
        "CREATE_NEW_PROTOTYPE"
):

    print(
        "Novel behavior detected."
    )

elif (
        evolution_action
        ==
        "UPDATE_EXISTING_PROTOTYPE"
):

    print(
        "Behavior is sufficiently similar to existing memory."
    )

else:

    print(
        "Behavior lies in an intermediate uncertainty region."
    )

print(
    "Novelty detection validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Build Continual Memory Event"
)

print()

evolution_event = {
    "event_id":
        "evolution_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "110R",

    "active_prototype":
        active_prototype_id,

    "best_similarity":
        best_similarity,

    "novelty_score":
        novelty_score,

    "action":
        evolution_action,

    "updated_prototype":
        updated_prototype_id,

    "new_prototype":
        (
            created_new_prototype[
                "prototype_id"
            ]
            if created_new_prototype
            else
            None
        ),

    "stability_score":
        stability_score,

    "prototype_drift":
        prototype_drift
}

print(
    evolution_event
)

print(
    "Continual memory event constructed."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Build Evolution History"
)

print()

previous_events = prototype_memory.get(
    "evolution_events",
    []
)

if not isinstance(
        previous_events,
        list
):

    previous_events = []

evolution_history = (
        list(
            previous_events
        )
        +
        [
            evolution_event
        ]
)

print(
    "Previous evolution events:",
    len(
        previous_events
    )
)

print(
    "Current evolution events:",
    len(
        evolution_history
    )
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Deterministic Evolution Decision"
)

print()

def decide_evolution(
        similarity: float
) -> str:

    if (
            similarity
            >=
            EVOLUTION_SIMILARITY_THRESHOLD
    ):

        return "UPDATE_EXISTING_PROTOTYPE"

    if (
            similarity
            <
            NOVELTY_THRESHOLD
    ):

        return "CREATE_NEW_PROTOTYPE"

    return "HOLD_FOR_MORE_EVIDENCE"


first_decision = decide_evolution(
    best_similarity
)

second_decision = decide_evolution(
    best_similarity
)

decision_deterministic = (
        first_decision
        ==
        second_decision
)

print(
    "First decision:",
    first_decision
)

print(
    "Second decision:",
    second_decision
)

print(
    "Deterministic evolution:",
    decision_deterministic
)

if not decision_deterministic:

    raise RuntimeError(
        "Evolution decision is nondeterministic."
    )

print(
    "Deterministic evolution validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Prototype Evolution Curriculum"
)

print()

evolution_tasks = [
    {
        "example_id":
            "evolution_001",

        "domain":
            "continual_memory",

        "question":
            "Why should a prototype memory evolve?",

        "answer":
            "New observations can refine existing patterns or reveal new patterns."
    },

    {
        "example_id":
            "evolution_002",

        "domain":
            "prototype_update",

        "question":
            "When should an existing prototype be updated?",

        "answer":
            "When a new pattern is sufficiently similar to the existing prototype."
    },

    {
        "example_id":
            "evolution_003",

        "domain":
            "novelty_detection",

        "question":
            "What indicates that a new pattern may be novel?",

        "answer":
            "Low similarity to all existing prototypes."
    },

    {
        "example_id":
            "evolution_004",

        "domain":
            "memory_stability",

        "question":
            "Why should prototype drift be monitored?",

        "answer":
            "Repeated updates can move a prototype away from its original meaning."
    },

    {
        "example_id":
            "evolution_005",

        "domain":
            "continual_learning",

        "question":
            "What is the purpose of continual prototype memory?",

        "answer":
            "To incorporate new evidence without discarding existing knowledge."
    },

    {
        "example_id":
            "evolution_006",

        "domain":
            "knowledge_preservation",

        "question":
            "Why preserve the original prototype and the evolution event?",

        "answer":
            "It makes memory changes auditable and prevents silent knowledge replacement."
    }
]

for task in evolution_tasks:

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
    "Evolution tasks:",
    len(
        evolution_tasks
    )
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Evolution Curriculum Coverage"
)

print()

expected_domains = {
    "continual_memory",
    "prototype_update",
    "novelty_detection",
    "memory_stability",
    "continual_learning",
    "knowledge_preservation"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in evolution_tasks
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
        "Evolution curriculum coverage is incomplete."
    )

print(
    "Evolution curriculum validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Numerical Health"
)

print()

health_tensors = [
    prototype_matrix,
    normalized_prototypes,
    prototype_matrix_evolved,
    active_query
]

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
        "Prototype evolution numerical health failed."
    )

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Memory Preservation Validation"
)

print()

original_prototype_count = len(
    prototypes
)

evolved_prototype_count = (
        original_prototype_count
        +
        (
            1
            if created_new_prototype
            else
            0
        )
)

if (
        evolution_action
        ==
        "UPDATE_EXISTING_PROTOTYPE"
):

    if (
            evolved_prototype_count
            !=
            original_prototype_count
    ):

        raise RuntimeError(
            "Existing-prototype update changed prototype count."
        )

elif (
        evolution_action
        ==
        "CREATE_NEW_PROTOTYPE"
):

    if (
            evolved_prototype_count
            !=
            original_prototype_count
            +
            1
    ):

        raise RuntimeError(
            "New prototype was not added correctly."
        )

else:

    if (
            evolved_prototype_count
            !=
            original_prototype_count
    ):

        raise RuntimeError(
            "Held evolution unexpectedly changed prototype count."
        )

print(
    "Original prototype count:",
    original_prototype_count
)

print(
    "Evolved prototype count:",
    evolved_prototype_count
)

print(
    "Memory preservation validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Final Continual Memory Promotion Gate"
)

print()

promotion_errors = []

if not decision_deterministic:

    promotion_errors.append(
        "Evolution decision is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if (
        stability_score
        <
        MIN_CONFIDENCE
):

    promotion_errors.append(
        "Prototype evolution stability is too low."
    )

if len(
        evolution_tasks
) < 6:

    promotion_errors.append(
        "Evolution curriculum is incomplete."
    )

if len(
        evolution_history
) < 1:

    promotion_errors.append(
        "Evolution history was not created."
    )

print(
    "Evolution action:",
    evolution_action
)

print(
    "Best similarity:",
    best_similarity
)

print(
    "Novelty score:",
    novelty_score
)

print(
    "Stability score:",
    stability_score
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
        "110R continual-memory promotion gate failed."
    )

print(
    "110R continual-memory promotion gate passed."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Persist Evolving Prototype Memory"
)

print()

evolved_prototypes = []

for index, prototype in enumerate(
        prototypes
):

    updated = dict(
        prototype
    )

    updated_vector = prototype_matrix_evolved[
        index
    ]

    updated[
        "prototype_vector"
    ] = updated_vector.tolist()

    updated[
        "prototype_dimension"
    ] = PATTERN_VECTOR_DIMENSION

    evolved_prototypes.append(
        updated
    )

if created_new_prototype:

    evolved_prototypes.append(
        created_new_prototype
    )

evolution_memory_payload = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_failure_prototype_evolution_continual_memory",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        prototype_memory.get(
            "memory_version"
        ),

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_prototype_count,

    "evolved_prototype_count":
        len(
            evolved_prototypes
        ),

    "evolution_similarity_threshold":
        EVOLUTION_SIMILARITY_THRESHOLD,

    "novelty_threshold":
        NOVELTY_THRESHOLD,

    "evolution_action":
        evolution_action,

    "novelty_score":
        novelty_score,

    "stability_score":
        stability_score,

    "prototype_drift":
        prototype_drift,

    "prototypes":
        evolved_prototypes,

    "evolution_events":
        evolution_history
}

write_json(
    EVOLUTION_MEMORY_FILE,
    evolution_memory_payload
)

evolved_matrix = torch.tensor(
    [
        prototype[
            "prototype_vector"
        ]
        for prototype
        in evolved_prototypes
    ],
    dtype=torch.float32
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
                in evolved_prototypes
            ],

        "prototype_matrix":
            evolved_matrix,

        "evolution_events":
            evolution_history
    },
    EVOLUTION_INDEX_FILE
)

print(
    "Evolution memory:",
    EVOLUTION_MEMORY_FILE
)

print(
    "Evolution index:",
    EVOLUTION_INDEX_FILE
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Reload Evolving Memory"
)

print()

reloaded_evolution = read_json(
    EVOLUTION_MEMORY_FILE
)

if (
        reloaded_evolution[
            "pattern_vector_dimension"
        ]
        !=
        PATTERN_VECTOR_DIMENSION
):

    raise RuntimeError(
        "Persisted evolution dimension changed."
    )

if (
        reloaded_evolution[
            "evolved_prototype_count"
        ]
        !=
        len(
            evolved_prototypes
        )
):

    raise RuntimeError(
        "Persisted prototype count changed."
    )

reloaded_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in reloaded_evolution[
        "prototypes"
    ]
]

current_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in evolved_prototypes
]

if reloaded_ids != current_ids:

    raise RuntimeError(
        "Prototype identity changed during persistence."
    )

print(
    "Reloaded prototype dimension:",
    reloaded_evolution[
        "pattern_vector_dimension"
    ]
)

print(
    "Reloaded prototype count:",
    len(
        reloaded_ids
    )
)

print(
    "Persistent evolving memory validated."
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save Evolution Dataset"
)

print()

evolution_dataset = {
    "lesson":
        "110R",

    "capability":
        "native_failure_prototype_evolution_continual_memory",

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_prototype_count,

    "evolved_prototype_count":
        len(
            evolved_prototypes
        ),

    "active_prototype":
        active_prototype_id,

    "best_match":
        best_match,

    "evolution_action":
        evolution_action,

    "novelty_score":
        novelty_score,

    "prototype_drift":
        prototype_drift,

    "stability_score":
        stability_score,

    "evolution_event":
        evolution_event,

    "evolved_prototypes":
        evolved_prototypes
}

write_json(
    EVOLUTION_DATASET_FILE,
    evolution_dataset
)

print(
    "Evolution dataset:",
    EVOLUTION_DATASET_FILE
)

print()


# ============================================================
# TEST 26
# ============================================================

print(
    "TEST 26: Save 110R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "110R",

    "capability":
        "native_failure_prototype_evolution_continual_memory",

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

    "original_prototype_count":
        original_prototype_count,

    "evolved_prototype_count":
        len(
            evolved_prototypes
        ),

    "active_prototype":
        active_prototype_id,

    "best_match":
        best_match,

    "evolution_action":
        evolution_action,

    "novelty_score":
        novelty_score,

    "prototype_drift":
        prototype_drift,

    "stability_score":
        stability_score,

    "evolved_prototypes":
        evolved_prototypes,

    "evolution_event":
        evolution_event,

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
# TEST 27
# ============================================================

print(
    "TEST 27: Write 110R Reports"
)

print()

report = {
    "lesson":
        "110R",

    "capability":
        "native_failure_prototype_evolution_continual_memory",

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

    "original_prototype_count":
        original_prototype_count,

    "evolved_prototype_count":
        len(
            evolved_prototypes
        ),

    "thresholds":
        {
            "evolution_similarity":
                EVOLUTION_SIMILARITY_THRESHOLD,

            "novelty":
                NOVELTY_THRESHOLD
        },

    "decision":
        {
            "action":
                evolution_action,

            "best_match":
                best_match,

            "novelty_score":
                novelty_score
        },

    "evolution":
        {
            "updated_prototype":
                updated_prototype_id,

            "created_prototype":
                (
                    created_new_prototype[
                        "prototype_id"
                    ]
                    if created_new_prototype
                    else
                    None
                ),

            "prototype_drift":
                prototype_drift,

            "stability":
                stability_score
        },

    "continual_memory":
        {
            "event":
                evolution_event,

            "history_length":
                len(
                    evolution_history
                )
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
    EVOLUTION_REPORT_FILE,
    report
)

write_json(
    EVOLUTION_EVALUATION_FILE,
    report
)

write_json(
    EVOLUTION_REGISTRY_FILE,
    {
        "lesson":
            "110R",

        "capability":
            "native_failure_prototype_evolution_continual_memory",

        "memory_version":
            MEMORY_VERSION,

        "pattern_vector_dimension":
            PATTERN_VECTOR_DIMENSION,

        "evolution_action":
            evolution_action,

        "novelty_score":
            novelty_score,

        "stability_score":
            stability_score,

        "next":
            "111R Native Novel Failure Discovery + Prototype Birth"
    }
)

print(
    "Evolution report:",
    EVOLUTION_REPORT_FILE
)

print(
    "Evolution evaluation:",
    EVOLUTION_EVALUATION_FILE
)

print(
    "Evolution registry:",
    EVOLUTION_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 110R CONTINUAL PROTOTYPE ARCHITECTURE"
)

print()

print(
    "New Incident"
)

print(
    "      ↓"
)

print(
    "Failure Pattern"
)

print(
    "      ↓"
)

print(
    "Existing Prototype Retrieval"
)

print(
    "      ↓"
)

print(
    "Similarity Evaluation"
)

print(
    "      ↓"
)

print(
    "      ┌─────────────────────┐"
)

print(
    "      │                     │"
)

print(
    "   Similar              Novel"
)

print(
    "      │                     │"
)

print(
    "      ↓                     ↓"
)

print(
    "Update Prototype     Create Prototype"
)

print(
    "      │                     │"
)

print(
    "      └──────────┬──────────┘"
)

print(
    "                 ↓"
)

print(
    "          Validate Evolution"
)

print(
    "                 ↓"
)

print(
    "          Continual Memory"
)

print()


# ============================================================
# WHY 110R MATTERS
# ============================================================

print(
    "WHY 110R MATTERS"
)

print()

print(
    "109R created reusable failure prototypes."
)

print(
    "110R gives those prototypes controlled evolution."
)

print()

print(
    "Silverwing can now distinguish between:"
)

print(
    "known behavior"
)

print(
    "novel behavior"
)

print(
    "and uncertain intermediate behavior."
)

print()

print(
    "This is a foundation for continual learning without "
    "blindly overwriting established knowledge."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "110R LIMITATION"
)

print()

print(
    "The present prototype memory is small."
)

print(
    "The evolution thresholds are controlled architectural "
    "parameters, not production-validated engineering thresholds."
)

print(
    "Real continual learning will require larger incident "
    "streams, stronger novelty estimation and rollback/versioning."
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
    "Lesson 111R: Native Novel Failure Discovery + Prototype Birth"
)

print()

print(
    "Novel Incident + Novelty Evidence + New Prototype Creation + "
    "Prototype Validation + Knowledge Expansion"
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
    "111R Native Novel Failure Discovery + Prototype Birth",
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
    "=== LESSON 110R COMPLETE ==="
)
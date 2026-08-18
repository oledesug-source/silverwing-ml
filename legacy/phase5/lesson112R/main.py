# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 112R
# Native Failure Prototype Validation + Cross-Case Reasoning
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
# 111R -> Native Novel Failure Discovery + Prototype Birth
# 112R -> Native Failure Prototype Validation + Cross-Case Reasoning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 112R validates whether multiple remembered failure cases
# provide consistent evidence for prototype reasoning.
#
# The system:
#
#   prototype memory
#        ↓
#   case representations
#        ↓
#   cross-case similarity
#        ↓
#   agreement detection
#        ↓
#   contradiction detection
#        ↓
#   prototype consistency
#        ↓
#   confidence
#        ↓
#   cross-case conclusion
#
# ============================================================
# CRITICAL COMPATIBILITY RULE
# ============================================================
#
# The representation dimension is discovered from 111R.
#
# No hard-coded downstream representation size is assumed.
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

MEMORY_VERSION = "112R.1"

CASE_AGREEMENT_THRESHOLD = 0.70

CASE_CONTRADICTION_THRESHOLD = 0.35

PROTOTYPE_CONSISTENCY_THRESHOLD = 0.60

CONFIDENCE_THRESHOLD = 0.50

TOP_K = 3

DETERMINISM_THRESHOLD = 1e-9

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

LESSON_111R = (
        PHASE5_DIR /
        "lesson111R"
)

SOURCE_NOVELTY_MEMORY_FILE = (
        LESSON_111R /
        "silverwing_novel_failure_memory.json"
)

SOURCE_NOVELTY_INDEX_FILE = (
        LESSON_111R /
        "silverwing_novel_failure_index.pt"
)

SOURCE_NOVELTY_DATASET_FILE = (
        LESSON_111R /
        "silverwing_novel_failure_dataset.json"
)

SOURCE_NOVELTY_REPORT_FILE = (
        LESSON_111R /
        "silverwing_novel_failure_report.json"
)

SOURCE_NOVELTY_REGISTRY_FILE = (
        LESSON_111R /
        "silverwing_novel_failure_registry.json"
)

SOURCE_NOVELTY_CHECKPOINT_PRIMARY = (
        LESSON_111R /
        "checkpoints" /
        "silverwing_novel_failure_best.pt"
)

SOURCE_NOVELTY_CHECKPOINT_CANDIDATE = (
        LESSON_111R /
        "checkpoints" /
        "silverwing_novel_failure_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROSS_CASE_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_memory.json"
)

CROSS_CASE_INDEX_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_index.pt"
)

CROSS_CASE_DATASET_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_dataset.json"
)

CROSS_CASE_REPORT_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_report.json"
)

CROSS_CASE_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_evaluation.json"
)

CROSS_CASE_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_cross_case_reasoning_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_cross_case_reasoning_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_cross_case_reasoning_best.pt"
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_NOVELTY_CHECKPOINT_PRIMARY,
        SOURCE_NOVELTY_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 111R checkpoint found."
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
    "PHASE 5 - LESSON 112R"
)

print(
    "Native Failure Prototype Validation + Cross-Case Reasoning"
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
    "110R -> Failure Prototype Evolution + Continual Memory",
    "111R -> Novel Failure Discovery + Prototype Birth",
    "112R -> Failure Prototype Validation + Cross-Case Reasoning"
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
    "Case agreement threshold:",
    CASE_AGREEMENT_THRESHOLD
)

print(
    "Case contradiction threshold:",
    CASE_CONTRADICTION_THRESHOLD
)

print(
    "Prototype consistency threshold:",
    PROTOTYPE_CONSISTENCY_THRESHOLD
)

print(
    "Confidence threshold:",
    CONFIDENCE_THRESHOLD
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
    "TEST 1: Verify 111R Novel-Memory Inputs"
)

print()

for path in [
    SOURCE_NOVELTY_MEMORY_FILE,
    SOURCE_NOVELTY_INDEX_FILE,
    SOURCE_NOVELTY_DATASET_FILE,
    SOURCE_NOVELTY_REPORT_FILE,
    SOURCE_NOVELTY_REGISTRY_FILE
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
    SOURCE_NOVELTY_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_NOVELTY_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_NOVELTY_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_NOVELTY_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_NOVELTY_REGISTRY_FILE
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
    "TEST 2: Load Continual Failure Memory"
)

print()

novelty_memory = read_json(
    SOURCE_NOVELTY_MEMORY_FILE
)

novelty_dataset = read_json(
    SOURCE_NOVELTY_DATASET_FILE
)

novelty_report = read_json(
    SOURCE_NOVELTY_REPORT_FILE
)

if not isinstance(
        novelty_memory,
        dict
):

    raise RuntimeError(
        "111R novelty memory is invalid."
    )

prototypes = novelty_memory.get(
    "prototypes"
)

if not isinstance(
        prototypes,
        list
):

    raise RuntimeError(
        "111R novelty memory contains no prototypes."
    )

if not prototypes:

    raise RuntimeError(
        "111R novelty memory is empty."
    )

print(
    "Memory version:",
    novelty_memory.get(
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
    "Expanded prototype count:",
    novelty_memory.get(
        "expanded_prototype_count"
    )
)

print(
    "Novelty decision:",
    novelty_memory.get(
        "decision"
    )
)

print(
    "Novelty score:",
    novelty_memory.get(
        "novelty_score"
    )
)

print(
    "Dataset loaded:",
    bool(
        novelty_dataset
    )
)

print(
    "Report loaded:",
    bool(
        novelty_report
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Discover Persisted Prototype Dimension"
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
                "has no valid vector."
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
    "Persisted dimensions:",
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
            "Prototype dimensions are inconsistent: "
            f"{sorted(unique_dimensions)}"
        )
    )

REPRESENTATION_DIMENSION = (
    vector_lengths[
        0
    ]
)

print(
    "Discovered representation dimension:",
    REPRESENTATION_DIMENSION
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
    "prototype_vector"
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

    if len(
            prototype[
                "prototype_vector"
            ]
    ) != REPRESENTATION_DIMENSION:

        schema_errors.append(
            {
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "error":
                    "representation dimension mismatch",

                "actual":
                    len(
                        prototype[
                            "prototype_vector"
                        ]
                    ),

                "expected":
                    REPRESENTATION_DIMENSION
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
        "112R prototype schema validation failed."
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

normalized_prototypes = F.normalize(
    prototype_matrix,
    p=2,
    dim=-1
)

print(
    "Prototype matrix:",
    tuple(
        prototype_matrix.shape
    )
)

print(
    "Normalized matrix:",
    tuple(
        normalized_prototypes.shape
    )
)

if not finite_tensor(
        normalized_prototypes
):

    raise RuntimeError(
        "Prototype representation matrix is invalid."
    )

print(
    "Prototype matrix validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Cross-Case Representations"
)

print()

case_records = []

for index, prototype in enumerate(
        prototypes
):

    vector = normalized_prototypes[
        index
    ]

    case_records.append(
        {
            "case_id":
                f"case_{index + 1:03d}",

            "prototype_id":
                prototype[
                    "prototype_id"
                ],

            "cluster_id":
                prototype[
                    "cluster_id"
                ],

            "vector":
                vector,

            "confidence":
                float(
                    prototype.get(
                        "prototype_confidence",
                        0.50
                    )
                ),

            "member_count":
                int(
                    prototype.get(
                        "member_count",
                        0
                    )
                )
        }
    )

for case in case_records:

    print(
        case[
            "case_id"
        ],
        "->",
        case[
            "prototype_id"
        ],
        "| cluster",
        case[
            "cluster_id"
        ]
    )

if len(
        case_records
) < 2:

    raise RuntimeError(
        "At least two cases are required for cross-case reasoning."
    )

print(
    "Cross-case representations:",
    len(
        case_records
    )
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Build Cross-Case Similarity Matrix"
)

print()

case_matrix = torch.stack(
    [
        case[
            "vector"
        ]
        for case
        in case_records
    ],
    dim=0
)

case_similarity = torch.matmul(
    case_matrix,
    case_matrix.T
)

print(
    "Cross-case similarity matrix:",
    tuple(
        case_similarity.shape
    )
)

if not finite_tensor(
        case_similarity
):

    raise RuntimeError(
        "Cross-case similarity matrix is invalid."
    )

print(
    "Cross-case similarity matrix validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Per-Case Prototype Validation"
)

print()

case_validation = []

for index, case in enumerate(
        case_records
):

    self_similarity = float(
        case_similarity[
            index,
            index
        ]
    )

    other_scores = [
        float(
            case_similarity[
                index,
                other
            ]
        )
        for other in range(
            len(
                case_records
            )
        )
        if other != index
    ]

    strongest_other = (
        max(
            other_scores
        )
        if other_scores
        else
        0.0
    )

    case_validation.append(
        {
            "case_id":
                case[
                    "case_id"
                ],

            "prototype_id":
                case[
                    "prototype_id"
                ],

            "self_similarity":
                self_similarity,

            "strongest_other_case":
                strongest_other,

            "confidence":
                case[
                    "confidence"
                ]
        }
    )

for result in case_validation:

    print(
        result
    )

print(
    "Per-case prototype validation completed."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Cross-Case Agreement Detection"
)

print()

agreement_pairs = []

for left in range(
        len(
            case_records
        )
):

    for right in range(
            left + 1,
            len(
                case_records
            )
    ):

        similarity = float(
            case_similarity[
                left,
                right
            ]
        )

        agreement = (
                similarity
                >=
                CASE_AGREEMENT_THRESHOLD
        )

        contradiction = (
                similarity
                <=
                CASE_CONTRADICTION_THRESHOLD
        )

        agreement_pairs.append(
            {
                "case_a":
                    case_records[
                        left
                    ][
                        "case_id"
                    ],

                "case_b":
                    case_records[
                        right
                    ][
                        "case_id"
                    ],

                "similarity":
                    similarity,

                "agreement":
                    agreement,

                "contradiction":
                    contradiction
            }
        )

for pair in agreement_pairs:

    print(
        pair
    )

agreement_count = sum(
    1
    for pair
    in agreement_pairs
    if pair[
        "agreement"
    ]
)

contradiction_count = sum(
    1
    for pair
    in agreement_pairs
    if pair[
        "contradiction"
    ]
)

print(
    "Agreement pairs:",
    agreement_count
)

print(
    "Contradiction pairs:",
    contradiction_count
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Prototype Consistency"
)

print()

prototype_consistency_values = []

for prototype in prototypes:

    member_confidence = float(
        prototype.get(
            "prototype_confidence",
            0.50
        )
    )

    member_count = int(
        prototype.get(
            "member_count",
            0
        )
    )

    membership_factor = clamp(
        member_count
        /
        3.0
    )

    consistency = safe_mean(
        [
            member_confidence,
            membership_factor
        ]
    )

    prototype_consistency_values.append(
        {
            "prototype_id":
                prototype[
                    "prototype_id"
                ],

            "member_count":
                member_count,

            "prototype_confidence":
                member_confidence,

            "membership_factor":
                membership_factor,

            "consistency":
                consistency
        }
    )

for result in prototype_consistency_values:

    print(
        result
    )

overall_prototype_consistency = safe_mean(
    [
        item[
            "consistency"
        ]
        for item
        in prototype_consistency_values
    ]
)

print(
    "Overall prototype consistency:",
    overall_prototype_consistency
)

if not math.isfinite(
        overall_prototype_consistency
):

    raise RuntimeError(
        "Prototype consistency is invalid."
    )

print(
    "Prototype consistency validated."
)

print()


# ============================================================
# CRITICAL COMPATIBILITY ALIAS
# ============================================================
#
# Earlier code called this result
# `prototype_consistency_results`
# while TEST 10 created
# `prototype_consistency_values`.
#
# Use one canonical object from here onward.
#

prototype_consistency_results = (
    prototype_consistency_values
)


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Cross-Case Evidence Score"
)

print()

if agreement_pairs:

    agreement_score = (
            agreement_count
            /
            len(
                agreement_pairs
            )
    )

    contradiction_penalty = (
            contradiction_count
            /
            len(
                agreement_pairs
            )
    )

else:

    agreement_score = 1.0

    contradiction_penalty = 0.0

cross_case_evidence_score = clamp(
    safe_mean(
        [
            agreement_score,
            overall_prototype_consistency,
            1.0 - contradiction_penalty
        ]
    )
)

print(
    "Agreement score:",
    agreement_score
)

print(
    "Contradiction penalty:",
    contradiction_penalty
)

print(
    "Cross-case evidence score:",
    cross_case_evidence_score
)

if not math.isfinite(
        cross_case_evidence_score
):

    raise RuntimeError(
        "Cross-case evidence score is invalid."
    )

print(
    "Cross-case evidence score validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Contradiction Analysis"
)

print()

contradiction_cases = [
    pair
    for pair
    in agreement_pairs
    if pair[
        "contradiction"
    ]
]

for pair in contradiction_cases:

    print(
        pair
    )

if contradiction_cases:

    contradiction_state = (
        "CONTRADICTORY_CASES_PRESENT"
    )

else:

    contradiction_state = (
        "NO_STRONG_CONTRADICTION"
    )

print(
    "Contradiction state:",
    contradiction_state
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Case-Level Confidence"
)

print()

case_confidences = [
    clamp(
        case[
            "confidence"
        ]
    )
    for case
    in case_records
]

case_confidence = safe_mean(
    case_confidences
)

diagnostic_confidence = clamp(
    safe_mean(
        [
            case_confidence,
            cross_case_evidence_score,
            overall_prototype_consistency
        ]
    )
)

print(
    "Mean case confidence:",
    case_confidence
)

print(
    "Cross-case evidence:",
    cross_case_evidence_score
)

print(
    "Prototype consistency:",
    overall_prototype_consistency
)

print(
    "Diagnostic confidence:",
    diagnostic_confidence
)

if (
        diagnostic_confidence
        <
        CONFIDENCE_THRESHOLD
):

    raise RuntimeError(
        "Diagnostic confidence is below threshold."
    )

print(
    "Case-level confidence validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Cross-Case Reasoning Conclusion"
)

print()

if (
        contradiction_state
        ==
        "CONTRADICTORY_CASES_PRESENT"
):

    cross_case_conclusion = (
        "The prototype memory contains conflicting case evidence. "
        "The current diagnosis should remain cautious until additional "
        "observations resolve the contradiction."
    )

elif (
        cross_case_evidence_score
        >=
        CASE_AGREEMENT_THRESHOLD
):

    cross_case_conclusion = (
        "Multiple stored cases provide mutually supportive evidence "
        "for the current prototype interpretation."
    )

else:

    cross_case_conclusion = (
        "The cases provide mixed but usable evidence; the prototype "
        "interpretation should remain provisional."
    )

print(
    "Cross-case conclusion:"
)

print(
    cross_case_conclusion
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Prototype Validation Against Cross-Case Evidence"
)

print()

prototype_validation_results = []

for prototype in prototypes:

    prototype_id = prototype[
        "prototype_id"
    ]

    related_cases = [
        case
        for case
        in case_records
        if case[
               "prototype_id"
           ]
           ==
           prototype_id
    ]

    related_scores = []

    for case in related_cases:

        case_index = next(
            index
            for index, candidate
            in enumerate(
                case_records
            )
            if candidate[
                "case_id"
            ]
            ==
            case[
                "case_id"
            ]
        )

        for other_index in range(
                len(
                    case_records
                )
        ):

            if (
                    other_index
                    ==
                    case_index
            ):

                continue

            related_scores.append(
                float(
                    case_similarity[
                        case_index,
                        other_index
                    ]
                )
            )

    cross_case_mean = (
        safe_mean(
            related_scores
        )
        if related_scores
        else
        1.0
    )

    validation_score = clamp(
        safe_mean(
            [
                float(
                    prototype.get(
                        "prototype_confidence",
                        0.50
                    )
                ),

                cross_case_mean
            ]
        )
    )

    prototype_validation_results.append(
        {
            "prototype_id":
                prototype_id,

            "cross_case_mean":
                cross_case_mean,

            "validation_score":
                validation_score
        }
    )

for result in prototype_validation_results:

    print(
        result
    )

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Cross-Case Ranking"
)

print()

case_rankings = []

for index, case in enumerate(
        case_records
):

    other_scores = [
        float(
            case_similarity[
                index,
                other
            ]
        )
        for other
        in range(
            len(
                case_records
            )
        )
        if other != index
    ]

    similarity_to_others = safe_mean(
        other_scores
    )

    case_rankings.append(
        {
            "case_id":
                case[
                    "case_id"
                ],

            "prototype_id":
                case[
                    "prototype_id"
                ],

            "mean_cross_case_similarity":
                similarity_to_others,

            "confidence":
                case[
                    "confidence"
                ]
        }
    )

case_rankings.sort(
    key=lambda item:
    (
        item[
            "mean_cross_case_similarity"
        ],
        item[
            "case_id"
        ]
    ),
    reverse=True
)

for ranking in case_rankings:

    print(
        ranking
    )

if not case_rankings:

    raise RuntimeError(
        "Cross-case ranking produced no cases."
    )

print(
    "Cross-case ranking validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Deterministic Cross-Case Reasoning"
)

print()


def calculate_cross_case_score(
        similarity_matrix: torch.Tensor
) -> float:

    count = similarity_matrix.shape[
        0
    ]

    pairs = []

    for left in range(
            count
    ):

        for right in range(
                left + 1,
                count
        ):

            pairs.append(
                float(
                    similarity_matrix[
                        left,
                        right
                    ]
                )
            )

    if not pairs:

        return 1.0

    agreement = sum(
        1
        for value
        in pairs
        if value
        >=
        CASE_AGREEMENT_THRESHOLD
    ) / len(
        pairs
    )

    contradiction = sum(
        1
        for value
        in pairs
        if value
        <=
        CASE_CONTRADICTION_THRESHOLD
    ) / len(
        pairs
    )

    return clamp(
        safe_mean(
            [
                agreement,
                1.0 - contradiction
            ]
        )
    )


first_reasoning_score = calculate_cross_case_score(
    case_similarity
)

second_reasoning_score = calculate_cross_case_score(
    case_similarity
)

reasoning_deterministic = (
        abs(
            first_reasoning_score
            -
            second_reasoning_score
        )
        <=
        DETERMINISM_THRESHOLD
)

print(
    "First score:",
    first_reasoning_score
)

print(
    "Second score:",
    second_reasoning_score
)

print(
    "Deterministic:",
    reasoning_deterministic
)

if not reasoning_deterministic:

    raise RuntimeError(
        "Cross-case reasoning is nondeterministic."
    )

print(
    "Deterministic cross-case reasoning validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Cross-Case Reasoning Curriculum"
)

print()

cross_case_tasks = [
    {
        "example_id":
            "crosscase_001",

        "domain":
            "case_comparison",

        "question":
            "Why compare multiple failure cases?",

        "answer":
            "To determine whether evidence is consistent across observations."
    },

    {
        "example_id":
            "crosscase_002",

        "domain":
            "prototype_validation",

        "question":
            "What makes a prototype more trustworthy?",

        "answer":
            "Consistent evidence across multiple related cases."
    },

    {
        "example_id":
            "crosscase_003",

        "domain":
            "contradiction_detection",

        "question":
            "Why detect contradictory cases?",

        "answer":
            "Conflicting evidence can reduce confidence in a diagnosis."
    },

    {
        "example_id":
            "crosscase_004",

        "domain":
            "diagnostic_confidence",

        "question":
            "What should diagnostic confidence depend on?",

        "answer":
            "Evidence strength, consistency and unresolved contradictions."
    },

    {
        "example_id":
            "crosscase_005",

        "domain":
            "case_based_reasoning",

        "question":
            "How does cross-case reasoning improve failure analysis?",

        "answer":
            "It allows the system to reason across several historical examples."
    },

    {
        "example_id":
            "crosscase_006",

        "domain":
            "engineering_intelligence",

        "question":
            "Why is cross-case validation useful in engineering?",

        "answer":
            "Repeated operating patterns can strengthen or weaken a proposed explanation."
    }
]

for task in cross_case_tasks:

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
    "Cross-case tasks:",
    len(
        cross_case_tasks
    )
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Cross-Case Curriculum Coverage"
)

print()

expected_domains = {
    "case_comparison",
    "prototype_validation",
    "contradiction_detection",
    "diagnostic_confidence",
    "case_based_reasoning",
    "engineering_intelligence"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in cross_case_tasks
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
        "Cross-case curriculum coverage is incomplete."
    )

print(
    "Cross-case curriculum validated."
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
    case_matrix,
    case_similarity
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
        "Cross-case reasoning numerical health failed."
    )

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Final Cross-Case Promotion Gate"
)

print()

promotion_errors = []

if not reasoning_deterministic:

    promotion_errors.append(
        "Cross-case reasoning is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if (
        diagnostic_confidence
        <
        CONFIDENCE_THRESHOLD
):

    promotion_errors.append(
        "Diagnostic confidence is below threshold."
    )

if len(
        cross_case_tasks
) < 6:

    promotion_errors.append(
        "Cross-case curriculum is incomplete."
    )

if len(
        case_records
) < 2:

    promotion_errors.append(
        "At least two cases are required."
    )

if not cross_case_conclusion:

    promotion_errors.append(
        "Cross-case conclusion is missing."
    )

print(
    "Representation dimension:",
    REPRESENTATION_DIMENSION
)

print(
    "Case count:",
    len(
        case_records
    )
)

print(
    "Cross-case evidence:",
    cross_case_evidence_score
)

print(
    "Diagnostic confidence:",
    diagnostic_confidence
)

print(
    "Contradiction state:",
    contradiction_state
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
        "112R cross-case promotion gate failed."
    )

print(
    "112R cross-case promotion gate passed."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Persist Cross-Case Reasoning Memory"
)

print()

#
# FIXED VARIABLE CONTRACT:
#
# TEST 10 creates:
#
#     prototype_consistency_values
#
# From this point onward the script uses:
#
#     prototype_consistency_results
#
# which is an explicit canonical alias.
#
# This prevents a serialization-stage NameError.
#

cross_case_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_failure_prototype_validation_cross_case_reasoning",

    "created_at":
        datetime.now().isoformat(),

    "representation_dimension":
        REPRESENTATION_DIMENSION,

    "case_count":
        len(
            case_records
        ),

    "cases":
        [
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "prototype_id":
                    case[
                        "prototype_id"
                    ],

                "cluster_id":
                    case[
                        "cluster_id"
                    ],

                "confidence":
                    case[
                        "confidence"
                    ],

                "member_count":
                    case[
                        "member_count"
                    ]
            }
            for case
            in case_records
        ],

    "agreement_pairs":
        agreement_pairs,

    "prototype_consistency":
        prototype_consistency_results,

    "cross_case_evidence_score":
        cross_case_evidence_score,

    "contradiction_state":
        contradiction_state,

    "diagnostic_confidence":
        diagnostic_confidence,

    "conclusion":
        cross_case_conclusion
}

write_json(
    CROSS_CASE_MEMORY_FILE,
    cross_case_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "representation_dimension":
            REPRESENTATION_DIMENSION,

        "case_ids":
            [
                case[
                    "case_id"
                ]
                for case
                in case_records
            ],

        "case_matrix":
            case_matrix,

        "case_similarity":
            case_similarity
    },
    CROSS_CASE_INDEX_FILE
)

print(
    "Cross-case memory:",
    CROSS_CASE_MEMORY_FILE
)

print(
    "Cross-case index:",
    CROSS_CASE_INDEX_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Reload Cross-Case Memory"
)

print()

reloaded_cross_case = read_json(
    CROSS_CASE_MEMORY_FILE
)

if (
        reloaded_cross_case[
            "representation_dimension"
        ]
        !=
        REPRESENTATION_DIMENSION
):

    raise RuntimeError(
        "Cross-case representation dimension changed."
    )

if (
        reloaded_cross_case[
            "case_count"
        ]
        !=
        len(
            case_records
        )
):

    raise RuntimeError(
        "Cross-case count changed after persistence."
    )

reloaded_consistency = (
    reloaded_cross_case.get(
        "prototype_consistency",
        []
    )
)

if len(
        reloaded_consistency
) != len(
    prototype_consistency_results
):

    raise RuntimeError(
        "Prototype consistency records changed during persistence."
    )

print(
    "Reloaded dimension:",
    reloaded_cross_case[
        "representation_dimension"
    ]
)

print(
    "Reloaded case count:",
    reloaded_cross_case[
        "case_count"
    ]
)

print(
    "Reloaded consistency records:",
    len(
        reloaded_consistency
    )
)

print(
    "Persistent cross-case memory validated."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save Cross-Case Dataset"
)

print()

cross_case_dataset = {
    "lesson":
        "112R",

    "capability":
        "native_failure_prototype_validation_cross_case_reasoning",

    "representation_dimension":
        REPRESENTATION_DIMENSION,

    "case_count":
        len(
            case_records
        ),

    "agreement_pairs":
        agreement_pairs,

    "contradiction_cases":
        contradiction_cases,

    "case_rankings":
        case_rankings,

    "prototype_consistency":
        prototype_consistency_results,

    "cross_case_evidence_score":
        cross_case_evidence_score,

    "diagnostic_confidence":
        diagnostic_confidence,

    "conclusion":
        cross_case_conclusion
}

write_json(
    CROSS_CASE_DATASET_FILE,
    cross_case_dataset
)

print(
    "Cross-case dataset:",
    CROSS_CASE_DATASET_FILE
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save 112R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "112R",

    "capability":
        "native_failure_prototype_validation_cross_case_reasoning",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "representation_dimension":
        REPRESENTATION_DIMENSION,

    "case_count":
        len(
            case_records
        ),

    "agreement_pairs":
        agreement_pairs,

    "contradiction_state":
        contradiction_state,

    "prototype_consistency":
        prototype_consistency_results,

    "cross_case_evidence_score":
        cross_case_evidence_score,

    "diagnostic_confidence":
        diagnostic_confidence,

    "case_rankings":
        case_rankings,

    "conclusion":
        cross_case_conclusion,

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
# TEST 26
# ============================================================

print(
    "TEST 26: Write 112R Reports"
)

print()

report = {
    "lesson":
        "112R",

    "capability":
        "native_failure_prototype_validation_cross_case_reasoning",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "representation_dimension":
        REPRESENTATION_DIMENSION,

    "case_count":
        len(
            case_records
        ),

    "agreement":
        {
            "pair_count":
                len(
                    agreement_pairs
                ),

            "agreement_count":
                agreement_count,

            "contradiction_count":
                contradiction_count,

            "score":
                agreement_score
        },

    "prototype_validation":
        {
            "overall_consistency":
                overall_prototype_consistency,

            "results":
                prototype_consistency_results
        },

    "reasoning":
        {
            "cross_case_evidence":
                cross_case_evidence_score,

            "diagnostic_confidence":
                diagnostic_confidence,

            "contradiction_state":
                contradiction_state,

            "conclusion":
                cross_case_conclusion
        },

    "ranking":
        case_rankings,

    "verification":
        {
            "deterministic":
                reasoning_deterministic
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
    CROSS_CASE_REPORT_FILE,
    report
)

write_json(
    CROSS_CASE_EVALUATION_FILE,
    report
)

write_json(
    CROSS_CASE_REGISTRY_FILE,
    {
        "lesson":
            "112R",

        "capability":
            "native_failure_prototype_validation_cross_case_reasoning",

        "memory_version":
            MEMORY_VERSION,

        "representation_dimension":
            REPRESENTATION_DIMENSION,

        "case_count":
            len(
                case_records
            ),

        "cross_case_evidence_score":
            cross_case_evidence_score,

        "diagnostic_confidence":
            diagnostic_confidence,

        "contradiction_state":
            contradiction_state,

        "next":
            "113R Native Contradiction Resolution + Evidence Arbitration"
    }
)

print(
    "Cross-case report:",
    CROSS_CASE_REPORT_FILE
)

print(
    "Cross-case evaluation:",
    CROSS_CASE_EVALUATION_FILE
)

print(
    "Cross-case registry:",
    CROSS_CASE_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 112R CROSS-CASE REASONING ARCHITECTURE"
)

print()

print(
    "Multiple Failure Cases"
)

print(
    "        ↓"
)

print(
    "Case Representations"
)

print(
    "        ↓"
)

print(
    "Prototype Retrieval"
)

print(
    "        ↓"
)

print(
    "Cross-Case Similarity"
)

print(
    "        ↓"
)

print(
    "Agreement / Contradiction"
)

print(
    "        ↓"
)

print(
    "Prototype Consistency"
)

print(
    "        ↓"
)

print(
    "Diagnostic Confidence"
)

print(
    "        ↓"
)

print(
    "Verified Cross-Case Conclusion"
)

print()


# ============================================================
# WHY 112R MATTERS
# ============================================================

print(
    "WHY 112R MATTERS"
)

print()

print(
    "109R organized failure patterns into prototypes."
)

print(
    "110R allowed those prototypes to evolve."
)

print(
    "111R allowed new patterns to be discovered."
)

print(
    "112R validates the resulting knowledge across multiple cases."
)

print()

print(
    "This is the transition from single-case reasoning "
    "to evidence arbitration across remembered cases."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "112R LIMITATION"
)

print()

print(
    "The current prototype memory remains small and controlled."
)

print(
    "Cross-case validation therefore establishes the reasoning "
    "architecture rather than production engineering diagnosis."
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
    "Lesson 113R: Native Contradiction Resolution + Evidence Arbitration"
)

print()

print(
    "Conflicting Cases + Evidence Weighting + Source Reliability + "
    "Contradiction Resolution + Final Diagnosis"
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
    "112R Native Failure Prototype Validation + Cross-Case Reasoning",
    " ↓",
    "113R Native Contradiction Resolution + Evidence Arbitration",
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
    "=== LESSON 112R COMPLETE ==="
)
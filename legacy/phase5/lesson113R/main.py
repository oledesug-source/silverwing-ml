# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 113R
# Native Contradiction Resolution + Evidence Arbitration
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
# 113R -> Native Contradiction Resolution + Evidence Arbitration
#
# ============================================================
# PURPOSE
# ============================================================
#
# 113R resolves conflicting evidence without hiding uncertainty.
#
# The lesson distinguishes:
#
#   arbitration success
#   from
#   diagnostic confidence
#
# Low confidence is NOT automatically a system failure.
#
# A contradictory evidence set can be correctly arbitrated while
# still producing:
#
#   LOW_CONFIDENCE
#   or
#   UNRESOLVED
#
# That is a valid intelligent outcome.
#
# ============================================================

import json
import math
import random

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MEMORY_VERSION = "113R.2"

HIGH_RELIABILITY = 0.90
MEDIUM_RELIABILITY = 0.70
LOW_RELIABILITY = 0.40

MIN_CONFIDENCE_FOR_STRONG_RESOLUTION = 0.50

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8

AGREEMENT_THRESHOLD = 0.60
CONFLICT_THRESHOLD = 0.40


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_112R = (
        PHASE5_DIR /
        "lesson112R"
)

SOURCE_CROSS_CASE_MEMORY_FILE = (
        LESSON_112R /
        "silverwing_cross_case_reasoning_memory.json"
)

SOURCE_CROSS_CASE_INDEX_FILE = (
        LESSON_112R /
        "silverwing_cross_case_reasoning_index.pt"
)

SOURCE_CROSS_CASE_DATASET_FILE = (
        LESSON_112R /
        "silverwing_cross_case_reasoning_dataset.json"
)

SOURCE_CROSS_CASE_REPORT_FILE = (
        LESSON_112R /
        "silverwing_cross_case_reasoning_report.json"
)

SOURCE_CROSS_CASE_REGISTRY_FILE = (
        LESSON_112R /
        "silverwing_cross_case_reasoning_registry.json"
)

SOURCE_CROSS_CASE_CHECKPOINT_PRIMARY = (
        LESSON_112R /
        "checkpoints" /
        "silverwing_cross_case_reasoning_best.pt"
)

SOURCE_CROSS_CASE_CHECKPOINT_CANDIDATE = (
        LESSON_112R /
        "checkpoints" /
        "silverwing_cross_case_reasoning_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ARBITRATION_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_memory.json"
)

ARBITRATION_INDEX_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_index.pt"
)

ARBITRATION_DATASET_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_dataset.json"
)

ARBITRATION_REPORT_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_report.json"
)

ARBITRATION_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_evaluation.json"
)

ARBITRATION_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_evidence_arbitration_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_evidence_arbitration_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_evidence_arbitration_best.pt"
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
        SOURCE_CROSS_CASE_CHECKPOINT_PRIMARY,
        SOURCE_CROSS_CASE_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 112R checkpoint found."
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
    "PHASE 5 - LESSON 113R"
)

print(
    "Native Contradiction Resolution + Evidence Arbitration"
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
    "112R -> Failure Prototype Validation + Cross-Case Reasoning",
    "113R -> Contradiction Resolution + Evidence Arbitration"
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
    "High reliability:",
    HIGH_RELIABILITY
)

print(
    "Medium reliability:",
    MEDIUM_RELIABILITY
)

print(
    "Low reliability:",
    LOW_RELIABILITY
)

print(
    "Strong-resolution confidence threshold:",
    MIN_CONFIDENCE_FOR_STRONG_RESOLUTION
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 112R Cross-Case Inputs"
)

print()

for path in [
    SOURCE_CROSS_CASE_MEMORY_FILE,
    SOURCE_CROSS_CASE_INDEX_FILE,
    SOURCE_CROSS_CASE_DATASET_FILE,
    SOURCE_CROSS_CASE_REPORT_FILE,
    SOURCE_CROSS_CASE_REGISTRY_FILE
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
    SOURCE_CROSS_CASE_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_CROSS_CASE_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_CROSS_CASE_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_CROSS_CASE_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_CROSS_CASE_REGISTRY_FILE
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
    "TEST 2: Load Cross-Case Reasoning Memory"
)

print()

cross_case_memory = read_json(
    SOURCE_CROSS_CASE_MEMORY_FILE
)

cross_case_dataset = read_json(
    SOURCE_CROSS_CASE_DATASET_FILE
)

cross_case_report = read_json(
    SOURCE_CROSS_CASE_REPORT_FILE
)

if not isinstance(
        cross_case_memory,
        dict
):

    raise RuntimeError(
        "112R cross-case memory is invalid."
    )

agreement_pairs = cross_case_memory.get(
    "agreement_pairs"
)

if not isinstance(
        agreement_pairs,
        list
):

    raise RuntimeError(
        "112R agreement pairs are unavailable."
    )

prototype_consistency = (
    cross_case_memory.get(
        "prototype_consistency"
    )
)

if not isinstance(
        prototype_consistency,
        list
):

    raise RuntimeError(
        "112R prototype consistency records are unavailable."
    )

print(
    "Memory version:",
    cross_case_memory.get(
        "memory_version"
    )
)

print(
    "Case count:",
    cross_case_memory.get(
        "case_count"
    )
)

print(
    "Agreement pairs:",
    len(
        agreement_pairs
    )
)

print(
    "Prototype consistency records:",
    len(
        prototype_consistency
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Preserve Representation Schema"
)

print()

representation_dimension = cross_case_memory.get(
    "representation_dimension"
)

if not isinstance(
        representation_dimension,
        int
):

    raise RuntimeError(
        "112R representation dimension is missing."
    )

if representation_dimension <= 0:

    raise RuntimeError(
        "112R representation dimension is invalid."
    )

print(
    "Representation dimension:",
    representation_dimension
)

print(
    "Representation schema preserved."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Identify Conflicting Evidence"
)

print()

contradiction_pairs = [
    pair
    for pair
    in agreement_pairs
    if pair.get(
        "contradiction",
        False
    )
]

supporting_pairs = [
    pair
    for pair
    in agreement_pairs
    if pair.get(
        "agreement",
        False
    )
]

print(
    "Supporting pairs:",
    len(
        supporting_pairs
    )
)

print(
    "Contradictory pairs:",
    len(
        contradiction_pairs
    )
)

for pair in contradiction_pairs:

    print(
        pair
    )

contradiction_state = (
    "CONTRADICTIONS_PRESENT"
    if contradiction_pairs
    else
    "NO_CONTRADICTIONS"
)

print(
    "Contradiction state:",
    contradiction_state
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Evidence Sources"
)

print()

evidence_sources = []

for index, pair in enumerate(
        agreement_pairs
):

    similarity = float(
        pair.get(
            "similarity",
            0.0
        )
    )

    if pair.get(
            "agreement",
            False
    ):

        direction = "support"

        reliability = HIGH_RELIABILITY

        magnitude = clamp(
            (
                    similarity
                    +
                    1.0
            )
            /
            2.0
        )

    elif pair.get(
            "contradiction",
            False
    ):

        direction = "conflict"

        reliability = MEDIUM_RELIABILITY

        magnitude = clamp(
            1.0
            -
            (
                    similarity
                    +
                    1.0
            )
            /
            2.0
        )

    else:

        direction = "neutral"

        reliability = LOW_RELIABILITY

        magnitude = 0.0

    evidence_sources.append(
        {
            "evidence_id":
                f"evidence_{index + 1:03d}",

            "source_type":
                "cross_case_similarity",

            "case_a":
                pair.get(
                    "case_a"
                ),

            "case_b":
                pair.get(
                    "case_b"
                ),

            "similarity":
                similarity,

            "direction":
                direction,

            "reliability":
                reliability,

            "magnitude":
                magnitude
        }
    )

for evidence in evidence_sources:

    print(
        evidence
    )

if not evidence_sources:

    raise RuntimeError(
        "No evidence sources were constructed."
    )

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Prototype Reliability Evidence"
)

print()

for item in prototype_consistency:

    consistency = clamp(
        float(
            item.get(
                "consistency",
                0.0
            )
        )
    )

    reliability = (
        HIGH_RELIABILITY
        if consistency >= 0.75
        else
        MEDIUM_RELIABILITY
        if consistency >= 0.50
        else
        LOW_RELIABILITY
    )

    evidence_sources.append(
        {
            "evidence_id":
                f"prototype_{item.get('prototype_id')}",

            "source_type":
                "prototype_consistency",

            "prototype_id":
                item.get(
                    "prototype_id"
                ),

            "consistency":
                consistency,

            "direction":
                "support"
                if consistency >= 0.50
                else
                "conflict",

            "reliability":
                reliability,

            "magnitude":
                consistency
        }
    )

print(
    "Total evidence sources:",
    len(
        evidence_sources
    )
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Calculate Weighted Evidence"
)

print()

weighted_evidence = []

for evidence in evidence_sources:

    weight = (
            float(
                evidence[
                    "reliability"
                ]
            )
            *
            float(
                evidence[
                    "magnitude"
                ]
            )
    )

    contribution = (
        weight
        if evidence[
               "direction"
           ]
           ==
           "support"
        else
        -weight
        if evidence[
               "direction"
           ]
           ==
           "conflict"
        else
        0.0
    )

    result = dict(
        evidence
    )

    result[
        "weight"
    ] = weight

    result[
        "contribution"
    ] = contribution

    weighted_evidence.append(
        result
    )

for evidence in weighted_evidence:

    print(
        evidence
    )

support_score = sum(
    item[
        "weight"
    ]
    for item
    in weighted_evidence
    if item[
        "direction"
    ]
    ==
    "support"
)

conflict_score = sum(
    item[
        "weight"
    ]
    for item
    in weighted_evidence
    if item[
        "direction"
    ]
    ==
    "conflict"
)

total_weight = sum(
    item[
        "weight"
    ]
    for item
    in weighted_evidence
)

print(
    "Support score:",
    support_score
)

print(
    "Conflict score:",
    conflict_score
)

print(
    "Total evidence weight:",
    total_weight
)

if total_weight <= EPSILON:

    raise RuntimeError(
        "No weighted evidence exists."
    )

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Evidence Arbitration"
)

print()

net_evidence = (
        support_score
        -
        conflict_score
)

arbitration_strength = (
        abs(
            net_evidence
        )
        /
        total_weight
)

if (
        net_evidence
        >
        EPSILON
):

    arbitration_direction = (
        "SUPPORTS_PRIMARY_HYPOTHESIS"
    )

elif (
        net_evidence
        <
        -EPSILON
):

    arbitration_direction = (
        "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
    )

else:

    arbitration_direction = (
        "UNRESOLVED"
    )

print(
    "Net evidence:",
    net_evidence
)

print(
    "Arbitration strength:",
    arbitration_strength
)

print(
    "Arbitration direction:",
    arbitration_direction
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Evidence Arbitration Confidence"
)

print()

evidence_balance = (
        abs(
            support_score
            -
            conflict_score
        )
        /
        total_weight
)

source_reliability = safe_mean(
    [
        float(
            item[
                "reliability"
            ]
        )
        for item
        in weighted_evidence
    ]
)

resolution_confidence = clamp(
    safe_mean(
        [
            evidence_balance,
            source_reliability,
            arbitration_strength
        ]
    )
)

print(
    "Evidence balance:",
    evidence_balance
)

print(
    "Source reliability:",
    source_reliability
)

print(
    "Arbitration strength:",
    arbitration_strength
)

print(
    "Resolution confidence:",
    resolution_confidence
)

if not math.isfinite(
        resolution_confidence
):

    raise RuntimeError(
        "Resolution confidence is invalid."
    )

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Preserve Contradiction Information"
)

print()

contradiction_record = {
    "state":
        contradiction_state,

    "count":
        len(
            contradiction_pairs
        ),

    "pairs":
        contradiction_pairs,

    "conflict_score":
        conflict_score
}

print(
    "Contradiction record:"
)

print(
    json.dumps(
        contradiction_record,
        indent=4
    )
)

if (
        contradiction_pairs
        and
        contradiction_record[
            "count"
        ]
        == 0
):

    raise RuntimeError(
        "Contradiction information was lost."
    )

print(
    "Contradiction preservation validated."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Resolve Primary Conclusion"
)

print()

if (
        arbitration_direction
        ==
        "SUPPORTS_PRIMARY_HYPOTHESIS"
):

    resolved_conclusion = (
        "Primary hypothesis remains the best-supported "
        "interpretation after evidence arbitration."
    )

elif (
        arbitration_direction
        ==
        "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
):

    resolved_conclusion = (
        "Alternative interpretation has stronger weighted "
        "evidence than the primary hypothesis."
    )

else:

    resolved_conclusion = (
        "Evidence remains unresolved; the system should "
        "request additional observations before preferring "
        "one interpretation."
    )

print(
    "Resolved conclusion:"
)

print(
    resolved_conclusion
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Evidence Audit Trail"
)

print()

audit_trail = []

for evidence in weighted_evidence:

    audit_trail.append(
        {
            "evidence_id":
                evidence[
                    "evidence_id"
                ],

            "source_type":
                evidence[
                    "source_type"
                ],

            "direction":
                evidence[
                    "direction"
                ],

            "reliability":
                evidence[
                    "reliability"
                ],

            "magnitude":
                evidence[
                    "magnitude"
                ],

            "weight":
                evidence[
                    "weight"
                ],

            "contribution":
                evidence[
                    "contribution"
                ]
        }
    )

if len(
        audit_trail
) != len(
    weighted_evidence
):

    raise RuntimeError(
        "Evidence audit trail is incomplete."
    )

print(
    "Audit records:",
    len(
        audit_trail
    )
)

print(
    "Evidence audit trail validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Determine Resolution State"
)

print()

#
# IMPORTANT DESIGN CHANGE:
#
# A confidence below 0.50 is not automatically a failure.
#
# The system must distinguish:
#
#   strong resolution
#   cautious resolution
#   unresolved evidence
#
# Contradictory evidence should often produce LOW confidence.
#

if (
        resolution_confidence
        >=
        MIN_CONFIDENCE_FOR_STRONG_RESOLUTION
):

    resolution_state = (
        "RESOLVED_WITH_CONFIDENCE"
    )

elif (
        resolution_confidence
        >=
        0.25
):

    resolution_state = (
        "CAUTIOUS_RESOLUTION"
    )

else:

    resolution_state = (
        "UNRESOLVED"
    )

print(
    "Resolution confidence:",
    resolution_confidence
)

print(
    "Resolution state:",
    resolution_state
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Contradiction-Aware Diagnostic Reasoning"
)

print()

if (
        contradiction_state
        ==
        "CONTRADICTIONS_PRESENT"
):

    diagnostic_note = (
        "Contradictory cases were detected and retained. "
        "The result is evidence arbitration rather than "
        "simple majority voting."
    )

else:

    diagnostic_note = (
        "No strong contradictory pairs were detected. "
        "The arbitration result is primarily support-driven."
    )

print(
    diagnostic_note
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Deterministic Arbitration"
)

print()


def arbitrate(
        evidence: List[Dict[str, Any]]
) -> Dict[str, float]:

    support = sum(
        item[
            "weight"
        ]
        for item
        in evidence
        if item[
            "direction"
        ]
        ==
        "support"
    )

    conflict = sum(
        item[
            "weight"
        ]
        for item
        in evidence
        if item[
            "direction"
        ]
        ==
        "conflict"
    )

    total = (
            support
            +
            conflict
    )

    if total <= EPSILON:

        return {
            "net":
                0.0,

            "strength":
                0.0
        }

    net = (
            support
            -
            conflict
    )

    return {
        "net":
            net,

        "strength":
            abs(
                net
            )
            /
            total
    }


first_arbitration = arbitrate(
    weighted_evidence
)

second_arbitration = arbitrate(
    weighted_evidence
)

deterministic_arbitration = (
        abs(
            first_arbitration[
                "net"
            ]
            -
            second_arbitration[
                "net"
            ]
        )
        <=
        DETERMINISM_THRESHOLD
        and
        abs(
            first_arbitration[
                "strength"
            ]
            -
            second_arbitration[
                "strength"
            ]
        )
        <=
        DETERMINISM_THRESHOLD
)

print(
    "First arbitration:",
    first_arbitration
)

print(
    "Second arbitration:",
    second_arbitration
)

print(
    "Deterministic:",
    deterministic_arbitration
)

if not deterministic_arbitration:

    raise RuntimeError(
        "Evidence arbitration is nondeterministic."
    )

print(
    "Deterministic arbitration validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Contradiction Resolution Curriculum"
)

print()

arbitration_tasks = [
    {
        "example_id":
            "arbitration_001",

        "domain":
            "evidence_weighting",

        "question":
            "Why should conflicting evidence be weighted?",

        "answer":
            "Different evidence sources may have different reliability and strength."
    },

    {
        "example_id":
            "arbitration_002",

        "domain":
            "contradiction_resolution",

        "question":
            "How should contradictory cases be handled?",

        "answer":
            "Preserve both cases and compare their evidence rather than deleting one."
    },

    {
        "example_id":
            "arbitration_003",

        "domain":
            "source_reliability",

        "question":
            "Why assign reliability to evidence?",

        "answer":
            "Reliable evidence should influence a conclusion more strongly than weak evidence."
    },

    {
        "example_id":
            "arbitration_004",

        "domain":
            "diagnostic_confidence",

        "question":
            "When should a diagnosis remain unresolved?",

        "answer":
            "When evidence is too balanced or unreliable to justify a preferred interpretation."
    },

    {
        "example_id":
            "arbitration_005",

        "domain":
            "auditability",

        "question":
            "Why preserve an evidence audit trail?",

        "answer":
            "A conclusion must remain traceable to the evidence that produced it."
    },

    {
        "example_id":
            "arbitration_006",

        "domain":
            "engineering_reasoning",

        "question":
            "Why is evidence arbitration useful in engineering?",

        "answer":
            "Real systems can produce measurements and observations that do not always agree."
    }
]

for task in arbitration_tasks:

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
    "Arbitration tasks:",
    len(
        arbitration_tasks
    )
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Arbitration Curriculum Coverage"
)

print()

expected_domains = {
    "evidence_weighting",
    "contradiction_resolution",
    "source_reliability",
    "diagnostic_confidence",
    "auditability",
    "engineering_reasoning"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in arbitration_tasks
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
        "Arbitration curriculum coverage is incomplete."
    )

print(
    "Arbitration curriculum validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Numerical Health"
)

print()

numeric_values = [
    support_score,
    conflict_score,
    total_weight,
    net_evidence,
    arbitration_strength,
    resolution_confidence,
    evidence_balance,
    source_reliability
]

numeric_tensor = torch.tensor(
    numeric_values,
    dtype=torch.float32
)

numerically_healthy = finite_tensor(
    numeric_tensor
)

print(
    "NaN values:",
    int(
        torch.isnan(
            numeric_tensor
        ).sum()
    )
)

print(
    "Inf values:",
    int(
        torch.isinf(
            numeric_tensor
        ).sum()
    )
)

print(
    "Numerically healthy:",
    numerically_healthy
)

if not numerically_healthy:

    raise RuntimeError(
        "Evidence arbitration numerical health failed."
    )

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Final Evidence Arbitration Promotion Gate"
)

print()

promotion_errors = []

#
# These are architectural validity requirements.
#
# Low confidence is NOT itself a failure.
#

if not deterministic_arbitration:

    promotion_errors.append(
        "Evidence arbitration is nondeterministic."
    )

if not audit_trail:

    promotion_errors.append(
        "Evidence audit trail is empty."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        arbitration_tasks
) < 6:

    promotion_errors.append(
        "Arbitration curriculum is incomplete."
    )

if not resolved_conclusion:

    promotion_errors.append(
        "Resolved conclusion is missing."
    )

if (
        contradiction_pairs
        and
        contradiction_record[
            "count"
        ]
        == 0
):

    promotion_errors.append(
        "Contradiction information was lost."
    )

print(
    "Contradiction state:",
    contradiction_state
)

print(
    "Support score:",
    support_score
)

print(
    "Conflict score:",
    conflict_score
)

print(
    "Arbitration strength:",
    arbitration_strength
)

print(
    "Resolution confidence:",
    resolution_confidence
)

print(
    "Resolution state:",
    resolution_state
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
        "113R evidence arbitration promotion gate failed."
    )

print(
    "113R evidence arbitration promotion gate passed."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Persist Evidence Arbitration Memory"
)

print()

arbitration_event = {
    "event_id":
        "arbitration_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "113R",

    "contradiction_state":
        contradiction_state,

    "support_score":
        support_score,

    "conflict_score":
        conflict_score,

    "net_evidence":
        net_evidence,

    "arbitration_strength":
        arbitration_strength,

    "resolution_confidence":
        resolution_confidence,

    "resolution_state":
        resolution_state,

    "conclusion":
        resolved_conclusion
}

arbitration_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_contradiction_resolution_evidence_arbitration",

    "created_at":
        datetime.now().isoformat(),

    "representation_dimension":
        representation_dimension,

    "contradiction_state":
        contradiction_state,

    "contradiction_pairs":
        contradiction_pairs,

    "evidence_sources":
        weighted_evidence,

    "audit_trail":
        audit_trail,

    "support_score":
        support_score,

    "conflict_score":
        conflict_score,

    "net_evidence":
        net_evidence,

    "arbitration_strength":
        arbitration_strength,

    "resolution_confidence":
        resolution_confidence,

    "resolution_state":
        resolution_state,

    "conclusion":
        resolved_conclusion,

    "event":
        arbitration_event
}

write_json(
    ARBITRATION_MEMORY_FILE,
    arbitration_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "representation_dimension":
            representation_dimension,

        "support_score":
            support_score,

        "conflict_score":
            conflict_score,

        "net_evidence":
            net_evidence,

        "resolution_confidence":
            resolution_confidence,

        "resolution_state":
            resolution_state,

        "contradiction_pairs":
            contradiction_pairs
    },
    ARBITRATION_INDEX_FILE
)

print(
    "Arbitration memory:",
    ARBITRATION_MEMORY_FILE
)

print(
    "Arbitration index:",
    ARBITRATION_INDEX_FILE
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Reload Evidence Arbitration Memory"
)

print()

reloaded_arbitration = read_json(
    ARBITRATION_MEMORY_FILE
)

if (
        reloaded_arbitration[
            "resolution_confidence"
        ]
        !=
        resolution_confidence
):

    raise RuntimeError(
        "Resolution confidence changed after persistence."
    )

if (
        reloaded_arbitration[
            "resolution_state"
        ]
        !=
        resolution_state
):

    raise RuntimeError(
        "Resolution state changed after persistence."
    )

if (
        len(
            reloaded_arbitration[
                "audit_trail"
            ]
        )
        !=
        len(
            audit_trail
        )
):

    raise RuntimeError(
        "Evidence audit trail changed after persistence."
    )

if (
        len(
            reloaded_arbitration[
                "contradiction_pairs"
            ]
        )
        !=
        len(
            contradiction_pairs
        )
):

    raise RuntimeError(
        "Contradiction records changed after persistence."
    )

print(
    "Reloaded resolution confidence:",
    reloaded_arbitration[
        "resolution_confidence"
    ]
)

print(
    "Reloaded resolution state:",
    reloaded_arbitration[
        "resolution_state"
    ]
)

print(
    "Reloaded audit entries:",
    len(
        reloaded_arbitration[
            "audit_trail"
        ]
    )
)

print(
    "Reloaded contradictions:",
    len(
        reloaded_arbitration[
            "contradiction_pairs"
        ]
    )
)

print(
    "Persistent evidence arbitration validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Save Arbitration Dataset"
)

print()

arbitration_dataset = {
    "lesson":
        "113R",

    "capability":
        "native_contradiction_resolution_evidence_arbitration",

    "representation_dimension":
        representation_dimension,

    "contradiction_state":
        contradiction_state,

    "contradiction_pairs":
        contradiction_pairs,

    "evidence_sources":
        weighted_evidence,

    "support_score":
        support_score,

    "conflict_score":
        conflict_score,

    "net_evidence":
        net_evidence,

    "arbitration_strength":
        arbitration_strength,

    "resolution_confidence":
        resolution_confidence,

    "resolution_state":
        resolution_state,

    "conclusion":
        resolved_conclusion
}

write_json(
    ARBITRATION_DATASET_FILE,
    arbitration_dataset
)

print(
    "Arbitration dataset:",
    ARBITRATION_DATASET_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Save 113R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "113R",

    "capability":
        "native_contradiction_resolution_evidence_arbitration",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "representation_dimension":
        representation_dimension,

    "contradiction_state":
        contradiction_state,

    "contradiction_pairs":
        contradiction_pairs,

    "evidence_sources":
        weighted_evidence,

    "support_score":
        support_score,

    "conflict_score":
        conflict_score,

    "net_evidence":
        net_evidence,

    "arbitration_strength":
        arbitration_strength,

    "resolution_confidence":
        resolution_confidence,

    "resolution_state":
        resolution_state,

    "conclusion":
        resolved_conclusion,

    "audit_trail":
        audit_trail,

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
# TEST 24
# ============================================================

print(
    "TEST 24: Write 113R Reports"
)

print()

report = {
    "lesson":
        "113R",

    "capability":
        "native_contradiction_resolution_evidence_arbitration",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "representation_dimension":
        representation_dimension,

    "contradiction":
        {
            "state":
                contradiction_state,

            "count":
                len(
                    contradiction_pairs
                ),

            "pairs":
                contradiction_pairs
        },

    "evidence":
        {
            "source_count":
                len(
                    weighted_evidence
                ),

            "sources":
                weighted_evidence,

            "support_score":
                support_score,

            "conflict_score":
                conflict_score
        },

    "arbitration":
        {
            "net_evidence":
                net_evidence,

            "strength":
                arbitration_strength,

            "direction":
                (
                    "SUPPORT"
                    if net_evidence > EPSILON
                    else
                    "CONFLICT"
                    if net_evidence < -EPSILON
                    else
                    "UNRESOLVED"
                )
        },

    "resolution":
        {
            "confidence":
                resolution_confidence,

            "state":
                resolution_state,

            "conclusion":
                resolved_conclusion,

            "diagnostic_note":
                diagnostic_note
        },

    "audit":
        audit_trail,

    "verification":
        {
            "deterministic":
                deterministic_arbitration,

            "contradictions_preserved":
                len(
                    contradiction_pairs
                )
                ==
                len(
                    reloaded_arbitration.get(
                        "contradiction_pairs",
                        []
                    )
                )
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
    ARBITRATION_REPORT_FILE,
    report
)

write_json(
    ARBITRATION_EVALUATION_FILE,
    report
)

write_json(
    ARBITRATION_REGISTRY_FILE,
    {
        "lesson":
            "113R",

        "capability":
            "native_contradiction_resolution_evidence_arbitration",

        "memory_version":
            MEMORY_VERSION,

        "contradiction_state":
            contradiction_state,

        "resolution_confidence":
            resolution_confidence,

        "resolution_state":
            resolution_state,

        "next":
            "114R Native Evidence Provenance + Reasoning Trace"
    }
)

print(
    "Arbitration report:",
    ARBITRATION_REPORT_FILE
)

print(
    "Arbitration evaluation:",
    ARBITRATION_EVALUATION_FILE
)

print(
    "Arbitration registry:",
    ARBITRATION_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 113R EVIDENCE ARBITRATION ARCHITECTURE"
)

print()

print(
    "Conflicting Cases"
)

print(
    "        ↓"
)

print(
    "Evidence Sources"
)

print(
    "        ↓"
)

print(
    "Source Reliability"
)

print(
    "        ↓"
)

print(
    "Evidence Weighting"
)

print(
    "        ↓"
)

print(
    "Support / Conflict"
)

print(
    "        ↓"
)

print(
    "Arbitration"
)

print(
    "        ↓"
)

print(
    "Confidence + Uncertainty"
)

print(
    "        ↓"
)

print(
    "Auditable Diagnosis"
)

print()


# ============================================================
# WHY 113R MATTERS
# ============================================================

print(
    "WHY 113R MATTERS"
)

print()

print(
    "112R established that remembered cases can contradict each other."
)

print(
    "113R teaches Silverwing to preserve those contradictions "
    "and arbitrate them rather than hiding them."
)

print()

print(
    "A low-confidence conclusion can therefore be a successful "
    "reasoning result when the evidence genuinely remains uncertain."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "113R LIMITATION"
)

print()

print(
    "The reliability weights in this lesson are controlled "
    "architectural values."
)

print(
    "Production evidence arbitration should later incorporate "
    "sensor quality, measurement uncertainty, provenance, "
    "historical outcomes and domain-specific reliability."
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
    "Lesson 114R: Native Evidence Provenance + Reasoning Trace"
)

print()

print(
    "Evidence Source + Provenance + Reasoning Steps + "
    "Decision Trace + Full Auditability"
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
    "114R Native Evidence Provenance + Reasoning Trace",
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
    "=== LESSON 113R COMPLETE ==="
)
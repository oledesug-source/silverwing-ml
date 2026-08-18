# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 118R
# Native Error Pattern Generalization + Preventive Reasoning
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
# 91R  -> Machine Learning Foundations
# 92R  -> Classical Machine Learning
# 93R  -> Neural Network Foundations
# 94R  -> Deep Learning
# 95R  -> Representation Learning
# 96R  -> Sequence Representation Learning
# 97R  -> Structured Representation Learning
# 98R  -> Advanced Sequence + Structured Learning
# 99R  -> Multimodal Representation Foundations
# 100R -> Cross-Modal Alignment + Retrieval
# 101R -> Hard-Negative Multimodal Learning
# 102R -> Multimodal Memory Integration
# 103R -> Memory Consolidation + Temporal Retrieval
# 104R -> Multimodal Memory Reasoning
# 105R -> Memory Prediction + State Forecasting
# 106R -> Predictive Memory + Anomaly Detection
# 107R -> Predictive Risk + Failure Reasoning
# 108R -> Failure Pattern Memory + Retrieval
# 109R -> Failure Pattern Clustering + Prototype Memory
# 110R -> Failure Prototype Evolution + Continual Memory
# 111R -> Novel Failure Discovery + Prototype Birth
# 112R -> Failure Prototype Validation + Cross-Case Reasoning
# 113R -> Contradiction Resolution + Evidence Arbitration
# 114R -> Evidence Provenance + Reasoning Trace
# 115R -> Reasoning Replay + Decision Verification
# 116R -> Independent Reasoning Validator + Error Detection
# 117R -> Reasoning Error Memory + Self-Correction
# 118R -> Error Pattern Generalization + Preventive Reasoning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 118R takes a verified reasoning error from 117R and turns it
# into a reusable prevention pattern.
#
# The system:
#
#   verified error memory
#          ↓
#   error pattern extraction
#          ↓
#   generalized pattern
#          ↓
#   prevention rule
#          ↓
#   pre-reasoning scan
#          ↓
#   risk assessment
#          ↓
#   prevention / continuation decision
#          ↓
#   verified outcome
#
# ============================================================
# IMPORTANT
# ============================================================
#
# This lesson does not claim that one synthetic error is enough
# to learn a production-grade prevention policy.
#
# It establishes the architecture and verifies that:
#
#   detected error
#       can become
#   reusable preventive knowledge
#
# ============================================================
# EXTERNAL LLM
# ============================================================
#
# NONE
#
# ============================================================

import hashlib
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

MEMORY_VERSION = "118R.1"

EPSILON = 1e-8

NUMERIC_TOLERANCE = 1e-6

DETERMINISM_TOLERANCE = 1e-9

PREVENTION_THRESHOLD = 0.50

HIGH_RISK_THRESHOLD = 0.75

MEDIUM_RISK_THRESHOLD = 0.40

TRACE_MINIMUM = 6


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_117R = (
        PHASE5_DIR /
        "lesson117R"
)

SOURCE_ERROR_MEMORY_FILE = (
        LESSON_117R /
        "silverwing_reasoning_error_memory.json"
)

SOURCE_ERROR_INDEX_FILE = (
        LESSON_117R /
        "silverwing_reasoning_error_index.pt"
)

SOURCE_ERROR_DATASET_FILE = (
        LESSON_117R /
        "silverwing_reasoning_error_dataset.json"
)

SOURCE_ERROR_REPORT_FILE = (
        LESSON_117R /
        "silverwing_reasoning_error_report.json"
)

SOURCE_ERROR_REGISTRY_FILE = (
        LESSON_117R /
        "silverwing_reasoning_error_registry.json"
)

SOURCE_ERROR_CHECKPOINT_PRIMARY = (
        LESSON_117R /
        "checkpoints" /
        "silverwing_reasoning_error_best.pt"
)

SOURCE_ERROR_CHECKPOINT_CANDIDATE = (
        LESSON_117R /
        "checkpoints" /
        "silverwing_reasoning_error_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREVENTION_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_memory.json"
)

PREVENTION_INDEX_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_index.pt"
)

PREVENTION_DATASET_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_dataset.json"
)

PREVENTION_REPORT_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_report.json"
)

PREVENTION_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_evaluation.json"
)

PREVENTION_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_preventive_reasoning_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_preventive_reasoning_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_preventive_reasoning_best.pt"
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


def stable_hash(
        value: Any
) -> str:

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


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


def nearly_equal(
        left: float,
        right: float,
        tolerance: float = NUMERIC_TOLERANCE
) -> bool:

    return (
            abs(
                float(left)
                -
                float(right)
            )
            <=
            tolerance
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
        SOURCE_ERROR_CHECKPOINT_PRIMARY,
        SOURCE_ERROR_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 117R checkpoint found."
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
    "PHASE 5 - LESSON 118R"
)

print(
    "Native Error Pattern Generalization + Preventive Reasoning"
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
    "113R -> Contradiction Resolution + Evidence Arbitration",
    "114R -> Evidence Provenance + Reasoning Trace",
    "115R -> Reasoning Replay + Decision Verification",
    "116R -> Independent Reasoning Validator + Error Detection",
    "117R -> Reasoning Error Memory + Self-Correction",
    "118R -> Error Pattern Generalization + Preventive Reasoning"
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
    "Prevention threshold:",
    PREVENTION_THRESHOLD
)

print(
    "High-risk threshold:",
    HIGH_RISK_THRESHOLD
)

print(
    "Medium-risk threshold:",
    MEDIUM_RISK_THRESHOLD
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 117R Error-Memory Inputs"
)

print()

for path in [
    SOURCE_ERROR_MEMORY_FILE,
    SOURCE_ERROR_INDEX_FILE,
    SOURCE_ERROR_DATASET_FILE,
    SOURCE_ERROR_REPORT_FILE,
    SOURCE_ERROR_REGISTRY_FILE
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
    SOURCE_ERROR_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_ERROR_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_ERROR_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_ERROR_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_ERROR_REGISTRY_FILE
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
    "TEST 2: Load Verified Error Memory"
)

print()

error_memory = read_json(
    SOURCE_ERROR_MEMORY_FILE
)

error_dataset = read_json(
    SOURCE_ERROR_DATASET_FILE
)

error_report = read_json(
    SOURCE_ERROR_REPORT_FILE
)

if not isinstance(
        error_memory,
        dict
):

    raise RuntimeError(
        "117R error memory is invalid."
    )

error_records = error_memory.get(
    "error_records"
)

correction_strategies = error_memory.get(
    "correction_strategies"
)

verification = error_memory.get(
    "verification"
)

if not isinstance(
        error_records,
        list
):

    raise RuntimeError(
        "117R error records are unavailable."
    )

if not isinstance(
        correction_strategies,
        list
):

    raise RuntimeError(
        "117R correction strategies are unavailable."
    )

if not isinstance(
        verification,
        dict
):

    raise RuntimeError(
        "117R verification record is unavailable."
    )

print(
    "Memory version:",
    error_memory.get(
        "memory_version"
    )
)

print(
    "Reasoning id:",
    error_memory.get(
        "reasoning_id"
    )
)

print(
    "Error records:",
    len(
        error_records
    )
)

print(
    "Correction strategies:",
    len(
        correction_strategies
    )
)

print(
    "Self-correction verified:",
    verification.get(
        "self_correction_verified"
    )
)

print(
    "Error before:",
    verification.get(
        "error_before"
    )
)

print(
    "Error after:",
    verification.get(
        "error_after"
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Validate Verified Error Record"
)

print()

required_error_fields = {
    "error_id",
    "source_lesson",
    "error_type",
    "evidence_id",
    "field",
    "expected_value",
    "observed_value",
    "affected_fields",
    "detection_method",
    "correction_required"
}

error_schema_errors = []

for record in error_records:

    missing = (
            required_error_fields
            -
            set(
                record.keys()
            )
    )

    if missing:

        error_schema_errors.append(
            {
                "error_id":
                    record.get(
                        "error_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

if error_schema_errors:

    print(
        json.dumps(
            error_schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "117R error-memory schema validation failed."
    )

print(
    "Verified error schema validated."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Confirm Correction Was Actually Verified"
)

print()

self_correction_verified = (
        verification.get(
            "self_correction_verified"
        )
        is True
)

error_before = float(
    verification.get(
        "error_before",
        0.0
    )
)

error_after = float(
    verification.get(
        "error_after",
        0.0
    )
)

correction_confidence = float(
    verification.get(
        "correction_confidence",
        0.0
    )
)

print(
    "Self-correction verified:",
    self_correction_verified
)

print(
    "Error before:",
    error_before
)

print(
    "Error after:",
    error_after
)

print(
    "Correction confidence:",
    correction_confidence
)

if not self_correction_verified:

    raise RuntimeError(
        "118R requires a verified 117R correction."
    )

if error_after > NUMERIC_TOLERANCE:

    raise RuntimeError(
        "117R correction still contains residual error."
    )

print(
    "117R verified correction accepted."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Extract Error Pattern"
)

print()

primary_error = error_records[
    0
]

primary_strategy = correction_strategies[
    0
]

generalized_error_pattern = {
    "pattern_id":
        "error_pattern_001",

    "source_error_type":
        primary_error[
            "error_type"
        ],

    "source_field":
        primary_error[
            "field"
        ],

    "observable_signature":
        {
            "field":
                primary_error[
                    "field"
                ],

            "expected_value":
                primary_error[
                    "expected_value"
                ],

            "observed_value":
                primary_error[
                    "observed_value"
                ]
        },

    "detection_method":
        primary_error[
            "detection_method"
        ],

    "verified_correction":
        {
            "strategy_id":
                primary_strategy[
                    "strategy_id"
                ],

            "strategy_type":
                primary_strategy[
                    "strategy_type"
                ]
        },

    "source_lesson":
        primary_error[
            "source_lesson"
        ],

    "generalization_scope":
        "evidence_representation_mismatch"
}

print(
    json.dumps(
        generalized_error_pattern,
        indent=4
    )
)

print(
    "Error pattern extracted."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Preventive Rule"
)

print()

prevention_rule = {
    "rule_id":
        "prevention_rule_001",

    "pattern_id":
        generalized_error_pattern[
            "pattern_id"
        ],

    "trigger":
        {
            "field":
                primary_error[
                    "field"
                ],

            "condition":
                "observed_value differs from provenance-backed value"
        },

    "action":
        "STOP_BEFORE_REASONING_COMPLETES",

    "validation":
        "recalculate_using_provenance_value",

    "promotion_requirement":
        "independent_verification",

    "risk_level":
        "HIGH",

    "description":
        (
            "When an evidence representation differs from its "
            "provenance-backed value, prevent downstream reasoning "
            "until the evidence is restored and independently verified."
        )
}

print(
    json.dumps(
        prevention_rule,
        indent=4
    )
)

print(
    "Preventive rule constructed."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Build Pre-Reasoning Validation Input"
)

print()

validation_case = {
    "case_id":
        "preventive_case_001",

    "evidence_id":
        primary_error[
            "evidence_id"
        ],

    "field":
        primary_error[
            "field"
        ],

    "expected_value":
        primary_error[
            "expected_value"
        ],

    "observed_value":
        primary_error[
            "observed_value"
        ],

    "rule_id":
        prevention_rule[
            "rule_id"
        ]
}

print(
    json.dumps(
        validation_case,
        indent=4
    )
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Execute Pre-Reasoning Scan"
)

print()

expected_value = float(
    validation_case[
        "expected_value"
    ]
)

observed_value = float(
    validation_case[
        "observed_value"
    ]
)

absolute_difference = abs(
    expected_value
    -
    observed_value
)

relative_difference = (
        absolute_difference
        /
        max(
            abs(
                expected_value
            ),
            EPSILON
        )
)

mismatch_detected = (
        absolute_difference
        >
        NUMERIC_TOLERANCE
)

print(
    "Expected value:",
    expected_value
)

print(
    "Observed value:",
    observed_value
)

print(
    "Absolute difference:",
    absolute_difference
)

print(
    "Relative difference:",
    relative_difference
)

print(
    "Mismatch detected:",
    mismatch_detected
)

if not mismatch_detected:

    raise RuntimeError(
        "Preventive scan failed to identify the known error pattern."
    )

print(
    "Pre-reasoning mismatch detected."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Calculate Preventive Risk"
)

print()

pattern_confidence = clamp(
    correction_confidence
)

magnitude_component = clamp(
    relative_difference
)

preventive_risk = clamp(
    safe_mean(
        [
            pattern_confidence,
            magnitude_component,
            1.0 if mismatch_detected else 0.0
        ]
    )
)

print(
    "Pattern confidence:",
    pattern_confidence
)

print(
    "Mismatch magnitude component:",
    magnitude_component
)

print(
    "Preventive risk:",
    preventive_risk
)

if not math.isfinite(
        preventive_risk
):

    raise RuntimeError(
        "Preventive risk is invalid."
    )

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Preventive Decision"
)

print()

if (
        preventive_risk
        >=
        HIGH_RISK_THRESHOLD
):

    preventive_decision = (
        "BLOCK_REASONING"
    )

elif (
        preventive_risk
        >=
        MEDIUM_RISK_THRESHOLD
):

    preventive_decision = (
        "REQUIRE_VALIDATION"
    )

else:

    preventive_decision = (
        "ALLOW_REASONING"
    )

print(
    "Preventive risk:",
    preventive_risk
)

print(
    "Preventive decision:",
    preventive_decision
)

if preventive_decision not in {
    "BLOCK_REASONING",
    "REQUIRE_VALIDATION",
    "ALLOW_REASONING"
}:

    raise RuntimeError(
        "Invalid preventive decision."
    )

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Apply Preventive Rule"
)

print()

prevention_applied = (
        preventive_decision
        ==
        "BLOCK_REASONING"
        or
        preventive_decision
        ==
        "REQUIRE_VALIDATION"
)

if mismatch_detected:

    if not prevention_applied:

        raise RuntimeError(
            "Known error pattern was detected but prevention did not activate."
        )

print(
    "Prevention applied:",
    prevention_applied
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Correct Before Downstream Reasoning"
)

print()

preventively_corrected_value = (
    expected_value
    if prevention_applied
    else
    observed_value
)

preventive_correction_error = abs(
    preventively_corrected_value
    -
    expected_value
)

print(
    "Observed value:",
    observed_value
)

print(
    "Preventively corrected value:",
    preventively_corrected_value
)

print(
    "Residual error:",
    preventive_correction_error
)

if (
        prevention_applied
        and
        preventive_correction_error
        >
        NUMERIC_TOLERANCE
):

    raise RuntimeError(
        "Preventive correction did not restore the validated source value."
    )

print(
    "Preventive correction validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Prevented-Reasoning Replay"
)

print()

#
# The prevention mechanism must stop reasoning before the
# corrupted value is allowed to propagate.
#
# The "safe" path uses the provenance-backed value.
#

safe_value = (
    preventively_corrected_value
)

safe_deviation = abs(
    safe_value
    -
    expected_value
)

prevention_success = (
        safe_deviation
        <=
        NUMERIC_TOLERANCE
)

print(
    "Safe value:",
    safe_value
)

print(
    "Safe deviation:",
    safe_deviation
)

print(
    "Prevention success:",
    prevention_success
)

if not prevention_success:

    raise RuntimeError(
        "Preventive reasoning failed to protect the downstream value."
    )

print(
    "Prevented reasoning replay validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Generalize Beyond Exact Error Instance"
)

print()

#
# Use a second deterministic case with a different numeric value
# but the same structural error signature.
#

generalization_case = {
    "case_id":
        "preventive_case_002",

    "evidence_id":
        "evidence_generalized_001",

    "field":
        primary_error[
            "field"
        ],

    "expected_value":
        0.25,

    "observed_value":
        0.0,

    "rule_id":
        prevention_rule[
            "rule_id"
        ]
}

generalized_difference = abs(
    generalization_case[
        "expected_value"
    ]
    -
    generalization_case[
        "observed_value"
    ]
)

generalized_mismatch = (
        generalized_difference
        >
        NUMERIC_TOLERANCE
        and
        generalization_case[
            "field"
        ]
        ==
        prevention_rule[
            "trigger"
        ][
            "field"
        ]
)

print(
    "Generalization case:",
    generalization_case
)

print(
    "Generalized difference:",
    generalized_difference
)

print(
    "Generalized mismatch:",
    generalized_mismatch
)

if not generalized_mismatch:

    raise RuntimeError(
        "Error pattern failed to generalize beyond the original numeric instance."
    )

print(
    "Error pattern generalized successfully."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Preventive Knowledge Confidence"
)

print()

generalization_confidence = clamp(
    safe_mean(
        [
            pattern_confidence,
            1.0 if generalized_mismatch else 0.0,
            1.0 if prevention_success else 0.0
        ]
    )
)

print(
    "Generalization confidence:",
    generalization_confidence
)

if (
        generalization_confidence
        <
        PREVENTION_THRESHOLD
):

    raise RuntimeError(
        "Preventive knowledge confidence is below threshold."
    )

print(
    "Preventive knowledge confidence validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Preventive Reasoning Curriculum"
)

print()

prevention_tasks = [
    {
        "example_id":
            "prevention_001",

        "domain":
            "error_pattern_generalization",

        "question":
            "Why generalize a verified reasoning error?",

        "answer":
            "A reusable pattern can prevent similar errors in future reasoning."
    },

    {
        "example_id":
            "prevention_002",

        "domain":
            "preventive_validation",

        "question":
            "Why scan evidence before reasoning completes?",

        "answer":
            "It can stop corrupted evidence from propagating into later decisions."
    },

    {
        "example_id":
            "prevention_003",

        "domain":
            "risk_prediction",

        "question":
            "What is preventive reasoning?",

        "answer":
            "Reasoning that identifies known risk patterns before they produce an invalid conclusion."
    },

    {
        "example_id":
            "prevention_004",

        "domain":
            "rule_generation",

        "question":
            "What should a prevention rule contain?",

        "answer":
            "A detectable trigger, a protective action and a validation requirement."
    },

    {
        "example_id":
            "prevention_005",

        "domain":
            "continual_learning",

        "question":
            "How does preventive memory support continual learning?",

        "answer":
            "Verified errors become reusable constraints for future reasoning."
    },

    {
        "example_id":
            "prevention_006",

        "domain":
            "engineering_reliability",

        "question":
            "Why is preventive reasoning important in engineering intelligence?",

        "answer":
            "Known failure modes should be detected before they influence a critical conclusion."
    }
]

for task in prevention_tasks:

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
    "Prevention tasks:",
    len(
        prevention_tasks
    )
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Prevention Curriculum Coverage"
)

print()

expected_domains = {
    "error_pattern_generalization",
    "preventive_validation",
    "risk_prediction",
    "rule_generation",
    "continual_learning",
    "engineering_reliability"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in prevention_tasks
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
        "Prevention curriculum coverage is incomplete."
    )

print(
    "Prevention curriculum validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Numerical Health"
)

print()

health_values = [
    error_before,
    error_after,
    correction_confidence,
    absolute_difference,
    relative_difference,
    preventive_risk,
    preventive_correction_error,
    generalized_difference,
    generalization_confidence
]

health_tensor = torch.tensor(
    health_values,
    dtype=torch.float32
)

numerically_healthy = finite_tensor(
    health_tensor
)

print(
    "NaN values:",
    int(
        torch.isnan(
            health_tensor
        ).sum()
    )
)

print(
    "Inf values:",
    int(
        torch.isinf(
            health_tensor
        ).sum()
    )
)

print(
    "Numerically healthy:",
    numerically_healthy
)

if not numerically_healthy:

    raise RuntimeError(
        "Preventive reasoning numerical health failed."
    )

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Deterministic Preventive Reasoning"
)

print()


def preventive_scan(
        expected: float,
        observed: float,
        threshold: float
) -> Dict[str, Any]:

    difference = abs(
        expected
        -
        observed
    )

    mismatch = (
            difference
            >
            threshold
    )

    if mismatch:

        decision = (
            "BLOCK_REASONING"
        )

    else:

        decision = (
            "ALLOW_REASONING"
        )

    return {
        "difference":
            difference,

        "mismatch":
            mismatch,

        "decision":
            decision
    }


first_scan = preventive_scan(
    expected_value,
    observed_value,
    NUMERIC_TOLERANCE
)

second_scan = preventive_scan(
    expected_value,
    observed_value,
    NUMERIC_TOLERANCE
)

preventive_deterministic = (
        stable_hash(
            first_scan
        )
        ==
        stable_hash(
            second_scan
        )
)

print(
    "First scan:",
    first_scan
)

print(
    "Second scan:",
    second_scan
)

print(
    "Deterministic:",
    preventive_deterministic
)

if not preventive_deterministic:

    raise RuntimeError(
        "Preventive reasoning is nondeterministic."
    )

print(
    "Deterministic preventive reasoning validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Final Preventive Reasoning Promotion Gate"
)

print()

promotion_errors = []

if not prevention_success:

    promotion_errors.append(
        "Prevention did not successfully protect downstream reasoning."
    )

if not generalized_mismatch:

    promotion_errors.append(
        "Error pattern did not generalize."
    )

if (
        generalization_confidence
        <
        PREVENTION_THRESHOLD
):

    promotion_errors.append(
        "Generalization confidence is below threshold."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if not preventive_deterministic:

    promotion_errors.append(
        "Preventive reasoning is nondeterministic."
    )

if len(
        prevention_tasks
) < 6:

    promotion_errors.append(
        "Prevention curriculum is incomplete."
    )

if not prevention_rule.get(
        "rule_id"
):

    promotion_errors.append(
        "Prevention rule has no identity."
    )

print(
    "Preventive risk:",
    preventive_risk
)

print(
    "Preventive decision:",
    preventive_decision
)

print(
    "Generalization confidence:",
    generalization_confidence
)

print(
    "Prevention success:",
    prevention_success
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
        "118R preventive reasoning promotion gate failed."
    )

print(
    "118R preventive reasoning promotion gate passed."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Persist Preventive Reasoning Memory"
)

print()

prevention_event = {
    "event_id":
        "prevention_118_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "118R",

    "source_error_id":
        primary_error[
            "error_id"
        ],

    "pattern_id":
        generalized_error_pattern[
            "pattern_id"
        ],

    "rule_id":
        prevention_rule[
            "rule_id"
        ],

    "preventive_risk":
        preventive_risk,

    "decision":
        preventive_decision,

    "generalization_confidence":
        generalization_confidence,

    "prevention_success":
        prevention_success
}

prevention_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_error_pattern_generalization_preventive_reasoning",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        error_memory.get(
            "memory_version"
        ),

    "error_patterns":
        [
            generalized_error_pattern
        ],

    "prevention_rules":
        [
            prevention_rule
        ],

    "validation_cases":
        [
            validation_case,
            generalization_case
        ],

    "events":
        [
            prevention_event
        ],

    "evaluation":
        {
            "preventive_risk":
                preventive_risk,

            "preventive_decision":
                preventive_decision,

            "prevention_success":
                prevention_success,

            "generalization_confidence":
                generalization_confidence
        }
}

write_json(
    PREVENTION_MEMORY_FILE,
    prevention_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "pattern_ids":
            [
                generalized_error_pattern[
                    "pattern_id"
                ]
            ],

        "rule_ids":
            [
                prevention_rule[
                    "rule_id"
                ]
            ],

        "preventive_risk":
            preventive_risk,

        "generalization_confidence":
            generalization_confidence,

        "prevention_success":
            prevention_success
    },
    PREVENTION_INDEX_FILE
)

print(
    "Prevention memory:",
    PREVENTION_MEMORY_FILE
)

print(
    "Prevention index:",
    PREVENTION_INDEX_FILE
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Reload Preventive Memory"
)

print()

reloaded_prevention = read_json(
    PREVENTION_MEMORY_FILE
)

if (
        len(
            reloaded_prevention[
                "error_patterns"
            ]
        )
        !=
        1
):

    raise RuntimeError(
        "Generalized error pattern count changed."
    )

if (
        len(
            reloaded_prevention[
                "prevention_rules"
            ]
        )
        !=
        1
):

    raise RuntimeError(
        "Prevention rule count changed."
    )

if not nearly_equal(
        reloaded_prevention[
            "evaluation"
        ][
            "generalization_confidence"
        ],
        generalization_confidence
):

    raise RuntimeError(
        "Generalization confidence changed after persistence."
    )

print(
    "Reloaded error patterns:",
    len(
        reloaded_prevention[
            "error_patterns"
        ]
    )
)

print(
    "Reloaded prevention rules:",
    len(
        reloaded_prevention[
            "prevention_rules"
        ]
    )
)

print(
    "Reloaded generalization confidence:",
    reloaded_prevention[
        "evaluation"
    ][
        "generalization_confidence"
    ]
)

print(
    "Persistent preventive memory validated."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Save Preventive-Reasoning Dataset"
)

print()

prevention_dataset = {
    "lesson":
        "118R",

    "capability":
        "native_error_pattern_generalization_preventive_reasoning",

    "source_error":
        primary_error,

    "source_correction":
        primary_strategy,

    "generalized_error_pattern":
        generalized_error_pattern,

    "prevention_rule":
        prevention_rule,

    "validation_case":
        validation_case,

    "generalization_case":
        generalization_case,

    "preventive_risk":
        preventive_risk,

    "preventive_decision":
        preventive_decision,

    "prevention_success":
        prevention_success,

    "generalization_confidence":
        generalization_confidence
}

write_json(
    PREVENTION_DATASET_FILE,
    prevention_dataset
)

print(
    "Preventive dataset:",
    PREVENTION_DATASET_FILE
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save 118R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "118R",

    "capability":
        "native_error_pattern_generalization_preventive_reasoning",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "error_pattern":
        generalized_error_pattern,

    "prevention_rule":
        prevention_rule,

    "preventive_risk":
        preventive_risk,

    "preventive_decision":
        preventive_decision,

    "validation_case":
        validation_case,

    "generalization_case":
        generalization_case,

    "prevention_success":
        prevention_success,

    "generalization_confidence":
        generalization_confidence,

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
    "TEST 25: Write 118R Reports"
)

print()

report = {
    "lesson":
        "118R",

    "capability":
        "native_error_pattern_generalization_preventive_reasoning",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "source_error":
        primary_error,

    "generalized_pattern":
        generalized_error_pattern,

    "prevention_rule":
        prevention_rule,

    "risk":
        {
            "preventive_risk":
                preventive_risk,

            "decision":
                preventive_decision
        },

    "validation":
        {
            "prevention_success":
                prevention_success,

            "generalized_mismatch":
                generalized_mismatch,

            "generalization_confidence":
                generalization_confidence
        },

    "determinism":
        {
            "preventive_reasoning":
                preventive_deterministic
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
    PREVENTION_REPORT_FILE,
    report
)

write_json(
    PREVENTION_EVALUATION_FILE,
    report
)

write_json(
    PREVENTION_REGISTRY_FILE,
    {
        "lesson":
            "118R",

        "capability":
            "native_error_pattern_generalization_preventive_reasoning",

        "memory_version":
            MEMORY_VERSION,

        "rule_id":
            prevention_rule[
                "rule_id"
            ],

        "generalization_confidence":
            generalization_confidence,

        "prevention_success":
            prevention_success,

        "next":
            "119R Native Predictive Error Prevention + Preemptive Validation"
    }
)

print(
    "Prevention report:",
    PREVENTION_REPORT_FILE
)

print(
    "Prevention evaluation:",
    PREVENTION_EVALUATION_FILE
)

print(
    "Prevention registry:",
    PREVENTION_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 118R PREVENTIVE REASONING ARCHITECTURE"
)

print()

print(
    "Verified Error Memory"
)

print(
    "        ↓"
)

print(
    "Error Pattern Extraction"
)

print(
    "        ↓"
)

print(
    "Generalized Error Pattern"
)

print(
    "        ↓"
)

print(
    "Prevention Rule"
)

print(
    "        ↓"
)

print(
    "Pre-Reasoning Scan"
)

print(
    "        ↓"
)

print(
    "Risk Assessment"
)

print(
    "        ↓"
)

print(
    "Block / Validate / Allow"
)

print(
    "        ↓"
)

print(
    "Protected Reasoning"
)

print()


# ============================================================
# WHY 118R MATTERS
# ============================================================

print(
    "WHY 118R MATTERS"
)

print()

print(
    "117R taught Silverwing to remember and correct a reasoning error."
)

print(
    "118R turns that remembered error into preventive knowledge."
)

print()

print(
    "The system begins moving from:"
)

print(
    "reactive correction"
)

print(
    "to:"
)

print(
    "preventive reasoning."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "118R LIMITATION"
)

print()

print(
    "The current prevention rule is derived from a controlled "
    "synthetic error pattern."
)

print(
    "Production prevention requires many independent historical "
    "errors, calibrated recurrence statistics and real outcome data."
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
    "Lesson 119R: Native Predictive Error Prevention + Preemptive Validation"
)

print()

print(
    "Error Recurrence Memory + Risk Prediction + "
    "Preemptive Validation + Early Intervention + "
    "Verified Prevention"
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
    "115R Native Reasoning Replay + Decision Verification",
    " ↓",
    "116R Native Independent Reasoning Validator + Error Detection",
    " ↓",
    "117R Native Reasoning Error Memory + Self-Correction",
    " ↓",
    "118R Native Error Pattern Generalization + Preventive Reasoning",
    " ↓",
    "119R Native Predictive Error Prevention + Preemptive Validation",
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
    "=== LESSON 118R COMPLETE ==="
)
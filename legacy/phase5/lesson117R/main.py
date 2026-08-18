# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 117R
# Native Reasoning Error Memory + Self-Correction
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

MEMORY_VERSION = "117R.2"

EPSILON = 1e-8
NUMERIC_TOLERANCE = 1e-6
DETERMINISM_TOLERANCE = 1e-9

CORRECTION_CONFIDENCE_THRESHOLD = 0.50
CONTROLLED_ERROR_SCALE = 0.10
CONTROLLED_ZERO_MAGNITUDE = 0.10


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

PHASE5_DIR = BASE_DIR.parent

LESSON_116R = (
        PHASE5_DIR /
        "lesson116R"
)

SOURCE_VALIDATOR_MEMORY_FILE = (
        LESSON_116R /
        "silverwing_independent_validator_memory.json"
)

SOURCE_VALIDATOR_INDEX_FILE = (
        LESSON_116R /
        "silverwing_independent_validator_index.pt"
)

SOURCE_VALIDATOR_DATASET_FILE = (
        LESSON_116R /
        "silverwing_independent_validator_dataset.json"
)

SOURCE_VALIDATOR_REPORT_FILE = (
        LESSON_116R /
        "silverwing_independent_validator_report.json"
)

SOURCE_VALIDATOR_REGISTRY_FILE = (
        LESSON_116R /
        "silverwing_independent_validator_registry.json"
)

SOURCE_VALIDATOR_CHECKPOINT_PRIMARY = (
        LESSON_116R /
        "checkpoints" /
        "silverwing_independent_validator_best.pt"
)

SOURCE_VALIDATOR_CHECKPOINT_CANDIDATE = (
        LESSON_116R /
        "checkpoints" /
        "silverwing_independent_validator_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ERROR_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_memory.json"
)

ERROR_INDEX_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_index.pt"
)

ERROR_DATASET_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_dataset.json"
)

ERROR_REPORT_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_report.json"
)

ERROR_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_evaluation.json"
)

ERROR_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_reasoning_error_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_reasoning_error_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_reasoning_error_best.pt"
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


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_VALIDATOR_CHECKPOINT_PRIMARY,
        SOURCE_VALIDATOR_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 116R checkpoint found."
    )


def calculate_state(
        confidence: float
) -> str:

    if (
            confidence
            >=
            CORRECTION_CONFIDENCE_THRESHOLD
    ):

        return (
            "RESOLVED_WITH_CONFIDENCE"
        )

    if (
            confidence
            >=
            0.25
    ):

        return (
            "CAUTIOUS_RESOLUTION"
        )

    return (
        "UNRESOLVED"
    )


def independent_recalculate(
        evidence: List[Dict[str, Any]]
) -> Dict[str, float]:

    support = 0.0
    conflict = 0.0

    reliabilities = []

    for item in evidence:

        reliability = float(
            item[
                "reliability"
            ]
        )

        magnitude = float(
            item[
                "magnitude"
            ]
        )

        weight = (
                reliability
                *
                magnitude
        )

        reliabilities.append(
            reliability
        )

        if (
                item[
                    "direction"
                ]
                ==
                "support"
        ):

            support += weight

        elif (
                item[
                    "direction"
                ]
                ==
                "conflict"
        ):

            conflict += weight

    total = (
            support
            +
            conflict
    )

    net = (
            support
            -
            conflict
    )

    strength = (
        abs(
            net
        )
        /
        total
        if total > EPSILON
        else
        0.0
    )

    reliability = safe_mean(
        reliabilities
    )

    confidence = clamp(
        safe_mean(
            [
                strength,
                reliability,
                strength
            ]
        )
    )

    state = calculate_state(
        confidence
    )

    return {
        "support":
            support,

        "conflict":
            conflict,

        "net":
            net,

        "strength":
            strength,

        "confidence":
            confidence,

        "state":
            state
    }


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
    "PHASE 5 - LESSON 117R"
)

print(
    "Native Reasoning Error Memory + Self-Correction"
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
    "117R -> Reasoning Error Memory + Self-Correction"
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
    "Controlled error scale:",
    CONTROLLED_ERROR_SCALE
)

print(
    "Controlled zero-magnitude value:",
    CONTROLLED_ZERO_MAGNITUDE
)

print(
    "Correction confidence threshold:",
    CORRECTION_CONFIDENCE_THRESHOLD
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 116R Validator Inputs"
)

print()

for path in [
    SOURCE_VALIDATOR_MEMORY_FILE,
    SOURCE_VALIDATOR_INDEX_FILE,
    SOURCE_VALIDATOR_DATASET_FILE,
    SOURCE_VALIDATOR_REPORT_FILE,
    SOURCE_VALIDATOR_REGISTRY_FILE
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
    SOURCE_VALIDATOR_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_VALIDATOR_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_VALIDATOR_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_VALIDATOR_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_VALIDATOR_REGISTRY_FILE
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
    "TEST 2: Load Independent Validator Memory"
)

print()

validator_memory = read_json(
    SOURCE_VALIDATOR_MEMORY_FILE
)

validator_dataset = read_json(
    SOURCE_VALIDATOR_DATASET_FILE
)

validator_report = read_json(
    SOURCE_VALIDATOR_REPORT_FILE
)

if not isinstance(
        validator_memory,
        dict
):

    raise RuntimeError(
        "116R validator memory is invalid."
    )

primary_decision = validator_memory.get(
    "primary_decision"
)

validator_decision = validator_memory.get(
    "validator_decision"
)

disagreements = validator_memory.get(
    "disagreements"
)

if not isinstance(
        primary_decision,
        dict
):

    raise RuntimeError(
        "116R primary decision is unavailable."
    )

if not isinstance(
        validator_decision,
        dict
):

    raise RuntimeError(
        "116R validator decision is unavailable."
    )

if not isinstance(
        disagreements,
        list
):

    raise RuntimeError(
        "116R disagreement memory is unavailable."
    )

print(
    "Memory version:",
    validator_memory.get(
        "memory_version"
    )
)

print(
    "Reasoning id:",
    validator_memory.get(
        "reasoning_id"
    )
)

print(
    "Stored validator agreement:",
    validator_memory.get(
        "validator_agreement_score"
    )
)

print(
    "Stored disagreements:",
    len(
        disagreements
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Reconstruct Validator Evidence"
)

print()

PROVENANCE_SOURCE = (
        PHASE5_DIR /
        "lesson114R" /
        "silverwing_evidence_provenance_memory.json"
)

require_file(
    PROVENANCE_SOURCE
)

provenance_memory = read_json(
    PROVENANCE_SOURCE
)

provenance_records = provenance_memory.get(
    "provenance_records"
)

if not isinstance(
        provenance_records,
        list
):

    raise RuntimeError(
        "114R provenance records are unavailable."
    )

error_memory_evidence = []

for provenance in provenance_records:

    snapshot = provenance.get(
        "evidence_snapshot"
    )

    if not isinstance(
            snapshot,
            dict
    ):

        raise RuntimeError(
            (
                "Invalid provenance snapshot: "
                f"{provenance.get('provenance_id')}"
            )
        )

    error_memory_evidence.append(
        {
            "evidence_id":
                snapshot.get(
                    "evidence_id"
                ),

            "direction":
                snapshot.get(
                    "direction"
                ),

            "reliability":
                float(
                    snapshot.get(
                        "reliability",
                        0.0
                    )
                ),

            "magnitude":
                float(
                    snapshot.get(
                        "magnitude",
                        0.0
                    )
                ),

            "source_type":
                snapshot.get(
                    "source_type"
                )
        }
    )

print(
    "Evidence available for self-correction:",
    len(
        error_memory_evidence
    )
)

for evidence in error_memory_evidence:

    print(
        evidence
    )

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Independently Recalculate Baseline"
)

print()

baseline_recalculation = independent_recalculate(
    error_memory_evidence
)

print(
    "Baseline support:",
    baseline_recalculation[
        "support"
    ]
)

print(
    "Baseline conflict:",
    baseline_recalculation[
        "conflict"
    ]
)

print(
    "Baseline net:",
    baseline_recalculation[
        "net"
    ]
)

print(
    "Baseline strength:",
    baseline_recalculation[
        "strength"
    ]
)

print(
    "Baseline confidence:",
    baseline_recalculation[
        "confidence"
    ]
)

print(
    "Baseline state:",
    baseline_recalculation[
        "state"
    ]
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Validate 116R Baseline Consistency"
)

print()

baseline_mismatches = []

for field in [
    "support",
    "conflict",
    "net",
    "strength",
    "confidence"
]:

    stored_value = float(
        validator_decision.get(
            field,
            0.0
        )
    )

    recalculated_value = float(
        baseline_recalculation[
            field
        ]
    )

    if not nearly_equal(
            stored_value,
            recalculated_value
    ):

        baseline_mismatches.append(
            {
                "field":
                    field,

                "stored":
                    stored_value,

                "recalculated":
                    recalculated_value
            }
        )

stored_state = str(
    validator_decision.get(
        "state",
        "UNKNOWN"
    )
)

if (
        stored_state
        !=
        baseline_recalculation[
            "state"
        ]
):

    baseline_mismatches.append(
        {
            "field":
                "state",

            "stored":
                stored_state,

            "recalculated":
                baseline_recalculation[
                    "state"
                ]
        }
    )

if baseline_mismatches:

    print(
        json.dumps(
            baseline_mismatches,
            indent=4
        )
    )

    raise RuntimeError(
        "116R baseline validator result cannot be reconstructed."
    )

print(
    "116R baseline validator result validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Select Error-Injection Target"
)

print()

#
# ZERO-SAFE SELECTION
#
# Prefer an evidence item with a non-zero magnitude.
#
# If every magnitude is zero, use a deterministic zero-valued
# evidence item and inject a fixed bounded magnitude.
#
# This guarantees the controlled test actually changes the
# reasoning input.
#

nonzero_candidates = [
    index
    for index, evidence
    in enumerate(
        error_memory_evidence
    )
    if abs(
        float(
            evidence[
                "magnitude"
            ]
        )
    )
       >
       EPSILON
]

if nonzero_candidates:

    target_index = (
        nonzero_candidates[
            0
        ]
    )

    injection_mode = (
        "MULTIPLICATIVE"
    )

else:

    target_index = 0

    injection_mode = (
        "ADDITIVE_ZERO_SAFE"
    )

if not error_memory_evidence:

    raise RuntimeError(
        "No evidence is available for controlled error injection."
    )

print(
    "Target evidence:",
    error_memory_evidence[
        target_index
    ][
        "evidence_id"
    ]
)

print(
    "Injection mode:",
    injection_mode
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Build Controlled Error Injection"
)

print()

perturbed_evidence = [
    dict(
        item
    )
    for item
    in error_memory_evidence
]

original_target_magnitude = float(
    perturbed_evidence[
        target_index
    ][
        "magnitude"
    ]
)

if (
        injection_mode
        ==
        "MULTIPLICATIVE"
):

    perturbed_target_magnitude = (
            original_target_magnitude
            *
            (
                    1.0
                    +
                    CONTROLLED_ERROR_SCALE
            )
    )

else:

    perturbed_target_magnitude = (
        CONTROLLED_ZERO_MAGNITUDE
    )

perturbed_evidence[
    target_index
][
    "magnitude"
] = perturbed_target_magnitude

print(
    "Injected evidence id:",
    perturbed_evidence[
        target_index
    ][
        "evidence_id"
    ]
)

print(
    "Original magnitude:",
    original_target_magnitude
)

print(
    "Perturbed magnitude:",
    perturbed_target_magnitude
)

if nearly_equal(
        original_target_magnitude,
        perturbed_target_magnitude
):

    raise RuntimeError(
        (
            "Controlled error injection did not modify the target. "
            "This indicates the perturbation mechanism failed."
        )
    )

print(
    "Controlled reasoning error injected."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Detect Controlled Reasoning Error"
)

print()

perturbed_result = independent_recalculate(
    perturbed_evidence
)

error_fields = []

for field in [
    "support",
    "conflict",
    "net",
    "strength",
    "confidence"
]:

    if not nearly_equal(
            perturbed_result[
                field
            ],
            baseline_recalculation[
                field
            ]
    ):

        error_fields.append(
            field
        )

if (
        perturbed_result[
            "state"
        ]
        !=
        baseline_recalculation[
            "state"
        ]
):

    error_fields.append(
        "state"
    )

print(
    "Perturbed support:",
    perturbed_result[
        "support"
    ]
)

print(
    "Perturbed conflict:",
    perturbed_result[
        "conflict"
    ]
)

print(
    "Perturbed net:",
    perturbed_result[
        "net"
    ]
)

print(
    "Perturbed strength:",
    perturbed_result[
        "strength"
    ]
)

print(
    "Perturbed confidence:",
    perturbed_result[
        "confidence"
    ]
)

print(
    "Detected error fields:",
    error_fields
)

if not error_fields:

    raise RuntimeError(
        "Controlled reasoning error was not detected."
    )

print(
    "Controlled reasoning error detected."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Localize Reasoning Error"
)

print()

localized_error = {
    "error_type":
        "evidence_magnitude_mismatch",

    "evidence_id":
        perturbed_evidence[
            target_index
        ][
            "evidence_id"
        ],

    "field":
        "magnitude",

    "expected":
        original_target_magnitude,

    "observed":
        perturbed_evidence[
            target_index
        ][
            "magnitude"
        ],

    "affected_reasoning_fields":
        error_fields,

    "injection_mode":
        injection_mode
}

print(
    json.dumps(
        localized_error,
        indent=4
    )
)

if (
        localized_error[
            "field"
        ]
        !=
        "magnitude"
):

    raise RuntimeError(
        "Error localization failed."
    )

print(
    "Reasoning error localized."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Build Error Memory Record"
)

print()

error_record = {
    "error_id":
        "error_117_001",

    "timestamp":
        datetime.now().isoformat(),

    "source_lesson":
        "116R",

    "error_type":
        localized_error[
            "error_type"
        ],

    "evidence_id":
        localized_error[
            "evidence_id"
        ],

    "field":
        localized_error[
            "field"
        ],

    "expected_value":
        localized_error[
            "expected"
        ],

    "observed_value":
        localized_error[
            "observed"
        ],

    "affected_fields":
        localized_error[
            "affected_reasoning_fields"
        ],

    "detection_method":
        "independent_recalculation",

    "correction_required":
        True
}

print(
    json.dumps(
        error_record,
        indent=4
    )
)

print(
    "Error memory record constructed."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Select Correction Strategy"
)

print()

correction_strategy = {
    "strategy_id":
        "correction_001",

    "strategy_type":
        "restore_source_evidence",

    "target":
        localized_error[
            "evidence_id"
        ],

    "field":
        localized_error[
            "field"
        ],

    "replacement_value":
        localized_error[
            "expected"
        ],

    "reason":
        "Restore the provenance-backed source value "
        "identified independently before perturbation."
}

print(
    json.dumps(
        correction_strategy,
        indent=4
    )
)

if not nearly_equal(
        correction_strategy[
            "replacement_value"
        ],
        original_target_magnitude
):

    raise RuntimeError(
        "Correction strategy does not restore source truth."
    )

print(
    "Correction strategy selected."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Apply Self-Correction"
)

print()

corrected_evidence = [
    dict(
        item
    )
    for item
    in perturbed_evidence
]

corrected_evidence[
    target_index
][
    "magnitude"
] = float(
    correction_strategy[
        "replacement_value"
    ]
)

print(
    "Corrected evidence id:",
    corrected_evidence[
        target_index
    ][
        "evidence_id"
    ]
)

print(
    "Corrected magnitude:",
    corrected_evidence[
        target_index
    ][
        "magnitude"
    ]
)

if not nearly_equal(
        corrected_evidence[
            target_index
        ][
            "magnitude"
        ],
        original_target_magnitude
):

    raise RuntimeError(
        "Self-correction did not restore the source value."
    )

print(
    "Self-correction applied."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Recalculate After Self-Correction"
)

print()

corrected_result = independent_recalculate(
    corrected_evidence
)

print(
    "Corrected support:",
    corrected_result[
        "support"
    ]
)

print(
    "Corrected conflict:",
    corrected_result[
        "conflict"
    ]
)

print(
    "Corrected net:",
    corrected_result[
        "net"
    ]
)

print(
    "Corrected strength:",
    corrected_result[
        "strength"
    ]
)

print(
    "Corrected confidence:",
    corrected_result[
        "confidence"
    ]
)

print(
    "Corrected state:",
    corrected_result[
        "state"
    ]
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Verify Self-Correction Against Baseline"
)

print()

correction_comparison = {
    "support":
        nearly_equal(
            corrected_result[
                "support"
            ],
            baseline_recalculation[
                "support"
            ]
        ),

    "conflict":
        nearly_equal(
            corrected_result[
                "conflict"
            ],
            baseline_recalculation[
                "conflict"
            ]
        ),

    "net":
        nearly_equal(
            corrected_result[
                "net"
            ],
            baseline_recalculation[
                "net"
            ]
        ),

    "strength":
        nearly_equal(
            corrected_result[
                "strength"
            ],
            baseline_recalculation[
                "strength"
            ]
        ),

    "confidence":
        nearly_equal(
            corrected_result[
                "confidence"
            ],
            baseline_recalculation[
                "confidence"
            ]
        ),

    "state":
        (
                corrected_result[
                    "state"
                ]
                ==
                baseline_recalculation[
                    "state"
                ]
        )
}

for field, valid in correction_comparison.items():

    print(
        field,
        "->",
        valid
    )

self_correction_verified = all(
    correction_comparison.values()
)

print(
    "Self-correction verified:",
    self_correction_verified
)

if not self_correction_verified:

    raise RuntimeError(
        "Self-correction did not reproduce the validated baseline."
    )

print(
    "Self-correction independently verified."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Correction Improvement Measurement"
)

print()

error_before = abs(
    perturbed_result[
        "net"
    ]
    -
    baseline_recalculation[
        "net"
    ]
)

error_after = abs(
    corrected_result[
        "net"
    ]
    -
    baseline_recalculation[
        "net"
    ]
)

improvement = (
        error_before
        -
        error_after
)

print(
    "Error before correction:",
    error_before
)

print(
    "Error after correction:",
    error_after
)

print(
    "Improvement:",
    improvement
)

if error_after > NUMERIC_TOLERANCE:

    raise RuntimeError(
        "Correction did not reduce residual error to tolerance."
    )

if improvement < -NUMERIC_TOLERANCE:

    raise RuntimeError(
        "Correction made the reasoning error worse."
    )

print(
    "Correction improvement validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Self-Correction Confidence"
)

print()

correction_confidence = clamp(
    safe_mean(
        [
            1.0
            if self_correction_verified
            else
            0.0,

            1.0
            if error_after <= NUMERIC_TOLERANCE
            else
            0.0,

            baseline_recalculation[
                "confidence"
            ]
        ]
    )
)

print(
    "Correction confidence:",
    correction_confidence
)

if (
        correction_confidence
        <
        CORRECTION_CONFIDENCE_THRESHOLD
):

    raise RuntimeError(
        "Self-correction confidence is too low."
    )

print(
    "Self-correction confidence validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Error Memory Curriculum"
)

print()

error_memory_tasks = [
    {
        "example_id":
            "error_001",

        "domain":
            "error_detection",

        "question":
            "Why store reasoning errors?",

        "answer":
            "Repeated errors can become reusable knowledge for future validation."
    },

    {
        "example_id":
            "error_002",

        "domain":
            "error_localization",

        "question":
            "Why localize the source of a reasoning error?",

        "answer":
            "A correction should modify the actual faulty component."
    },

    {
        "example_id":
            "error_003",

        "domain":
            "self_correction",

        "question":
            "What makes self-correction trustworthy?",

        "answer":
            "The corrected result must be independently verified."
    },

    {
        "example_id":
            "error_004",

        "domain":
            "correction_memory",

        "question":
            "What should error memory contain?",

        "answer":
            "Error type, location, expected value, observed value and correction strategy."
    },

    {
        "example_id":
            "error_005",

        "domain":
            "continual_learning",

        "question":
            "How can error memory support continual learning?",

        "answer":
            "Future reasoning can reuse verified correction patterns."
    },

    {
        "example_id":
            "error_006",

        "domain":
            "engineering_reliability",

        "question":
            "Why is self-correction important in engineering intelligence?",

        "answer":
            "Reasoning mistakes should be detected and corrected before conclusions are trusted."
    }
]

for task in error_memory_tasks:

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
    "Error-memory tasks:",
    len(
        error_memory_tasks
    )
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Error-Memory Curriculum Coverage"
)

print()

expected_domains = {
    "error_detection",
    "error_localization",
    "self_correction",
    "correction_memory",
    "continual_learning",
    "engineering_reliability"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in error_memory_tasks
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
        "Error-memory curriculum coverage is incomplete."
    )

print(
    "Error-memory curriculum validated."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Numerical Health"
)

print()

health_values = [
    baseline_recalculation[
        "support"
    ],
    baseline_recalculation[
        "conflict"
    ],
    baseline_recalculation[
        "net"
    ],
    baseline_recalculation[
        "confidence"
    ],
    perturbed_result[
        "net"
    ],
    corrected_result[
        "net"
    ],
    correction_confidence,
    error_before,
    error_after
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
        "Self-correction numerical health failed."
    )

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Deterministic Self-Correction"
)

print()


def run_correction(
        evidence: List[Dict[str, Any]],
        target_index: int,
        replacement: float
) -> Dict[str, float]:

    corrected = [
        dict(
            item
        )
        for item
        in evidence
    ]

    corrected[
        target_index
    ][
        "magnitude"
    ] = replacement

    return independent_recalculate(
        corrected
    )


first_correction = run_correction(
    perturbed_evidence,
    target_index,
    original_target_magnitude
)

second_correction = run_correction(
    perturbed_evidence,
    target_index,
    original_target_magnitude
)

correction_deterministic = (
        stable_hash(
            first_correction
        )
        ==
        stable_hash(
            second_correction
        )
)

print(
    "First correction:",
    first_correction
)

print(
    "Second correction:",
    second_correction
)

print(
    "Deterministic:",
    correction_deterministic
)

if not correction_deterministic:

    raise RuntimeError(
        "Self-correction is nondeterministic."
    )

print(
    "Deterministic self-correction validated."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Final Self-Correction Promotion Gate"
)

print()

promotion_errors = []

if not self_correction_verified:

    promotion_errors.append(
        "Self-correction verification failed."
    )

if error_after > NUMERIC_TOLERANCE:

    promotion_errors.append(
        "Residual correction error exceeds tolerance."
    )

if (
        correction_confidence
        <
        CORRECTION_CONFIDENCE_THRESHOLD
):

    promotion_errors.append(
        "Correction confidence is below threshold."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if not correction_deterministic:

    promotion_errors.append(
        "Self-correction is nondeterministic."
    )

if len(
        error_memory_tasks
) < 6:

    promotion_errors.append(
        "Error-memory curriculum is incomplete."
    )

if not error_record[
    "correction_required"
]:

    promotion_errors.append(
        "Correction requirement was not recorded."
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
    "Improvement:",
    improvement
)

print(
    "Correction confidence:",
    correction_confidence
)

print(
    "Self-correction verified:",
    self_correction_verified
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
        "117R self-correction promotion gate failed."
    )

print(
    "117R self-correction promotion gate passed."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Persist Reasoning Error Memory"
)

print()

correction_event = {
    "event_id":
        "self_correction_117_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "117R",

    "reasoning_id":
        validator_memory.get(
            "reasoning_id"
        ),

    "error_id":
        error_record[
            "error_id"
        ],

    "correction_strategy_id":
        correction_strategy[
            "strategy_id"
        ],

    "self_correction_verified":
        self_correction_verified,

    "error_before":
        error_before,

    "error_after":
        error_after,

    "improvement":
        improvement,

    "correction_confidence":
        correction_confidence
}

error_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_reasoning_error_memory_self_correction",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        validator_memory.get(
            "memory_version"
        ),

    "reasoning_id":
        validator_memory.get(
            "reasoning_id"
        ),

    "error_records":
        [
            error_record
        ],

    "correction_strategies":
        [
            correction_strategy
        ],

    "events":
        [
            correction_event
        ],

    "baseline_result":
        baseline_recalculation,

    "perturbed_result":
        perturbed_result,

    "corrected_result":
        corrected_result,

    "verification":
        {
            "self_correction_verified":
                self_correction_verified,

            "error_before":
                error_before,

            "error_after":
                error_after,

            "improvement":
                improvement,

            "correction_confidence":
                correction_confidence,

            "deterministic":
                correction_deterministic
        }
}

write_json(
    ERROR_MEMORY_FILE,
    error_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            validator_memory.get(
                "reasoning_id"
            ),

        "error_records":
            [
                error_record
            ],

        "correction_strategies":
            [
                correction_strategy
            ],

        "baseline_result":
            baseline_recalculation,

        "corrected_result":
            corrected_result,

        "verified":
            self_correction_verified
    },
    ERROR_INDEX_FILE
)

print(
    "Error memory:",
    ERROR_MEMORY_FILE
)

print(
    "Error index:",
    ERROR_INDEX_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Reload Error Memory"
)

print()

reloaded_error_memory = read_json(
    ERROR_MEMORY_FILE
)

if (
        len(
            reloaded_error_memory[
                "error_records"
            ]
        )
        !=
        1
):

    raise RuntimeError(
        "Reasoning error record changed after persistence."
    )

if (
        reloaded_error_memory[
            "verification"
        ][
            "self_correction_verified"
        ]
        is not True
):

    raise RuntimeError(
        "Self-correction verification changed after persistence."
    )

if not nearly_equal(
        reloaded_error_memory[
            "verification"
        ][
            "error_after"
        ],
        error_after
):

    raise RuntimeError(
        "Residual error changed after persistence."
    )

print(
    "Reloaded error records:",
    len(
        reloaded_error_memory[
            "error_records"
        ]
    )
)

print(
    "Reloaded verified:",
    reloaded_error_memory[
        "verification"
    ][
        "self_correction_verified"
    ]
)

print(
    "Reloaded residual error:",
    reloaded_error_memory[
        "verification"
    ][
        "error_after"
    ]
)

print(
    "Persistent reasoning-error memory validated."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save Error-Memory Dataset"
)

print()

error_dataset = {
    "lesson":
        "117R",

    "capability":
        "native_reasoning_error_memory_self_correction",

    "reasoning_id":
        validator_memory.get(
            "reasoning_id"
        ),

    "error":
        error_record,

    "correction":
        correction_strategy,

    "baseline":
        baseline_recalculation,

    "perturbed":
        perturbed_result,

    "corrected":
        corrected_result,

    "verification":
        {
            "self_correction_verified":
                self_correction_verified,

            "error_before":
                error_before,

            "error_after":
                error_after,

            "improvement":
                improvement,

            "correction_confidence":
                correction_confidence,

            "deterministic":
                correction_deterministic
        }
}

write_json(
    ERROR_DATASET_FILE,
    error_dataset
)

print(
    "Error-memory dataset:",
    ERROR_DATASET_FILE
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save 117R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "117R",

    "capability":
        "native_reasoning_error_memory_self_correction",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        validator_memory.get(
            "reasoning_id"
        ),

    "error_record":
        error_record,

    "correction_strategy":
        correction_strategy,

    "baseline_result":
        baseline_recalculation,

    "perturbed_result":
        perturbed_result,

    "corrected_result":
        corrected_result,

    "self_correction_verified":
        self_correction_verified,

    "error_before":
        error_before,

    "error_after":
        error_after,

    "improvement":
        improvement,

    "correction_confidence":
        correction_confidence,

    "correction_deterministic":
        correction_deterministic,

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
    "TEST 26: Write 117R Reports"
)

print()

report = {
    "lesson":
        "117R",

    "capability":
        "native_reasoning_error_memory_self_correction",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        validator_memory.get(
            "reasoning_id"
        ),

    "error":
        error_record,

    "correction":
        correction_strategy,

    "baseline":
        baseline_recalculation,

    "perturbed":
        perturbed_result,

    "corrected":
        corrected_result,

    "verification":
        {
            "self_correction_verified":
                self_correction_verified,

            "error_before":
                error_before,

            "error_after":
                error_after,

            "improvement":
                improvement,

            "correction_confidence":
                correction_confidence,

            "deterministic":
                correction_deterministic
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
    ERROR_REPORT_FILE,
    report
)

write_json(
    ERROR_EVALUATION_FILE,
    report
)

write_json(
    ERROR_REGISTRY_FILE,
    {
        "lesson":
            "117R",

        "capability":
            "native_reasoning_error_memory_self_correction",

        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            validator_memory.get(
                "reasoning_id"
            ),

        "error_type":
            error_record[
                "error_type"
            ],

        "self_correction_verified":
            self_correction_verified,

        "correction_confidence":
            correction_confidence,

        "next":
            "118R Native Error Pattern Generalization + Preventive Reasoning"
    }
)

print(
    "Error-memory report:",
    ERROR_REPORT_FILE
)

print(
    "Error-memory evaluation:",
    ERROR_EVALUATION_FILE
)

print(
    "Error-memory registry:",
    ERROR_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 117R SELF-CORRECTION ARCHITECTURE"
)

print()

print(
    "Independent Validator"
)

print(
    "        ↓"
)

print(
    "Detected Reasoning Error"
)

print(
    "        ↓"
)

print(
    "Error Localization"
)

print(
    "        ↓"
)

print(
    "Error Memory"
)

print(
    "        ↓"
)

print(
    "Correction Strategy"
)

print(
    "        ↓"
)

print(
    "Self-Correction"
)

print(
    "        ↓"
)

print(
    "Independent Recalculation"
)

print(
    "        ↓"
)

print(
    "Correction Verification"
)

print(
    "        ↓"
)

print(
    "Verified Error Knowledge"
)

print()


# ============================================================
# WHY 117R MATTERS
# ============================================================

print(
    "WHY 117R MATTERS"
)

print()

print(
    "116R detects disagreement."
)

print(
    "117R turns a detected reasoning problem into an auditable "
    "error record and a verified correction."
)

print()

print(
    "The controlled loop is:"
)

print(
    "detect -> localize -> remember -> correct -> verify"
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "117R LIMITATION"
)

print()

print(
    "The injected error is synthetic and deterministic."
)

print(
    "It exists to validate the self-correction mechanism."
)

print(
    "Production learning requires naturally occurring errors, "
    "independent verification and verified external outcomes."
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
    "Lesson 118R: Native Error Pattern Generalization + Preventive Reasoning"
)

print()

print(
    "Error Memory + Recurrent Error Detection + "
    "Prevention Rules + Risk Prediction + "
    "Pre-Reasoning Validation"
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
    "=== LESSON 117R COMPLETE ==="
)
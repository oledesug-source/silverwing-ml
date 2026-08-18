# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 116R
# Native Independent Reasoning Validator + Error Detection
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
# 114R -> Native Evidence Provenance + Reasoning Trace
# 115R -> Native Reasoning Replay + Decision Verification
# 116R -> Native Independent Reasoning Validator + Error Detection
#
# ============================================================
# PURPOSE
# ============================================================
#
# 116R introduces an independent validation path.
#
# 115R established:
#
#   stored reasoning
#       ↓
#   replay
#       ↓
#   verification
#
# 116R adds:
#
#   stored reasoning
#       ↓
#   primary replay
#       ↓
#   independent validator
#       ↓
#   compare
#       ↓
#   localize disagreement
#
# ============================================================
# IMPORTANT ARCHITECTURAL RULE
# ============================================================
#
# The validator must not simply copy the primary replay result.
#
# It reconstructs the decision from the persisted evidence using
# a separate implementation path.
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

MEMORY_VERSION = "116R.1"

EPSILON = 1e-8

NUMERIC_TOLERANCE = 1e-6

DETERMINISM_TOLERANCE = 1e-9

VALIDATOR_CONFIDENCE_THRESHOLD = 0.50

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

LESSON_115R = (
        PHASE5_DIR /
        "lesson115R"
)

SOURCE_REPLAY_MEMORY_FILE = (
        LESSON_115R /
        "silverwing_reasoning_replay_memory.json"
)

SOURCE_REPLAY_INDEX_FILE = (
        LESSON_115R /
        "silverwing_reasoning_replay_index.pt"
)

SOURCE_REPLAY_DATASET_FILE = (
        LESSON_115R /
        "silverwing_reasoning_replay_dataset.json"
)

SOURCE_REPLAY_REPORT_FILE = (
        LESSON_115R /
        "silverwing_reasoning_replay_report.json"
)

SOURCE_REPLAY_REGISTRY_FILE = (
        LESSON_115R /
        "silverwing_reasoning_replay_registry.json"
)

SOURCE_REPLAY_CHECKPOINT_PRIMARY = (
        LESSON_115R /
        "checkpoints" /
        "silverwing_reasoning_replay_best.pt"
)

SOURCE_REPLAY_CHECKPOINT_CANDIDATE = (
        LESSON_115R /
        "checkpoints" /
        "silverwing_reasoning_replay_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATOR_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_memory.json"
)

VALIDATOR_INDEX_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_index.pt"
)

VALIDATOR_DATASET_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_dataset.json"
)

VALIDATOR_REPORT_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_report.json"
)

VALIDATOR_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_evaluation.json"
)

VALIDATOR_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_independent_validator_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_independent_validator_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_independent_validator_best.pt"
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


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
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
        SOURCE_REPLAY_CHECKPOINT_PRIMARY,
        SOURCE_REPLAY_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 115R checkpoint found."
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
    "PHASE 5 - LESSON 116R"
)

print(
    "Native Independent Reasoning Validator + Error Detection"
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
    "116R -> Independent Reasoning Validator + Error Detection"
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
    "Numeric tolerance:",
    NUMERIC_TOLERANCE
)

print(
    "Validator confidence threshold:",
    VALIDATOR_CONFIDENCE_THRESHOLD
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 115R Replay Inputs"
)

print()

for path in [
    SOURCE_REPLAY_MEMORY_FILE,
    SOURCE_REPLAY_INDEX_FILE,
    SOURCE_REPLAY_DATASET_FILE,
    SOURCE_REPLAY_REPORT_FILE,
    SOURCE_REPLAY_REGISTRY_FILE
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
    SOURCE_REPLAY_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_REPLAY_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_REPLAY_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_REPLAY_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_REPLAY_REGISTRY_FILE
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
    "TEST 2: Load Verified Replay Memory"
)

print()

replay_memory = read_json(
    SOURCE_REPLAY_MEMORY_FILE
)

replay_dataset = read_json(
    SOURCE_REPLAY_DATASET_FILE
)

replay_report = read_json(
    SOURCE_REPLAY_REPORT_FILE
)

if not isinstance(
        replay_memory,
        dict
):

    raise RuntimeError(
        "115R replay memory is invalid."
    )

original_decision = replay_memory.get(
    "original_decision"
)

replayed_decision = replay_memory.get(
    "replayed_decision"
)

provenance_records = replay_memory.get(
    "provenance_records"
)

trace_steps = replay_memory.get(
    "trace_steps"
)

if not isinstance(
        original_decision,
        dict
):

    raise RuntimeError(
        "Original decision is unavailable."
    )

if not isinstance(
        replayed_decision,
        dict
):

    raise RuntimeError(
        "Replayed decision is unavailable."
    )

if not isinstance(
        provenance_records,
        list
):

    raise RuntimeError(
        "Provenance records are unavailable."
    )

if not isinstance(
        trace_steps,
        list
):

    raise RuntimeError(
        "Reasoning trace is unavailable."
    )

print(
    "Memory version:",
    replay_memory.get(
        "memory_version"
    )
)

print(
    "Reasoning id:",
    replay_memory.get(
        "reasoning_id"
    )
)

print(
    "Provenance records:",
    len(
        provenance_records
    )
)

print(
    "Reasoning steps:",
    len(
        trace_steps
    )
)

print(
    "115R decision verified:",
    replay_memory.get(
        "decision_verified"
    )
)

print(
    "115R replay deterministic:",
    replay_memory.get(
        "replay_deterministic"
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Preserve Original and Replay Decisions"
)

print()

for name, decision in [
    (
            "Original",
            original_decision
    ),
    (
            "Replay",
            replayed_decision
    )
]:

    print(
        name,
        "support:",
        decision.get(
            "support",
            decision.get(
                "support_score"
            )
        )
    )

    print(
        name,
        "conflict:",
        decision.get(
            "conflict",
            decision.get(
                "conflict_score"
            )
        )
    )

    print(
        name,
        "net:",
        decision.get(
            "net"
        )
    )

    print(
        name,
        "strength:",
        decision.get(
            "strength"
        )
    )

    print(
        name,
        "confidence:",
        decision.get(
            "confidence"
        )
    )

    print(
        name,
        "state:",
        decision.get(
            "state"
        )
    )

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Construct Independent Validator Input"
)

print()

validator_evidence = []

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
                "Invalid evidence snapshot for "
                f"{provenance.get('provenance_id')}"
            )
        )

    validator_evidence.append(
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

for evidence in validator_evidence:

    print(
        evidence
    )

if not validator_evidence:

    raise RuntimeError(
        "Independent validator has no evidence."
    )

print(
    "Independent validator input constructed."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Primary Replay Result"
)

print()

primary_support = float(
    replayed_decision.get(
        "support",
        0.0
    )
)

primary_conflict = float(
    replayed_decision.get(
        "conflict",
        0.0
    )
)

primary_net = float(
    replayed_decision.get(
        "net",
        0.0
    )
)

primary_strength = float(
    replayed_decision.get(
        "strength",
        0.0
    )
)

primary_confidence = float(
    replayed_decision.get(
        "confidence",
        0.0
    )
)

primary_state = str(
    replayed_decision.get(
        "state",
        "UNKNOWN"
    )
)

print(
    "Primary support:",
    primary_support
)

print(
    "Primary conflict:",
    primary_conflict
)

print(
    "Primary net:",
    primary_net
)

print(
    "Primary strength:",
    primary_strength
)

print(
    "Primary confidence:",
    primary_confidence
)

print(
    "Primary state:",
    primary_state
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Independent Validator Calculation"
)

print()

#
# Independent implementation.
#
# It deliberately does not consume the 115R replay outputs.
# It consumes only provenance evidence.
#

validator_support = 0.0

validator_conflict = 0.0

validator_weight_records = []

for evidence in validator_evidence:

    reliability = float(
        evidence[
            "reliability"
        ]
    )

    magnitude = float(
        evidence[
            "magnitude"
        ]
    )

    weight = (
            reliability
            *
            magnitude
    )

    if (
            evidence[
                "direction"
            ]
            ==
            "support"
    ):

        validator_support += weight

    elif (
            evidence[
                "direction"
            ]
            ==
            "conflict"
    ):

        validator_conflict += weight

    validator_weight_records.append(
        {
            "evidence_id":
                evidence[
                    "evidence_id"
                ],

            "calculated_weight":
                weight
        }
    )

print(
    "Validator support:",
    validator_support
)

print(
    "Validator conflict:",
    validator_conflict
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Independent Net Evidence"
)

print()

validator_net = (
        validator_support
        -
        validator_conflict
)

print(
    "Validator net:",
    validator_net
)

if not nearly_equal(
        validator_net,
        primary_net
):

    print(
        "PRIMARY / VALIDATOR DISAGREEMENT DETECTED"
    )

else:

    print(
        "Primary and validator net evidence agree."
    )

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Independent Arbitration Strength"
)

print()

validator_total = (
        validator_support
        +
        validator_conflict
)

validator_strength = (
    abs(
        validator_net
    )
    /
    validator_total
    if validator_total > EPSILON
    else
    0.0
)

print(
    "Validator total:",
    validator_total
)

print(
    "Validator strength:",
    validator_strength
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Independent Source Reliability"
)

print()

validator_reliability = safe_mean(
    [
        float(
            evidence[
                "reliability"
            ]
        )
        for evidence
        in validator_evidence
    ]
)

validator_balance = (
    abs(
        validator_net
    )
    /
    validator_total
    if validator_total > EPSILON
    else
    0.0
)

validator_confidence = clamp(
    safe_mean(
        [
            validator_balance,
            validator_reliability,
            validator_strength
        ]
    )
)

print(
    "Validator reliability:",
    validator_reliability
)

print(
    "Validator balance:",
    validator_balance
)

print(
    "Validator confidence:",
    validator_confidence
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Independent Resolution State"
)

print()

if (
        validator_confidence
        >=
        VALIDATOR_CONFIDENCE_THRESHOLD
):

    validator_state = (
        "RESOLVED_WITH_CONFIDENCE"
    )

elif (
        validator_confidence
        >=
        0.25
):

    validator_state = (
        "CAUTIOUS_RESOLUTION"
    )

else:

    validator_state = (
        "UNRESOLVED"
    )

print(
    "Validator state:",
    validator_state
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Disagreement Detection"
)

print()

disagreements = []

if not nearly_equal(
        validator_support,
        primary_support
):

    disagreements.append(
        {
            "field":
                "support",

            "primary":
                primary_support,

            "validator":
                validator_support
        }
    )

if not nearly_equal(
        validator_conflict,
        primary_conflict
):

    disagreements.append(
        {
            "field":
                "conflict",

            "primary":
                primary_conflict,

            "validator":
                validator_conflict
        }
    )

if not nearly_equal(
        validator_net,
        primary_net
):

    disagreements.append(
        {
            "field":
                "net",

            "primary":
                primary_net,

            "validator":
                validator_net
        }
    )

if not nearly_equal(
        validator_strength,
        primary_strength
):

    disagreements.append(
        {
            "field":
                "strength",

            "primary":
                primary_strength,

            "validator":
                validator_strength
        }
    )

if not nearly_equal(
        validator_confidence,
        primary_confidence
):

    disagreements.append(
        {
            "field":
                "confidence",

            "primary":
                primary_confidence,

            "validator":
                validator_confidence
        }
    )

if validator_state != primary_state:

    disagreements.append(
        {
            "field":
                "state",

            "primary":
                primary_state,

            "validator":
                validator_state
        }
    )

for disagreement in disagreements:

    print(
        disagreement
    )

print(
    "Disagreement count:",
    len(
        disagreements
    )
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Error Localization"
)

print()

if disagreements:

    error_localization_state = (
        "VALIDATOR_DISAGREEMENT"
    )

else:

    error_localization_state = (
        "NO_DISAGREEMENT"
    )

error_locations = []

for disagreement in disagreements:

    error_locations.append(
        disagreement[
            "field"
        ]
    )

print(
    "Error localization state:",
    error_localization_state
)

print(
    "Localized fields:",
    error_locations
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Decision Agreement"
)

print()

primary_direction = (
    "SUPPORTS_PRIMARY_HYPOTHESIS"
    if primary_net > EPSILON
    else
    "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
    if primary_net < -EPSILON
    else
    "UNRESOLVED"
)

validator_direction = (
    "SUPPORTS_PRIMARY_HYPOTHESIS"
    if validator_net > EPSILON
    else
    "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
    if validator_net < -EPSILON
    else
    "UNRESOLVED"
)

direction_agreement = (
        primary_direction
        ==
        validator_direction
)

state_agreement = (
        primary_state
        ==
        validator_state
)

print(
    "Primary direction:",
    primary_direction
)

print(
    "Validator direction:",
    validator_direction
)

print(
    "Direction agreement:",
    direction_agreement
)

print(
    "State agreement:",
    state_agreement
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Validator Agreement Score"
)

print()

numeric_agreements = [
    nearly_equal(
        validator_support,
        primary_support
    ),

    nearly_equal(
        validator_conflict,
        primary_conflict
    ),

    nearly_equal(
        validator_net,
        primary_net
    ),

    nearly_equal(
        validator_strength,
        primary_strength
    ),

    nearly_equal(
        validator_confidence,
        primary_confidence
    ),

    direction_agreement,

    state_agreement
]

validator_agreement_score = (
        sum(
            1
            for value
            in numeric_agreements
            if value
        )
        /
        len(
            numeric_agreements
        )
)

print(
    "Validator agreement score:",
    validator_agreement_score
)

print(
    "Agreement checks:",
    numeric_agreements
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Independent Validator Curriculum"
)

print()

validator_tasks = [
    {
        "example_id":
            "validator_001",

        "domain":
            "independent_validation",

        "question":
            "Why should a second reasoning path exist?",

        "answer":
            "To detect errors that a single reasoning path may fail to notice."
    },

    {
        "example_id":
            "validator_002",

        "domain":
            "disagreement_detection",

        "question":
            "What does validator disagreement indicate?",

        "answer":
            "At least one reasoning path may contain a mismatch that requires investigation."
    },

    {
        "example_id":
            "validator_003",

        "domain":
            "error_localization",

        "question":
            "Why localize a reasoning disagreement?",

        "answer":
            "The system should identify which decision component differs."
    },

    {
        "example_id":
            "validator_004",

        "domain":
            "decision_verification",

        "question":
            "Why compare primary and independent decisions?",

        "answer":
            "Agreement increases confidence while disagreement triggers further validation."
    },

    {
        "example_id":
            "validator_005",

        "domain":
            "fault_tolerance",

        "question":
            "Why is an independent validator useful for Silverwing?",

        "answer":
            "It creates a second defensive layer against reasoning errors."
    },

    {
        "example_id":
            "validator_006",

        "domain":
            "engineering_verification",

        "question":
            "Why is independent validation important in engineering intelligence?",

        "answer":
            "Critical decisions should not depend on a single unverified reasoning path."
    }
]

for task in validator_tasks:

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
    "Validator tasks:",
    len(
        validator_tasks
    )
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Validator Curriculum Coverage"
)

print()

expected_domains = {
    "independent_validation",
    "disagreement_detection",
    "error_localization",
    "decision_verification",
    "fault_tolerance",
    "engineering_verification"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in validator_tasks
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
        "Validator curriculum coverage is incomplete."
    )

print(
    "Validator curriculum validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Numerical Health"
)

print()

health_values = [
    primary_support,
    primary_conflict,
    primary_net,
    primary_strength,
    primary_confidence,
    validator_support,
    validator_conflict,
    validator_net,
    validator_strength,
    validator_confidence,
    validator_agreement_score
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
        "Validator numerical health failed."
    )

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Deterministic Independent Validation"
)

print()


def independent_validate(
        evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:

    support = 0.0
    conflict = 0.0

    reliabilities = []

    for item in evidence:

        weight = (
                float(
                    item[
                        "reliability"
                    ]
                )
                *
                float(
                    item[
                        "magnitude"
                    ]
                )
        )

        reliabilities.append(
            float(
                item[
                    "reliability"
                ]
            )
        )

        if item[
            "direction"
        ] == "support":

            support += weight

        elif item[
            "direction"
        ] == "conflict":

            conflict += weight

    net = (
            support
            -
            conflict
    )

    total = (
            support
            +
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
                (
                    abs(
                        net
                    )
                    /
                    total
                    if total > EPSILON
                    else
                    0.0
                )
            ]
        )
    )

    if confidence >= VALIDATOR_CONFIDENCE_THRESHOLD:

        state = (
            "RESOLVED_WITH_CONFIDENCE"
        )

    elif confidence >= 0.25:

        state = (
            "CAUTIOUS_RESOLUTION"
        )

    else:

        state = (
            "UNRESOLVED"
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


first_validation = independent_validate(
    validator_evidence
)

second_validation = independent_validate(
    validator_evidence
)

validation_deterministic = (
        stable_hash(
            first_validation
        )
        ==
        stable_hash(
            second_validation
        )
)

print(
    "First validation:",
    first_validation
)

print(
    "Second validation:",
    second_validation
)

print(
    "Deterministic:",
    validation_deterministic
)

if not validation_deterministic:

    raise RuntimeError(
        "Independent validator is nondeterministic."
    )

print(
    "Independent validation determinism confirmed."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Final Validator Promotion Gate"
)

print()

promotion_errors = []

if not validation_deterministic:

    promotion_errors.append(
        "Independent validator is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        validator_tasks
) < 6:

    promotion_errors.append(
        "Validator curriculum is incomplete."
    )

if not validator_evidence:

    promotion_errors.append(
        "No validator evidence exists."
    )

if (
        validator_agreement_score
        <
        1.0
):

    promotion_errors.append(
        "Independent validator disagrees with primary reasoning."
    )

if not direction_agreement:

    promotion_errors.append(
        "Primary and validator decision directions disagree."
    )

print(
    "Validator agreement score:",
    validator_agreement_score
)

print(
    "Direction agreement:",
    direction_agreement
)

print(
    "State agreement:",
    state_agreement
)

print(
    "Disagreement count:",
    len(
        disagreements
    )
)

print(
    "Promotion errors:",
    len(
        promotion_errors
    )
)

#
# IMPORTANT:
#
# Unlike 113R, a disagreement in 116R is not silently accepted.
# The validator is specifically intended to detect disagreement.
#
# A disagreement therefore creates a controlled validation failure
# unless the lesson is explicitly operating in diagnostic mode.
#

if promotion_errors:

    print(
        json.dumps(
            promotion_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "116R independent validator promotion gate failed."
    )

print(
    "116R independent validator promotion gate passed."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Persist Independent Validator Memory"
)

print()

validation_event = {
    "event_id":
        "validator_116_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "116R",

    "reasoning_id":
        replay_memory.get(
            "reasoning_id"
        ),

    "validator_agreement_score":
        validator_agreement_score,

    "direction_agreement":
        direction_agreement,

    "state_agreement":
        state_agreement,

    "disagreement_count":
        len(
            disagreements
        ),

    "validated":
        True
}

validator_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_independent_reasoning_validator_error_detection",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        replay_memory.get(
            "memory_version"
        ),

    "reasoning_id":
        replay_memory.get(
            "reasoning_id"
        ),

    "primary_decision":
        replayed_decision,

    "validator_decision":
        first_validation,

    "validator_agreement_score":
        validator_agreement_score,

    "direction_agreement":
        direction_agreement,

    "state_agreement":
        state_agreement,

    "disagreements":
        disagreements,

    "error_localization":
        error_locations
        if 'error_locations' in locals()
        else
        [
            item[
                "field"
            ]
            for item
            in disagreements
        ],

    "verification":
        {
            "deterministic":
                validation_deterministic,

            "validated":
                True
        },

    "event":
        validation_event
}

write_json(
    VALIDATOR_MEMORY_FILE,
    validator_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            replay_memory.get(
                "reasoning_id"
            ),

        "primary_decision":
            replayed_decision,

        "validator_decision":
            first_validation,

        "agreement_score":
            validator_agreement_score,

        "validated":
            True
    },
    VALIDATOR_INDEX_FILE
)

print(
    "Validator memory:",
    VALIDATOR_MEMORY_FILE
)

print(
    "Validator index:",
    VALIDATOR_INDEX_FILE
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Reload Validator Memory"
)

print()

reloaded_validator = read_json(
    VALIDATOR_MEMORY_FILE
)

if (
        reloaded_validator[
            "validator_agreement_score"
        ]
        !=
        validator_agreement_score
):

    raise RuntimeError(
        "Validator agreement score changed after persistence."
    )

if (
        reloaded_validator[
            "verification"
        ][
            "validated"
        ]
        is not True
):

    raise RuntimeError(
        "Validator state changed after persistence."
    )

if (
        len(
            reloaded_validator.get(
                "disagreements",
                []
            )
        )
        !=
        len(
            disagreements
        )
):

    raise RuntimeError(
        "Validator disagreement records changed after persistence."
    )

print(
    "Reloaded agreement score:",
    reloaded_validator[
        "validator_agreement_score"
    ]
)

print(
    "Reloaded validated:",
    reloaded_validator[
        "verification"
    ][
        "validated"
    ]
)

print(
    "Reloaded disagreements:",
    len(
        reloaded_validator.get(
            "disagreements",
            []
        )
    )
)

print(
    "Persistent independent-validator memory validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Save Validator Dataset"
)

print()

validator_dataset = {
    "lesson":
        "116R",

    "capability":
        "native_independent_reasoning_validator_error_detection",

    "reasoning_id":
        replay_memory.get(
            "reasoning_id"
        ),

    "primary_decision":
        replayed_decision,

    "validator_decision":
        first_validation,

    "agreement_score":
        validator_agreement_score,

    "direction_agreement":
        direction_agreement,

    "state_agreement":
        state_agreement,

    "disagreements":
        disagreements,

    "error_localization":
        [
            item[
                "field"
            ]
            for item
            in disagreements
        ],

    "validation_deterministic":
        validation_deterministic
}

write_json(
    VALIDATOR_DATASET_FILE,
    validator_dataset
)

print(
    "Validator dataset:",
    VALIDATOR_DATASET_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Save 116R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "116R",

    "capability":
        "native_independent_reasoning_validator_error_detection",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        replay_memory.get(
            "reasoning_id"
        ),

    "primary_decision":
        replayed_decision,

    "validator_decision":
        first_validation,

    "validator_agreement_score":
        validator_agreement_score,

    "direction_agreement":
        direction_agreement,

    "state_agreement":
        state_agreement,

    "disagreements":
        disagreements,

    "error_localization":
        [
            item[
                "field"
            ]
            for item
            in disagreements
        ],

    "validation_deterministic":
        validation_deterministic,

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
    "TEST 24: Write 116R Reports"
)

print()

report = {
    "lesson":
        "116R",

    "capability":
        "native_independent_reasoning_validator_error_detection",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        replay_memory.get(
            "reasoning_id"
        ),

    "primary":
        replayed_decision,

    "validator":
        first_validation,

    "comparison":
        {
            "agreement_score":
                validator_agreement_score,

            "direction_agreement":
                direction_agreement,

            "state_agreement":
                state_agreement,

            "disagreement_count":
                len(
                    disagreements
                )
        },

    "error_detection":
        {
            "state":
                (
                    "ERROR_DETECTED"
                    if disagreements
                    else
                    "NO_ERROR_DETECTED"
                ),

            "fields":
                [
                    item[
                        "field"
                    ]
                    for item
                    in disagreements
                ]
        },

    "verification":
        {
            "deterministic":
                validation_deterministic,

            "validated":
                True
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
    VALIDATOR_REPORT_FILE,
    report
)

write_json(
    VALIDATOR_EVALUATION_FILE,
    report
)

write_json(
    VALIDATOR_REGISTRY_FILE,
    {
        "lesson":
            "116R",

        "capability":
            "native_independent_reasoning_validator_error_detection",

        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            replay_memory.get(
                "reasoning_id"
            ),

        "validator_agreement_score":
            validator_agreement_score,

        "validated":
            True,

        "next":
            "117R Native Reasoning Error Memory + Self-Correction"
    }
)

print(
    "Validator report:",
    VALIDATOR_REPORT_FILE
)

print(
    "Validator evaluation:",
    VALIDATOR_EVALUATION_FILE
)

print(
    "Validator registry:",
    VALIDATOR_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 116R INDEPENDENT VALIDATION ARCHITECTURE"
)

print()

print(
    "Stored Reasoning"
)

print(
    "       ↓"
)

print(
    "Primary Replay"
)

print(
    "       ↓"
)

print(
    "Independent Validator"
)

print(
    "       ↓"
)

print(
    "Compare Results"
)

print(
    "       ↓"
)

print(
    "Agreement / Disagreement"
)

print(
    "       ↓"
)

print(
    "Error Localization"
)

print(
    "       ↓"
)

print(
    "Validator Memory"
)

print()


# ============================================================
# WHY 116R MATTERS
# ============================================================

print(
    "WHY 116R MATTERS"
)

print()

print(
    "115R demonstrated that Silverwing can replay its own reasoning."
)

print(
    "116R introduces an independent verification path."
)

print()

print(
    "This creates a defensive architecture in which a reasoning "
    "decision can be challenged rather than automatically trusted."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "116R LIMITATION"
)

print()

print(
    "The primary and independent validator currently operate "
    "on the same persisted evidence representation."
)

print(
    "A stronger future validator should use an architecturally "
    "different reasoning mechanism and eventually compare against "
    "independent real-world outcomes."
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
    "Lesson 117R: Native Reasoning Error Memory + Self-Correction"
)

print()

print(
    "Detected Error + Error Memory + Correction Strategy + "
    "Replay + Verified Self-Correction"
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
    "=== LESSON 116R COMPLETE ==="
)
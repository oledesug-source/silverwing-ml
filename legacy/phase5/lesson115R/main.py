# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 115R
# Native Reasoning Replay + Decision Verification
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# 115R replays the reasoning stored by 114R and independently
# verifies that:
#
#   evidence
#      ↓
#   weights
#      ↓
#   support/conflict
#      ↓
#   net evidence
#      ↓
#   arbitration strength
#      ↓
#   confidence
#      ↓
#   resolution state
#
# still produces the same decision.
#
# ============================================================
# CRITICAL ARCHITECTURAL RULE
# ============================================================
#
# 115R does not trust the stored final conclusion by itself.
#
# It reconstructs the decision from the stored provenance
# evidence and compares the replayed result with the original.
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

MEMORY_VERSION = "115R.1"

EPSILON = 1e-8

NUMERIC_TOLERANCE = 1e-6

DETERMINISM_TOLERANCE = 1e-9

REPLAY_CONFIDENCE_THRESHOLD = 0.50

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

LESSON_114R = (
        PHASE5_DIR /
        "lesson114R"
)

SOURCE_PROVENANCE_MEMORY_FILE = (
        LESSON_114R /
        "silverwing_evidence_provenance_memory.json"
)

SOURCE_PROVENANCE_INDEX_FILE = (
        LESSON_114R /
        "silverwing_evidence_provenance_index.pt"
)

SOURCE_PROVENANCE_DATASET_FILE = (
        LESSON_114R /
        "silverwing_evidence_provenance_dataset.json"
)

SOURCE_PROVENANCE_REPORT_FILE = (
        LESSON_114R /
        "silverwing_evidence_provenance_report.json"
)

SOURCE_PROVENANCE_REGISTRY_FILE = (
        LESSON_114R /
        "silverwing_evidence_provenance_registry.json"
)

SOURCE_PROVENANCE_CHECKPOINT_PRIMARY = (
        LESSON_114R /
        "checkpoints" /
        "silverwing_evidence_provenance_best.pt"
)

SOURCE_PROVENANCE_CHECKPOINT_CANDIDATE = (
        LESSON_114R /
        "checkpoints" /
        "silverwing_evidence_provenance_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPLAY_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_memory.json"
)

REPLAY_INDEX_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_index.pt"
)

REPLAY_DATASET_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_dataset.json"
)

REPLAY_REPORT_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_report.json"
)

REPLAY_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_evaluation.json"
)

REPLAY_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_reasoning_replay_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_reasoning_replay_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_reasoning_replay_best.pt"
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


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_PROVENANCE_CHECKPOINT_PRIMARY,
        SOURCE_PROVENANCE_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 114R checkpoint found."
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
    "PHASE 5 - LESSON 115R"
)

print(
    "Native Reasoning Replay + Decision Verification"
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
    "115R -> Reasoning Replay + Decision Verification"
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
    "Determinism tolerance:",
    DETERMINISM_TOLERANCE
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 114R Provenance Inputs"
)

print()

for path in [
    SOURCE_PROVENANCE_MEMORY_FILE,
    SOURCE_PROVENANCE_INDEX_FILE,
    SOURCE_PROVENANCE_DATASET_FILE,
    SOURCE_PROVENANCE_REPORT_FILE,
    SOURCE_PROVENANCE_REGISTRY_FILE
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
    SOURCE_PROVENANCE_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_PROVENANCE_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_PROVENANCE_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_PROVENANCE_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_PROVENANCE_REGISTRY_FILE
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
    "TEST 2: Load Provenance Memory"
)

print()

provenance_memory = read_json(
    SOURCE_PROVENANCE_MEMORY_FILE
)

provenance_dataset = read_json(
    SOURCE_PROVENANCE_DATASET_FILE
)

provenance_report = read_json(
    SOURCE_PROVENANCE_REPORT_FILE
)

if not isinstance(
        provenance_memory,
        dict
):

    raise RuntimeError(
        "114R provenance memory is invalid."
    )

provenance_records = provenance_memory.get(
    "provenance_records"
)

reasoning_steps = provenance_memory.get(
    "reasoning_steps"
)

decision_contributions = provenance_memory.get(
    "decision_contributions"
)

contradictions = provenance_memory.get(
    "contradictions"
)

if not isinstance(
        provenance_records,
        list
):

    raise RuntimeError(
        "Provenance records are unavailable."
    )

if not isinstance(
        reasoning_steps,
        list
):

    raise RuntimeError(
        "Reasoning steps are unavailable."
    )

if not isinstance(
        decision_contributions,
        list
):

    raise RuntimeError(
        "Decision contributions are unavailable."
    )

if not isinstance(
        contradictions,
        list
):

    raise RuntimeError(
        "Contradiction records are unavailable."
    )

print(
    "Memory version:",
    provenance_memory.get(
        "memory_version"
    )
)

print(
    "Reasoning id:",
    provenance_memory.get(
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
        reasoning_steps
    )
)

print(
    "Decision contributions:",
    len(
        decision_contributions
    )
)

print(
    "Contradictions:",
    len(
        contradictions
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Load Original Decision"
)

print()

stored_decision = provenance_memory.get(
    "decision"
)

if not isinstance(
        stored_decision,
        dict
):

    raise RuntimeError(
        "Stored 114R decision is unavailable."
    )

original_support = float(
    stored_decision.get(
        "support_score",
        0.0
    )
)

original_conflict = float(
    stored_decision.get(
        "conflict_score",
        0.0
    )
)

original_net = float(
    stored_decision.get(
        "net_evidence",
        0.0
    )
)

original_strength = float(
    stored_decision.get(
        "arbitration_strength",
        0.0
    )
)

original_confidence = float(
    stored_decision.get(
        "resolution_confidence",
        0.0
    )
)

original_state = str(
    stored_decision.get(
        "resolution_state",
        "UNKNOWN"
    )
)

original_conclusion = str(
    stored_decision.get(
        "conclusion",
        ""
    )
)

print(
    "Original support:",
    original_support
)

print(
    "Original conflict:",
    original_conflict
)

print(
    "Original net:",
    original_net
)

print(
    "Original strength:",
    original_strength
)

print(
    "Original confidence:",
    original_confidence
)

print(
    "Original state:",
    original_state
)

print(
    "Original conclusion:",
    original_conclusion
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Stored Trace Structure"
)

print()

trace_errors = []

required_step_fields = {
    "trace_id",
    "step_id",
    "operation",
    "input",
    "output"
}

for step in reasoning_steps:

    missing = (
            required_step_fields
            -
            set(
                step.keys()
            )
    )

    if missing:

        trace_errors.append(
            {
                "step_id":
                    step.get(
                        "step_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

for contribution in decision_contributions:

    required = {
        "evidence_id",
        "provenance_id",
        "direction",
        "weight",
        "contribution"
    }

    missing = (
            required
            -
            set(
                contribution.keys()
            )
    )

    if missing:

        trace_errors.append(
            {
                "evidence_id":
                    contribution.get(
                        "evidence_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

if trace_errors:

    print(
        json.dumps(
            trace_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Stored reasoning trace structure is invalid."
    )

print(
    "Stored reasoning trace structure validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Independent Evidence Reconstruction"
)

print()

replay_evidence = []

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
                "Missing evidence snapshot for "
                f"{provenance.get('provenance_id')}"
            )
        )

    replay_evidence.append(
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

            "stored_weight":
                float(
                    snapshot.get(
                        "weight",
                        0.0
                    )
                ),

            "stored_contribution":
                float(
                    snapshot.get(
                        "contribution",
                        0.0
                    )
                )
        }
    )

for evidence in replay_evidence:

    print(
        evidence
    )

if not replay_evidence:

    raise RuntimeError(
        "No evidence available for replay."
    )

print(
    "Independent evidence reconstruction completed."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Recalculate Evidence Weights"
)

print()

weight_errors = []

for evidence in replay_evidence:

    calculated_weight = (
            evidence[
                "reliability"
            ]
            *
            evidence[
                "magnitude"
            ]
    )

    stored_weight = evidence[
        "stored_weight"
    ]

    if not nearly_equal(
            calculated_weight,
            stored_weight
    ):

        weight_errors.append(
            {
                "evidence_id":
                    evidence[
                        "evidence_id"
                    ],

                "calculated":
                    calculated_weight,

                "stored":
                    stored_weight
            }
        )

    evidence[
        "replayed_weight"
    ] = calculated_weight

if weight_errors:

    print(
        json.dumps(
            weight_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Evidence weight replay failed."
    )

print(
    "All evidence weights reproduced."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Recalculate Evidence Contributions"
)

print()

contribution_errors = []

for evidence in replay_evidence:

    weight = evidence[
        "replayed_weight"
    ]

    direction = evidence[
        "direction"
    ]

    calculated_contribution = (
        weight
        if direction == "support"
        else
        -weight
        if direction == "conflict"
        else
        0.0
    )

    stored_contribution = evidence[
        "stored_contribution"
    ]

    if not nearly_equal(
            calculated_contribution,
            stored_contribution
    ):

        contribution_errors.append(
            {
                "evidence_id":
                    evidence[
                        "evidence_id"
                    ],

                "calculated":
                    calculated_contribution,

                "stored":
                    stored_contribution
            }
        )

    evidence[
        "replayed_contribution"
    ] = calculated_contribution

if contribution_errors:

    print(
        json.dumps(
            contribution_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Evidence contribution replay failed."
    )

print(
    "All evidence contributions reproduced."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Recalculate Support and Conflict"
)

print()

replay_support = sum(
    evidence[
        "replayed_weight"
    ]
    for evidence
    in replay_evidence
    if evidence[
        "direction"
    ]
    ==
    "support"
)

replay_conflict = sum(
    evidence[
        "replayed_weight"
    ]
    for evidence
    in replay_evidence
    if evidence[
        "direction"
    ]
    ==
    "conflict"
)

print(
    "Replay support:",
    replay_support
)

print(
    "Replay conflict:",
    replay_conflict
)

print()

if not nearly_equal(
        replay_support,
        original_support
):

    raise RuntimeError(
        "Replay support score differs from original."
    )

if not nearly_equal(
        replay_conflict,
        original_conflict
):

    raise RuntimeError(
        "Replay conflict score differs from original."
    )

print(
    "Support and conflict scores reproduced."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Recalculate Net Evidence"
)

print()

replay_net = (
        replay_support
        -
        replay_conflict
)

print(
    "Replay net evidence:",
    replay_net
)

print(
    "Original net evidence:",
    original_net
)

if not nearly_equal(
        replay_net,
        original_net
):

    raise RuntimeError(
        "Replay net evidence differs from original."
    )

print(
    "Net evidence reproduced."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Recalculate Arbitration Strength"
)

print()

replay_total_weight = (
        replay_support
        +
        replay_conflict
)

if replay_total_weight <= EPSILON:

    replay_strength = 0.0

else:

    replay_strength = (
            abs(
                replay_net
            )
            /
            replay_total_weight
    )

print(
    "Replay total weight:",
    replay_total_weight
)

print(
    "Replay arbitration strength:",
    replay_strength
)

print(
    "Original arbitration strength:",
    original_strength
)

if not nearly_equal(
        replay_strength,
        original_strength
):

    raise RuntimeError(
        "Replay arbitration strength differs from original."
    )

print(
    "Arbitration strength reproduced."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Recalculate Resolution Confidence"
)

print()

replay_balance = (
    abs(
        replay_net
    )
    /
    replay_total_weight
    if replay_total_weight > EPSILON
    else
    0.0
)

reliability_values = [
    evidence[
        "reliability"
    ]
    for evidence
    in replay_evidence
]

replay_source_reliability = safe_mean(
    reliability_values
)

replay_confidence = clamp(
    safe_mean(
        [
            replay_balance,
            replay_source_reliability,
            replay_strength
        ]
    )
)

print(
    "Replay evidence balance:",
    replay_balance
)

print(
    "Replay source reliability:",
    replay_source_reliability
)

print(
    "Replay confidence:",
    replay_confidence
)

print(
    "Original confidence:",
    original_confidence
)

if not nearly_equal(
        replay_confidence,
        original_confidence
):

    raise RuntimeError(
        "Replay confidence differs from original."
    )

print(
    "Resolution confidence reproduced."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Recalculate Resolution State"
)

print()

if (
        replay_confidence
        >=
        REPLAY_CONFIDENCE_THRESHOLD
):

    replay_state = (
        "RESOLVED_WITH_CONFIDENCE"
    )

elif (
        replay_confidence
        >=
        0.25
):

    replay_state = (
        "CAUTIOUS_RESOLUTION"
    )

else:

    replay_state = (
        "UNRESOLVED"
    )

print(
    "Replay state:",
    replay_state
)

print(
    "Original state:",
    original_state
)

if replay_state != original_state:

    raise RuntimeError(
        "Replay resolution state differs from original."
    )

print(
    "Resolution state reproduced."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Reconstruct Decision Direction"
)

print()

if (
        replay_net
        >
        EPSILON
):

    replay_direction = (
        "SUPPORTS_PRIMARY_HYPOTHESIS"
    )

elif (
        replay_net
        <
        -EPSILON
):

    replay_direction = (
        "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
    )

else:

    replay_direction = (
        "UNRESOLVED"
    )

print(
    "Replay direction:",
    replay_direction
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Independent Decision Verification"
)

print()

stored_direction = (
    "SUPPORTS_PRIMARY_HYPOTHESIS"
    if original_net > EPSILON
    else
    "SUPPORTS_ALTERNATIVE_HYPOTHESIS"
    if original_net < -EPSILON
    else
    "UNRESOLVED"
)

direction_verified = (
        replay_direction
        ==
        stored_direction
)

numeric_decision_verified = all(
    [
        nearly_equal(
            replay_support,
            original_support
        ),
        nearly_equal(
            replay_conflict,
            original_conflict
        ),
        nearly_equal(
            replay_net,
            original_net
        ),
        nearly_equal(
            replay_strength,
            original_strength
        ),
        nearly_equal(
            replay_confidence,
            original_confidence
        ),
        replay_state
        ==
        original_state
    ]
)

decision_verified = (
        direction_verified
        and
        numeric_decision_verified
)

print(
    "Direction verified:",
    direction_verified
)

print(
    "Numeric decision verified:",
    numeric_decision_verified
)

print(
    "Decision verified:",
    decision_verified
)

if not decision_verified:

    raise RuntimeError(
        "Independent decision verification failed."
    )

print(
    "Independent decision verification passed."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Replay Trace Verification"
)

print()

trace_outputs = {
    "support":
        replay_support,

    "conflict":
        replay_conflict,

    "net":
        replay_net,

    "strength":
        replay_strength,

    "confidence":
        replay_confidence,

    "state":
        replay_state
}

trace_checks = {
    "support_matches":
        nearly_equal(
            trace_outputs[
                "support"
            ],
            original_support
        ),

    "conflict_matches":
        nearly_equal(
            trace_outputs[
                "conflict"
            ],
            original_conflict
        ),

    "net_matches":
        nearly_equal(
            trace_outputs[
                "net"
            ],
            original_net
        ),

    "strength_matches":
        nearly_equal(
            trace_outputs[
                "strength"
            ],
            original_strength
        ),

    "confidence_matches":
        nearly_equal(
            trace_outputs[
                "confidence"
            ],
            original_confidence
        ),

    "state_matches":
        trace_outputs[
            "state"
        ]
        ==
        original_state
}

for name, valid in trace_checks.items():

    print(
        name,
        "->",
        valid
    )

if not all(
        trace_checks.values()
):

    raise RuntimeError(
        "Reasoning trace replay verification failed."
    )

print(
    "Reasoning trace replay validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Verify Provenance Integrity During Replay"
)

print()

integrity_results = []

for provenance in provenance_records:

    source_identity = provenance.get(
        "source_identity"
    )

    stored_hash = provenance.get(
        "source_hash"
    )

    recomputed_hash = stable_hash(
        source_identity
    )

    valid = (
            stored_hash
            ==
            recomputed_hash
    )

    integrity_results.append(
        {
            "provenance_id":
                provenance.get(
                    "provenance_id"
                ),

            "valid":
                valid
        }
    )

for result in integrity_results:

    print(
        result
    )

if not all(
        result[
            "valid"
        ]
        for result
        in integrity_results
):

    raise RuntimeError(
        "Provenance integrity failed during replay."
    )

print(
    "Provenance integrity preserved during replay."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Replay Determinism"
)

print()


def replay_decision(
        evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:

    support = 0.0

    conflict = 0.0

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
        [
            float(
                item[
                    "reliability"
                ]
            )
            for item
            in evidence
        ]
    )

    balance = (
        abs(
            net
        )
        /
        total
        if total > EPSILON
        else
        0.0
    )

    confidence = clamp(
        safe_mean(
            [
                balance,
                reliability,
                strength
            ]
        )
    )

    if (
            confidence
            >=
            REPLAY_CONFIDENCE_THRESHOLD
    ):

        state = (
            "RESOLVED_WITH_CONFIDENCE"
        )

    elif (
            confidence
            >=
            0.25
    ):

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


first_replay = replay_decision(
    replay_evidence
)

second_replay = replay_decision(
    replay_evidence
)

replay_deterministic = (
        stable_hash(
            first_replay
        )
        ==
        stable_hash(
            second_replay
        )
)

print(
    "First replay:",
    first_replay
)

print(
    "Second replay:",
    second_replay
)

print(
    "Replay deterministic:",
    replay_deterministic
)

if not replay_deterministic:

    raise RuntimeError(
        "Reasoning replay is nondeterministic."
    )

print(
    "Reasoning replay determinism validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Native Replay Curriculum"
)

print()

replay_tasks = [
    {
        "example_id":
            "replay_001",

        "domain":
            "reasoning_replay",

        "question":
            "Why replay stored reasoning?",

        "answer":
            "To verify that the stored evidence still produces the recorded decision."
    },

    {
        "example_id":
            "replay_002",

        "domain":
            "decision_verification",

        "question":
            "What should independent replay compare?",

        "answer":
            "Recomputed evidence scores, confidence and resolution state against the original."
    },

    {
        "example_id":
            "replay_003",

        "domain":
            "numerical_verification",

        "question":
            "Why compare numerical reasoning outputs?",

        "answer":
            "A silent numerical change can alter the final decision."
    },

    {
        "example_id":
            "replay_004",

        "domain":
            "provenance_integrity",

        "question":
            "Why verify provenance during replay?",

        "answer":
            "The replay must use the same identified evidence that produced the original result."
    },

    {
        "example_id":
            "replay_005",

        "domain":
            "deterministic_reasoning",

        "question":
            "Why should deterministic reasoning be tested?",

        "answer":
            "Repeated replay should produce the same result when the inputs are unchanged."
    },

    {
        "example_id":
            "replay_006",

        "domain":
            "engineering_verification",

        "question":
            "Why is decision replay useful in engineering intelligence?",

        "answer":
            "It allows previous diagnostic decisions to be independently checked."
    }
]

for task in replay_tasks:

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
    "Replay tasks:",
    len(
        replay_tasks
    )
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Replay Curriculum Coverage"
)

print()

expected_domains = {
    "reasoning_replay",
    "decision_verification",
    "numerical_verification",
    "provenance_integrity",
    "deterministic_reasoning",
    "engineering_verification"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in replay_tasks
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
        "Replay curriculum coverage is incomplete."
    )

print(
    "Replay curriculum validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Numerical Health"
)

print()

health_values = [
    original_support,
    original_conflict,
    original_net,
    original_strength,
    original_confidence,
    replay_support,
    replay_conflict,
    replay_net,
    replay_strength,
    replay_confidence
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
        "Replay numerical health failed."
    )

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Final Reasoning Verification Gate"
)

print()

promotion_errors = []

if not decision_verified:

    promotion_errors.append(
        "Independent decision verification failed."
    )

if not replay_deterministic:

    promotion_errors.append(
        "Reasoning replay is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        reasoning_steps
) < TRACE_MINIMUM:

    promotion_errors.append(
        "Stored reasoning trace is incomplete."
    )

if len(
        provenance_records
) == 0:

    promotion_errors.append(
        "No provenance records exist."
    )

if len(
        decision_contributions
) == 0:

    promotion_errors.append(
        "No decision contributions exist."
    )

if not all(
        result[
            "valid"
        ]
        for result
        in integrity_results
):

    promotion_errors.append(
        "Provenance integrity failed."
    )

print(
    "Decision verified:",
    decision_verified
)

print(
    "Replay deterministic:",
    replay_deterministic
)

print(
    "Trace steps:",
    len(
        reasoning_steps
    )
)

print(
    "Provenance records:",
    len(
        provenance_records
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
        "115R reasoning verification gate failed."
    )

print(
    "115R reasoning verification gate passed."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Persist Verified Replay Memory"
)

print()

verification_event = {
    "event_id":
        "replay_115_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "115R",

    "reasoning_id":
        provenance_memory.get(
            "reasoning_id"
        ),

    "decision_verified":
        decision_verified,

    "replay_deterministic":
        replay_deterministic,

    "original_state":
        original_state,

    "replayed_state":
        replay_state,

    "original_confidence":
        original_confidence,

    "replayed_confidence":
        replay_confidence
}

replay_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_reasoning_replay_decision_verification",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        provenance_memory.get(
            "memory_version"
        ),

    "reasoning_id":
        provenance_memory.get(
            "reasoning_id"
        ),

    "original_decision":
        stored_decision,

    "replayed_decision":
        first_replay,

    "decision_verified":
        decision_verified,

    "replay_deterministic":
        replay_deterministic,

    "trace_steps":
        reasoning_steps,

    "provenance_records":
        provenance_records,

    "verification_event":
        verification_event
}

write_json(
    REPLAY_MEMORY_FILE,
    replay_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            provenance_memory.get(
                "reasoning_id"
            ),

        "original_decision":
            stored_decision,

        "replayed_decision":
            first_replay,

        "decision_verified":
            decision_verified,

        "replay_deterministic":
            replay_deterministic
    },
    REPLAY_INDEX_FILE
)

print(
    "Replay memory:",
    REPLAY_MEMORY_FILE
)

print(
    "Replay index:",
    REPLAY_INDEX_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Reload Verified Replay Memory"
)

print()

reloaded_replay = read_json(
    REPLAY_MEMORY_FILE
)

if (
        reloaded_replay[
            "reasoning_id"
        ]
        !=
        provenance_memory.get(
            "reasoning_id"
        )
):

    raise RuntimeError(
        "Reasoning identity changed after replay persistence."
    )

if (
        reloaded_replay[
            "decision_verified"
        ]
        is not True
):

    raise RuntimeError(
        "Verified decision flag changed after persistence."
    )

if (
        reloaded_replay[
            "replay_deterministic"
        ]
        is not True
):

    raise RuntimeError(
        "Replay deterministic flag changed after persistence."
    )

print(
    "Reloaded reasoning id:",
    reloaded_replay[
        "reasoning_id"
    ]
)

print(
    "Decision verified:",
    reloaded_replay[
        "decision_verified"
    ]
)

print(
    "Replay deterministic:",
    reloaded_replay[
        "replay_deterministic"
    ]
)

print(
    "Persistent replay memory validated."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save Replay Dataset"
)

print()

replay_dataset = {
    "lesson":
        "115R",

    "capability":
        "native_reasoning_replay_decision_verification",

    "reasoning_id":
        provenance_memory.get(
            "reasoning_id"
        ),

    "original_decision":
        stored_decision,

    "replayed_decision":
        first_replay,

    "decision_verified":
        decision_verified,

    "replay_deterministic":
        replay_deterministic,

    "numeric_comparison":
        {
            "support_difference":
                abs(
                    replay_support
                    -
                    original_support
                ),

            "conflict_difference":
                abs(
                    replay_conflict
                    -
                    original_conflict
                ),

            "net_difference":
                abs(
                    replay_net
                    -
                    original_net
                ),

            "strength_difference":
                abs(
                    replay_strength
                    -
                    original_strength
                ),

            "confidence_difference":
                abs(
                    replay_confidence
                    -
                    original_confidence
                )
        },

    "trace_verification":
        trace_checks,

    "provenance_integrity":
        integrity_results
}

write_json(
    REPLAY_DATASET_FILE,
    replay_dataset
)

print(
    "Replay dataset:",
    REPLAY_DATASET_FILE
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save 115R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "115R",

    "capability":
        "native_reasoning_replay_decision_verification",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        provenance_memory.get(
            "reasoning_id"
        ),

    "original_decision":
        stored_decision,

    "replayed_decision":
        first_replay,

    "decision_verified":
        decision_verified,

    "replay_deterministic":
        replay_deterministic,

    "trace_checks":
        trace_checks,

    "integrity_results":
        integrity_results,

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
    "TEST 26: Write 115R Reports"
)

print()

report = {
    "lesson":
        "115R",

    "capability":
        "native_reasoning_replay_decision_verification",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        provenance_memory.get(
            "reasoning_id"
        ),

    "original":
        stored_decision,

    "replay":
        first_replay,

    "verification":
        {
            "decision_verified":
                decision_verified,

            "replay_deterministic":
                replay_deterministic,

            "trace_checks":
                trace_checks,

            "provenance_integrity":
                all(
                    result[
                        "valid"
                    ]
                    for result
                    in integrity_results
                )
        },

    "numeric_differences":
        {
            "support":
                abs(
                    replay_support
                    -
                    original_support
                ),

            "conflict":
                abs(
                    replay_conflict
                    -
                    original_conflict
                ),

            "net":
                abs(
                    replay_net
                    -
                    original_net
                ),

            "strength":
                abs(
                    replay_strength
                    -
                    original_strength
                ),

            "confidence":
                abs(
                    replay_confidence
                    -
                    original_confidence
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
    REPLAY_REPORT_FILE,
    report
)

write_json(
    REPLAY_EVALUATION_FILE,
    report
)

write_json(
    REPLAY_REGISTRY_FILE,
    {
        "lesson":
            "115R",

        "capability":
            "native_reasoning_replay_decision_verification",

        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            provenance_memory.get(
                "reasoning_id"
            ),

        "decision_verified":
            decision_verified,

        "replay_deterministic":
            replay_deterministic,

        "next":
            "116R Native Independent Reasoning Validator + Error Detection"
    }
)

print(
    "Replay report:",
    REPLAY_REPORT_FILE
)

print(
    "Replay evaluation:",
    REPLAY_EVALUATION_FILE
)

print(
    "Replay registry:",
    REPLAY_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 115R REASONING VERIFICATION ARCHITECTURE"
)

print()

print(
    "Stored Provenance"
)

print(
    "        ↓"
)

print(
    "Evidence Reconstruction"
)

print(
    "        ↓"
)

print(
    "Independent Replay"
)

print(
    "        ↓"
)

print(
    "Recalculate Evidence Weights"
)

print(
    "        ↓"
)

print(
    "Recalculate Decision"
)

print(
    "        ↓"
)

print(
    "Compare With Original"
)

print(
    "        ↓"
)

print(
    "Decision Verification"
)

print(
    "        ↓"
)

print(
    "Verified Reasoning Memory"
)

print()


# ============================================================
# WHY 115R MATTERS
# ============================================================

print(
    "WHY 115R MATTERS"
)

print()

print(
    "113R gave Silverwing evidence arbitration."
)

print(
    "114R gave Silverwing provenance and a reasoning trace."
)

print(
    "115R closes the loop by replaying that stored reasoning."
)

print()

print(
    "Silverwing can now test whether a previous decision can "
    "be independently reconstructed from its own recorded evidence."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "115R LIMITATION"
)

print()

print(
    "Replay verification proves consistency with the stored "
    "reasoning inputs; it does not prove that those inputs "
    "were physically correct in the real world."
)

print(
    "Future validation must therefore introduce independent "
    "validators and external outcome comparisons."
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
    "Lesson 116R: Native Independent Reasoning Validator + Error Detection"
)

print()

print(
    "Independent Validator + Disagreement Detection + "
    "Reasoning Error Localization + Validator Memory"
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
    "=== LESSON 115R COMPLETE ==="
)
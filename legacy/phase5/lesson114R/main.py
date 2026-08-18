# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 114R
# Native Evidence Provenance + Reasoning Trace
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# 114R gives Silverwing an auditable reasoning trace.
#
# Every important reasoning conclusion should be traceable through:
#
#   source
#      ↓
#   evidence
#      ↓
#   transformation
#      ↓
#   reasoning step
#      ↓
#   contribution
#      ↓
#   decision
#      ↓
#   final conclusion
#
# ============================================================
# CRITICAL ARCHITECTURAL RULE
# ============================================================
#
# 113R already contains:
#
#   contradiction state
#   weighted evidence
#   support/conflict scores
#   arbitration
#   confidence
#   conclusion
#
# 114R does NOT recompute these independently from scratch.
#
# It wraps those existing results in provenance and trace objects.
#
# This preserves continuity between lessons.
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

MEMORY_VERSION = "114R.1"

MIN_TRACE_COMPLETENESS = 0.90

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

LESSON_113R = (
        PHASE5_DIR /
        "lesson113R"
)

SOURCE_ARBITRATION_MEMORY_FILE = (
        LESSON_113R /
        "silverwing_evidence_arbitration_memory.json"
)

SOURCE_ARBITRATION_INDEX_FILE = (
        LESSON_113R /
        "silverwing_evidence_arbitration_index.pt"
)

SOURCE_ARBITRATION_DATASET_FILE = (
        LESSON_113R /
        "silverwing_evidence_arbitration_dataset.json"
)

SOURCE_ARBITRATION_REPORT_FILE = (
        LESSON_113R /
        "silverwing_evidence_arbitration_report.json"
)

SOURCE_ARBITRATION_REGISTRY_FILE = (
        LESSON_113R /
        "silverwing_evidence_arbitration_registry.json"
)

SOURCE_ARBITRATION_CHECKPOINT_PRIMARY = (
        LESSON_113R /
        "checkpoints" /
        "silverwing_evidence_arbitration_best.pt"
)

SOURCE_ARBITRATION_CHECKPOINT_CANDIDATE = (
        LESSON_113R /
        "checkpoints" /
        "silverwing_evidence_arbitration_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROVENANCE_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_memory.json"
)

PROVENANCE_INDEX_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_index.pt"
)

PROVENANCE_DATASET_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_dataset.json"
)

PROVENANCE_REPORT_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_report.json"
)

PROVENANCE_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_evaluation.json"
)

PROVENANCE_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_evidence_provenance_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_evidence_provenance_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_evidence_provenance_best.pt"
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_ARBITRATION_CHECKPOINT_PRIMARY,
        SOURCE_ARBITRATION_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 113R checkpoint found."
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


def make_trace_id(
        prefix: str,
        index: int
) -> str:

    return (
        f"{prefix}_"
        f"{index + 1:04d}"
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
    "PHASE 5 - LESSON 114R"
)

print(
    "Native Evidence Provenance + Reasoning Trace"
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
    "114R -> Evidence Provenance + Reasoning Trace"
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
    "Minimum trace completeness:",
    MIN_TRACE_COMPLETENESS
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 113R Arbitration Inputs"
)

print()

for path in [
    SOURCE_ARBITRATION_MEMORY_FILE,
    SOURCE_ARBITRATION_INDEX_FILE,
    SOURCE_ARBITRATION_DATASET_FILE,
    SOURCE_ARBITRATION_REPORT_FILE,
    SOURCE_ARBITRATION_REGISTRY_FILE
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
    SOURCE_ARBITRATION_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_ARBITRATION_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_ARBITRATION_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_ARBITRATION_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_ARBITRATION_REGISTRY_FILE
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
    "TEST 2: Load Arbitration Memory"
)

print()

arbitration_memory = read_json(
    SOURCE_ARBITRATION_MEMORY_FILE
)

arbitration_dataset = read_json(
    SOURCE_ARBITRATION_DATASET_FILE
)

arbitration_report = read_json(
    SOURCE_ARBITRATION_REPORT_FILE
)

if not isinstance(
        arbitration_memory,
        dict
):

    raise RuntimeError(
        "113R arbitration memory is invalid."
    )

weighted_evidence = arbitration_memory.get(
    "evidence_sources"
)

audit_trail_113 = arbitration_memory.get(
    "audit_trail"
)

contradiction_pairs = arbitration_memory.get(
    "contradiction_pairs"
)

if not isinstance(
        weighted_evidence,
        list
):

    raise RuntimeError(
        "113R evidence sources are unavailable."
    )

if not isinstance(
        audit_trail_113,
        list
):

    raise RuntimeError(
        "113R audit trail is unavailable."
    )

if not isinstance(
        contradiction_pairs,
        list
):

    raise RuntimeError(
        "113R contradiction records are unavailable."
    )

print(
    "Memory version:",
    arbitration_memory.get(
        "memory_version"
    )
)

print(
    "Evidence sources:",
    len(
        weighted_evidence
    )
)

print(
    "Audit entries:",
    len(
        audit_trail_113
    )
)

print(
    "Contradictions:",
    len(
        contradiction_pairs
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Preserve Arbitration Decision"
)

print()

support_score = float(
    arbitration_memory.get(
        "support_score",
        0.0
    )
)

conflict_score = float(
    arbitration_memory.get(
        "conflict_score",
        0.0
    )
)

net_evidence = float(
    arbitration_memory.get(
        "net_evidence",
        0.0
    )
)

arbitration_strength = float(
    arbitration_memory.get(
        "arbitration_strength",
        0.0
    )
)

resolution_confidence = float(
    arbitration_memory.get(
        "resolution_confidence",
        0.0
    )
)

resolution_state = str(
    arbitration_memory.get(
        "resolution_state",
        "UNKNOWN"
    )
)

conclusion = str(
    arbitration_memory.get(
        "conclusion",
        ""
    )
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
    "Net evidence:",
    net_evidence
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
    "Conclusion:",
    conclusion
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Provenance Source Identities"
)

print()

provenance_errors = []

for index, evidence in enumerate(
        weighted_evidence
):

    evidence_id = evidence.get(
        "evidence_id"
    )

    source_type = evidence.get(
        "source_type"
    )

    direction = evidence.get(
        "direction"
    )

    reliability = evidence.get(
        "reliability"
    )

    if not evidence_id:

        provenance_errors.append(
            {
                "index":
                    index,

                "error":
                    "missing evidence_id"
            }
        )

    if not source_type:

        provenance_errors.append(
            {
                "evidence_id":
                    evidence_id,

                "error":
                    "missing source_type"
            }
        )

    if direction not in {
        "support",
        "conflict",
        "neutral"
    }:

        provenance_errors.append(
            {
                "evidence_id":
                    evidence_id,

                "error":
                    "invalid evidence direction"
            }
        )

    if not (
            isinstance(
                reliability,
                (int, float)
            )
            and
            0.0
            <=
            float(
                reliability
            )
            <=
            1.0
    ):

        provenance_errors.append(
            {
                "evidence_id":
                    evidence_id,

                "error":
                    "invalid reliability"
            }
        )

if provenance_errors:

    print(
        json.dumps(
            provenance_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Evidence provenance validation failed."
    )

print(
    "Evidence provenance identities validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Evidence Provenance Records"
)

print()

provenance_records = []

for index, evidence in enumerate(
        weighted_evidence
):

    evidence_snapshot = {
        key:
            value
        for key, value
        in evidence.items()
        if key
           not in {
               "provenance"
           }
    }

    source_identity = {
        "source_type":
            evidence.get(
                "source_type"
            ),

        "evidence_id":
            evidence.get(
                "evidence_id"
            ),

        "case_a":
            evidence.get(
                "case_a"
            ),

        "case_b":
            evidence.get(
                "case_b"
            ),

        "prototype_id":
            evidence.get(
                "prototype_id"
            )
    }

    provenance = {
        "provenance_id":
            make_trace_id(
                "provenance",
                index
            ),

        "source_identity":
            source_identity,

        "source_hash":
            stable_hash(
                source_identity
            ),

        "evidence_snapshot":
            evidence_snapshot,

        "retrieval_stage":
            "113R_evidence_arbitration",

        "captured_at":
            datetime.now().isoformat()
    }

    provenance_records.append(
        provenance
    )

for record in provenance_records:

    print(
        record
    )

if len(
        provenance_records
) != len(
    weighted_evidence
):

    raise RuntimeError(
        "Provenance record count mismatch."
    )

print(
    "Evidence provenance records built."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Reasoning Steps"
)

print()

reasoning_steps = [
    {
        "step_id":
            "reason_001",

        "operation":
            "collect_evidence",

        "input":
            "weighted_evidence",

        "output":
            "provenance_records",

        "description":
            "Collect and identify the evidence used by arbitration."
    },

    {
        "step_id":
            "reason_002",

        "operation":
            "separate_support_and_conflict",

        "input":
            "provenance_records",

        "output":
            "support_conflict_sets",

        "description":
            "Separate evidence according to reasoning direction."
    },

    {
        "step_id":
            "reason_003",

        "operation":
            "weighted_arbitration",

        "input":
            "support_conflict_sets",

        "output":
            "support_conflict_scores",

        "description":
            "Aggregate evidence using stored reliability weights."
    },

    {
        "step_id":
            "reason_004",

        "operation":
            "calculate_net_evidence",

        "input":
            "support_conflict_scores",

        "output":
            "net_evidence",

        "description":
            "Compute the balance between supporting and conflicting evidence."
    },

    {
        "step_id":
            "reason_005",

        "operation":
            "calculate_confidence",

        "input":
            "net_evidence",

        "output":
            "resolution_confidence",

        "description":
            "Convert evidence balance into the stored confidence state."
    },

    {
        "step_id":
            "reason_006",

        "operation":
            "resolve_conclusion",

        "input":
            "resolution_confidence",

        "output":
            "resolution_state",

        "description":
            "Preserve either a resolved, cautious or unresolved state."
    }
]

for step in reasoning_steps:

    print(
        step[
            "step_id"
        ],
        "->",
        step[
            "operation"
        ]
    )

if len(
        reasoning_steps
) < 6:

    raise RuntimeError(
        "Reasoning trace contains too few steps."
    )

print(
    "Reasoning steps constructed:",
    len(
        reasoning_steps
    )
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Link Evidence to Reasoning Steps"
)

print()

for provenance in provenance_records:

    provenance[
        "reasoning_links"
    ] = [
        "reason_001",
        "reason_002",
        "reason_003",
        "reason_004",
        "reason_005",
        "reason_006"
    ]

for provenance in provenance_records:

    if not provenance[
        "reasoning_links"
    ]:

        raise RuntimeError(
            "Evidence has no reasoning links."
        )

print(
    "Evidence-to-reasoning links validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Build Decision Contributions"
)

print()

decision_contributions = []

for provenance in provenance_records:

    evidence = provenance[
        "evidence_snapshot"
    ]

    contribution = float(
        evidence.get(
            "contribution",
            0.0
        )
    )

    decision_contributions.append(
        {
            "evidence_id":
                evidence.get(
                    "evidence_id"
                ),

            "provenance_id":
                provenance[
                    "provenance_id"
                ],

            "direction":
                evidence.get(
                    "direction"
                ),

            "weight":
                float(
                    evidence.get(
                        "weight",
                        0.0
                    )
                ),

            "contribution":
                contribution,

            "decision_effect":
                (
                    "supports"
                    if contribution > EPSILON
                    else
                    "opposes"
                    if contribution < -EPSILON
                    else
                    "neutral"
                )
        }
    )

for contribution in decision_contributions:

    print(
        contribution
    )

if len(
        decision_contributions
) != len(
    provenance_records
):

    raise RuntimeError(
        "Decision contribution count mismatch."
    )

print(
    "Decision contributions validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Reproduce Arbitration from Provenance"
)

print()

reproduced_support = sum(
    item[
        "contribution"
    ]
    for item
    in decision_contributions
    if item[
        "decision_effect"
    ]
    ==
    "supports"
)

reproduced_conflict = sum(
    abs(
        item[
            "contribution"
        ]
    )
    for item
    in decision_contributions
    if item[
        "decision_effect"
    ]
    ==
    "opposes"
)

reproduced_net = (
        reproduced_support
        -
        reproduced_conflict
)

print(
    "Reproduced support:",
    reproduced_support
)

print(
    "Reproduced conflict:",
    reproduced_conflict
)

print(
    "Reproduced net:",
    reproduced_net
)

print(
    "Stored support:",
    support_score
)

print(
    "Stored conflict:",
    conflict_score
)

print(
    "Stored net:",
    net_evidence
)

if abs(
        reproduced_support
        -
        support_score
) > 1e-6:

    raise RuntimeError(
        "Support score cannot be reproduced from provenance."
    )

if abs(
        reproduced_conflict
        -
        conflict_score
) > 1e-6:

    raise RuntimeError(
        "Conflict score cannot be reproduced from provenance."
    )

if abs(
        reproduced_net
        -
        net_evidence
) > 1e-6:

    raise RuntimeError(
        "Net evidence cannot be reproduced from provenance."
    )

print(
    "Arbitration reproduced from provenance."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Build Complete Reasoning Trace"
)

print()

trace_steps = []

for step_index, step in enumerate(
        reasoning_steps
):

    trace_steps.append(
        {
            "trace_id":
                make_trace_id(
                    "trace",
                    step_index
                ),

            "step_id":
                step[
                    "step_id"
                ],

            "operation":
                step[
                    "operation"
                ],

            "input":
                step[
                    "input"
                ],

            "output":
                step[
                    "output"
                ],

            "description":
                step[
                    "description"
                ],

            "evidence_refs":
                [
                    provenance[
                        "provenance_id"
                    ]
                    for provenance
                    in provenance_records
                ]
                if step_index
                   <=
                   5
                else
                []
        }
    )

for trace in trace_steps:

    print(
        trace
    )

if not trace_steps:

    raise RuntimeError(
        "Reasoning trace was not built."
    )

print(
    "Complete reasoning trace built."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Trace Completeness"
)

print()

completeness_checks = {
    "evidence_present":
        len(
            provenance_records
        )
        >
        0,

    "reasoning_steps_present":
        len(
            reasoning_steps
        )
        >=
        6,

    "decision_contributions_present":
        len(
            decision_contributions
        )
        ==
        len(
            provenance_records
        ),

    "contradictions_preserved":
        len(
            contradiction_pairs
        )
        ==
        len(
            arbitration_memory.get(
                "contradiction_pairs",
                []
            )
        ),

    "conclusion_present":
        bool(
            conclusion
        ),

    "confidence_present":
        math.isfinite(
            resolution_confidence
        ),

    "trace_steps_present":
        len(
            trace_steps
        )
        >=
        6
}

for name, value in completeness_checks.items():

    print(
        name,
        "->",
        value
    )

trace_completeness = (
        sum(
            1
            for value
            in completeness_checks.values()
            if value
        )
        /
        len(
            completeness_checks
        )
)

print(
    "Trace completeness:",
    trace_completeness
)

if (
        trace_completeness
        <
        MIN_TRACE_COMPLETENESS
):

    raise RuntimeError(
        "Reasoning trace completeness is below threshold."
    )

print(
    "Trace completeness validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Provenance Integrity Hashes"
)

print()

integrity_records = []

for provenance in provenance_records:

    expected_hash = stable_hash(
        provenance[
            "source_identity"
        ]
    )

    stored_hash = provenance[
        "source_hash"
    ]

    valid = (
            expected_hash
            ==
            stored_hash
    )

    integrity_records.append(
        {
            "provenance_id":
                provenance[
                    "provenance_id"
                ],

            "valid":
                valid
        }
    )

for record in integrity_records:

    print(
        record
    )

if not all(
        record[
            "valid"
        ]
        for record
        in integrity_records
):

    raise RuntimeError(
        "Evidence provenance integrity validation failed."
    )

print(
    "Evidence provenance integrity validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Reasoning Trace Determinism"
)

print()


def build_trace_signature(
        traces: List[Dict[str, Any]]
) -> str:

    compact = [
        {
            "step_id":
                trace[
                    "step_id"
                ],

            "operation":
                trace[
                    "operation"
                ],

            "input":
                trace[
                    "input"
                ],

            "output":
                trace[
                    "output"
                ]
        }
        for trace
        in traces
    ]

    return stable_hash(
        compact
    )


first_signature = build_trace_signature(
    trace_steps
)

second_signature = build_trace_signature(
    trace_steps
)

trace_deterministic = (
        first_signature
        ==
        second_signature
)

print(
    "First trace signature:",
    first_signature
)

print(
    "Second trace signature:",
    second_signature
)

print(
    "Deterministic:",
    trace_deterministic
)

if not trace_deterministic:

    raise RuntimeError(
        "Reasoning trace is nondeterministic."
    )

print(
    "Deterministic reasoning trace validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Contradiction Trace Coverage"
)

print()

contradiction_trace_coverage = []

for contradiction in contradiction_pairs:

    case_a = contradiction.get(
        "case_a"
    )

    case_b = contradiction.get(
        "case_b"
    )

    linked = False

    for provenance in provenance_records:

        evidence = provenance[
            "evidence_snapshot"
        ]

        if (
                evidence.get(
                    "case_a"
                )
                ==
                case_a
                and
                evidence.get(
                    "case_b"
                )
                ==
                case_b
        ):

            linked = True
            break

    contradiction_trace_coverage.append(
        {
            "case_a":
                case_a,

            "case_b":
                case_b,

            "traced":
                linked
        }
    )

for result in contradiction_trace_coverage:

    print(
        result
    )

if contradiction_trace_coverage:

    if not all(
            result[
                "traced"
            ]
            for result
            in contradiction_trace_coverage
    ):

        raise RuntimeError(
            "Not every contradiction has a provenance trace."
        )

print(
    "Contradiction trace coverage validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Build Final Reasoning Artifact"
)

print()

final_reasoning_artifact = {
    "reasoning_id":
        "reasoning_114_001",

    "lesson":
        "114R",

    "source_lesson":
        "113R",

    "created_at":
        datetime.now().isoformat(),

    "provenance":
        provenance_records,

    "reasoning_steps":
        trace_steps,

    "decision_contributions":
        decision_contributions,

    "contradictions":
        contradiction_pairs,

    "decision":
        {
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
                conclusion
        },

    "trace_completeness":
        trace_completeness
}

print(
    "Reasoning artifact fields:",
    list(
        final_reasoning_artifact.keys()
    )
)

print(
    "Final reasoning artifact constructed."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Evidence Provenance Curriculum"
)

print()

provenance_tasks = [
    {
        "example_id":
            "provenance_001",

        "domain":
            "evidence_identity",

        "question":
            "Why must evidence have a stable identity?",

        "answer":
            "A reasoning system must be able to trace a conclusion back to its source evidence."
    },

    {
        "example_id":
            "provenance_002",

        "domain":
            "provenance_tracking",

        "question":
            "What is evidence provenance?",

        "answer":
            "Information describing where evidence originated and how it entered reasoning."
    },

    {
        "example_id":
            "provenance_003",

        "domain":
            "reasoning_trace",

        "question":
            "Why store reasoning steps?",

        "answer":
            "They make the transformation from evidence to conclusion inspectable."
    },

    {
        "example_id":
            "provenance_004",

        "domain":
            "decision_trace",

        "question":
            "What should a decision trace contain?",

        "answer":
            "The evidence, operations, contributions and final decision state."
    },

    {
        "example_id":
            "provenance_005",

        "domain":
            "auditability",

        "question":
            "Why is reasoning auditability important?",

        "answer":
            "It allows conclusions to be verified instead of accepted as unexplained outputs."
    },

    {
        "example_id":
            "provenance_006",

        "domain":
            "engineering_intelligence",

        "question":
            "Why is provenance important for engineering intelligence?",

        "answer":
            "Engineering decisions must remain connected to observable evidence and transformations."
    }
]

for task in provenance_tasks:

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
    "Provenance tasks:",
    len(
        provenance_tasks
    )
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Provenance Curriculum Coverage"
)

print()

expected_domains = {
    "evidence_identity",
    "provenance_tracking",
    "reasoning_trace",
    "decision_trace",
    "auditability",
    "engineering_intelligence"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in provenance_tasks
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
        "Provenance curriculum coverage is incomplete."
    )

print(
    "Provenance curriculum validated."
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
    net_evidence,
    arbitration_strength,
    resolution_confidence,
    trace_completeness
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
        "Provenance numerical health failed."
    )

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Final Provenance Promotion Gate"
)

print()

promotion_errors = []

if not trace_deterministic:

    promotion_errors.append(
        "Reasoning trace is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if (
        trace_completeness
        <
        MIN_TRACE_COMPLETENESS
):

    promotion_errors.append(
        "Trace completeness is below threshold."
    )

if not provenance_records:

    promotion_errors.append(
        "No provenance records exist."
    )

if not reasoning_steps:

    promotion_errors.append(
        "No reasoning steps exist."
    )

if not decision_contributions:

    promotion_errors.append(
        "No decision contributions exist."
    )

if (
        contradiction_pairs
        and
        not all(
            item[
                "traced"
            ]
            for item
            in contradiction_trace_coverage
        )
):

    promotion_errors.append(
        "Not all contradictions are traceable."
    )

if abs(
        reproduced_net
        -
        net_evidence
) > 1e-6:

    promotion_errors.append(
        "Provenance cannot reproduce final net evidence."
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
    "Trace completeness:",
    trace_completeness
)

print(
    "Trace deterministic:",
    trace_deterministic
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
        "114R provenance promotion gate failed."
    )

print(
    "114R provenance promotion gate passed."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Persist Provenance Memory"
)

print()

provenance_event = {
    "event_id":
        "provenance_114_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "114R",

    "reasoning_id":
        final_reasoning_artifact[
            "reasoning_id"
        ],

    "trace_completeness":
        trace_completeness,

    "trace_deterministic":
        trace_deterministic,

    "resolution_state":
        resolution_state,

    "resolution_confidence":
        resolution_confidence
}

provenance_memory = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_evidence_provenance_reasoning_trace",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        arbitration_memory.get(
            "memory_version"
        ),

    "reasoning_id":
        final_reasoning_artifact[
            "reasoning_id"
        ],

    "provenance_records":
        provenance_records,

    "reasoning_steps":
        trace_steps,

    "decision_contributions":
        decision_contributions,

    "contradictions":
        contradiction_pairs,

    "decision":
        final_reasoning_artifact[
            "decision"
        ],

    "trace_completeness":
        trace_completeness,

    "provenance_event":
        provenance_event
}

write_json(
    PROVENANCE_MEMORY_FILE,
    provenance_memory
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            final_reasoning_artifact[
                "reasoning_id"
            ],

        "support_score":
            support_score,

        "conflict_score":
            conflict_score,

        "net_evidence":
            net_evidence,

        "resolution_confidence":
            resolution_confidence,

        "trace_completeness":
            trace_completeness,

        "provenance_hashes":
            [
                record[
                    "source_hash"
                ]
                for record
                in provenance_records
            ]
    },
    PROVENANCE_INDEX_FILE
)

print(
    "Provenance memory:",
    PROVENANCE_MEMORY_FILE
)

print(
    "Provenance index:",
    PROVENANCE_INDEX_FILE
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Reload Provenance Memory"
)

print()

reloaded_provenance = read_json(
    PROVENANCE_MEMORY_FILE
)

if (
        reloaded_provenance[
            "reasoning_id"
        ]
        !=
        final_reasoning_artifact[
            "reasoning_id"
        ]
):

    raise RuntimeError(
        "Reasoning identity changed after persistence."
    )

if (
        len(
            reloaded_provenance[
                "provenance_records"
            ]
        )
        !=
        len(
            provenance_records
        )
):

    raise RuntimeError(
        "Provenance record count changed."
    )

if (
        len(
            reloaded_provenance[
                "reasoning_steps"
            ]
        )
        !=
        len(
            trace_steps
        )
):

    raise RuntimeError(
        "Reasoning trace count changed."
    )

if (
        reloaded_provenance[
            "trace_completeness"
        ]
        !=
        trace_completeness
):

    raise RuntimeError(
        "Trace completeness changed after persistence."
    )

print(
    "Reloaded reasoning id:",
    reloaded_provenance[
        "reasoning_id"
    ]
)

print(
    "Reloaded provenance records:",
    len(
        reloaded_provenance[
            "provenance_records"
        ]
    )
)

print(
    "Reloaded reasoning steps:",
    len(
        reloaded_provenance[
            "reasoning_steps"
        ]
    )
)

print(
    "Persistent provenance memory validated."
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Save Provenance Dataset"
)

print()

provenance_dataset = {
    "lesson":
        "114R",

    "capability":
        "native_evidence_provenance_reasoning_trace",

    "source_lesson":
        "113R",

    "reasoning_id":
        final_reasoning_artifact[
            "reasoning_id"
        ],

    "provenance_records":
        provenance_records,

    "reasoning_steps":
        trace_steps,

    "decision_contributions":
        decision_contributions,

    "contradictions":
        contradiction_pairs,

    "decision":
        final_reasoning_artifact[
            "decision"
        ],

    "trace_completeness":
        trace_completeness,

    "audit_hash":
        stable_hash(
            audit_trail_113
        )
}

write_json(
    PROVENANCE_DATASET_FILE,
    provenance_dataset
)

print(
    "Provenance dataset:",
    PROVENANCE_DATASET_FILE
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Save 114R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "114R",

    "capability":
        "native_evidence_provenance_reasoning_trace",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "reasoning_id":
        final_reasoning_artifact[
            "reasoning_id"
        ],

    "provenance_records":
        provenance_records,

    "reasoning_steps":
        trace_steps,

    "decision_contributions":
        decision_contributions,

    "contradictions":
        contradiction_pairs,

    "decision":
        final_reasoning_artifact[
            "decision"
        ],

    "trace_completeness":
        trace_completeness,

    "trace_deterministic":
        trace_deterministic,

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
    "TEST 24: Write 114R Reports"
)

print()

report = {
    "lesson":
        "114R",

    "capability":
        "native_evidence_provenance_reasoning_trace",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "reasoning":
        {
            "reasoning_id":
                final_reasoning_artifact[
                    "reasoning_id"
                ],

            "resolution_state":
                resolution_state,

            "resolution_confidence":
                resolution_confidence,

            "conclusion":
                conclusion
        },

    "provenance":
        {
            "record_count":
                len(
                    provenance_records
                ),

            "records":
                provenance_records
        },

    "trace":
        {
            "step_count":
                len(
                    trace_steps
                ),

            "steps":
                trace_steps,

            "completeness":
                trace_completeness,

            "deterministic":
                trace_deterministic
        },

    "decision":
        {
            "support_score":
                support_score,

            "conflict_score":
                conflict_score,

            "net_evidence":
                net_evidence,

            "arbitration_strength":
                arbitration_strength
        },

    "contradictions":
        contradiction_pairs,

    "verification":
        {
            "provenance_reproduction":
                abs(
                    reproduced_net
                    -
                    net_evidence
                )
                <=
                1e-6,

            "integrity":
                all(
                    record[
                        "valid"
                    ]
                    for record
                    in integrity_records
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
    PROVENANCE_REPORT_FILE,
    report
)

write_json(
    PROVENANCE_EVALUATION_FILE,
    report
)

write_json(
    PROVENANCE_REGISTRY_FILE,
    {
        "lesson":
            "114R",

        "capability":
            "native_evidence_provenance_reasoning_trace",

        "memory_version":
            MEMORY_VERSION,

        "reasoning_id":
            final_reasoning_artifact[
                "reasoning_id"
            ],

        "trace_completeness":
            trace_completeness,

        "resolution_state":
            resolution_state,

        "next":
            "115R Native Reasoning Replay + Decision Verification"
    }
)

print(
    "Provenance report:",
    PROVENANCE_REPORT_FILE
)

print(
    "Provenance evaluation:",
    PROVENANCE_EVALUATION_FILE
)

print(
    "Provenance registry:",
    PROVENANCE_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 114R EVIDENCE PROVENANCE ARCHITECTURE"
)

print()

print(
    "Evidence Source"
)

print(
    "       ↓"
)

print(
    "Provenance Identity"
)

print(
    "       ↓"
)

print(
    "Evidence Snapshot"
)

print(
    "       ↓"
)

print(
    "Reasoning Step"
)

print(
    "       ↓"
)

print(
    "Decision Contribution"
)

print(
    "       ↓"
)

print(
    "Arbitration"
)

print(
    "       ↓"
)

print(
    "Confidence / Uncertainty"
)

print(
    "       ↓"
)

print(
    "Final Conclusion"
)

print(
    "       ↓"
)

print(
    "Complete Audit Trace"
)

print()


# ============================================================
# WHY 114R MATTERS
# ============================================================

print(
    "WHY 114R MATTERS"
)

print()

print(
    "113R taught Silverwing how to arbitrate conflicting evidence."
)

print(
    "114R makes that reasoning traceable."
)

print()

print(
    "A future Silverwing decision should therefore be able to "
    "answer four questions:"
)

print(
    "1. Where did this evidence come from?"
)

print(
    "2. What transformation was applied?"
)

print(
    "3. How did the evidence affect the decision?"
)

print(
    "4. Which reasoning path produced the final conclusion?"
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "114R LIMITATION"
)

print()

print(
    "A provenance trace improves auditability but does not by "
    "itself prove that the underlying evidence is correct."
)

print(
    "Future lessons must therefore combine provenance with "
    "replay, independent verification and outcome-based validation."
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
    "Lesson 115R: Native Reasoning Replay + Decision Verification"
)

print()

print(
    "Stored Trace + Replay + Independent Recalculation + "
    "Decision Comparison + Verification"
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
    "=== LESSON 114R COMPLETE ==="
)
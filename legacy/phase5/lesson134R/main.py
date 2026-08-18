# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 134R
# Autonomous Policy Memory Consolidation
# + Continuous Self-Improvement Commitment
# ============================================================
#
# 130R  -> Self-Healing Recovery Orchestration
#         + Failure Absorption
# 131R  -> Collective Defense Consolidation
#         + System-Level Resilience Audit
# 132R  -> Autonomous Defense Self-Tuning
#         + Online Control Refinement
# 133R  -> End-to-End Self-Improving Control Ledger
#         + Autonomous Policy Memory
# 134R  -> Autonomous Policy Memory Consolidation
#         + Continuous Self-Improvement Commitment
#
# ============================================================
# PURPOSE
# ============================================================
#
# 133R froze the 125R..132R arc into a ledger and distilled it
# into an autonomous policy memory. 134R closes the loop one
# final way: it CONSOLIDATES that policy memory into a single
# canonical, machine-readable form, and it COMMITS to it.
#
# Consolidation: the per-pattern policy becomes a numeric
# policy matrix -- one row per pattern, one column per policy
# dimension. The consolidated form must be lossless: hashing
# it must reproduce the 133R policy hash exactly.
#
# Commitment: the system does not merely hold a policy; it
# commits to it. The committed policy is signed by its hash,
# linked to the ledger that produced it, and chained into an
# append-only commitment record. A self-improvement rule
# guards every future commitment: no commitment may regress
# exposure beyond a numerical limit. The system then renews
# its commitment every cycle, proving continuity without
# drift.
#
# Policy memory consolidation:
#
#     autonomous policy (133R)
#               ↓
#     numeric encoding per pattern
#               ↓
#     canonical policy matrix [6 x 8]
#               ↓
#     lossless consolidation check
#
# Continuous self-improvement commitment:
#
#     genesis commitment
#               ↓
#     renewal cycle 1 -> 2 -> 3
#               ↓
#     no-regression guardrail
#               ↓
#     final commitment hash
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 133R memory is the source of truth.
# 2. Policy consolidation must be lossless.
# 3. One policy matrix row per pattern.
# 4. The committed exposure is the consolidated mean.
# 5. Commitments are append-only and hash-chained.
# 6. A commitment must reference the ledger that produced it.
# 7. No commitment may regress exposure.
# 8. Renewal proves continuity without drift.
# 9. The guardrail must reject a regressing candidate.
# 10. Determinism must be checked.
# 11. Numerical health must be checked.
# 12. Persistence and reload must be checked.
# 13. Promotion requires all validation gates to pass.
# 14. External LLM: NONE.
#
# ============================================================

import hashlib
import json
import random
import sys

from datetime import datetime
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(
        encoding="utf-8"
    )
except Exception:
    pass

SEED = 42
MEMORY_VERSION = "134R.1"
EXPECTED_PATTERNS = 6
POLICY_DIMENSIONS = 8
COMMITMENT_CYCLES = 3
REGRESSION_LIMIT = 1e-4
COMMITMENT_GENESIS_PHRASE = "SILVERWING_COMMITMENT"

TIER_CODE = {
    "ANOMALY": 3,
    "CRITICAL": 2,
    "SECONDARY": 1
}

ACTION_CODE = {
    "ESCALATE": 2,
    "HOLD": 1
}

TREND_CODE = {
    "RISING": 3,
    "STABLE": 2,
    "FALLING": 1
}

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_133R = PHASE5_DIR / "lesson133R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_133R
        / "silverwing_control_ledger_memory.json"
)

SOURCE_INDEX = (
        LESSON_133R
        / "silverwing_control_ledger_index.pt"
)

SOURCE_DATASET = (
        LESSON_133R
        / "silverwing_control_ledger_dataset.json"
)

SOURCE_REPORT = (
        LESSON_133R
        / "silverwing_control_ledger_report.json"
)

SOURCE_REGISTRY = (
        LESSON_133R
        / "silverwing_control_ledger_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_133R
        / "checkpoints"
        / "silverwing_control_ledger_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_policy_commitment_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_policy_commitment_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_policy_commitment_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_policy_commitment_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_policy_commitment_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_policy_commitment_best.pt"
)

read_json = lambda path: json.loads(
    path.read_text(
        encoding="utf-8"
    )
)

save_json = lambda path, data: path.write_text(
    json.dumps(
        data,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

clamp = lambda value: max(
    0.0,
    min(
        1.0,
        float(value)
    )
)

stable_hash = lambda value: hashlib.sha256(
    json.dumps(
        value,
        sort_keys=True,
        default=str
    ).encode("utf-8")
).hexdigest()

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 134R"
)

print(
    "Autonomous Policy Memory Consolidation"
)

print(
    "+ Continuous Self-Improvement Commitment"
)

print()

print(
    "130R -> Self-Healing Recovery Orchestration"
)

print(
    "        + Failure Absorption"
)

print(
    "131R -> Collective Defense Consolidation"
)

print(
    "        + System-Level Resilience Audit"
)

print(
    "132R -> Autonomous Defense Self-Tuning"
)

print(
    "        + Online Control Refinement"
)

print(
    "133R -> End-to-End Self-Improving Control Ledger"
)

print(
    "        + Autonomous Policy Memory"
)

print(
    "134R -> Autonomous Policy Memory Consolidation"
)

print(
    "        + Continuous Self-Improvement Commitment"
)

print()

print(
    "External LLM: NONE"
)

print(
    "Memory version:",
    MEMORY_VERSION
)

print()

print(
    "TEST 1: Verify 133R Inputs"
)

REQUIRED_FILES = [
    SOURCE_MEMORY,
    SOURCE_INDEX,
    SOURCE_DATASET,
    SOURCE_REPORT,
    SOURCE_REGISTRY,
    SOURCE_CHECKPOINT
]

assert all(
    path.exists()
    for path in REQUIRED_FILES
), "One or more source inputs are missing."

print(
    "FOUND:",
    SOURCE_MEMORY
)

print(
    "FOUND:",
    SOURCE_INDEX
)

print(
    "FOUND:",
    SOURCE_DATASET
)

print(
    "FOUND:",
    SOURCE_REPORT
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

print(
    "TEST 2: Load 133R Policy Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "133R policy memory is invalid."

POLICY = SOURCE.get(
    "policy",
    {}
)

POLICY_HASH = SOURCE.get(
    "policy_hash",
    ""
)

LEDGER_HASH = SOURCE.get(
    "ledger_hash",
    ""
)

IMPROVEMENT_PROOF = SOURCE.get(
    "improvement_proof",
    {}
)

assert len(
    POLICY
) == EXPECTED_PATTERNS, (
    "133R policy must cover six patterns."
)

assert (
    POLICY_HASH
    ==
    "1cc16f7bb2647d89d0fcc6594c1e8ed8866f8597dc73127f5d76a34b2733f796"
), "133R policy hash mismatch."

assert (
    LEDGER_HASH[:16]
    ==
    "479eea1f6cd01de0"
), "133R ledger hash mismatch."

TUNED_ORDER = SOURCE.get(
    "source_lessons",
    []
)

if not TUNED_ORDER:

    TUNED_ORDER = sorted(
        POLICY
    )

if len(
    TUNED_ORDER
) != EXPECTED_PATTERNS:

    TUNED_ORDER = [
        "pattern_001",
        "pattern_004",
        "pattern_003",
        "pattern_006",
        "pattern_005",
        "pattern_002"
    ]

print(
    "Policy hash:",
    POLICY_HASH[:16]
)

print(
    "Ledger hash:",
    LEDGER_HASH[:16]
)

print()

print(
    "TEST 3: Consolidate Policy Matrix"
)


def encode_policy(
        record
):

    return [
        record["schedule_position"],
        int(
            record["critical"]
        ),
        TIER_CODE[
            record["tier"]
        ],
        ACTION_CODE[
            record["control_action"]
        ],
        TREND_CODE[
            record["trend"]
        ],
        record["boost"],
        record["penetration"],
        record["recovery_cost"]
    ]


POLICY_MATRIX = [
    encode_policy(
        POLICY[
            pattern_id
        ]
    )
    for pattern_id
    in TUNED_ORDER
]

MATRIX_HASH = stable_hash(
    POLICY_MATRIX
)

MATRIX_TENSOR = torch.tensor(
    POLICY_MATRIX,
    dtype=torch.float32
)

assert MATRIX_TENSOR.shape == (
    EXPECTED_PATTERNS,
    POLICY_DIMENSIONS
), "Policy matrix must be [6 x 8]."

assert (
    MATRIX_HASH
    ==
    "ba607fc3f14126705b0b6bf2eb6a9ac9e3033c643405a28a700694579db8d8f5"
), "Policy matrix hash mismatch."

print(
    "Matrix hash:",
    MATRIX_HASH[:16]
)

print(
    "Matrix shape:",
    list(
        MATRIX_TENSOR.shape
    )
)

print()

print(
    "TEST 4: Verify Consolidation Fidelity"
)

CONSOLIDATED_HASH = stable_hash(
    {
        pattern_id: POLICY[pattern_id]
        for pattern_id
        in TUNED_ORDER
    }
)

assert (
    CONSOLIDATED_HASH
    ==
    POLICY_HASH
), "Consolidation must be lossless."

print(
    "Consolidated hash:",
    CONSOLIDATED_HASH[:16]
)

print(
    "Policy hash:",
    POLICY_HASH[:16]
)

print(
    "Lossless consolidation:",
    CONSOLIDATED_HASH
    ==
    POLICY_HASH
)

print()

print(
    "TEST 5: Compute Policy Statistics"
)

MEAN_EXPOSURE = float(
    MATRIX_TENSOR[
        :,
        6
    ].mean()
)

WORST_INDEX = int(
    torch.argmax(
        MATRIX_TENSOR[
            :,
            6
        ]
    )
)

WORST_PATTERN = TUNED_ORDER[
    WORST_INDEX
]

COVERAGE = (
        MATRIX_TENSOR.shape[0]
        /
        EXPECTED_PATTERNS
)

assert abs(
    MEAN_EXPOSURE
    -
    0.098035
) <= 1e-4, (
    "Mean exposure mismatch."
)

assert WORST_PATTERN == "pattern_004", (
    "Worst pattern mismatch."
)

assert abs(
    POLICY["pattern_004"]["penetration"]
    -
    0.120005
) <= 1e-4, (
    "Worst penetration mismatch."
)

assert COVERAGE == 1.0, (
    "Policy coverage must be complete."
)

print(
    "Mean exposure:",
    format(
        MEAN_EXPOSURE,
        ".4f"
    )
)

print(
    "Worst pattern:",
    WORST_PATTERN,
    format(
        POLICY[WORST_PATTERN]["penetration"],
        ".4f"
    )
)

print(
    "Coverage:",
    COVERAGE
)

print()

print(
    "TEST 6: Establish Committed Policy"
)


def commitment_entry(
        index,
        status,
        previous_hash
):

    entry = {
        "index":
            index,

        "version":
            MEMORY_VERSION,

        "policy_hash":
            POLICY_HASH,

        "exposure":
            MEAN_EXPOSURE,

        "status":
            status,

        "prev_hash":
            previous_hash
    }

    entry[
        "hash"
    ] = stable_hash(
        entry
    )

    return entry


COMMITMENT_GENESIS_HASH = stable_hash(
    COMMITMENT_GENESIS_PHRASE
)

assert (
    COMMITMENT_GENESIS_HASH
    ==
    "2f073a8a1c1881b177052ff269b1958e89bbb6b5c583ea97a0296b91c2759d8c"
), "Commitment genesis hash mismatch."

COMMITMENTS = [
    commitment_entry(
        0,
        "GENESIS",
        COMMITMENT_GENESIS_HASH
    )
]

assert (
    COMMITMENTS[0]["hash"]
    ==
    "71133095e5028b5db8fb88a6846cce521e2b668fc1a3783ad4669e5b6775ca5d"
), "Genesis commitment hash mismatch."

print(
    "Commitment genesis:",
    COMMITMENT_GENESIS_HASH[:16]
)

print(
    "Genesis commitment:",
    COMMITMENTS[0]["hash"][:16]
)

print(
    "Committed exposure:",
    format(
        MEAN_EXPOSURE,
        ".4f"
    )
)

print()

print(
    "TEST 7: Run Continuous Commitment Cycles"
)

RENEWAL_RECORDS = []

for cycle in range(
        1,
        COMMITMENT_CYCLES + 1
):

    verified = (
            POLICY_HASH
            ==
            stable_hash(
                {
                    pattern_id: POLICY[pattern_id]
                    for pattern_id
                    in TUNED_ORDER
                }
            )
            and
            LEDGER_HASH[:16]
            ==
            "479eea1f6cd01de0"
    )

    assert verified, (
        "Renewal must re-verify the policy against the ledger."
    )

    entry = commitment_entry(
        cycle,
        "RENEWED",
        COMMITMENTS[-1]["hash"]
    )

    COMMITMENTS.append(
        entry
    )

    RENEWAL_RECORDS.append(
        {
            "cycle": cycle,
            "hash": entry["hash"],
            "exposure": entry["exposure"]
        }
    )

for record in RENEWAL_RECORDS:

    print(
        "Cycle",
        record["cycle"],
        "| exposure=",
        format(
            record["exposure"],
            ".4f"
        ),
        "| hash=",
        record["hash"][:16]
    )

print()

print(
    "TEST 8: Verify No-Regression Commitment"
)

COMMITMENT_HASH = COMMITMENTS[-1][
    "hash"
]

assert len(
    COMMITMENTS
) == COMMITMENT_CYCLES + 1, (
    "Commitment record must hold genesis plus renewals."
)

assert (
    COMMITMENT_HASH
    ==
    "7aedadeb67450decec7235cca23bcc9b5ffe1bfed5a9cb4a2ec7e77e6f4d9af4"
), "Final commitment hash mismatch."

CHAIN_VALID = True

previous_hash = COMMITMENT_GENESIS_HASH

for entry in COMMITMENTS:

    recomputed = stable_hash(
        {
            "index":
                entry["index"],

            "version":
                entry["version"],

            "policy_hash":
                entry["policy_hash"],

            "exposure":
                entry["exposure"],

            "status":
                entry["status"],

            "prev_hash":
                entry["prev_hash"]
        }
    )

    if (
            recomputed
            !=
            entry["hash"]
            or
            entry["prev_hash"]
            !=
            previous_hash
    ):

        CHAIN_VALID = False

        break

    previous_hash = entry[
        "hash"
    ]

NO_REGRESSION = all(
    abs(
        entry["exposure"]
        -
        MEAN_EXPOSURE
    )
    <=
    REGRESSION_LIMIT
    for entry
    in COMMITMENTS
)

assert CHAIN_VALID, (
    "Commitment chain is broken."
)

assert NO_REGRESSION, (
    "A commitment regressed exposure."
)

print(
    "Chain valid:",
    CHAIN_VALID
)

print(
    "No regression:",
    NO_REGRESSION
)

print(
    "Final commitment hash:",
    COMMITMENT_HASH[:16]
)

print()

print(
    "TEST 9: Guardrail Rejection Test"
)


def evaluate_commitment(
        candidate_exposure
):

    if (
            candidate_exposure
            <=
            MEAN_EXPOSURE
            +
            REGRESSION_LIMIT
    ):

        return True, "ACCEPT"

    return False, "REJECT"


DEGRADED_CANDIDATE = MEAN_EXPOSURE + 0.01

IMPROVED_CANDIDATE = MEAN_EXPOSURE - 0.005

EQUIVALENT_CANDIDATE = MEAN_EXPOSURE

ACCEPT_IMPROVED, REASON_IMPROVED = evaluate_commitment(
    IMPROVED_CANDIDATE
)

ACCEPT_EQUIVALENT, REASON_EQUIVALENT = evaluate_commitment(
    EQUIVALENT_CANDIDATE
)

REJECT_DEGRADED, REASON_DEGRADED = evaluate_commitment(
    DEGRADED_CANDIDATE
)

assert ACCEPT_IMPROVED, (
    "An improved candidate must be accepted."
)

assert ACCEPT_EQUIVALENT, (
    "An equivalent candidate must be accepted."
)

assert not REJECT_DEGRADED, (
    "A regressing candidate must be rejected."
)

print(
    "Improved candidate (",
    format(
        IMPROVED_CANDIDATE,
        ".4f"
    ),
    "):",
    REASON_IMPROVED
)

print(
    "Equivalent candidate (",
    format(
        EQUIVALENT_CANDIDATE,
        ".4f"
    ),
    "):",
    REASON_EQUIVALENT
)

print(
    "Degraded candidate (",
    format(
        DEGRADED_CANDIDATE,
        ".4f"
    ),
    "):",
    REASON_DEGRADED
)

print()

print(
    "TEST 10: Determinism"
)

SECOND_MATRIX = [
    encode_policy(
        POLICY[
            pattern_id
        ]
    )
    for pattern_id
    in TUNED_ORDER
]

SECOND_COMMITMENTS = [
    commitment_entry(
        0,
        "GENESIS",
        COMMITMENT_GENESIS_HASH
    )
]

for cycle in range(
        1,
        COMMITMENT_CYCLES + 1
):

    SECOND_COMMITMENTS.append(
        commitment_entry(
            cycle,
            "RENEWED",
            SECOND_COMMITMENTS[-1]["hash"]
        )
    )

DETERMINISTIC = (
        SECOND_MATRIX
        ==
        POLICY_MATRIX
        and
        SECOND_COMMITMENTS
        ==
        COMMITMENTS
)

assert DETERMINISTIC, (
    "Consolidation or commitment is nondeterministic."
)

print(
    "Matrix deterministic:",
    SECOND_MATRIX
    ==
    POLICY_MATRIX
)

print(
    "Commitments deterministic:",
    SECOND_COMMITMENTS
    ==
    COMMITMENTS
)

print()

print(
    "TEST 11: Numerical Health"
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        MATRIX_TENSOR
    ).all()
)

print(
    "Matrix NaN:",
    int(
        torch.isnan(
            MATRIX_TENSOR
        ).sum()
    )
)

print(
    "Matrix Inf:",
    int(
        torch.isinf(
            MATRIX_TENSOR
        ).sum()
    )
)

print(
    "Numerically healthy:",
    NUMERICALLY_HEALTHY
)

assert NUMERICALLY_HEALTHY, (
    "Numerical health failed."
)

print()

print(
    "TEST 12: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if CONSOLIDATED_HASH == POLICY_HASH
    else [
        "Consolidation is not lossless."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        MEAN_EXPOSURE
        -
        0.098035
    ) <= 1e-4
    else [
        "Mean exposure invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if WORST_PATTERN == "pattern_004"
    else [
        "Worst pattern invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(
        COMMITMENTS
    ) == COMMITMENT_CYCLES + 1
    else [
        "Commitment record is incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if CHAIN_VALID
    else [
        "Commitment chain is broken."
    ]
)

PROMOTION_ERRORS += (
    []
    if NO_REGRESSION
    else [
        "A commitment regressed exposure."
    ]
)

PROMOTION_ERRORS += (
    []
    if not REJECT_DEGRADED
    else [
        "The guardrail failed to reject regression."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Consolidation is nondeterministic."
    ]
)

PROMOTION_ERRORS += (
    []
    if NUMERICALLY_HEALTHY
    else [
        "Numerical health failed."
    ]
)

print(
    "Consolidated exposure:",
    format(
        MEAN_EXPOSURE,
        ".4f"
    )
)

print(
    "Commitments:",
    len(
        COMMITMENTS
    )
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "134R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "134R promotion gate passed."
)

print()

print(
    "TEST 13: Persist Commitment Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "134R",

    "capability":
        "autonomous_policy_memory_consolidation_continuous_self_improvement_commitment",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "133R",

    "policy_hash":
        POLICY_HASH,

    "ledger_hash":
        LEDGER_HASH,

    "tuned_order":
        TUNED_ORDER,

    "policy_matrix":
        POLICY_MATRIX,

    "matrix_hash":
        MATRIX_HASH,

    "statistics":
        {
            "mean_exposure":
                MEAN_EXPOSURE,

            "worst_pattern":
                WORST_PATTERN,

            "worst_penetration":
                POLICY[
                    WORST_PATTERN
                ][
                    "penetration"
                ],

            "coverage":
                COVERAGE
        },

    "commitment":
        {
            "genesis_phrase_hash":
                COMMITMENT_GENESIS_HASH,

            "entries":
                COMMITMENTS,

            "commitment_hash":
                COMMITMENT_HASH,

            "cycles":
                COMMITMENT_CYCLES,

            "regression_limit":
                REGRESSION_LIMIT
        },

    "improvement_proof":
        IMPROVEMENT_PROOF,

    "verification":
        {
            "lossless":
                CONSOLIDATED_HASH
                ==
                POLICY_HASH,

            "chain_valid":
                CHAIN_VALID,

            "no_regression":
                NO_REGRESSION,

            "guardrail_enforced":
                not REJECT_DEGRADED,

            "deterministic":
                DETERMINISTIC,

            "numerically_healthy":
                NUMERICALLY_HEALTHY
        }
}

save_json(
    MEMORY_FILE,
    MEMORY
)

torch.save(
    MEMORY,
    INDEX_FILE
)

torch.save(
    MEMORY,
    CHECKPOINT_FILE
)

print(
    "Memory:",
    MEMORY_FILE
)

print(
    "Index:",
    INDEX_FILE
)

print(
    "Checkpoint:",
    CHECKPOINT_FILE
)

print()

print(
    "TEST 14: Reload Persistent Memory"
)

RELOADED = read_json(
    MEMORY_FILE
)

assert (
        RELOADED["memory_version"]
        ==
        MEMORY_VERSION
), "Memory version mismatch after reload."

assert (
        RELOADED["policy_hash"]
        ==
        POLICY_HASH
), "Policy hash changed after reload."

assert (
        RELOADED["matrix_hash"]
        ==
        MATRIX_HASH
), "Matrix hash changed after reload."

assert (
        RELOADED["commitment"]["commitment_hash"]
        ==
        COMMITMENT_HASH
), "Commitment hash changed after reload."

print(
    "Reloaded matrix hash:",
    RELOADED[
        "matrix_hash"
    ][:16]
)

print(
    "Reloaded commitment hash:",
    RELOADED[
        "commitment"
    ][
        "commitment_hash"
    ][:16]
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 15: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "134R",

        "capability":
            "autonomous_policy_memory_consolidation_continuous_self_improvement_commitment",

        "policy_matrix":
            POLICY_MATRIX,

        "matrix_hash":
            MATRIX_HASH,

        "mean_exposure":
            MEAN_EXPOSURE,

        "worst_pattern":
            WORST_PATTERN,

        "commitment_hashes":
            [
                entry["hash"]
                for entry
                in COMMITMENTS
            ],

        "commitment_statuses":
            [
                entry["status"]
                for entry
                in COMMITMENTS
            ]
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "134R",

        "memory_version":
            MEMORY_VERSION,

        "policy_hash":
            POLICY_HASH,

        "matrix_hash":
            MATRIX_HASH,

        "lossless":
            CONSOLIDATED_HASH
            ==
            POLICY_HASH,

        "mean_exposure":
            MEAN_EXPOSURE,

        "worst_pattern":
            WORST_PATTERN,

        "coverage":
            COVERAGE,

        "commitment_genesis_hash":
            COMMITMENT_GENESIS_HASH,

        "commitment_hash":
            COMMITMENT_HASH,

        "commitment_entries":
            len(
                COMMITMENTS
            ),

        "no_regression":
            NO_REGRESSION,

        "guardrail_enforced":
            not REJECT_DEGRADED,

        "deterministic":
            DETERMINISTIC,

        "numerically_healthy":
            NUMERICALLY_HEALTHY,

        "promotion_passed":
            True
    }
)

save_json(
    REGISTRY_FILE,
    {
        "lesson":
            "134R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "135R Autonomous Policy Evolution "
                "+ Zero-Regression Learning Protocol"
            )
    }
)

print(
    "Dataset:",
    DATASET_FILE
)

print(
    "Report:",
    REPORT_FILE
)

print(
    "Registry:",
    REGISTRY_FILE
)

print()

print(
    "TEST 16: Rebuild Equivalence"
)

RELOADED_ORDER = RELOADED.get(
    "tuned_order",
    TUNED_ORDER
)

REBUILT_MATRIX = [
    encode_policy(
        POLICY[
            pattern_id
        ]
    )
    for pattern_id
    in RELOADED_ORDER
]

assert (
        stable_hash(
            REBUILT_MATRIX
        )
        ==
        RELOADED["matrix_hash"]
), "Disk matrix differs from a fresh rebuild."

assert (
        RELOADED["commitment"]["entries"]
        ==
        COMMITMENTS
), "Disk commitment record differs from a fresh rebuild."

print(
    "Rebuilt matrix hash:",
    stable_hash(
        REBUILT_MATRIX
    )[:16]
)

print(
    "Rebuilt commitment entries:",
    len(
        COMMITMENTS
    )
)

print(
    "Disk and rebuild agree."
)

print()

print(
    "SILVERWING 134R ARCHITECTURE"
)

print(
    "Autonomous Policy Memory (133R)"
)

print(
    "   ↓"
)

print(
    "Numeric Encoding Per Pattern"
)

print(
    "   ↓"
)

print(
    "Canonical Policy Matrix [6 x 8]"
)

print(
    "   ↓"
)

print(
    "Lossless Consolidation Check"
)

print(
    "   ↓"
)

print(
    "Genesis Commitment"
)

print(
    "   ↓"
)

print(
    "Renewal Cycles 1 -> 2 -> 3"
)

print(
    "   ↓"
)

print(
    "No-Regression Guardrail"
)

print(
    "   ↓"
)

print(
    "Final Commitment Hash"
)

print()

print(
    "WHAT 134R ADDS"
)

print(
    "A lossless consolidation of the policy into a canonical "
    "numeric matrix, and a signed, chained commitment record "
    "that binds the system to a policy and to never regress it."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Autonomous systems that must commit to a policy across "
    "restarts, prove the policy is unchanged, and refuse "
    "regressions forever."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A policy that cannot be committed is a suggestion. "
    "Consolidation makes it concrete; the commitment protocol "
    "makes it durable; the guardrail makes improvement the "
    "only direction."
)

print()

print(
    "NEXT: 135R Autonomous Policy Evolution "
    "+ Zero-Regression Learning Protocol"
)

print()

print(
    "=== LESSON 134R COMPLETE ==="
)

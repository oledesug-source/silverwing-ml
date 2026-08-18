# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 135R
# Autonomous Policy Evolution
# + Zero-Regression Learning Protocol
# ============================================================
#
# 132R  -> Autonomous Defense Self-Tuning
#         + Online Control Refinement
# 133R  -> End-to-End Self-Improving Control Ledger
#         + Autonomous Policy Memory
# 134R  -> Autonomous Policy Memory Consolidation
#         + Continuous Self-Improvement Commitment
# 135R  -> Autonomous Policy Evolution
#         + Zero-Regression Learning Protocol
#
# ============================================================
# PURPOSE
# ============================================================
#
# 134R committed to a policy and promised never to regress it.
# 135R asks the question that commitment was designed for:
# can the policy IMPROVE, generation by generation, without
# ever taking a step backward?
#
# Policy evolution: the system tightens its own target. Each
# generation lowers the penetration tolerance, re-tunes the
# patterns that now violate it (and only those), and produces
# a candidate policy. The zero-regression learning protocol
# then decides: a candidate is adopted only if its exposure is
# no worse than the committed baseline. Every generation here
# reduces exposure, so every generation is adopted and chained
# into an evolution lineage.
#
# Policy evolution:
#
#     committed policy (134R)
#               ↓
#     generation target tightened
#               ↓
#     violating patterns re-tuned
#               ↓
#     candidate exposure computed
#               ↓
#     zero-regression protocol
#               ↓
#     adopted generation -> lineage
#
# Zero-regression learning protocol:
#
#     candidate exposure <= committed + limit ?
#           ├─ yes -> ACCEPT
#           └─ no  -> REJECT
#
# The lineage is chained to the 134R commitment, so the
# evolution is provably continuous with the committed policy.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 134R memory is the source of truth.
# 2. Evolution operates on the consolidated policy matrix.
# 3. Each generation tightens the penetration target.
# 4. Only patterns above the target are re-tuned.
# 5. Re-tuning uses the log-domain fixed point of 132R.
# 6. A candidate is adopted only if it does not regress.
# 7. Every generation must reduce exposure.
# 8. The lineage is chained to the 134R commitment.
# 9. The protocol must reject a regressing candidate.
# 10. Determinism must be checked.
# 11. Numerical health must be checked.
# 12. Persistence and reload must be checked.
# 13. Promotion requires all validation gates to pass.
# 14. External LLM: NONE.
#
# ============================================================

import hashlib
import json
import math
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
MEMORY_VERSION = "135R.1"
EXPECTED_PATTERNS = 6
REGRESSION_LIMIT = 1e-4
GENERATION_TARGETS = [
    0.115,
    0.110,
    0.105
]

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_134R = PHASE5_DIR / "lesson134R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_134R
        / "silverwing_policy_commitment_memory.json"
)

SOURCE_INDEX = (
        LESSON_134R
        / "silverwing_policy_commitment_index.pt"
)

SOURCE_DATASET = (
        LESSON_134R
        / "silverwing_policy_commitment_dataset.json"
)

SOURCE_REPORT = (
        LESSON_134R
        / "silverwing_policy_commitment_report.json"
)

SOURCE_REGISTRY = (
        LESSON_134R
        / "silverwing_policy_commitment_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_134R
        / "checkpoints"
        / "silverwing_policy_commitment_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_policy_evolution_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_policy_evolution_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_policy_evolution_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_policy_evolution_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_policy_evolution_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_policy_evolution_best.pt"
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
    "PHASE 5 - LESSON 135R"
)

print(
    "Autonomous Policy Evolution"
)

print(
    "+ Zero-Regression Learning Protocol"
)

print()

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

print(
    "135R -> Autonomous Policy Evolution"
)

print(
    "        + Zero-Regression Learning Protocol"
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
    "TEST 1: Verify 134R Inputs"
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
    "TEST 2: Load 134R Commitment Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "134R commitment memory is invalid."

POLICY_MATRIX = SOURCE.get(
    "policy_matrix",
    []
)

TUNED_ORDER = SOURCE.get(
    "tuned_order",
    []
)

STATISTICS = SOURCE.get(
    "statistics",
    {}
)

COMMITMENT = SOURCE.get(
    "commitment",
    {}
)

assert len(
    POLICY_MATRIX
) == EXPECTED_PATTERNS, (
    "134R matrix must cover six patterns."
)

assert len(
    TUNED_ORDER
) == EXPECTED_PATTERNS, (
    "134R order must cover six patterns."
)

COMMITTED_EXPOSURE = STATISTICS[
    "mean_exposure"
]

COMMITMENT_HASH = COMMITMENT[
    "commitment_hash"
]

assert (
    COMMITMENT_HASH[:16]
    ==
    "7aedadeb67450dec"
), "134R commitment hash mismatch."

print(
    "Committed exposure:",
    format(
        COMMITTED_EXPOSURE,
        ".4f"
    )
)

print(
    "Commitment hash:",
    COMMITMENT_HASH[:16]
)

print()

print(
    "TEST 3: Recover Policy State From Matrix"
)

BOOST = {}

PENETRATION = {}

for index, pattern_id in enumerate(
        TUNED_ORDER
):

    BOOST[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        5
    ]

    PENETRATION[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        6
    ]

assert abs(
    BOOST["pattern_004"]
    -
    1.480908
) <= 1e-4, (
    "pattern_004 committed boost mismatch."
)

assert abs(
    PENETRATION["pattern_004"]
    -
    0.120005
) <= 1e-4, (
    "pattern_004 committed penetration mismatch."
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "| boost=",
        format(
            BOOST[
                pattern_id
            ],
            ".4f"
        ),
        "| penetration=",
        format(
            PENETRATION[
                pattern_id
            ],
            ".4f"
        )
    )

print()

print(
    "TEST 4: Define Evolution + Zero-Regression Protocol"
)


def re_tune(
        pattern_id,
        target
):

    return (
            BOOST[pattern_id]
            *
            math.log(
                target
            )
            /
            math.log(
                PENETRATION[pattern_id]
            )
    )


def evaluate_candidate(
        candidate_exposure,
        baseline_exposure
):

    if (
            candidate_exposure
            <=
            baseline_exposure
            +
            REGRESSION_LIMIT
    ):

        return True, "ACCEPT"

    return False, "REJECT"


print(
    "Re-tuning: log-domain fixed point of 132R"
)

print(
    "Protocol: accept only non-regressing candidates"
)

print(
    "Regression limit:",
    REGRESSION_LIMIT
)

print()

print(
    "TEST 5: Run Policy Evolution"
)

GENERATIONS = []

for generation, target in enumerate(
        GENERATION_TARGETS,
        start=1
):

    changed = []

    for pattern_id in TUNED_ORDER:

        if (
                PENETRATION[pattern_id]
                >
                target
        ):

            BOOST[pattern_id] = re_tune(
                pattern_id,
                target
            )

            PENETRATION[pattern_id] = target

            changed.append(
                pattern_id
            )

    exposure = float(
        torch.tensor(
            list(
                PENETRATION.values()
            ),
            dtype=torch.float32
        ).mean()
    )

    adopted, verdict = evaluate_candidate(
        exposure,
        COMMITTED_EXPOSURE
    )

    assert adopted, (
        "Evolution must never regress exposure."
    )

    GENERATIONS.append(
        {
            "generation":
                generation,

            "target":
                target,

            "changed":
                changed,

            "exposure":
                exposure,

            "adopted":
                adopted,

            "verdict":
                verdict
        }
    )

    print(
        "Generation",
        generation,
        "| target=",
        target,
        "| changed=",
        changed,
        "| exposure=",
        format(
            exposure,
            ".4f"
        )
    )

print()

print(
    "TEST 6: Verify Evolved Exposure Trajectory"
)

EXPOSURE_TRAJECTORY = [
    COMMITTED_EXPOSURE
]

for record in GENERATIONS:

    EXPOSURE_TRAJECTORY.append(
        record["exposure"]
    )

assert abs(
    EXPOSURE_TRAJECTORY[1]
    -
    0.097200
) <= 1e-4, (
    "Generation 1 exposure mismatch."
)

assert abs(
    EXPOSURE_TRAJECTORY[2]
    -
    0.095028
) <= 1e-4, (
    "Generation 2 exposure mismatch."
)

assert abs(
    EXPOSURE_TRAJECTORY[3]
    -
    0.092178
) <= 1e-4, (
    "Generation 3 exposure mismatch."
)

assert EXPOSURE_TRAJECTORY == sorted(
    EXPOSURE_TRAJECTORY,
    reverse=True
), "Exposure must fall every generation."

print(
    "Trajectory:",
    [
        format(
            value,
            ".4f"
        )
        for value
        in EXPOSURE_TRAJECTORY
    ]
)

print(
    "Total reduction:",
    format(
        EXPOSURE_TRAJECTORY[0]
        -
        EXPOSURE_TRAJECTORY[-1],
        ".4f"
    )
)

print()

print(
    "TEST 7: Verify Evolution Scope"
)

assert GENERATIONS[0]["changed"] == [
    "pattern_004"
], "Generation 1 scope mismatch."

assert GENERATIONS[1]["changed"] == [
    "pattern_004",
    "pattern_005",
    "pattern_002"
], "Generation 2 scope mismatch."

assert GENERATIONS[2]["changed"] == [
    "pattern_001",
    "pattern_004",
    "pattern_005",
    "pattern_002"
], "Generation 3 scope mismatch."

for record in GENERATIONS:

    print(
        "Generation",
        record["generation"],
        "changed only:",
        record["changed"]
    )

print()

print(
    "TEST 8: Verify Evolution Lineage"
)

EVOLUTION_LINEAGE = []

previous_hash = COMMITMENT_HASH

for record in GENERATIONS:

    entry = {
        "generation":
            record["generation"],

        "target":
            record["target"],

        "adopted":
            record["adopted"],

        "exposure":
            record["exposure"],

        "prev_hash":
            previous_hash
    }

    entry[
        "hash"
    ] = stable_hash(
        entry
    )

    EVOLUTION_LINEAGE.append(
        entry
    )

    previous_hash = entry[
        "hash"
    ]

EVOLUTION_HASH = previous_hash

assert (
    EVOLUTION_HASH
    ==
    "46e8964bf2f0f5eff44ff94ace7d284c549eda707a67a32ea89a66581b24b565"
), "Evolution lineage hash mismatch."

CHAIN_VALID = True

previous_hash = COMMITMENT_HASH

for entry in EVOLUTION_LINEAGE:

    recomputed = stable_hash(
        {
            "generation":
                entry["generation"],

            "target":
                entry["target"],

            "adopted":
                entry["adopted"],

            "exposure":
                entry["exposure"],

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

assert CHAIN_VALID, (
    "Evolution lineage is broken."
)

assert EVOLUTION_HASH == previous_hash, (
    "Evolution hash does not match the final generation."
)

print(
    "Lineage chained to 134R commitment:",
    EVOLUTION_LINEAGE[0][
        "prev_hash"
    ][:16]
)

print(
    "Chain valid:",
    CHAIN_VALID
)

print(
    "Evolution hash:",
    EVOLUTION_HASH[:16]
)

print()

print(
    "TEST 9: Zero-Regression Protocol Tests"
)

ACCEPTED_IMPROVED, VERDICT_IMPROVED = evaluate_candidate(
    COMMITTED_EXPOSURE - 0.005,
    COMMITTED_EXPOSURE
)

ACCEPTED_EQUAL, VERDICT_EQUAL = evaluate_candidate(
    COMMITTED_EXPOSURE,
    COMMITTED_EXPOSURE
)

DEGRADED_PENETRATION = dict(
    PENETRATION
)

DEGRADED_PENETRATION[
    "pattern_004"
] = 0.238902

DEGRADED_EXPOSURE = float(
    torch.tensor(
        list(
            DEGRADED_PENETRATION.values()
        ),
        dtype=torch.float32
    ).mean()
)

REJECTED_DEGRADED, VERDICT_DEGRADED = evaluate_candidate(
    DEGRADED_EXPOSURE,
    COMMITTED_EXPOSURE
)

assert ACCEPTED_IMPROVED, (
    "An improved candidate must be accepted."
)

assert ACCEPTED_EQUAL, (
    "An equal candidate must be accepted."
)

assert not REJECTED_DEGRADED, (
    "A regressing candidate must be rejected."
)

print(
    "Improved candidate (",
    format(
        COMMITTED_EXPOSURE - 0.005,
        ".4f"
    ),
    "):",
    VERDICT_IMPROVED
)

print(
    "Equal candidate (",
    format(
        COMMITTED_EXPOSURE,
        ".4f"
    ),
    "):",
    VERDICT_EQUAL
)

print(
    "Degraded candidate (",
    format(
        DEGRADED_EXPOSURE,
        ".4f"
    ),
    "):",
    VERDICT_DEGRADED
)

print()

print(
    "TEST 10: Verify Final Evolved Policy"
)

assert abs(
    BOOST["pattern_004"]
    -
    1.574205
) <= 1e-4, (
    "pattern_004 evolved boost mismatch."
)

assert abs(
    BOOST["pattern_001"]
    -
    1.008873
) <= 1e-4, (
    "pattern_001 evolved boost mismatch."
)

assert abs(
    BOOST["pattern_005"]
    -
    1.041648
) <= 1e-4, (
    "pattern_005 evolved boost mismatch."
)

assert abs(
    BOOST["pattern_002"]
    -
    1.034237
) <= 1e-4, (
    "pattern_002 evolved boost mismatch."
)

assert BOOST["pattern_003"] == 1.0, (
    "pattern_003 must not be re-tuned."
)

assert BOOST["pattern_006"] == 1.0, (
    "pattern_006 must not be re-tuned."
)

assert all(
    abs(
        PENETRATION[pattern_id]
        -
        0.105
    ) <= 1e-4
    for pattern_id
    in [
        "pattern_001",
        "pattern_004",
        "pattern_005",
        "pattern_002"
    ]
), "Re-tuned patterns must sit at the final target."

print(
    "Final boosts:",
    {
        pattern_id: format(
            BOOST[pattern_id],
            ".4f"
        )
        for pattern_id
        in TUNED_ORDER
    }
)

print(
    "Final penetrations:",
    {
        pattern_id: format(
            PENETRATION[pattern_id],
            ".4f"
        )
        for pattern_id
        in TUNED_ORDER
    }
)

print()

print(
    "TEST 11: Determinism"
)

SECOND_BOOST = {}

SECOND_PENETRATION = {}

for index, pattern_id in enumerate(
        TUNED_ORDER
):

    SECOND_BOOST[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        5
    ]

    SECOND_PENETRATION[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        6
    ]

for generation, target in enumerate(
        GENERATION_TARGETS,
        start=1
):

    for pattern_id in TUNED_ORDER:

        if (
                SECOND_PENETRATION[pattern_id]
                >
                target
        ):

            SECOND_BOOST[pattern_id] = (
                    SECOND_BOOST[pattern_id]
                    *
                    math.log(
                        target
                    )
                    /
                    math.log(
                        SECOND_PENETRATION[pattern_id]
                    )
            )

            SECOND_PENETRATION[pattern_id] = target

DETERMINISTIC = (
        SECOND_BOOST
        ==
        BOOST
        and
        SECOND_PENETRATION
        ==
        PENETRATION
)

assert DETERMINISTIC, (
    "Policy evolution is nondeterministic."
)

print(
    "Boosts deterministic:",
    SECOND_BOOST
    ==
    BOOST
)

print(
    "Penetrations deterministic:",
    SECOND_PENETRATION
    ==
    PENETRATION
)

print()

print(
    "TEST 12: Numerical Health"
)

BOOST_TENSOR = torch.tensor(
    list(
        BOOST.values()
    ),
    dtype=torch.float32
)

PENETRATION_TENSOR = torch.tensor(
    list(
        PENETRATION.values()
    ),
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        BOOST_TENSOR
    ).all()
    and
    torch.isfinite(
        PENETRATION_TENSOR
    ).all()
)

print(
    "Boost NaN:",
    int(
        torch.isnan(
            BOOST_TENSOR
        ).sum()
    )
)

print(
    "Penetration Inf:",
    int(
        torch.isinf(
            PENETRATION_TENSOR
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
    "TEST 13: Final Promotion Gate"
)

PROMOTION_ERRORS = []

EXPOSURE_REDUCTION = (
        EXPOSURE_TRAJECTORY[0]
        -
        EXPOSURE_TRAJECTORY[-1]
)

PROMOTION_ERRORS += (
    []
    if EXPOSURE_TRAJECTORY == sorted(
        EXPOSURE_TRAJECTORY,
        reverse=True
    )
    else [
        "Exposure must fall every generation."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        EXPOSURE_REDUCTION
        -
        0.005857
    ) <= 1e-4
    else [
        "Evolution reduction invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if GENERATIONS[0]["changed"] == [
        "pattern_004"
    ]
    and GENERATIONS[1]["changed"] == [
        "pattern_004",
        "pattern_005",
        "pattern_002"
    ]
    and GENERATIONS[2]["changed"] == [
        "pattern_001",
        "pattern_004",
        "pattern_005",
        "pattern_002"
    ]
    else [
        "Evolution touched an already-compliant pattern."
    ]
)

PROMOTION_ERRORS += (
    []
    if CHAIN_VALID
    else [
        "Evolution lineage is broken."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        record["adopted"]
        for record
        in GENERATIONS
    )
    else [
        "A generation regressed exposure."
    ]
)

PROMOTION_ERRORS += (
    []
    if not REJECTED_DEGRADED
    else [
        "The protocol failed to reject regression."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Policy evolution is nondeterministic."
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
    "Committed exposure:",
    format(
        COMMITTED_EXPOSURE,
        ".4f"
    )
)

print(
    "Evolved exposure:",
    format(
        EXPOSURE_TRAJECTORY[-1],
        ".4f"
    )
)

print(
    "Reduction:",
    format(
        EXPOSURE_REDUCTION,
        ".4f"
    )
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "135R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "135R promotion gate passed."
)

print()

print(
    "TEST 14: Persist Evolution Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "135R",

    "capability":
        "autonomous_policy_evolution_zero_regression_learning_protocol",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "134R",

    "committed_exposure":
        COMMITTED_EXPOSURE,

    "commitment_hash":
        COMMITMENT_HASH,

    "tuned_order":
        TUNED_ORDER,

    "generation_targets":
        GENERATION_TARGETS,

    "generations":
        GENERATIONS,

    "exposure_trajectory":
        EXPOSURE_TRAJECTORY,

    "exposure_reduction":
        EXPOSURE_REDUCTION,

    "evolution_lineage":
        EVOLUTION_LINEAGE,

    "evolution_hash":
        EVOLUTION_HASH,

    "evolved_boosts":
        BOOST,

    "evolved_penetrations":
        PENETRATION,

    "regression_limit":
        REGRESSION_LIMIT,

    "verification":
        {
            "monotone_improvement":
                EXPOSURE_TRAJECTORY
                ==
                sorted(
                    EXPOSURE_TRAJECTORY,
                    reverse=True
                ),

            "chain_valid":
                CHAIN_VALID,

            "all_adopted":
                all(
                    record["adopted"]
                    for record
                    in GENERATIONS
                ),

            "guardrail_enforced":
                not REJECTED_DEGRADED,

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
    "TEST 15: Reload Persistent Memory"
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
        RELOADED["evolution_hash"]
        ==
        EVOLUTION_HASH
), "Evolution hash changed after reload."

assert (
        RELOADED["evolved_boosts"]
        ==
        BOOST
), "Evolved boosts changed after reload."

assert (
        RELOADED["exposure_trajectory"]
        ==
        EXPOSURE_TRAJECTORY
), "Exposure trajectory changed after reload."

print(
    "Reloaded evolution hash:",
    RELOADED[
        "evolution_hash"
    ][:16]
)

print(
    "Reloaded final exposure:",
    format(
        RELOADED[
            "exposure_trajectory"
        ][
            -1
        ],
        ".4f"
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 16: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "135R",

        "capability":
            "autonomous_policy_evolution_zero_regression_learning_protocol",

        "generation_targets":
            GENERATION_TARGETS,

        "exposure_trajectory":
            EXPOSURE_TRAJECTORY,

        "changed_per_generation":
            {
                record["generation"]: record["changed"]
                for record
                in GENERATIONS
            },

        "evolved_boosts":
            BOOST,

        "evolved_penetrations":
            PENETRATION,

        "evolution_hash":
            EVOLUTION_HASH
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "135R",

        "memory_version":
            MEMORY_VERSION,

        "committed_exposure":
            COMMITTED_EXPOSURE,

        "evolved_exposure":
            EXPOSURE_TRAJECTORY[-1],

        "exposure_reduction":
            EXPOSURE_REDUCTION,

        "generations":
            len(
                GENERATIONS
            ),

        "monotone_improvement":
            EXPOSURE_TRAJECTORY
            ==
            sorted(
                EXPOSURE_TRAJECTORY,
                reverse=True
            ),

        "chain_valid":
            CHAIN_VALID,

        "evolution_hash":
            EVOLUTION_HASH,

        "guardrail_enforced":
            not REJECTED_DEGRADED,

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
            "135R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "136R Autonomous Cross-Cycle Evolution Governance "
                "+ Bounded Innovation Protocol"
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
    "SILVERWING 135R ARCHITECTURE"
)

print(
    "Committed Policy (134R)"
)

print(
    "   ↓"
)

print(
    "Generation Target Tightened"
)

print(
    "   ↓"
)

print(
    "Violating Patterns Re-tuned"
)

print(
    "   ↓"
)

print(
    "Candidate Exposure Computed"
)

print(
    "   ↓"
)

print(
    "Zero-Regression Protocol"
)

print(
    "   ↓"
)

print(
    "Adopted Generation -> Lineage"
)

print(
    "   ↓"
)

print(
    "Exposure 0.0980 -> 0.0922"
)

print()

print(
    "WHAT 135R ADDS"
)

print(
    "A learning protocol under which the policy improves "
    "generation by generation -- tightening its own target, "
    "touching only what it must, and refusing every regressing "
    "candidate -- all chained to the 134R commitment."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Live systems that must keep improving while proving they "
    "never get worse, generation after generation."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Improvement without a guardrail is gambling. The "
    "zero-regression protocol makes evolution a controlled, "
    "provable, one-way ratchet."
)

print()

print(
    "NEXT: 136R Autonomous Cross-Cycle Evolution Governance "
    "+ Bounded Innovation Protocol"
)

print()

print(
    "=== LESSON 135R COMPLETE ==="
)

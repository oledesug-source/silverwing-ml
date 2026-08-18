# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 136R
# Autonomous Cross-Cycle Evolution Governance
# + Bounded Innovation Protocol
# ============================================================
#
# 133R  -> End-to-End Self-Improving Control Ledger
#         + Autonomous Policy Memory
# 134R  -> Autonomous Policy Memory Consolidation
#         + Continuous Self-Improvement Commitment
# 135R  -> Autonomous Policy Evolution
#         + Zero-Regression Learning Protocol
# 136R  -> Autonomous Cross-Cycle Evolution Governance
#         + Bounded Innovation Protocol
#
# ============================================================
# PURPOSE
# ============================================================
#
# 135R let the policy evolve under a zero-regression protocol.
# Evolution that only refuses to get worse is not yet safe:
# a single generation might leap too far, over-reach its
# scope, or break the chain to the committed past. 136R
# governs the evolution itself.
#
# Cross-cycle governance audits the entire lineage -- from the
# 134R commitment through every 135R generation -- against
# three rules. Innovation must be BOUNDED (no generation may
# move any boost further than a fixed limit, or shift exposure
# more than a fixed step). Innovation must be SCOPED (only
# patterns that violate the new target may change). And the
# lineage must be CONTINUOUS (every generation chained to the
# commitment that produced it).
#
# The bounded innovation protocol is the decision rule that
# enforces all of this: a candidate update is accepted only if
# every per-pattern delta is within bound, the exposure step is
# within bound, and the update does not regress.
#
# Cross-cycle evolution governance:
#
#     134R commitment -> 135R generations
#               ↓
#     per-generation snapshots
#               ↓
#     boost deltas + exposure deltas
#               ↓
#     bounded innovation protocol
#               ↓
#     governance verdict per generation
#
# Bounded innovation protocol:
#
#     deltas within bounds and no regression ?
#           ├─ yes -> ACCEPT
#           └─ no  -> REJECT
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 134R and 135R memories are the sources of truth.
# 2. Governance audits every generation of the lineage.
# 3. Innovation must be bounded: boost delta within limit.
# 4. Innovation must be bounded: exposure step within limit.
# 5. Innovation must be scoped: only violators change.
# 6. The lineage must chain to the 134R commitment.
# 7. The protocol must reject every out-of-bound update.
# 8. The final policy must remain compliant.
# 9. Determinism must be checked.
# 10. Numerical health must be checked.
# 11. Persistence and reload must be checked.
# 12. Promotion requires all validation gates to pass.
# 13. External LLM: NONE.
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
MEMORY_VERSION = "136R.1"
EXPECTED_PATTERNS = 6
BOOST_BOUND = 0.05
EXPOSURE_BOUND = 0.005
REGRESSION_LIMIT = 1e-4
FINAL_TARGET = 0.105

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_134R = PHASE5_DIR / "lesson134R"
LESSON_135R = PHASE5_DIR / "lesson135R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_COMMITMENT_MEMORY = (
        LESSON_134R
        / "silverwing_policy_commitment_memory.json"
)

SOURCE_EVOLUTION_MEMORY = (
        LESSON_135R
        / "silverwing_policy_evolution_memory.json"
)

SOURCE_EVOLUTION_INDEX = (
        LESSON_135R
        / "silverwing_policy_evolution_index.pt"
)

SOURCE_EVOLUTION_DATASET = (
        LESSON_135R
        / "silverwing_policy_evolution_dataset.json"
)

SOURCE_EVOLUTION_REPORT = (
        LESSON_135R
        / "silverwing_policy_evolution_report.json"
)

SOURCE_EVOLUTION_REGISTRY = (
        LESSON_135R
        / "silverwing_policy_evolution_registry.json"
)

SOURCE_EVOLUTION_CHECKPOINT = (
        LESSON_135R
        / "checkpoints"
        / "silverwing_policy_evolution_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_evolution_governance_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_evolution_governance_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_evolution_governance_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_evolution_governance_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_evolution_governance_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_evolution_governance_best.pt"
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
    "PHASE 5 - LESSON 136R"
)

print(
    "Autonomous Cross-Cycle Evolution Governance"
)

print(
    "+ Bounded Innovation Protocol"
)

print()

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

print(
    "136R -> Autonomous Cross-Cycle Evolution Governance"
)

print(
    "        + Bounded Innovation Protocol"
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
    "TEST 1: Verify 136R Inputs"
)

REQUIRED_FILES = [
    SOURCE_COMMITMENT_MEMORY,
    SOURCE_EVOLUTION_MEMORY,
    SOURCE_EVOLUTION_INDEX,
    SOURCE_EVOLUTION_DATASET,
    SOURCE_EVOLUTION_REPORT,
    SOURCE_EVOLUTION_REGISTRY,
    SOURCE_EVOLUTION_CHECKPOINT
]

assert all(
    path.exists()
    for path in REQUIRED_FILES
), "One or more source inputs are missing."

for path in REQUIRED_FILES:

    print(
        "FOUND:",
        path.name
    )

print()

print(
    "TEST 2: Load Source Memories"
)

COMMITMENT_SOURCE = read_json(
    SOURCE_COMMITMENT_MEMORY
)

EVOLUTION_SOURCE = read_json(
    SOURCE_EVOLUTION_MEMORY
)

assert isinstance(
    COMMITMENT_SOURCE,
    dict
), "134R commitment memory is invalid."

assert isinstance(
    EVOLUTION_SOURCE,
    dict
), "135R evolution memory is invalid."

TUNED_ORDER = EVOLUTION_SOURCE.get(
    "tuned_order",
    []
)

assert len(
    TUNED_ORDER
) == EXPECTED_PATTERNS, (
    "Order must cover six patterns."
)

POLICY_MATRIX = COMMITMENT_SOURCE.get(
    "policy_matrix",
    []
)

GENERATIONS = EVOLUTION_SOURCE.get(
    "generations",
    []
)

TRAJECTORY = EVOLUTION_SOURCE.get(
    "exposure_trajectory",
    []
)

EVOLVED_BOOSTS = EVOLUTION_SOURCE.get(
    "evolved_boosts",
    {}
)

EVOLVED_PENETRATIONS = EVOLUTION_SOURCE.get(
    "evolved_penetrations",
    {}
)

EVOLUTION_HASH = EVOLUTION_SOURCE.get(
    "evolution_hash",
    ""
)

COMMITMENT_HASH = EVOLUTION_SOURCE.get(
    "commitment_hash",
    ""
)

LINEAGE = EVOLUTION_SOURCE.get(
    "evolution_lineage",
    []
)

assert len(
    GENERATIONS
) == 3, "135R must provide exactly three generations."

assert (
    EVOLUTION_HASH
    ==
    "46e8964bf2f0f5eff44ff94ace7d284c549eda707a67a32ea89a66581b24b565"
), "135R evolution hash mismatch."

print(
    "135R evolution hash:",
    EVOLUTION_HASH[:16]
)

print(
    "135R committed hash:",
    COMMITMENT_HASH[:16]
)

print()

print(
    "TEST 3: Reconstruct Generation Snapshots"
)


def re_tune(
        boost,
        penetration,
        target
):

    return (
            boost
            *
            math.log(
                target
            )
            /
            math.log(
                penetration
            )
    )


SNAPSHOTS = []

boost = {}

penetration = {}

for index, pattern_id in enumerate(
        TUNED_ORDER
):

    boost[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        5
    ]

    penetration[
        pattern_id
    ] = POLICY_MATRIX[
        index
    ][
        6
    ]

SNAPSHOTS.append(
    {
        "generation": 0,
        "boosts": dict(
            boost
        ),
        "penetrations": dict(
            penetration
        )
    }
)

for record in GENERATIONS:

    for pattern_id in record[
        "changed"
    ]:

        boost[
            pattern_id
        ] = re_tune(
            boost[pattern_id],
            penetration[pattern_id],
            record["target"]
        )

        penetration[
            pattern_id
        ] = record[
            "target"
        ]

    SNAPSHOTS.append(
        {
            "generation":
                record["generation"],

            "boosts":
                dict(
                    boost
                ),

            "penetrations":
                dict(
                    penetration
                )
        }
    )

assert abs(
    SNAPSHOTS[-1]["boosts"]["pattern_004"]
    -
    1.574205
) <= 1e-4, (
    "Final evolved boost mismatch."
)

assert abs(
    SNAPSHOTS[-1]["penetrations"]["pattern_004"]
    -
    0.105
) <= 1e-4, (
    "Final evolved penetration mismatch."
)

print(
    "Snapshots:",
    [
        snapshot["generation"]
        for snapshot
        in SNAPSHOTS
    ]
)

print(
    "Reconstruction matches 135R evolved policy."
)

print()

print(
    "TEST 4: Define Governance Frame"
)

GOVERNANCE_FRAME = {
    "boost_bound":
        BOOST_BOUND,

    "exposure_bound":
        EXPOSURE_BOUND,

    "regression_limit":
        REGRESSION_LIMIT,

    "final_target":
        FINAL_TARGET
}

print(
    "Boost bound:",
    BOOST_BOUND
)

print(
    "Exposure bound:",
    EXPOSURE_BOUND
)

print(
    "Final target:",
    FINAL_TARGET
)

print()

print(
    "TEST 5: Compute Innovation Deltas"
)


def bounded_innovation_verdict(
        max_boost_delta,
        exposure_delta
):

    reasons = []

    if max_boost_delta > BOOST_BOUND:

        reasons.append(
            "boost delta out of bound"
        )

    if abs(
            exposure_delta
    ) > EXPOSURE_BOUND:

        reasons.append(
            "exposure step out of bound"
        )

    if exposure_delta > REGRESSION_LIMIT:

        reasons.append(
            "regression"
        )

    if reasons:

        return False, "; ".join(
            reasons
        )

    return True, "ACCEPT"


GENERATION_DELTAS = []

previous_boosts = SNAPSHOTS[0][
    "boosts"
]

for snapshot in SNAPSHOTS[1:]:

    generation = snapshot[
        "generation"
    ]

    max_boost_delta = max(
        abs(
            snapshot["boosts"][pattern_id]
            -
            previous_boosts[pattern_id]
        )
        for pattern_id
        in TUNED_ORDER
    )

    exposure_delta = (
            TRAJECTORY[generation]
            -
            TRAJECTORY[generation - 1]
    )

    GENERATION_DELTAS.append(
        {
            "generation":
                generation,

            "max_boost_delta":
                max_boost_delta,

            "exposure_delta":
                exposure_delta
        }
    )

    previous_boosts = snapshot[
        "boosts"
    ]

for record in GENERATION_DELTAS:

    print(
        "Generation",
        record["generation"],
        "| max boost delta=",
        format(
            record["max_boost_delta"],
            ".4f"
        ),
        "| exposure delta=",
        format(
            record["exposure_delta"],
            ".4f"
        )
    )

print()

print(
    "TEST 6: Run Governance Audit"
)

AUDIT = []

previous_boosts = SNAPSHOTS[0][
    "boosts"
]

previous_penetrations = SNAPSHOTS[0][
    "penetrations"
]

for delta_record in GENERATION_DELTAS:

    generation = delta_record[
        "generation"
    ]

    generation_record = GENERATIONS[
        generation - 1
    ]

    target = generation_record[
        "target"
    ]

    changed = generation_record[
        "changed"
    ]

    expected_scope = sorted(
        pattern_id
        for pattern_id
        in TUNED_ORDER
        if previous_penetrations[
            pattern_id
        ]
        > target
    )

    scope_valid = (
            sorted(
                changed
            )
            ==
            expected_scope
    )

    governed, reason = bounded_innovation_verdict(
        delta_record["max_boost_delta"],
        delta_record["exposure_delta"]
    )

    AUDIT.append(
        {
            "generation":
                generation,

            "target":
                target,

            "changed":
                changed,

            "scope_valid":
                scope_valid,

            "max_boost_delta":
                delta_record[
                    "max_boost_delta"
                ],

            "exposure_delta":
                delta_record[
                    "exposure_delta"
                ],

            "governed":
                governed,

            "reason":
                reason
        }
    )

    previous_boosts = SNAPSHOTS[
        generation
    ][
        "boosts"
    ]

    previous_penetrations = SNAPSHOTS[
        generation
    ][
        "penetrations"
    ]

for record in AUDIT:

    print(
        "Generation",
        record["generation"],
        "| governed=",
        record["governed"],
        "| scope=",
        record["scope_valid"],
        "|",
        record["reason"]
    )

print()

print(
    "TEST 7: Verify Bounded Innovation"
)

assert all(
    record["max_boost_delta"]
    <=
    BOOST_BOUND
    +
    1e-6
    for record
    in AUDIT
), "A generation exceeded the boost bound."

assert all(
    abs(
        record["exposure_delta"]
    )
    <=
    EXPOSURE_BOUND
    +
    1e-6
    for record
    in AUDIT
), "A generation exceeded the exposure bound."

assert abs(
    AUDIT[0]["max_boost_delta"]
    -
    0.029756
) <= 1e-4, (
    "Generation 1 boost delta mismatch."
)

assert abs(
    AUDIT[1]["max_boost_delta"]
    -
    0.031048
) <= 1e-4, (
    "Generation 2 boost delta mismatch."
)

assert abs(
    AUDIT[2]["max_boost_delta"]
    -
    0.032493
) <= 1e-4, (
    "Generation 3 boost delta mismatch."
)

print(
    "All boost deltas within bound:",
    max(
        record["max_boost_delta"]
        for record
        in AUDIT
    )
    <=
    BOOST_BOUND
)

print(
    "All exposure steps within bound:",
    max(
        abs(
            record["exposure_delta"]
        )
        for record
        in AUDIT
    )
    <=
    EXPOSURE_BOUND
)

print()

print(
    "TEST 8: Verify Cross-Cycle Continuity"
)

assert (
        EVOLUTION_HASH
        ==
        EVOLUTION_SOURCE["evolution_hash"]
), "Evolution hash must match 135R."

assert (
        COMMITMENT_HASH
        ==
        COMMITMENT_SOURCE[
            "commitment"
        ][
            "commitment_hash"
        ]
), "135R must reference the 134R commitment."

LINEAGE_VALID = True

previous_hash = COMMITMENT_HASH

for entry in LINEAGE:

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

        LINEAGE_VALID = False

        break

    previous_hash = entry[
        "hash"
    ]

assert LINEAGE_VALID, (
    "135R lineage must chain back to the 134R commitment."
)

assert (
        LINEAGE[0]["prev_hash"]
        ==
        COMMITMENT_HASH
), "First generation must chain to the commitment."

print(
    "Lineage genesis:",
    LINEAGE[0]["prev_hash"][:16]
)

print(
    "Lineage valid:",
    LINEAGE_VALID
)

print()

print(
    "TEST 9: Verify Scope Governance"
)

assert all(
    record["scope_valid"]
    for record
    in AUDIT
), "A generation changed an already-compliant pattern."

for record in AUDIT:

    print(
        "Generation",
        record["generation"],
        "changed exactly:",
        record["changed"]
    )

print()

print(
    "TEST 10: Verify Final Compliance"
)

FINAL_COMPLIANT = all(
    EVOLVED_PENETRATIONS[
        pattern_id
    ]
    <=
    FINAL_TARGET
    +
    1e-4
    for pattern_id
    in TUNED_ORDER
)

assert FINAL_COMPLIANT, (
    "The evolved policy must sit under the final target."
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "| penetration=",
        format(
            EVOLVED_PENETRATIONS[
                pattern_id
            ],
            ".4f"
        ),
        "|",
        "SAFE"
        if EVOLVED_PENETRATIONS[
            pattern_id
        ]
        <= FINAL_TARGET
        + 1e-4
        else "UNSAFE"
    )

print()

print(
    "TEST 11: Bounded Innovation Protocol Tests"
)

VALID_BOOST_DELTA = 0.032

VALID_EXPOSURE_DELTA = -0.0028

OUT_OF_BOUND_BOOST = 0.5

OUT_OF_BOUND_JUMP = -0.01

REGRESSION_DELTA = 0.005

ACCEPTED, REASON_VALID = bounded_innovation_verdict(
    VALID_BOOST_DELTA,
    VALID_EXPOSURE_DELTA
)

REJECTED_BOOST, REASON_BOOST = bounded_innovation_verdict(
    OUT_OF_BOUND_BOOST,
    VALID_EXPOSURE_DELTA
)

REJECTED_JUMP, REASON_JUMP = bounded_innovation_verdict(
    VALID_BOOST_DELTA,
    OUT_OF_BOUND_JUMP
)

REJECTED_REGRESSION, REASON_REGRESSION = bounded_innovation_verdict(
    VALID_BOOST_DELTA,
    REGRESSION_DELTA
)

assert ACCEPTED, (
    "A bounded innovation must be accepted."
)

assert not REJECTED_BOOST, (
    "An out-of-bound boost must be rejected."
)

assert not REJECTED_JUMP, (
    "An out-of-bound exposure jump must be rejected."
)

assert not REJECTED_REGRESSION, (
    "A regressing step must be rejected."
)

print(
    "Bounded boost (",
    VALID_BOOST_DELTA,
    "):",
    REASON_VALID
)

print(
    "Out-of-bound boost (",
    OUT_OF_BOUND_BOOST,
    "):",
    REASON_BOOST
)

print(
    "Out-of-bound jump (",
    OUT_OF_BOUND_JUMP,
    "):",
    REASON_JUMP
)

print(
    "Regression step (",
    REGRESSION_DELTA,
    "):",
    REASON_REGRESSION
)

print()

print(
    "TEST 12: Determinism"
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

for record in GENERATIONS:

    for pattern_id in record[
        "changed"
    ]:

        SECOND_BOOST[
            pattern_id
        ] = re_tune(
            SECOND_BOOST[pattern_id],
            SECOND_PENETRATION[pattern_id],
            record["target"]
        )

        SECOND_PENETRATION[
            pattern_id
        ] = record[
            "target"
        ]

DETERMINISTIC = (
        SECOND_BOOST
        ==
        EVOLVED_BOOSTS
        and
        SECOND_PENETRATION
        ==
        EVOLVED_PENETRATIONS
)

assert DETERMINISTIC, (
    "Governance reconstruction is nondeterministic."
)

print(
    "Boosts deterministic:",
    SECOND_BOOST
    ==
    EVOLVED_BOOSTS
)

print(
    "Penetrations deterministic:",
    SECOND_PENETRATION
    ==
    EVOLVED_PENETRATIONS
)

print()

print(
    "TEST 13: Numerical Health"
)

BOOST_TENSOR = torch.tensor(
    list(
        EVOLVED_BOOSTS.values()
    ),
    dtype=torch.float32
)

PENETRATION_TENSOR = torch.tensor(
    list(
        EVOLVED_PENETRATIONS.values()
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
    "TEST 14: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if all(
        record["governed"]
        for record
        in AUDIT
    )
    else [
        "A generation failed the governance audit."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        record["scope_valid"]
        for record
        in AUDIT
    )
    else [
        "A generation violated scope governance."
    ]
)

PROMOTION_ERRORS += (
    []
    if LINEAGE_VALID
    else [
        "Lineage continuity failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if FINAL_COMPLIANT
    else [
        "Evolved policy is not compliant."
    ]
)

PROMOTION_ERRORS += (
    []
    if not REJECTED_BOOST
    and not REJECTED_JUMP
    and not REJECTED_REGRESSION
    else [
        "The bounded innovation protocol failed a rejection."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Governance reconstruction is nondeterministic."
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
    "Generations governed:",
    sum(
        int(
            record["governed"]
        )
        for record
        in AUDIT
    ),
    "/",
    len(
        AUDIT
    )
)

print(
    "Final compliance:",
    FINAL_COMPLIANT
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "136R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "136R promotion gate passed."
)

print()

print(
    "TEST 15: Persist Governance Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "136R",

    "capability":
        "autonomous_cross_cycle_evolution_governance_bounded_innovation_protocol",

    "created_at":
        datetime.now().isoformat(),

    "source_lessons":
        [
            "134R",
            "135R"
        ],

    "governance_frame":
        GOVERNANCE_FRAME,

    "evolution_hash":
        EVOLUTION_HASH,

    "commitment_hash":
        COMMITMENT_HASH,

    "tuned_order":
        TUNED_ORDER,

    "generation_deltas":
        GENERATION_DELTAS,

    "audit":
        AUDIT,

    "final_policy":
        {
            "boosts":
                EVOLVED_BOOSTS,

            "penetrations":
                EVOLVED_PENETRATIONS
        },

    "verification":
        {
            "all_governed":
                all(
                    record["governed"]
                    for record
                    in AUDIT
                ),

            "all_scoped":
                all(
                    record["scope_valid"]
                    for record
                    in AUDIT
                ),

            "lineage_continuous":
                LINEAGE_VALID,

            "final_compliant":
                FINAL_COMPLIANT,

            "protocol_enforced":
                not REJECTED_BOOST
                and not REJECTED_JUMP
                and not REJECTED_REGRESSION,

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
    "TEST 16: Reload Persistent Memory"
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
        RELOADED["audit"]
        ==
        AUDIT
), "Audit changed after reload."

assert (
        RELOADED["final_policy"]
        ==
        MEMORY["final_policy"]
), "Final policy changed after reload."

print(
    "Reloaded governed generations:",
    sum(
        int(
            record["governed"]
        )
        for record
        in RELOADED[
            "audit"
        ]
    ),
    "/",
    len(
        RELOADED[
            "audit"
        ]
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 17: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "136R",

        "capability":
            "autonomous_cross_cycle_evolution_governance_bounded_innovation_protocol",

        "generation_deltas":
            GENERATION_DELTAS,

        "audit_verdicts":
            [
                {
                    "generation":
                        record["generation"],

                    "governed":
                        record["governed"],

                    "scope_valid":
                        record["scope_valid"]
                }
                for record
                in AUDIT
            ],

        "final_boosts":
            EVOLVED_BOOSTS,

        "final_penetrations":
            EVOLVED_PENETRATIONS
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "136R",

        "memory_version":
            MEMORY_VERSION,

        "generations_audited":
            len(
                AUDIT
            ),

        "all_governed":
            all(
                record["governed"]
                for record
                in AUDIT
            ),

        "max_boost_delta":
            max(
                record["max_boost_delta"]
                for record
                in AUDIT
            ),

        "max_exposure_step":
            max(
                abs(
                    record["exposure_delta"]
                )
                for record
                in AUDIT
            ),

        "boost_bound":
            BOOST_BOUND,

        "exposure_bound":
            EXPOSURE_BOUND,

        "lineage_continuous":
            LINEAGE_VALID,

        "final_compliant":
            FINAL_COMPLIANT,

        "protocol_enforced":
            not REJECTED_BOOST
            and not REJECTED_JUMP
            and not REJECTED_REGRESSION,

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
            "136R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "137R Autonomous Governance Ledger "
                "+ Innovation Accountability Chain"
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
    "SILVERWING 136R ARCHITECTURE"
)

print(
    "134R Commitment -> 135R Generations"
)

print(
    "   ↓"
)

print(
    "Per-Generation Snapshots"
)

print(
    "   ↓"
)

print(
    "Boost Deltas + Exposure Deltas"
)

print(
    "   ↓"
)

print(
    "Bounded Innovation Protocol"
)

print(
    "   ↓"
)

print(
    "Governance Verdict Per Generation"
)

print(
    "   ↓"
)

print(
    "All Generations Governed + Compliant"
)

print()

print(
    "WHAT 136R ADDS"
)

print(
    "A governance layer over evolution: every generation is "
    "audited for bounded innovation, scoped changes and "
    "continuity with the committed past, and the bounded "
    "innovation protocol rejects any out-of-bound update."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Autonomous systems that must improve safely -- where "
    "unbounded innovation is as dangerous as regression."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Evolution without governance is drift. Bounded "
    "innovation keeps every step small, scoped and chained, "
    "so improvement is provably safe."
)

print()

print(
    "NEXT: 137R Autonomous Governance Ledger "
    "+ Innovation Accountability Chain"
)

print()

print(
    "=== LESSON 136R COMPLETE ==="
)

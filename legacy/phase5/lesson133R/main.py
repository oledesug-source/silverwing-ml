# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 133R
# End-to-End Self-Improving Control Ledger
# + Autonomous Policy Memory
# ============================================================
#
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
# 128R  -> Uncertainty-Aware Preventive Execution
#         + Probabilistic Guardrails
# 129R  -> Anomaly-First Adaptive Scheduling
#         + Critical Path Defense
# 130R  -> Self-Healing Recovery Orchestration
#         + Failure Absorption
# 131R  -> Collective Defense Consolidation
#         + System-Level Resilience Audit
# 132R  -> Autonomous Defense Self-Tuning
#         + Online Control Refinement
# 133R  -> End-to-End Self-Improving Control Ledger
#         + Autonomous Policy Memory
#
# ============================================================
# PURPOSE
# ============================================================
#
# 132R proved the defense can tune itself. 133R asks the
# closing question: how does the system REMEMBER what it
# learned? The answers, from 125R to 132R, must not evaporate
# between runs. 133R freezes the entire arc into an
# end-to-end control ledger: a hash-chained, tamper-evident
# record with one entry per lesson. And it distills the
# ledger into an autonomous policy memory: a single per-pattern
# policy that the defense operates on from now on.
#
# The control ledger:
#
#     genesis (root of trust)
#               ↓
#     entry 125R -> entry 126R -> ... -> entry 132R
#               ↓
#     each entry chains the previous hash
#               ↓
#     final ledger hash (tamper-evident)
#
# The autonomous policy memory:
#
#     trend (125R) + tier (129R) + action (127R)
#               ↓
#     schedule position (129R) + recovery (130R)
#               ↓
#     tuned boost + penetration (132R)
#               ↓
#     one autonomous policy per pattern
#
# The ledger also records the self-improvement proof: the
# closed-loop gain, the resilience grade, and the exposure
# reduction are all part of the chain, so the improvement is
# itself provable and persistent.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. The 125R..132R memories are the sources of truth.
# 2. Every lesson contributes one entry to the ledger.
# 3. Each entry is chained to the previous entry's hash.
# 4. The genesis hash is the root of trust.
# 5. The ledger must verify by recomputation.
# 6. The ledger must cover lessons 125R through 132R.
# 7. The policy memory is derived only from the chain.
# 8. Policy is one entry per pattern, six patterns total.
# 9. The ledger must record the improvement proof.
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
MEMORY_VERSION = "133R.1"
EXPECTED_PATTERNS = 6
EXPECTED_LESSONS = 8
GENESIS_PHRASE = "SILVERWING"

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_LESSONS = [
    (
        "125R",
        PHASE5_DIR
        / "lesson125R"
        / "silverwing_cross_cycle_trending_memory.json"
    ),
    (
        "126R",
        PHASE5_DIR
        / "lesson126R"
        / "silverwing_control_loop_governance_memory.json"
    ),
    (
        "127R",
        PHASE5_DIR
        / "lesson127R"
        / "silverwing_multi_layer_defense_memory.json"
    ),
    (
        "128R",
        PHASE5_DIR
        / "lesson128R"
        / "silverwing_uncertainty_guardrails_memory.json"
    ),
    (
        "129R",
        PHASE5_DIR
        / "lesson129R"
        / "silverwing_anomaly_scheduling_memory.json"
    ),
    (
        "130R",
        PHASE5_DIR
        / "lesson130R"
        / "silverwing_recovery_orchestration_memory.json"
    ),
    (
        "131R",
        PHASE5_DIR
        / "lesson131R"
        / "silverwing_resilience_audit_memory.json"
    ),
    (
        "132R",
        PHASE5_DIR
        / "lesson132R"
        / "silverwing_self_tuning_memory.json"
    )
]

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_control_ledger_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_control_ledger_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_control_ledger_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_control_ledger_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_control_ledger_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_control_ledger_best.pt"
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
    "PHASE 5 - LESSON 133R"
)

print(
    "End-to-End Self-Improving Control Ledger"
)

print(
    "+ Autonomous Policy Memory"
)

print()

print(
    "127R -> Multi-Layer Defense Simulation + Adaptive Control"
)

print(
    "128R -> Uncertainty-Aware Preventive Execution"
)

print(
    "        + Probabilistic Guardrails"
)

print(
    "129R -> Anomaly-First Adaptive Scheduling"
)

print(
    "        + Critical Path Defense"
)

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

assert len(
    SOURCE_LESSONS
) == EXPECTED_LESSONS, (
    "Ledger must cover eight source lessons."
)

assert all(
    path.exists()
    for _, path
    in SOURCE_LESSONS
), "One or more source memories are missing."

for lesson, path in SOURCE_LESSONS:

    print(
        "FOUND:",
        lesson,
        "|",
        path.name
    )

print()

print(
    "TEST 2: Load Source Memories"
)

SOURCES = {}

for lesson, path in SOURCE_LESSONS:

    memory = read_json(
        path
    )

    assert isinstance(
        memory,
        dict
    ), lesson + " memory is invalid."

    SOURCES[
        lesson
    ] = memory

    print(
        lesson,
        "->",
        memory.get(
            "capability"
        )
    )

print()

print(
    "TEST 3: Extract Lesson Contributions"
)


def extract_contribution(
        lesson,
        memory
):

    if lesson == "125R":

        return {
            "trends":
                memory["trends"],

            "tuned_order":
                memory["tuned_order"]
        }

    if lesson == "126R":

        return {
            "approved_rehearsal":
                memory[
                    "approved_rehearsal"
                ][
                    "total_residual"
                ],

            "residual_budget":
                memory[
                    "residual_budget"
                ],

            "critical_set":
                memory[
                    "critical_set"
                ]
        }

    if lesson == "127R":

        return {
            "control_actions":
                memory[
                    "control_actions"
                ],

            "baseline_mean_stop_rate":
                memory[
                    "baseline_mean_stop_rate"
                ],

            "controlled_mean_stop_rate":
                memory[
                    "controlled_mean_stop_rate"
                ],

            "penetration_threshold":
                memory[
                    "penetration_threshold"
                ]
        }

    if lesson == "128R":

        return {
            "controlled_gate":
                memory[
                    "controlled_gate"
                ][
                    "blocked"
                ],

            "guardrail_threshold":
                memory[
                    "guardrail_frame"
                ][
                    "guardrail_threshold"
                ]
        }

    if lesson == "129R":

        return {
            "anomaly_schedule":
                memory[
                    "anomaly_schedule"
                ],

            "critical_path":
                memory[
                    "critical_path"
                ],

            "defense_tiers":
                memory[
                    "defense_tiers"
                ],

            "anomaly_set":
                memory[
                    "anomaly_set"
                ]
        }

    if lesson == "130R":

        return {
            "absorption_capacity":
                memory[
                    "absorption_capacity"
                ],

            "absorbed":
                memory[
                    "breach_scenario"
                ][
                    "absorbed"
                ],

            "recovery_actions":
                {
                    pattern_id: outcome["action"]
                    for pattern_id, outcome
                    in memory[
                        "breach_scenario"
                    ][
                        "recovery"
                    ].items()
                },

            "root_recovery":
                memory[
                    "root_cascade_scenario"
                ][
                    "recovery"
                ][
                    "action"
                ]
        }

    if lesson == "131R":

        return {
            "resilience_score":
                memory[
                    "resilience_audit"
                ][
                    "score"
                ],

            "resilience_grade":
                memory[
                    "resilience_audit"
                ][
                    "grade"
                ],

            "collective_exposure":
                memory[
                    "system_metrics"
                ][
                    "collective_exposure"
                ]
        }

    if lesson == "132R":

        return {
            "converged_boost":
                memory[
                    "convergence"
                ][
                    "converged_boost"
                ],

            "static_exposure":
                memory[
                    "stress_scenario"
                ][
                    "static_exposure"
                ],

            "tuned_exposure":
                memory[
                    "convergence"
                ][
                    "tuned_exposure"
                ],

            "exposure_reduction":
                memory[
                    "convergence"
                ][
                    "exposure_reduction"
                ]
        }

    assert False, "Unknown source lesson."


CONTRIBUTIONS = {}

for lesson, memory in SOURCES.items():

    CONTRIBUTIONS[
        lesson
    ] = (
        memory["capability"],
        extract_contribution(
            lesson,
            memory
        )
    )

assert len(
    CONTRIBUTIONS
) == EXPECTED_LESSONS, (
    "Every lesson must contribute to the ledger."
)

print(
    "Contributions extracted for:",
    sorted(
        CONTRIBUTIONS
    )
)

print()

print(
    "TEST 4: Build Control Ledger"
)


def build_ledger(
        contributions
):

    genesis_hash = stable_hash(
        GENESIS_PHRASE
    )

    ledger = []

    previous_hash = genesis_hash

    for lesson in sorted(
            contributions
    ):

        capability, contribution = (
            contributions[
                lesson
            ]
        )

        entry = {
            "lesson":
                lesson,

            "capability":
                capability,

            "contribution":
                contribution,

            "prev_hash":
                previous_hash
        }

        entry_hash = stable_hash(
            entry
        )

        entry[
            "entry_hash"
        ] = entry_hash

        ledger.append(
            entry
        )

        previous_hash = entry_hash

    return ledger, previous_hash, genesis_hash


LEDGER, LEDGER_HASH, GENESIS_HASH = build_ledger(
    CONTRIBUTIONS
)

assert len(
    LEDGER
) == EXPECTED_LESSONS, (
    "Ledger must hold exactly eight entries."
)

assert GENESIS_HASH[:16] == "576fdde124060d5e", (
    "Genesis hash mismatch."
)

for entry in LEDGER:

    print(
        entry["lesson"],
        "|",
        entry["entry_hash"][:16],
        "| prev",
        entry["prev_hash"][:16]
    )

print(
    "Genesis:",
    GENESIS_HASH[:16]
)

print()

print(
    "TEST 5: Verify Ledger Chain Integrity"
)

CHAIN_VALID = True

previous_hash = GENESIS_HASH

for entry in LEDGER:

    recomputed = stable_hash(
        {
            "lesson":
                entry["lesson"],

            "capability":
                entry["capability"],

            "contribution":
                entry["contribution"],

            "prev_hash":
                entry["prev_hash"]
        }
    )

    if (
            recomputed
            !=
            entry[
                "entry_hash"
            ]
            or
            entry[
                "prev_hash"
            ]
            !=
            previous_hash
    ):

        CHAIN_VALID = False

        break

    previous_hash = entry[
        "entry_hash"
    ]

assert CHAIN_VALID, (
    "Ledger chain is broken."
)

assert LEDGER_HASH == previous_hash, (
    "Ledger hash does not match the final entry."
)

assert (
    LEDGER_HASH[:16]
    ==
    "479eea1f6cd01de0"
), "Final ledger hash mismatch."

print(
    "Chain integrity:",
    CHAIN_VALID
)

print(
    "Ledger hash:",
    LEDGER_HASH[:16]
)

print(
    "All eight entries verified by recomputation."
)

print()

print(
    "TEST 6: Derive Autonomous Policy Memory"
)

M_125 = SOURCES["125R"]
M_127 = SOURCES["127R"]
M_129 = SOURCES["129R"]
M_130 = SOURCES["130R"]
M_132 = SOURCES["132R"]

TUNED_ORDER = M_125["tuned_order"]

ANOMALY_SCHEDULE = M_129[
    "anomaly_schedule"
]

CRITICAL_PATTERNS = set(
    M_129[
        "critical_path"
    ][
        "patterns"
    ]
)

TOLERANCE = M_127[
    "penetration_threshold"
]

TREND_BY_PATTERN = {
    record["pattern_id"]: record["trend"]
    for record
    in M_125["trends"]
}

BREACH_RECOVERY = {
    pattern_id: outcome
    for pattern_id, outcome
    in M_130[
        "breach_scenario"
    ][
        "recovery"
    ].items()
}

ROOT_RECOVERY = M_130[
    "root_cascade_scenario"
][
    "recovery"
]


def derive_policy(
        pattern_id
):

    if pattern_id in BREACH_RECOVERY:

        recovery_action = (
            BREACH_RECOVERY[
                pattern_id
            ][
                "action"
            ]
        )

        recovery_cost = (
            BREACH_RECOVERY[
                pattern_id
            ][
                "cost"
            ]
        )

    elif pattern_id == ROOT_RECOVERY[
            "pattern"
    ]:

        recovery_action = ROOT_RECOVERY[
            "action"
        ]

        recovery_cost = ROOT_RECOVERY[
            "cost"
        ]

    else:

        recovery_action = "NONE"

        recovery_cost = 0.0

    refined = M_132[
        "refined_control"
    ][
        pattern_id
    ]

    return {
        "trend":
            TREND_BY_PATTERN[
                pattern_id
            ],

        "tier":
            M_129[
                "defense_tiers"
            ][
                pattern_id
            ],

        "control_action":
            M_127[
                "control_actions"
            ][
                pattern_id
            ],

        "schedule_position":
            ANOMALY_SCHEDULE.index(
                pattern_id
            ),

        "critical":
            pattern_id
            in
            CRITICAL_PATTERNS,

        "boost":
            refined["boost"],

        "penetration":
            refined["penetration"],

        "recovery_action":
            recovery_action,

        "recovery_cost":
            recovery_cost,

        "compliant":
            refined["penetration"]
            <=
            TOLERANCE
            +
            1e-4
    }


POLICY = {
    pattern_id: derive_policy(
        pattern_id
    )
    for pattern_id
    in TUNED_ORDER
}

POLICY_HASH = stable_hash(
    POLICY
)

assert len(
    POLICY
) == EXPECTED_PATTERNS, (
    "Policy memory must cover six patterns."
)

print(
    "Policy derived from the chain for:",
    TUNED_ORDER
)

print()

print(
    "TEST 7: Verify Policy Accuracy"
)

assert POLICY["pattern_001"] == {
    "trend": "FALLING",
    "tier": "CRITICAL",
    "control_action": "HOLD",
    "schedule_position": 0,
    "critical": True,
    "boost": 1.0,
    "penetration": 0.10710215090658792,
    "recovery_action": "ISOLATE",
    "recovery_cost": 3.0,
    "compliant": True
}, "pattern_001 policy mismatch."

assert POLICY["pattern_004"]["trend"] == "RISING", (
    "pattern_004 trend mismatch."
)

assert POLICY["pattern_004"]["tier"] == "ANOMALY", (
    "pattern_004 tier mismatch."
)

assert POLICY["pattern_004"]["control_action"] == "ESCALATE", (
    "pattern_004 action mismatch."
)

assert abs(
    POLICY["pattern_004"]["boost"]
    -
    1.480908
) <= 1e-4, (
    "pattern_004 tuned boost mismatch."
)

assert POLICY["pattern_004"][
    "recovery_action"
] == "REDUNDANT_CHANNEL", (
    "pattern_004 recovery mismatch."
)

assert POLICY["pattern_002"][
    "recovery_action"
] == "NONE", (
    "pattern_002 recovery mismatch."
)

assert POLICY["pattern_005"][
    "critical"
] is False, (
    "pattern_005 must not be critical."
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "|",
        POLICY[
            pattern_id
        ][
            "trend"
        ],
        "|",
        POLICY[
            pattern_id
        ][
            "tier"
        ],
        "|",
        POLICY[
            pattern_id
        ][
            "control_action"
        ],
        "| pos=",
        POLICY[
            pattern_id
        ][
            "schedule_position"
        ],
        "| boost=",
        format(
            POLICY[
                pattern_id
            ][
                "boost"
            ],
            ".4f"
        ),
        "| recovery=",
        POLICY[
            pattern_id
        ][
            "recovery_action"
        ]
    )

print()

print(
    "TEST 8: Verify Policy Consistency"
)

assert all(
    POLICY[
        pattern_id
    ][
        "control_action"
    ]
    ==
    (
        "ESCALATE"
        if POLICY[
            pattern_id
        ][
            "tier"
        ]
        ==
        "ANOMALY"
        else "HOLD"
    )
    for pattern_id
    in TUNED_ORDER
), "Control action must follow tier."

assert all(
    POLICY[
        pattern_id
    ][
        "compliant"
    ]
    for pattern_id
    in TUNED_ORDER
), "Every pattern must be compliant under the policy."

assert all(
    POLICY[
        pattern_id
    ][
        "boost"
    ]
    >=
    1.0
    for pattern_id
    in TUNED_ORDER
), "No reinforcement may fall below unity."

assert sorted(
    [
        POLICY[
            pattern_id
        ][
            "schedule_position"
        ]
        for pattern_id
        in TUNED_ORDER
    ]
) == list(
    range(
        EXPECTED_PATTERNS
    )
), "Schedule positions must form a valid permutation."

assert all(
    POLICY[
        pattern_id
    ][
        "schedule_position"
    ]
    ==
    ANOMALY_SCHEDULE.index(
        pattern_id
    )
    for pattern_id
    in TUNED_ORDER
), "Schedule positions must match the 129R schedule."

print(
    "Action-tier consistency:",
    all(
        POLICY[
            pattern_id
        ][
            "control_action"
        ]
        ==
        (
            "ESCALATE"
            if POLICY[
                pattern_id
            ][
                "tier"
            ]
            ==
            "ANOMALY"
            else "HOLD"
        )
        for pattern_id
        in TUNED_ORDER
    )
)

print(
    "All compliant:",
    all(
        POLICY[
            pattern_id
        ][
            "compliant"
        ]
        for pattern_id
        in TUNED_ORDER
    )
)

print()

print(
    "TEST 9: Verify Self-Improvement Proof"
)

CLOSED_LOOP_GAIN = (
        M_127["controlled_mean_stop_rate"]
        -
        M_127["baseline_mean_stop_rate"]
)

IMPROVEMENT_PROOF = {
    "closed_loop_gain":
        CLOSED_LOOP_GAIN,

    "resilience_score":
        SOURCES[
            "131R"
        ][
            "resilience_audit"
        ][
            "score"
        ],

    "resilience_grade":
        SOURCES[
            "131R"
        ][
            "resilience_audit"
        ][
            "grade"
        ],

    "static_exposure":
        SOURCES[
            "132R"
        ][
            "stress_scenario"
        ][
            "static_exposure"
        ],

    "tuned_exposure":
        SOURCES[
            "132R"
        ][
            "convergence"
        ][
            "tuned_exposure"
        ],

    "exposure_reduction":
        SOURCES[
            "132R"
        ][
            "convergence"
        ][
            "exposure_reduction"
        ],

    "absorption_capacity":
        SOURCES[
            "130R"
        ][
            "absorption_capacity"
        ]
}

assert abs(
    CLOSED_LOOP_GAIN
    -
    0.044084
) <= 1e-4, (
    "Closed-loop gain mismatch."
)

assert (
        IMPROVEMENT_PROOF["resilience_score"]
        ==
        1.0
), "Resilience score mismatch."

assert (
        IMPROVEMENT_PROOF["resilience_grade"]
        ==
        "RESILIENT"
), "Resilience grade mismatch."

assert abs(
    IMPROVEMENT_PROOF["exposure_reduction"]
    -
    0.019816
) <= 1e-4, (
    "Exposure reduction mismatch."
)

assert (
        IMPROVEMENT_PROOF["tuned_exposure"]
        <
        IMPROVEMENT_PROOF["static_exposure"]
), "Policy must improve on the static baseline."

print(
    "Closed-loop gain:",
    format(
        CLOSED_LOOP_GAIN,
        ".4f"
    )
)

print(
    "Resilience:",
    IMPROVEMENT_PROOF[
        "resilience_score"
    ],
    "/",
    IMPROVEMENT_PROOF[
        "resilience_grade"
    ]
)

print(
    "Exposure:",
    format(
        IMPROVEMENT_PROOF["static_exposure"],
        ".4f"
    ),
    "->",
    format(
        IMPROVEMENT_PROOF["tuned_exposure"],
        ".4f"
    )
)

print()

print(
    "TEST 10: Determinism"
)

SECOND_LEDGER, SECOND_LEDGER_HASH, SECOND_GENESIS = build_ledger(
    CONTRIBUTIONS
)

SECOND_POLICY = {
    pattern_id: derive_policy(
        pattern_id
    )
    for pattern_id
    in TUNED_ORDER
}

assert (
        SECOND_LEDGER_HASH
        ==
        LEDGER_HASH
), "Ledger rebuild is nondeterministic."

assert (
        SECOND_GENESIS
        ==
        GENESIS_HASH
), "Genesis rebuild is nondeterministic."

assert (
        SECOND_LEDGER
        ==
        LEDGER
), "Ledger contents are nondeterministic."

assert (
        SECOND_POLICY
        ==
        POLICY
), "Policy derivation is nondeterministic."

print(
    "Ledger deterministic:",
    SECOND_LEDGER_HASH
    ==
    LEDGER_HASH
)

print(
    "Policy deterministic:",
    SECOND_POLICY
    ==
    POLICY
)

print()

print(
    "TEST 11: Numerical Health"
)

PENETRATION_TENSOR = torch.tensor(
    [
        POLICY[
            pattern_id
        ][
            "penetration"
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

COST_TENSOR = torch.tensor(
    [
        POLICY[
            pattern_id
        ][
            "recovery_cost"
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        PENETRATION_TENSOR
    ).all()
    and
    torch.isfinite(
        COST_TENSOR
    ).all()
)

print(
    "Penetration NaN:",
    int(
        torch.isnan(
            PENETRATION_TENSOR
        ).sum()
    )
)

print(
    "Recovery cost Inf:",
    int(
        torch.isinf(
            COST_TENSOR
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
    if len(
        LEDGER
    ) == EXPECTED_LESSONS
    else [
        "Ledger does not cover all lessons."
    ]
)

PROMOTION_ERRORS += (
    []
    if CHAIN_VALID
    else [
        "Ledger chain integrity failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(
        POLICY
    ) == EXPECTED_PATTERNS
    else [
        "Policy does not cover all patterns."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        POLICY[
            pattern_id
        ][
            "compliant"
        ]
        for pattern_id
        in TUNED_ORDER
    )
    else [
        "Policy leaves a pattern non-compliant."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        CLOSED_LOOP_GAIN
        -
        0.044084
    ) <= 1e-4
    else [
        "Closed-loop gain invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if (
            IMPROVEMENT_PROOF["resilience_grade"]
            ==
            "RESILIENT"
    )
    else [
        "Resilience grade invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if (
            IMPROVEMENT_PROOF["tuned_exposure"]
            <
            IMPROVEMENT_PROOF["static_exposure"]
    )
    else [
        "Policy does not improve exposure."
    ]
)

PROMOTION_ERRORS += (
    []
    if (
            SECOND_LEDGER_HASH
            ==
            LEDGER_HASH
    )
    else [
        "Ledger is nondeterministic."
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
    "Ledger entries:",
    len(
        LEDGER
    )
)

print(
    "Policy patterns:",
    len(
        POLICY
    )
)

print(
    "Improvement:",
    format(
        IMPROVEMENT_PROOF["tuned_exposure"],
        ".4f"
    ),
    "<",
    format(
        IMPROVEMENT_PROOF["static_exposure"],
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
        "133R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "133R promotion gate passed."
)

print()

print(
    "TEST 13: Persist Ledger Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "133R",

    "capability":
        "end_to_end_self_improving_control_ledger_autonomous_policy_memory",

    "created_at":
        datetime.now().isoformat(),

    "source_lessons":
        [
            lesson
            for lesson, _
            in SOURCE_LESSONS
        ],

    "genesis_hash":
        GENESIS_HASH,

    "ledger":
        LEDGER,

    "ledger_hash":
        LEDGER_HASH,

    "policy":
        POLICY,

    "policy_hash":
        POLICY_HASH,

    "improvement_proof":
        IMPROVEMENT_PROOF,

    "verification":
        {
            "chain_integrity":
                CHAIN_VALID,

            "policy_complete":
                len(
                    POLICY
                )
                ==
                EXPECTED_PATTERNS,

            "all_compliant":
                all(
                    POLICY[
                        pattern_id
                    ][
                        "compliant"
                    ]
                    for pattern_id
                    in TUNED_ORDER
                ),

            "deterministic":
                SECOND_LEDGER_HASH
                ==
                LEDGER_HASH,

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
        RELOADED["ledger_hash"]
        ==
        LEDGER_HASH
), "Ledger hash changed after reload."

assert (
        RELOADED["policy"]
        ==
        POLICY
), "Policy changed after reload."

assert (
        RELOADED["policy_hash"]
        ==
        POLICY_HASH
), "Policy hash changed after reload."

print(
    "Reloaded ledger hash:",
    RELOADED[
        "ledger_hash"
    ][:16]
)

print(
    "Reloaded policy hash:",
    RELOADED[
        "policy_hash"
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
            "133R",

        "capability":
            "end_to_end_self_improving_control_ledger_autonomous_policy_memory",

        "ledger_lessons":
            [
                entry["lesson"]
                for entry
                in LEDGER
            ],

        "ledger_hash":
            LEDGER_HASH,

        "policy_boosts":
            {
                pattern_id:
                    POLICY[
                        pattern_id
                    ][
                        "boost"
                    ]
                for pattern_id
                in TUNED_ORDER
            },

        "policy_actions":
            {
                pattern_id:
                    POLICY[
                        pattern_id
                    ][
                        "control_action"
                    ]
                for pattern_id
                in TUNED_ORDER
            },

        "policy_recovery":
            {
                pattern_id:
                    POLICY[
                        pattern_id
                    ][
                        "recovery_action"
                    ]
                for pattern_id
                in TUNED_ORDER
            },

        "closed_loop_gain":
            CLOSED_LOOP_GAIN,

        "resilience_grade":
            IMPROVEMENT_PROOF[
                "resilience_grade"
            ],

        "exposure_reduction":
            IMPROVEMENT_PROOF[
                "exposure_reduction"
            ]
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "133R",

        "memory_version":
            MEMORY_VERSION,

        "ledger_entries":
            len(
                LEDGER
            ),

        "chain_integrity":
            CHAIN_VALID,

        "genesis_hash":
            GENESIS_HASH[:16],

        "ledger_hash":
            LEDGER_HASH,

        "policy_patterns":
            len(
                POLICY
            ),

        "policy_hash":
            POLICY_HASH,

        "all_compliant":
            all(
                POLICY[
                    pattern_id
                ][
                    "compliant"
                ]
                for pattern_id
                in TUNED_ORDER
            ),

        "closed_loop_gain":
            CLOSED_LOOP_GAIN,

        "resilience_score":
            IMPROVEMENT_PROOF[
                "resilience_score"
            ],

        "resilience_grade":
            IMPROVEMENT_PROOF[
                "resilience_grade"
            ],

        "static_exposure":
            IMPROVEMENT_PROOF[
                "static_exposure"
            ],

        "tuned_exposure":
            IMPROVEMENT_PROOF[
                "tuned_exposure"
            ],

        "exposure_reduction":
            IMPROVEMENT_PROOF[
                "exposure_reduction"
            ],

        "deterministic":
            SECOND_LEDGER_HASH
            ==
            LEDGER_HASH,

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
            "133R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "134R Autonomous Policy Memory Consolidation "
                "+ Continuous Self-Improvement Commitment"
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

REBUILT_LEDGER, REBUILT_LEDGER_HASH, REBUILT_GENESIS = build_ledger(
    CONTRIBUTIONS
)

assert (
        REBUILT_LEDGER_HASH
        ==
        RELOADED["ledger_hash"]
), "Disk ledger differs from a fresh rebuild."

assert (
        REBUILT_GENESIS
        ==
        RELOADED["genesis_hash"]
), "Disk genesis differs from a fresh rebuild."

assert (
        POLICY_HASH
        ==
        RELOADED["policy_hash"]
), "Disk policy differs from a fresh derivation."

print(
    "Rebuilt ledger hash:",
    REBUILT_LEDGER_HASH[:16]
)

print(
    "Rebuilt policy hash:",
    POLICY_HASH[:16]
)

print(
    "Disk and rebuild agree."
)

print()

print(
    "SILVERWING 133R ARCHITECTURE"
)

print(
    "Genesis (Root of Trust)"
)

print(
    "   ↓"
)

print(
    "Entry 125R -> 126R -> 127R -> 128R"
)

print(
    "   ↓"
)

print(
    "Entry 129R -> 130R -> 131R -> 132R"
)

print(
    "   ↓"
)

print(
    "Final Ledger Hash (Tamper-Evident)"
)

print(
    "   ↓"
)

print(
    "Autonomous Policy Memory (Per Pattern)"
)

print(
    "   ↓"
)

print(
    "Improvement Proof (Gain + Grade + Reduction)"
)

print()

print(
    "WHAT 133R ADDS"
)

print(
    "A tamper-evident, hash-chained ledger of every lesson "
    "from 125R to 132R, and an autonomous policy memory that "
    "distills the chain into one compliant policy per pattern."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Any self-improving system that must prove where its "
    "policy came from, and keep that proof auditable across "
    "runs."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Learning without memory is noise. The ledger makes the "
    "learning legible and verifiable, and the policy memory "
    "makes it executable."
)

print()

print(
    "NEXT: 134R Autonomous Policy Memory Consolidation "
    "+ Continuous Self-Improvement Commitment"
)

print()

print(
    "=== LESSON 133R COMPLETE ==="
)

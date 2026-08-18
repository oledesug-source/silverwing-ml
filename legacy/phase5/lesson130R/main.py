# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 130R
# Self-Healing Recovery Orchestration + Failure Absorption
# ============================================================
#
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
# 128R  -> Uncertainty-Aware Preventive Execution
#         + Probabilistic Guardrails
# 129R  -> Anomaly-First Adaptive Scheduling
#         + Critical Path Defense
# 130R  -> Self-Healing Recovery Orchestration
#         + Failure Absorption
#
# ============================================================
# PURPOSE
# ============================================================
#
# 129R decided the ORDER of defense attention. 130R asks what
# happens when a pattern FAILS anyway. Prevention is not
# perfect: a critical pattern is breached, its risk spills onto
# the patterns that depend on it, and the system must heal
# itself. Recovery orchestration decides, deterministically,
# which recovery action to take and at what cost. Failure
# absorption is the system's capacity to contain a breach:
# healthy neighbors absorb the spillover before it becomes a
# cascade.
#
# Self-healing recovery orchestration:
#
#     base penetration (anomaly score)
#               ↓
#     failure injection
#               ↓
#     spillover to dependents
#               ↓
#     recovery playbook (cost-ranked)
#               ↓
#     seeded recovery trials
#               ↓
#     absorbed residuals
#
# Failure absorption:
#
#     breach
#       ↓
#     spillover ratio × (1 - child penetration)
#       ↓
#     child absorbs until FAILURE_THRESHOLD
#       ↓
#     recovery closes the residual
#
# The orchestrator is deterministic: every trial is seeded, so
# a given breach always resolves the same way.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 129R memory is the source of truth.
# 2. Base penetration of a pattern is its 129R anomaly score.
# 3. A breach sets penetration to 1.0.
# 4. Spillover raises each dependent's penetration.
# 5. A pattern is FAILED when penetration >= threshold.
# 6. Recovery trials are seeded per pattern (deterministic).
# 7. Recovery actions are tried in increasing cost order.
# 8. Absorption capacity is the total healthy headroom.
# 9. Recovery must close every residual below the safe ceiling.
# 10. A root breach must be contained without a cascade.
# 11. Determinism must be checked.
# 12. Numerical health must be checked.
# 13. Persistence and reload must be checked.
# 14. Promotion requires all validation gates to pass.
# 15. External LLM: NONE.
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
MEMORY_VERSION = "130R.1"
EXPECTED_PATTERNS = 6
SPILLOVER_RATIO = 0.35
FAILURE_THRESHOLD = 0.80
SAFE_CEILING = 0.70
BREACH_PATTERN = "pattern_003"
ROOT_BREACH_PATTERN = "pattern_001"

PLAYBOOK = [
    {
        "action": "RETRY",
        "cost": 1.0,
        "success": 0.60,
        "recovery": 0.30
    },
    {
        "action": "REDUNDANT_CHANNEL",
        "cost": 2.0,
        "success": 0.75,
        "recovery": 0.45
    },
    {
        "action": "ISOLATE",
        "cost": 3.0,
        "success": 0.85,
        "recovery": 0.55
    },
    {
        "action": "REBUILD",
        "cost": 4.0,
        "success": 0.95,
        "recovery": 0.70
    },
    {
        "action": "QUARANTINE",
        "cost": 5.0,
        "success": 1.00,
        "recovery": 0.90
    }
]

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_129R = PHASE5_DIR / "lesson129R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_memory.json"
)

SOURCE_INDEX = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_index.pt"
)

SOURCE_DATASET = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_dataset.json"
)

SOURCE_REPORT = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_report.json"
)

SOURCE_REGISTRY = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_129R
        / "checkpoints"
        / "silverwing_anomaly_scheduling_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_recovery_orchestration_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_recovery_orchestration_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_recovery_orchestration_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_recovery_orchestration_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_recovery_orchestration_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_recovery_orchestration_best.pt"
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
    "PHASE 5 - LESSON 130R"
)

print(
    "Self-Healing Recovery Orchestration"
)

print(
    "+ Failure Absorption"
)

print()

print(
    "125R -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning"
)

print(
    "126R -> Preventive Control Loop Governance + Policy Rehearsal"
)

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
    "TEST 1: Verify 129R Inputs"
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
    "TEST 2: Load 129R Scheduling Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "129R scheduling memory is invalid."

ANOMALY_SCORES = SOURCE.get(
    "anomaly_scores",
    {}
)

ANOMALY_SCHEDULE = SOURCE.get(
    "anomaly_schedule",
    []
)

DEPENDENCIES = SOURCE.get(
    "dependencies",
    {}
)

ACTION_MAP = SOURCE.get(
    "action_map",
    {}
)

ANOMALY_SET = SOURCE.get(
    "anomaly_set",
    []
)

CRITICAL_PATH = SOURCE.get(
    "critical_path",
    {}
)

TUNED_RECORDS = SOURCE.get(
    "tuned_records",
    []
)

TUNED_ORDER = SOURCE.get(
    "tuned_order",
    []
)

assert len(
    ANOMALY_SCORES
) == EXPECTED_PATTERNS, (
    "129R must supply exactly six anomaly scores."
)

assert len(
    ANOMALY_SCHEDULE
) == EXPECTED_PATTERNS, (
    "129R anomaly schedule must cover six patterns."
)

assert len(
    DEPENDENCIES
) == EXPECTED_PATTERNS, (
    "129R dependencies must cover six patterns."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Anomaly schedule:",
    ANOMALY_SCHEDULE
)

print()

print(
    "TEST 3: Rebuild Dependency Graph"
)

CHILDREN = {}

for pattern_id, dependency in DEPENDENCIES.items():

    if dependency is not None:

        CHILDREN.setdefault(
            ACTION_MAP[
                dependency
            ],
            []
        ).append(
            pattern_id
        )

assert CHILDREN == {
    "pattern_001": [
        "pattern_005",
        "pattern_002"
    ],
    "pattern_002": [
        "pattern_003"
    ],
    "pattern_003": [
        "pattern_004",
        "pattern_006"
    ]
}, "Dependency graph mismatch."

assert BREACH_PATTERN in CHILDREN[
    "pattern_002"
], "pattern_003 must be downstream of pattern_002."

print(
    "Dependency graph:",
    CHILDREN
)

print()

print(
    "TEST 4: Derive Baseline Defense State"
)

BASE_PENETRATION = dict(
    ANOMALY_SCORES
)

HEALTHY_HEADROOM = {
    pattern_id: (
        1.0
        -
        BASE_PENETRATION[
            pattern_id
        ]
    )
    for pattern_id
    in ANOMALY_SCORES
}

ABSORPTION_CAPACITY = sum(
    HEALTHY_HEADROOM.values()
)

assert abs(
    ABSORPTION_CAPACITY
    -
    1.747219
) <= 1e-4, (
    "Absorption capacity mismatch."
)

print(
    "Base penetration:",
    dict(
        map(
            lambda pair: (
                pair[0],
                format(
                    pair[1],
                    ".4f"
                )
            ),
            BASE_PENETRATION.items()
        )
    )
)

print(
    "Absorption capacity:",
    format(
        ABSORPTION_CAPACITY,
        ".4f"
    )
)

print()

print(
    "TEST 5: Inject Failure at Critical Branch"
)

BREACHED = dict(
    BASE_PENETRATION
)

BREACHED[
    BREACH_PATTERN
] = 1.0

assert BREACHED[
    BREACH_PATTERN
] >= FAILURE_THRESHOLD, (
    "Breach must exceed the failure threshold."
)

assert BREACH_PATTERN in ANOMALY_SET, (
    "Breach must target an anomaly."
)

assert BREACH_PATTERN in CRITICAL_PATH[
    "patterns"
], "Breach must target a critical pattern."

print(
    BREACH_PATTERN,
    "penetration set to",
    format(
        BREACHED[
            BREACH_PATTERN
        ],
        ".4f"
    )
)

print()

print(
    "TEST 6: Propagate Spillover to Dependents"
)

for child in CHILDREN[
    BREACH_PATTERN
]:

    BREACHED[
        child
    ] = clamp(
        BREACHED[
            child
        ]
        +
        SPILLOVER_RATIO
        *
        (
                1.0
                -
                BREACHED[
                    child
                ]
        )
    )

assert abs(
    BREACHED["pattern_004"]
    -
    0.913927
) <= 1e-4, "pattern_004 spillover mismatch."

assert abs(
    BREACHED["pattern_006"]
    -
    0.839356
) <= 1e-4, "pattern_006 spillover mismatch."

FAILED_SET = {
    pattern_id
    for pattern_id, penetration
    in BREACHED.items()
    if penetration >= FAILURE_THRESHOLD
}

assert FAILED_SET == {
    "pattern_003",
    "pattern_004",
    "pattern_006"
}, "Failed set mismatch after spillover."

print(
    "Spillover ratio:",
    SPILLOVER_RATIO
)

print(
    "Failed set:",
    sorted(
        FAILED_SET
    )
)

print()

print(
    "TEST 7: Build Recovery Playbook"
)

assert len(
    PLAYBOOK
) == 5, "Recovery playbook must have five actions."

assert [
    step["cost"]
    for step
    in PLAYBOOK
] == [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
], "Recovery actions must be cost-ranked."

assert PLAYBOOK[-1][
    "success"
] == 1.00, (
    "Last resort must be guaranteed."
)

print(
    "Playbook:",
    [
        (
            step["action"],
            "cost=" + str(step["cost"]),
            "succ=" + str(step["success"]),
            "recovery=" + str(step["recovery"])
        )
        for step
        in PLAYBOOK
    ]
)

print()

print(
    "TEST 8: Orchestrate Recovery"
)


def orchestrate_recovery(
        penetration,
        pattern_id
):

    index = ANOMALY_SCHEDULE.index(
        pattern_id
    )

    generator = torch.Generator().manual_seed(
        SEED + index * 7
    )

    cost = 0.0

    for step in PLAYBOOK:

        roll = torch.rand(
            1,
            generator=generator
        ).item()

        cost += step["cost"]

        if roll < step["success"]:

            return {
                "pattern":
                    pattern_id,

                "action":
                    step["action"],

                "cost":
                    step["cost"],

                "roll":
                    round(
                        roll,
                        4
                    ),

                "residual":
                    penetration
                    *
                    (
                            1.0
                            -
                            step["recovery"]
                    )
            }

    assert False, (
        "Guaranteed last resort never succeeded."
    )


RECOVERY = {}

for pattern_id in ANOMALY_SCHEDULE:

    if pattern_id in FAILED_SET:

        RECOVERY[
            pattern_id
        ] = orchestrate_recovery(
            BREACHED[
                pattern_id
            ],
            pattern_id
        )

assert RECOVERY["pattern_003"][
    "action"
] == "REDUNDANT_CHANNEL", (
    "pattern_003 recovery action mismatch."
)

assert RECOVERY["pattern_004"][
    "action"
] == "REDUNDANT_CHANNEL", (
    "pattern_004 recovery action mismatch."
)

assert RECOVERY["pattern_006"][
    "action"
] == "RETRY", (
    "pattern_006 recovery action mismatch."
)

for pattern_id in RECOVERY:

    print(
        pattern_id,
        "|",
        RECOVERY[
            pattern_id
        ][
            "action"
        ],
        "| cost=",
        RECOVERY[
            pattern_id
        ][
            "cost"
        ],
        "| roll=",
        RECOVERY[
            pattern_id
        ][
            "roll"
        ],
        "| residual=",
        format(
            RECOVERY[
                pattern_id
            ][
                "residual"
            ],
            ".4f"
        )
    )

print()

print(
    "TEST 9: Validate Recovery Outcomes"
)

assert abs(
    RECOVERY["pattern_003"]["residual"]
    -
    0.55
) <= 1e-4, "pattern_003 residual mismatch."

assert abs(
    RECOVERY["pattern_004"]["residual"]
    -
    0.502660
) <= 1e-4, "pattern_004 residual mismatch."

assert abs(
    RECOVERY["pattern_006"]["residual"]
    -
    0.587549
) <= 1e-4, "pattern_006 residual mismatch."

POST_RECOVERY = dict(
    BREACHED
)

for pattern_id, outcome in RECOVERY.items():

    POST_RECOVERY[
        pattern_id
    ] = outcome[
        "residual"
    ]

assert all(
    POST_RECOVERY[
        pattern_id
    ]
    <=
    SAFE_CEILING
    for pattern_id
    in RECOVERY
), "Every residual must be below the safe ceiling."

print(
    "Safe ceiling:",
    SAFE_CEILING
)

for pattern_id in RECOVERY:

    print(
        pattern_id,
        "|",
        format(
            BREACHED[
                pattern_id
            ],
            ".4f"
        ),
        "->",
        format(
            POST_RECOVERY[
                pattern_id
            ],
            ".4f"
        ),
        "| SAFE"
        if POST_RECOVERY[
            pattern_id
        ] <= SAFE_CEILING
        else "| UNSAFE"
    )

print()

print(
    "TEST 10: Measure Failure Absorption"
)

PRE_RECOVERY_TOTAL = sum(
    BREACHED.values()
)

POST_RECOVERY_TOTAL = sum(
    POST_RECOVERY.values()
)

ABSORBED = (
        PRE_RECOVERY_TOTAL
        -
        POST_RECOVERY_TOTAL
)

assert abs(
    ABSORBED
    -
    1.113074
) <= 1e-4, "Absorbed residual mismatch."

assert ABSORBED <= ABSORPTION_CAPACITY, (
    "Absorption must not exceed capacity."
)

assert (
        POST_RECOVERY_TOTAL
        <
        PRE_RECOVERY_TOTAL
), "Recovery must reduce total penetration."

ABSORPTION_RATE = (
        ABSORBED
        /
        PRE_RECOVERY_TOTAL
)

assert ABSORPTION_RATE > 0.20, (
    "Absorption rate must be meaningful."
)

print(
    "Pre-recovery total:",
    format(
        PRE_RECOVERY_TOTAL,
        ".4f"
    )
)

print(
    "Post-recovery total:",
    format(
        POST_RECOVERY_TOTAL,
        ".4f"
    )
)

print(
    "Absorbed:",
    format(
        ABSORBED,
        ".4f"
    )
)

print(
    "Absorption rate:",
    format(
        ABSORPTION_RATE,
        ".4f"
    )
)

print()

print(
    "TEST 11: Root Cascade Scenario"
)

BASELINE_FAILED_SET = {
    pattern_id
    for pattern_id, penetration
    in BASE_PENETRATION.items()
    if penetration >= FAILURE_THRESHOLD
}

assert BASELINE_FAILED_SET == {
    "pattern_004"
}, "Baseline failed set mismatch."

print(
    "Baseline failed set:",
    sorted(
        BASELINE_FAILED_SET
    )
)

ROOT_BREACHED = dict(
    BASE_PENETRATION
)

ROOT_BREACHED[
    ROOT_BREACH_PATTERN
] = 1.0

for child in CHILDREN[
    ROOT_BREACH_PATTERN
]:

    ROOT_BREACHED[
        child
    ] = clamp(
        ROOT_BREACHED[
            child
        ]
        +
        SPILLOVER_RATIO
        *
        (
                1.0
                -
                ROOT_BREACHED[
                    child
                ]
        )
    )

assert abs(
    ROOT_BREACHED["pattern_005"]
    -
    0.779526
) <= 1e-4, "pattern_005 root spillover mismatch."

assert abs(
    ROOT_BREACHED["pattern_002"]
    -
    0.726844
) <= 1e-4, "pattern_002 root spillover mismatch."

ROOT_FAILED_SET = {
    pattern_id
    for pattern_id, penetration
    in ROOT_BREACHED.items()
    if penetration >= FAILURE_THRESHOLD
}

INDUCED_FAILURES = (
        ROOT_FAILED_SET
        -
        BASELINE_FAILED_SET
)

assert INDUCED_FAILURES == {
    "pattern_001"
}, "Only the root must be a newly induced failure."

assert ROOT_FAILED_SET == (
        BASELINE_FAILED_SET
        |
        {
            "pattern_001"
        }
), "Root cascade must add only the root to the failed set."

ROOT_RECOVERY = orchestrate_recovery(
    ROOT_BREACHED[
        ROOT_BREACH_PATTERN
    ],
    ROOT_BREACH_PATTERN
)

assert ROOT_RECOVERY[
    "action"
] == "ISOLATE", (
    "Root recovery action mismatch."
)

assert abs(
    ROOT_RECOVERY[
        "residual"
    ]
    -
    0.45
) <= 1e-4, "Root residual mismatch."

print(
    "Root spillover to pattern_005:",
    format(
        ROOT_BREACHED[
            "pattern_005"
        ],
        ".4f"
    )
)

print(
    "Root spillover to pattern_002:",
    format(
        ROOT_BREACHED[
            "pattern_002"
        ],
        ".4f"
    )
)

print(
    "Root failed set:",
    sorted(
        ROOT_FAILED_SET
    )
)

print(
    "Induced failures:",
    sorted(
        INDUCED_FAILURES
    )
)

print(
    "Root recovery:",
    ROOT_RECOVERY[
        "action"
    ],
    "->",
    format(
        ROOT_RECOVERY[
            "residual"
        ],
        ".4f"
    )
)

print()

print(
    "TEST 12: Budget Accounting"
)

TOTAL_RECOVERY_COST = sum(
    outcome["cost"]
    for outcome
    in RECOVERY.values()
)

assert abs(
    TOTAL_RECOVERY_COST
    -
    5.0
) <= 1e-4, "Total recovery cost mismatch."

assert TOTAL_RECOVERY_COST < (
    sum(
        step["cost"]
        for step
        in PLAYBOOK
    )
), "Recovery must cost less than full quarantine."

print(
    "Total recovery cost:",
    TOTAL_RECOVERY_COST
)

print(
    "Root recovery cost:",
    ROOT_RECOVERY[
        "cost"
    ]
)

print()

print(
    "TEST 13: Determinism"
)

SECOND_ORCHESTRATION = {}

for pattern_id in FAILED_SET:

    SECOND_ORCHESTRATION[
        pattern_id
    ] = orchestrate_recovery(
        BREACHED[
            pattern_id
        ],
        pattern_id
    )

DETERMINISTIC = (
        RECOVERY
        ==
        SECOND_ORCHESTRATION
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Recovery orchestration is nondeterministic."
)

print(
    "Deterministic recovery validated."
)

print()

print(
    "TEST 14: Numerical Health"
)

PENETRATION_TENSOR = torch.tensor(
    [
        BREACHED[
            pattern_id
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

RESIDUAL_TENSOR = torch.tensor(
    [
        POST_RECOVERY[
            pattern_id
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
        RESIDUAL_TENSOR
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
    "Residual Inf:",
    int(
        torch.isinf(
            RESIDUAL_TENSOR
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
    "TEST 15: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if FAILED_SET == {
        "pattern_003",
        "pattern_004",
        "pattern_006"
    }
    else [
        "Failed set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        ABSORBED
        -
        1.113074
    ) <= 1e-4
    else [
        "Absorption measurement invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if ABSORBED <= ABSORPTION_CAPACITY
    else [
        "Absorption exceeded capacity."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        POST_RECOVERY[
            pattern_id
        ]
        <=
        SAFE_CEILING
        for pattern_id
        in RECOVERY
    )
    else [
        "Recovery left an unsafe residual."
    ]
)

PROMOTION_ERRORS += (
    []
    if INDUCED_FAILURES == {
        "pattern_001"
    }
    else [
        "Root cascade was not contained."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        TOTAL_RECOVERY_COST
        -
        5.0
    ) <= 1e-4
    else [
        "Recovery cost accounting invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Recovery orchestration nondeterministic."
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
    "Failed set:",
    sorted(
        FAILED_SET
    )
)

print(
    "Absorbed:",
    format(
        ABSORBED,
        ".4f"
    )
)

print(
    "Total recovery cost:",
    TOTAL_RECOVERY_COST
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "130R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "130R promotion gate passed."
)

print()

print(
    "TEST 16: Persist Recovery Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "130R",

    "capability":
        "self_healing_recovery_orchestration_failure_absorption",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "129R",

    "anomaly_scores":
        ANOMALY_SCORES,

    "anomaly_schedule":
        ANOMALY_SCHEDULE,

    "dependencies":
        DEPENDENCIES,

    "action_map":
        ACTION_MAP,

    "children":
        CHILDREN,

    "anomaly_set":
        ANOMALY_SET,

    "critical_path":
        CRITICAL_PATH,

    "tuned_records":
        TUNED_RECORDS,

    "tuned_order":
        TUNED_ORDER,

    "recovery_frame":
        {
            "spillover_ratio":
                SPILLOVER_RATIO,

            "failure_threshold":
                FAILURE_THRESHOLD,

            "safe_ceiling":
                SAFE_CEILING,

            "playbook":
                PLAYBOOK
        },

    "base_penetration":
        BASE_PENETRATION,

    "absorption_capacity":
        ABSORPTION_CAPACITY,

    "breach_scenario":
        {
            "breach_pattern":
                BREACH_PATTERN,

            "pre_recovery":
                BREACHED,

            "failed_set":
                sorted(
                    FAILED_SET
                ),

            "recovery":
                RECOVERY,

            "post_recovery":
                POST_RECOVERY,

            "absorbed":
                ABSORBED,

            "absorption_rate":
                ABSORPTION_RATE,

            "pre_recovery_total":
                PRE_RECOVERY_TOTAL,

            "post_recovery_total":
                POST_RECOVERY_TOTAL
        },

    "root_cascade_scenario":
        {
            "root_pattern":
                ROOT_BREACH_PATTERN,

            "baseline_failed_set":
                sorted(
                    BASELINE_FAILED_SET
                ),

            "pre_recovery":
                ROOT_BREACHED,

            "failed_set":
                sorted(
                    ROOT_FAILED_SET
                ),

            "induced_failures":
                sorted(
                    INDUCED_FAILURES
                ),

            "recovery":
                ROOT_RECOVERY
        },

    "verification":
        {
            "all_residuals_safe":
                all(
                    POST_RECOVERY[
                        pattern_id
                    ]
                    <=
                    SAFE_CEILING
                    for pattern_id
                    in RECOVERY
                ),

            "cascade_contained":
                INDUCED_FAILURES
                ==
                {
                    ROOT_BREACH_PATTERN
                },

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
    "TEST 17: Reload Persistent Memory"
)

RELOADED = read_json(
    MEMORY_FILE
)

assert (
        RELOADED[
            "memory_version"
        ]
        ==
        MEMORY_VERSION
), "Memory version mismatch after reload."

assert (
        RELOADED["breach_scenario"]["failed_set"]
        ==
        sorted(
            FAILED_SET
        )
), "Failed set changed after reload."

assert (
        RELOADED["breach_scenario"]["recovery"]
        ==
        RECOVERY
), "Recovery outcomes changed after reload."

assert (
        RELOADED["root_cascade_scenario"]["induced_failures"]
        ==
        [
            ROOT_BREACH_PATTERN
        ]
), "Root cascade changed after reload."

print(
    "Reloaded failed set:",
    RELOADED[
        "breach_scenario"
    ][
        "failed_set"
    ]
)

print(
    "Reloaded absorbed:",
    format(
        RELOADED[
            "breach_scenario"
        ][
            "absorbed"
        ],
        ".4f"
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 18: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "130R",

        "capability":
            "self_healing_recovery_orchestration_failure_absorption",

        "recovery_frame":
            {
                "spillover_ratio":
                    SPILLOVER_RATIO,

                "failure_threshold":
                    FAILURE_THRESHOLD,

                "safe_ceiling":
                    SAFE_CEILING,

                "playbook":
                    PLAYBOOK
            },

        "base_penetration":
            BASE_PENETRATION,

        "absorption_capacity":
            ABSORPTION_CAPACITY,

        "breach_failed_set":
            sorted(
                FAILED_SET
            ),

        "breach_recovery":
            {
                pattern_id: outcome["action"]
                for pattern_id, outcome
                in RECOVERY.items()
            },

        "breach_residuals":
            {
                pattern_id:
                    outcome["residual"]
                for pattern_id, outcome
                in RECOVERY.items()
            },

        "absorbed":
            ABSORBED,

        "root_induced_failures":
            sorted(
                INDUCED_FAILURES
            ),

        "root_recovery_action":
            ROOT_RECOVERY["action"]
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "130R",

        "memory_version":
            MEMORY_VERSION,

        "breach_pattern":
            BREACH_PATTERN,

        "failed_set":
            sorted(
                FAILED_SET
            ),

        "recovery_actions":
            {
                pattern_id: outcome["action"]
                for pattern_id, outcome
                in RECOVERY.items()
            },

        "pre_recovery_total":
            PRE_RECOVERY_TOTAL,

        "post_recovery_total":
            POST_RECOVERY_TOTAL,

        "absorbed":
            ABSORBED,

        "absorption_rate":
            ABSORPTION_RATE,

        "absorption_capacity":
            ABSORPTION_CAPACITY,

        "total_recovery_cost":
            TOTAL_RECOVERY_COST,

        "root_cascade_contained":
            INDUCED_FAILURES
            ==
            {
                ROOT_BREACH_PATTERN
            },

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
            "130R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "131R Collective Defense Consolidation + "
                "System-Level Resilience Audit"
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
    "SILVERWING 130R ARCHITECTURE"
)

print(
    "Base Penetration (Anomaly Score)"
)

print(
    "   ↓"
)

print(
    "Failure Injection"
)

print(
    "   ↓"
)

print(
    "Spillover to Dependents"
)

print(
    "   ↓"
)

print(
    "Recovery Playbook (Cost-Ranked)"
)

print(
    "   ↓"
)

print(
    "Seeded Recovery Trials"
)

print(
    "   ↓"
)

print(
    "Absorbed Residuals"
)

print(
    "   ↓"
)

print(
    "Root Cascade -> Contained"
)

print()

print(
    "WHAT 130R ADDS"
)

print(
    "A deterministic recovery orchestrator that answers a "
    "breach with the cheapest reliable action, and a failure "
    "absorption model that proves healthy dependents can "
    "contain a spillover before it becomes a cascade."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems where a single breach must not cascade, and "
    "recovery must be decided under uncertainty with a "
    "reproducible budget."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Prevention reduces risk; self-healing contains what "
    "prevention misses. Deterministic recovery makes a breach "
    "a bounded, budgeted, reproducible event."
)

print()

print(
    "NEXT: 131R Collective Defense Consolidation "
    "+ System-Level Resilience Audit"
)

print()

print(
    "=== LESSON 130R COMPLETE ==="
)

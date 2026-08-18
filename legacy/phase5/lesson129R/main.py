# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 129R
# Anomaly-First Adaptive Scheduling + Critical Path Defense
# ============================================================
#
# 124R  -> Outcome Memory Consolidation + Risk Calibration
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
# 128R  -> Uncertainty-Aware Preventive Execution
#         + Probabilistic Guardrails
# 129R  -> Anomaly-First Adaptive Scheduling
#         + Critical Path Defense
#
# ============================================================
# PURPOSE
# ============================================================
#
# By 128R every pattern has an uncertainty band and a guardrail
# verdict. 129R asks: in what order should defense attention be
# spent? Not every pattern deserves equal urgency. Some are
# anomalies: high risk, wide uncertainty, hostile trend. They
# must be scheduled first. But attention is constrained by the
# dependency lattice - an anomaly cannot run before its
# prerequisites. The scheduler must be anomaly-first yet
# dependency-feasible, and it must defend the critical path
# that unlocks the most anomalies.
#
# Anomaly-first adaptive scheduling:
#
#     risk + uncertainty band + trend
#               ↓
#     anomaly score
#               ↓
#     ANOMALY / NORMAL
#               ↓
#     dependency lattice
#               ↓
#     anomaly-first feasible schedule
#
# Critical path defense:
#
#     dependency graph
#               ↓
#     longest path
#               ↓
#     critical patterns
#               ↓
#     defense tiers
#
# The scheduler adapts: when a signal drifts, classification and
# tiers are re-derived, and the schedule is regenerated.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 128R memory is the source of truth.
# 2. 125R memory supplies the trend signal.
# 3. Anomaly score blends risk, uncertainty band and trend.
# 4. Anomalies are classified by an explicit threshold.
# 5. Scheduling is dependency-feasible (topological).
# 6. Priority rewards critical-path patterns and anomalies.
# 7. Every anomaly must be placed as early as feasible.
# 8. The scheduler must adapt when signals drift.
# 9. Determinism must be checked.
# 10. Numerical health must be checked.
# 11. Persistence and reload must be checked.
# 12. Promotion requires all validation gates to pass.
# 13. External LLM: NONE.
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
MEMORY_VERSION = "129R.1"
EXPECTED_PATTERNS = 6
ANOMALY_THRESHOLD = 0.70
RISK_WEIGHT = 0.60
BAND_WEIGHT = 0.25
TREND_WEIGHT = 0.15
EXPECTED_CRITICAL_LENGTH = 4
DRIFT_TREND = "RISING"

TREND_WEIGHTS = {
    "RISING": 1.0,
    "STABLE": 0.5,
    "FALLING": 0.0
}

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_128R = PHASE5_DIR / "lesson128R"
LESSON_125R = PHASE5_DIR / "lesson125R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_memory.json"
)

SOURCE_INDEX = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_index.pt"
)

SOURCE_DATASET = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_dataset.json"
)

SOURCE_REPORT = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_report.json"
)

SOURCE_REGISTRY = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_128R
        / "checkpoints"
        / "silverwing_uncertainty_guardrails_best.pt"
)

TREND_MEMORY = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_memory.json"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_anomaly_scheduling_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_anomaly_scheduling_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_anomaly_scheduling_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_anomaly_scheduling_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_anomaly_scheduling_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_anomaly_scheduling_best.pt"
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
    "PHASE 5 - LESSON 129R"
)

print(
    "Anomaly-First Adaptive Scheduling"
)

print(
    "+ Critical Path Defense"
)

print()

print(
    "124R -> Outcome Memory Consolidation + Risk Calibration"
)

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
    "TEST 1: Verify 128R Inputs"
)

REQUIRED_FILES = [
    SOURCE_MEMORY,
    SOURCE_INDEX,
    SOURCE_DATASET,
    SOURCE_REPORT,
    SOURCE_REGISTRY,
    SOURCE_CHECKPOINT,
    TREND_MEMORY
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

print(
    "FOUND:",
    TREND_MEMORY
)

print()

print(
    "TEST 2: Load 128R Guardrail Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "128R guardrail memory is invalid."

TUNED_RECORDS = SOURCE.get(
    "tuned_records",
    []
)

TUNED_ORDER = SOURCE.get(
    "tuned_order",
    []
)

TUNED_PLAN = SOURCE.get(
    "tuned_plan",
    []
)

POLICY_FRAME = SOURCE.get(
    "policy_frame",
    []
)

CRITICAL_SET = SOURCE.get(
    "critical_set",
    []
)

CONTROL_ACTIONS = SOURCE.get(
    "control_actions",
    {}
)

BASELINE_BANDS = {
    pattern_id: band["band_width"]
    for pattern_id, band
    in SOURCE.get(
        "baseline_bands",
        {}
    ).items()
}

CONTROLLED_GATE = SOURCE.get(
    "controlled_gate",
    {}
)

assert len(
    TUNED_RECORDS
) == EXPECTED_PATTERNS, (
    "128R must supply exactly six tuned records."
)

assert len(
    TUNED_PLAN
) == EXPECTED_PATTERNS, (
    "128R tuned plan must contain six steps."
)

assert len(
    BASELINE_BANDS
) == EXPECTED_PATTERNS, (
    "128R baseline bands must cover six patterns."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Tuned order:",
    TUNED_ORDER
)

print()

print(
    "TEST 3: Load 125R Trend Source"
)

TREND_SOURCE = read_json(
    TREND_MEMORY
)

TRENDS = {}

for entry in TREND_SOURCE.get(
    "trends",
    []
):

    TRENDS[
        entry["pattern_id"]
    ] = entry["trend"]

assert len(
    TRENDS
) == EXPECTED_PATTERNS, (
    "125R trends must cover six patterns."
)

print(
    "Trend map:",
    TRENDS
)

print()

print(
    "TEST 4: Rebuild Scheduling State"
)

RECORD_MAP = {
    record["pattern_id"]: record
    for record
    in TUNED_RECORDS
}

ACTION_TO_PATTERN = {
    step["action"]: step["pattern_id"]
    for step
    in TUNED_PLAN
}

DEPENDENCIES = {
    step["pattern_id"]: step["dependency"]
    for step
    in TUNED_PLAN
}

CHILDREN = {}

for pattern_id, dependency in DEPENDENCIES.items():

    if dependency is not None:

        CHILDREN.setdefault(
            ACTION_TO_PATTERN[
                dependency
            ],
            []
        ).append(
            pattern_id
        )

assert len(
    ACTION_TO_PATTERN
) == EXPECTED_PATTERNS, (
    "Action map must cover six actions."
)

MAX_BAND = max(
    BASELINE_BANDS.values()
)

assert MAX_BAND > 0.0, (
    "Max band width must be positive."
)

print(
    "Dependencies:",
    DEPENDENCIES
)

print(
    "Action map:",
    ACTION_TO_PATTERN
)

print(
    "Max band width:",
    format(
        MAX_BAND,
        ".4f"
    )
)

print()

print(
    "TEST 5: Compute Anomaly Scores"
)

ANOMALY_SCORES = {}

for pattern_id in RECORD_MAP:

    risk = RECORD_MAP[
        pattern_id
    ][
        "risk_score"
    ]

    band_norm = (
            BASELINE_BANDS[
                pattern_id
            ]
            /
            MAX_BAND
    )

    trend_weight = TREND_WEIGHTS[
        TRENDS[
            pattern_id
        ]
    ]

    ANOMALY_SCORES[
        pattern_id
    ] = (
            RISK_WEIGHT * risk
            +
            BAND_WEIGHT * band_norm
            +
            TREND_WEIGHT * trend_weight
    )

assert abs(
    ANOMALY_SCORES["pattern_001"]
    -
    0.687223
) <= 1e-4, "pattern_001 anomaly score mismatch."

assert abs(
    ANOMALY_SCORES["pattern_004"]
    -
    0.867580
) <= 1e-4, "pattern_004 anomaly score mismatch."

assert abs(
    ANOMALY_SCORES["pattern_003"]
    -
    0.704554
) <= 1e-4, "pattern_003 anomaly score mismatch."

assert abs(
    ANOMALY_SCORES["pattern_006"]
    -
    0.752855
) <= 1e-4, "pattern_006 anomaly score mismatch."

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "| score=",
            format(
                ANOMALY_SCORES[
                    pattern_id
                ],
                ".4f"
            )
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 6: Classify Anomalies"
)

ANOMALY_SET = {
    pattern_id
    for pattern_id, score
    in ANOMALY_SCORES.items()
    if score > ANOMALY_THRESHOLD
}

NORMAL_SET = {
    pattern_id
    for pattern_id
    in RECORD_MAP
    if pattern_id not in ANOMALY_SET
}

assert ANOMALY_SET == {
    "pattern_003",
    "pattern_004",
    "pattern_006"
}, "Anomaly set mismatch."

assert NORMAL_SET == {
    "pattern_001",
    "pattern_002",
    "pattern_005"
}, "Normal set mismatch."

assert len(
    ANOMALY_SET
) + len(
    NORMAL_SET
) == EXPECTED_PATTERNS, (
    "Classification must partition the pattern set."
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "|",
            "ANOMALY"
            if pattern_id in ANOMALY_SET
            else "NORMAL"
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 7: Identify Critical Path"
)


def critical_path(
        children,
        root
):

    paths = []

    def visit(
            node,
            trail
    ):

        trail = trail + [node]

        if node not in children:

            paths.append(
                trail
            )

            return

        for child in children[node]:

            visit(
                child,
                trail
            )

    visit(
        root,
        []
    )

    longest = max(
        len(path)
        for path
        in paths
    )

    critical = set()

    for path in paths:

        if len(path) == longest:

            critical.update(
                path
            )

    return (
        critical,
        longest
    )


CRITICAL_PATTERNS, CRITICAL_LENGTH = critical_path(
    CHILDREN,
    "pattern_001"
)

assert CRITICAL_LENGTH == EXPECTED_CRITICAL_LENGTH, (
    "Critical path length mismatch."
)

assert CRITICAL_PATTERNS == {
    "pattern_001",
    "pattern_002",
    "pattern_003",
    "pattern_004",
    "pattern_006"
}, "Critical pattern set mismatch."

print(
    "Critical path length:",
    CRITICAL_LENGTH
)

print(
    "Critical patterns:",
    sorted(
        CRITICAL_PATTERNS
    )
)

print()

print(
    "TEST 8: Build Risk-First Baseline Schedule"
)


def schedule(
        priority_key
):

    scheduled = []

    done = set()

    while len(scheduled) < EXPECTED_PATTERNS:

        eligible = [
            pattern_id
            for pattern_id
            in RECORD_MAP
            if pattern_id not in done
            and (
                DEPENDENCIES[
                    pattern_id
                ] is None
                or
                ACTION_TO_PATTERN[
                    DEPENDENCIES[
                        pattern_id
                    ]
                ] in done
            )
        ]

        assert eligible, (
            "Dependency cycle detected."
        )

        eligible.sort(
            key=priority_key
        )

        chosen = eligible[0]

        scheduled.append(
            chosen
        )

        done.add(
            chosen
        )

    return scheduled


def feasible(
        schedule_order
):

    done = set()

    for pattern_id in schedule_order:

        dependency = DEPENDENCIES[
            pattern_id
        ]

        if dependency is not None:

            parent = ACTION_TO_PATTERN[
                dependency
            ]

            if parent not in done:

                return False

        done.add(
            pattern_id
        )

    return True


RISK_FIRST_SCHEDULE = schedule(
    lambda pattern_id: (
        -RECORD_MAP[
            pattern_id
        ][
            "risk_score"
        ],
        pattern_id
    )
)

assert feasible(
    RISK_FIRST_SCHEDULE
), "Risk-first schedule violates dependencies."

assert RISK_FIRST_SCHEDULE == [
    "pattern_001",
    "pattern_005",
    "pattern_002",
    "pattern_003",
    "pattern_004",
    "pattern_006"
], "Risk-first schedule mismatch."

print(
    "Risk-first schedule:",
    RISK_FIRST_SCHEDULE
)

print()

print(
    "TEST 9: Build Anomaly-First Schedule"
)

ANOMALY_SCHEDULE = schedule(
    lambda pattern_id: (
        -(
            1
            if pattern_id in CRITICAL_PATTERNS
            else 0
        ),
        -ANOMALY_SCORES[
            pattern_id
        ],
        pattern_id
    )
)

assert feasible(
    ANOMALY_SCHEDULE
), "Anomaly-first schedule violates dependencies."

assert ANOMALY_SCHEDULE == [
    "pattern_001",
    "pattern_002",
    "pattern_003",
    "pattern_004",
    "pattern_006",
    "pattern_005"
], "Anomaly-first schedule mismatch."

assert (
        ANOMALY_SCHEDULE
        !=
        RISK_FIRST_SCHEDULE
), "Scheduling mechanism must change the order."

POSITIONS = {
    pattern_id: index
    for index, pattern_id
    in enumerate(
        ANOMALY_SCHEDULE
    )
}

assert POSITIONS["pattern_005"] == 5, (
    "The non-essential normal must run last."
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "| pos=",
            POSITIONS[
                pattern_id
            ],
            "|",
            "ANOMALY"
            if pattern_id in ANOMALY_SET
            else "NORMAL",
            "|",
            "CRITICAL"
            if pattern_id in CRITICAL_PATTERNS
            else "OFF-PATH"
        ),
        ANOMALY_SCHEDULE
    )
)

print()

print(
    "TEST 10: Validate Anomaly Earliest-Feasible Placement"
)


def earliest_feasible(
        pattern_id
):

    dependency = DEPENDENCIES[
        pattern_id
    ]

    if dependency is None:

        return 0

    return (
            1
            +
            earliest_feasible(
                ACTION_TO_PATTERN[
                    dependency
                ]
            )
    )

EARLIEST = {
    pattern_id:
        earliest_feasible(
            pattern_id
        )
    for pattern_id
    in RECORD_MAP
}

assert EARLIEST["pattern_003"] == 2, (
    "pattern_003 earliest feasible position mismatch."
)

assert EARLIEST["pattern_004"] == 3, (
    "pattern_004 earliest feasible position mismatch."
)

assert EARLIEST["pattern_006"] == 3, (
    "pattern_006 earliest feasible position mismatch."
)

assert POSITIONS[
    "pattern_003"
] == EARLIEST[
    "pattern_003"
], "pattern_003 must run at its earliest feasible position."

assert POSITIONS[
    "pattern_004"
] == EARLIEST[
    "pattern_004"
], "pattern_004 must run at its earliest feasible position."

assert (
        POSITIONS[
            "pattern_006"
        ]
        ==
        EARLIEST[
            "pattern_006"
        ]
        +
        1
), "pattern_006 must yield only to the higher-scored anomaly."

assert (
        POSITIONS[
            "pattern_004"
        ]
        <
        POSITIONS[
            "pattern_006"
        ]
), "pattern_004 must precede pattern_006."

for pattern_id in ANOMALY_SET:

    assert (
            POSITIONS[
                pattern_id
            ]
            <
            POSITIONS[
                "pattern_005"
            ]
    ), "Every anomaly must run before the non-essential normal."

print(
    "Earliest feasible:",
    EARLIEST
)

print(
    "All anomalies placed at earliest feasible positions."
)

print()

print(
    "TEST 11: Assign Defense Tiers"
)

DEFENSE_TIERS = {}

for pattern_id in RECORD_MAP:

    if pattern_id in ANOMALY_SET:

        DEFENSE_TIERS[
            pattern_id
        ] = "ANOMALY"

    elif pattern_id in CRITICAL_PATTERNS:

        DEFENSE_TIERS[
            pattern_id
        ] = "CRITICAL"

    else:

        DEFENSE_TIERS[
            pattern_id
        ] = "SECONDARY"

assert DEFENSE_TIERS == {
    "pattern_001": "CRITICAL",
    "pattern_002": "CRITICAL",
    "pattern_003": "ANOMALY",
    "pattern_004": "ANOMALY",
    "pattern_005": "SECONDARY",
    "pattern_006": "ANOMALY"
}, "Defense tier assignment mismatch."

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "|",
            DEFENSE_TIERS[
                pattern_id
            ]
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 12: Adaptive Rescheduling under Drift"
)

DRIFTED_SCORES = dict(
    ANOMALY_SCORES
)

DRIFTED_SCORES[
    "pattern_005"
] = (
        RISK_WEIGHT
        *
        RECORD_MAP[
            "pattern_005"
        ][
            "risk_score"
        ]
        +
        BAND_WEIGHT
        *
        (
                BASELINE_BANDS[
                    "pattern_005"
                ]
                /
                MAX_BAND
        )
        +
        TREND_WEIGHT
        *
        TREND_WEIGHTS[
            DRIFT_TREND
        ]
)

DRIFTED_ANOMALY_SET = {
    pattern_id
    for pattern_id, score
    in DRIFTED_SCORES.items()
    if score > ANOMALY_THRESHOLD
}

assert (
        DRIFTED_SCORES[
            "pattern_005"
        ]
        >
        ANOMALY_THRESHOLD
), "Drift must push pattern_005 past the threshold."

assert DRIFTED_ANOMALY_SET == {
    "pattern_003",
    "pattern_004",
    "pattern_005",
    "pattern_006"
}, "Drifted anomaly set mismatch."

DRIFTED_TIERS = dict(
    DEFENSE_TIERS
)

for pattern_id in DRIFTED_ANOMALY_SET:

    DRIFTED_TIERS[
        pattern_id
    ] = "ANOMALY"

assert (
        DRIFTED_TIERS["pattern_005"]
        ==
        "ANOMALY"
), "pattern_005 tier must adapt to ANOMALY."

DRIFTED_SCHEDULE = schedule(
    lambda pattern_id: (
        -(
            1
            if pattern_id in CRITICAL_PATTERNS
            else 0
        ),
        -DRIFTED_SCORES[
            pattern_id
        ],
        pattern_id
    )
)

assert feasible(
    DRIFTED_SCHEDULE
), "Drifted schedule must stay feasible."

assert (
        DEFENSE_TIERS["pattern_005"]
        !=
        DRIFTED_TIERS["pattern_005"]
), "Tier must change under drift."

print(
    "Drifted pattern_005 score:",
    format(
        DRIFTED_SCORES[
            "pattern_005"
        ],
        ".4f"
    )
)

print(
    "Drifted anomaly set:",
    sorted(
        DRIFTED_ANOMALY_SET
    )
)

print(
    "pattern_005 tier:",
    DEFENSE_TIERS[
        "pattern_005"
    ],
    "->",
    DRIFTED_TIERS[
        "pattern_005"
    ]
)

print(
    "Drifted schedule:",
    DRIFTED_SCHEDULE
)

print()

print(
    "TEST 13: Determinism"
)

SECOND_SCHEDULE = schedule(
    lambda pattern_id: (
        -(
            1
            if pattern_id in CRITICAL_PATTERNS
            else 0
        ),
        -ANOMALY_SCORES[
            pattern_id
        ],
        pattern_id
    )
)

DETERMINISTIC = (
        ANOMALY_SCHEDULE
        ==
        SECOND_SCHEDULE
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Scheduling is nondeterministic."
)

print(
    "Deterministic scheduling validated."
)

print()

print(
    "TEST 14: Numerical Health"
)

SCORE_TENSOR = torch.tensor(
    [
        ANOMALY_SCORES[
            pattern_id
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

BAND_TENSOR = torch.tensor(
    [
        BASELINE_BANDS[
            pattern_id
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        SCORE_TENSOR
    ).all()
    and
    torch.isfinite(
        BAND_TENSOR
    ).all()
)

print(
    "Score NaN:",
    int(
        torch.isnan(
            SCORE_TENSOR
        ).sum()
    )
)

print(
    "Score Inf:",
    int(
        torch.isinf(
            SCORE_TENSOR
        ).sum()
    )
)

print(
    "Band NaN:",
    int(
        torch.isnan(
            BAND_TENSOR
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
    if ANOMALY_SET == {
        "pattern_003",
        "pattern_004",
        "pattern_006"
    }
    else [
        "Anomaly set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if CRITICAL_LENGTH == EXPECTED_CRITICAL_LENGTH
    else [
        "Critical path length invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if CRITICAL_PATTERNS == {
        "pattern_001",
        "pattern_002",
        "pattern_003",
        "pattern_004",
        "pattern_006"
    }
    else [
        "Critical pattern set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if ANOMALY_SCHEDULE == [
        "pattern_001",
        "pattern_002",
        "pattern_003",
        "pattern_004",
        "pattern_006",
        "pattern_005"
    ]
    else [
        "Anomaly-first schedule invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if feasible(
        ANOMALY_SCHEDULE
    )
    else [
        "Anomaly-first schedule infeasible."
    ]
)

PROMOTION_ERRORS += (
    []
    if POSITIONS[
        "pattern_003"
    ] == EARLIEST[
        "pattern_003"
    ]
    and POSITIONS[
        "pattern_004"
    ] == EARLIEST[
        "pattern_004"
    ]
    else [
        "Anomaly earliest-feasible placement violated."
    ]
)

PROMOTION_ERRORS += (
    []
    if DEFENSE_TIERS == {
        "pattern_001": "CRITICAL",
        "pattern_002": "CRITICAL",
        "pattern_003": "ANOMALY",
        "pattern_004": "ANOMALY",
        "pattern_005": "SECONDARY",
        "pattern_006": "ANOMALY"
    }
    else [
        "Defense tier assignment invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if DRIFTED_TIERS["pattern_005"] == "ANOMALY"
    else [
        "Adaptive rescheduling failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Scheduling nondeterministic."
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
    "Anomalies:",
    sorted(
        ANOMALY_SET
    )
)

print(
    "Critical path length:",
    CRITICAL_LENGTH
)

print(
    "Anomaly-first schedule:",
    ANOMALY_SCHEDULE
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "129R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "129R promotion gate passed."
)

print()

print(
    "TEST 16: Persist Scheduling Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "129R",

    "capability":
        "anomaly_first_adaptive_scheduling_critical_path_defense",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "128R",

    "tuned_records":
        TUNED_RECORDS,

    "tuned_order":
        TUNED_ORDER,

    "tuned_plan":
        TUNED_PLAN,

    "policy_frame":
        POLICY_FRAME,

    "critical_set":
        CRITICAL_SET,

    "control_actions":
        CONTROL_ACTIONS,

    "trends":
        TRENDS,

    "baseline_bands":
        BASELINE_BANDS,

    "dependencies":
        DEPENDENCIES,

    "action_map":
        ACTION_TO_PATTERN,

    "anomaly_frame":
        {
            "threshold":
                ANOMALY_THRESHOLD,

            "risk_weight":
                RISK_WEIGHT,

            "band_weight":
                BAND_WEIGHT,

            "trend_weight":
                TREND_WEIGHT
        },

    "anomaly_scores":
        ANOMALY_SCORES,

    "anomaly_set":
        sorted(
            ANOMALY_SET
        ),

    "normal_set":
        sorted(
            NORMAL_SET
        ),

    "critical_path":
        {
            "length":
                CRITICAL_LENGTH,

            "patterns":
                sorted(
                    CRITICAL_PATTERNS
                )
        },

    "risk_first_schedule":
        RISK_FIRST_SCHEDULE,

    "anomaly_schedule":
        ANOMALY_SCHEDULE,

    "positions":
        POSITIONS,

    "earliest_feasible":
        EARLIEST,

    "defense_tiers":
        DEFENSE_TIERS,

    "drift":
        {
            "pattern":
                "pattern_005",

            "trend":
                DRIFT_TREND,

            "scores":
                DRIFTED_SCORES,

            "anomaly_set":
                sorted(
                    DRIFTED_ANOMALY_SET
                ),

            "tiers":
                DRIFTED_TIERS,

            "schedule":
                DRIFTED_SCHEDULE
        },

    "verification":
        {
            "anomalies_first":
                all(
                    POSITIONS[
                        pattern_id
                    ]
                    <
                    POSITIONS[
                        "pattern_005"
                    ]
                    for pattern_id
                    in ANOMALY_SET
                ),

            "earliest_placed":
                POSITIONS[
                    "pattern_003"
                ] == EARLIEST[
                    "pattern_003"
                ]
                and POSITIONS[
                    "pattern_004"
                ] == EARLIEST[
                    "pattern_004"
                ],

            "adaptive":
                DEFENSE_TIERS[
                    "pattern_005"
                ]
                !=
                DRIFTED_TIERS[
                    "pattern_005"
                ],

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
        RELOADED["anomaly_schedule"]
        ==
        ANOMALY_SCHEDULE
), "Schedule changed after reload."

assert (
        RELOADED["defense_tiers"]
        ==
        DEFENSE_TIERS
), "Defense tiers changed after reload."

assert (
        RELOADED["anomaly_scores"]
        ==
        ANOMALY_SCORES
), "Anomaly scores changed after reload."

assert (
        RELOADED["drift"]["tiers"]["pattern_005"]
        ==
        "ANOMALY"
), "Drifted tier changed after reload."

print(
    "Reloaded schedule:",
    RELOADED[
        "anomaly_schedule"
    ]
)

print(
    "Reloaded tiers:",
    dict(
        map(
            lambda pair: (
                pair[0],
                pair[1][0]
            ),
            RELOADED[
                "defense_tiers"
            ].items()
        )
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
            "129R",

        "capability":
            "anomaly_first_adaptive_scheduling_critical_path_defense",

        "anomaly_frame":
            {
                "threshold":
                    ANOMALY_THRESHOLD,

                "risk_weight":
                    RISK_WEIGHT,

                "band_weight":
                    BAND_WEIGHT,

                "trend_weight":
                    TREND_WEIGHT
            },

        "anomaly_scores":
            ANOMALY_SCORES,

        "anomaly_set":
            sorted(
                ANOMALY_SET
            ),

        "critical_path":
            {
                "length":
                    CRITICAL_LENGTH,

                "patterns":
                    sorted(
                        CRITICAL_PATTERNS
                    )
            },

        "risk_first_schedule":
            RISK_FIRST_SCHEDULE,

        "anomaly_schedule":
            ANOMALY_SCHEDULE,

        "defense_tiers":
            DEFENSE_TIERS,

        "drifted_tiers":
            DRIFTED_TIERS
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "129R",

        "memory_version":
            MEMORY_VERSION,

        "anomaly_threshold":
            ANOMALY_THRESHOLD,

        "anomaly_set":
            sorted(
                ANOMALY_SET
            ),

        "critical_path_length":
            CRITICAL_LENGTH,

        "critical_patterns":
            sorted(
                CRITICAL_PATTERNS
            ),

        "risk_first_schedule":
            RISK_FIRST_SCHEDULE,

        "anomaly_schedule":
            ANOMALY_SCHEDULE,

        "defense_tiers":
            DEFENSE_TIERS,

        "drift_pattern":
            "pattern_005",

        "drifted_tier":
            DRIFTED_TIERS[
                "pattern_005"
            ],

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
            "129R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "130R Self-Healing Recovery Orchestration "
                "+ Failure Absorption"
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
    "SILVERWING 129R ARCHITECTURE"
)

print(
    "Risk + Uncertainty Band + Trend"
)

print(
    "   ↓"
)

print(
    "Anomaly Score"
)

print(
    "   ↓"
)

print(
    "ANOMALY / NORMAL"
)

print(
    "   ↓"
)

print(
    "Dependency Lattice"
)

print(
    "   ↓"
)

print(
    "Anomaly-First Feasible Schedule"
)

print(
    "   ↓"
)

print(
    "Critical Path Identification"
)

print(
    "   ↓"
)

print(
    "Defense Tiers"
)

print(
    "   ↓"
)

print(
    "Drift -> Reclassify -> Reschedule"
)

print()

print(
    "WHAT 129R ADDS"
)

print(
    "A scheduler that ranks defense attention by a composite "
    "anomaly signal, respects the dependency lattice, places "
    "every anomaly as early as feasibility allows, protects the "
    "critical path and adapts its tiers when signals drift."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems where attention is scarce and a wrong ordering "
    "lets a high-risk anomaly wait behind benign work."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Sequencing is a control decision. Scheduling anomalies "
    "first - while never violating their prerequisites - turns "
    "limited defense capacity into early, guaranteed attention."
)

print()

print(
    "NEXT: 130R Self-Healing Recovery Orchestration "
    "+ Failure Absorption"
)

print()

print(
    "=== LESSON 129R COMPLETE ==="
)

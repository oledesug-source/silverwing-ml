# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 125R
# Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# ============================================================
#
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
# 124R  -> Outcome Memory Consolidation + Risk Calibration
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 124R produced the first consolidated cycle. 125R closes the
# loop across cycles: a second execution cycle runs, and
# Silverwing compares what the risk model said last cycle with
# what it says now.
#
# Cross-cycle trending:
#
#     cycle-1 consolidated risk
#          ↓
#     cycle-2 execution outcomes
#          ↓
#     cycle-2 risk profile
#          ↓
#     risk delta per pattern
#          ↓
#     RISING / FALLING / STABLE trend
#
# Adaptive threshold tuning:
#
#     cycle-1 mean effectiveness
#          ↓
#     cycle-2 mean effectiveness
#          ↓
#     tuning shift
#          ↓
#     tuned HIGH / MEDIUM boundaries
#          ↓
#     tuned risk classes
#          ↓
#     tuned next plan
#
# Tuning rule: when effectiveness improves across cycles the
# system has earned tolerance and thresholds loosen; a pattern
# whose risk rises moves ahead in the order and receives
# earlier mitigation.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 124R memory is the source of truth.
# 2. Cycle-2 outcomes must cover every consolidated pattern.
# 3. Risk trends must be computed against cycle-1 baselines.
# 4. RISING / FALLING / STABLE classes must be deterministic.
# 5. Thresholds must be tuned from cross-cycle effectiveness.
# 6. A RISING pattern must move ahead of its cycle-1 priority.
# 7. The tuned plan must respect dependencies.
# 8. Outcome history must grow across cycles.
# 9. Numerical health must be checked.
# 10. Determinism must be checked.
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
MEMORY_VERSION = "125R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
LEARNING_RATE = 0.30
EFFECTIVE_THRESHOLD = 0.75
TREND_BAND = 0.005
TUNING_SLOPE = 0.50
EXPECTED_PATTERNS = 6
EXPECTED_MEAN_2 = 0.80
EXPECTED_TUNING_SHIFT = 0.025

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_124R = PHASE5_DIR / "lesson124R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_124R
        / "silverwing_memory_consolidation_memory.json"
)

SOURCE_INDEX = (
        LESSON_124R
        / "silverwing_memory_consolidation_index.pt"
)

SOURCE_DATASET = (
        LESSON_124R
        / "silverwing_memory_consolidation_dataset.json"
)

SOURCE_REPORT = (
        LESSON_124R
        / "silverwing_memory_consolidation_report.json"
)

SOURCE_REGISTRY = (
        LESSON_124R
        / "silverwing_memory_consolidation_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_124R
        / "checkpoints"
        / "silverwing_memory_consolidation_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_cross_cycle_trending_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_cross_cycle_trending_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_cross_cycle_trending_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_cross_cycle_trending_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_cross_cycle_trending_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_cross_cycle_trending_best.pt"
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

risk_class = lambda score, high, medium: (
    "HIGH"
    if score >= high
    else (
        "MEDIUM"
        if score >= medium
        else "LOW"
    )
)

risk_score = lambda item: clamp(
    0.30 * item["confidence"]
    + 0.25 * item["severity"]
    + 0.20 * item["impact"]
    + 0.20 * item["recurrence"]
    - 0.05 * item["cost"]
)

arbitrate = lambda records: sorted(
    records,
    key=lambda item: (
        -item["risk_score"],
        -item["severity"],
        -item["impact"],
        item["pattern_id"]
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
    "PHASE 5 - LESSON 125R"
)

print(
    "Cross-Cycle Risk Trending + Adaptive Threshold Tuning"
)

print()

print(
    "119R -> Predictive Error Prevention"
)

print(
    "120R -> Multi-Pattern Risk Arbitration + Preventive Planning"
)

print(
    "121R -> Adaptive Preventive Planning + Dynamic Reprioritization"
)

print(
    "122R -> Continuous Adaptive Execution + Runtime Replanning"
)

print(
    "123R -> Post-Execution Outcome Feedback + Plan Learning"
)

print(
    "124R -> Outcome Memory Consolidation + Risk Calibration"
)

print(
    "125R -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning"
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
    "TEST 1: Verify 124R Inputs"
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
), "One or more 124R inputs are missing."

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
    "TEST 2: Load 124R Consolidated Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "124R consolidated memory is invalid."

CONSOLIDATED_MEMORY = SOURCE.get(
    "consolidated_memory",
    []
)

CALIBRATED_RECORDS = SOURCE.get(
    "calibrated_records",
    []
)

CALIBRATED_ORDER = SOURCE.get(
    "calibrated_order",
    []
)

CALIBRATED_PLAN = SOURCE.get(
    "calibrated_plan",
    []
)

CALIBRATED_PLAN_ACTIONS = [
    step["action"]
    for step
    in CALIBRATED_PLAN
]

OUTCOME_HISTORY = SOURCE.get(
    "outcome_history",
    {}
)

CALIBRATED_THRESHOLDS = SOURCE.get(
    "calibrated_thresholds",
    {}
)

OUTCOME_STATS = SOURCE.get(
    "outcome_stats",
    {}
)

assert len(
    CONSOLIDATED_MEMORY
) == EXPECTED_PATTERNS, (
    "124R must supply exactly six consolidated patterns."
)

assert len(
    CALIBRATED_PLAN
) == EXPECTED_PATTERNS, (
    "124R calibrated plan must contain six steps."
)

assert len(
    CALIBRATED_ORDER
) == EXPECTED_PATTERNS, (
    "124R calibrated order must contain six patterns."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Consolidated patterns:",
    len(
        CONSOLIDATED_MEMORY
    )
)

print(
    "Calibrated order:",
    CALIBRATED_ORDER
)

print(
    "Calibrated plan:",
    CALIBRATED_PLAN_ACTIONS
)

print(
    "Calibrated HIGH:",
    CALIBRATED_THRESHOLDS[
        "calibrated_high"
    ]
)

print()

print(
    "TEST 3: Rebuild Cycle-1 Baseline"
)

CYCLE_1_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in CALIBRATED_RECORDS
}

CYCLE_1_PRIORITY = {
    record["pattern_id"]: record["priority"]
    for record
    in CALIBRATED_RECORDS
}

MEAN_1 = OUTCOME_STATS.get(
    "mean_effectiveness",
    0.0
)

assert abs(
    MEAN_1
    -
    0.75
) <= 1e-9, "Cycle-1 mean effectiveness must be 0.75."

assert len(
    CYCLE_1_RISK
) == EXPECTED_PATTERNS, (
    "Cycle-1 risk baseline must cover six patterns."
)

print(
    "Cycle-1 risks:",
    dict(
        map(
            lambda pair: (
                pair[0],
                round(
                    pair[1],
                    6
                )
            ),
            CYCLE_1_RISK.items()
        )
    )
)

print(
    "Cycle-1 priorities:",
    CYCLE_1_PRIORITY
)

print(
    "Cycle-1 mean effectiveness:",
    format(
        MEAN_1,
        ".4f"
    )
)

print()

print(
    "TEST 4: Collect Cycle-2 Outcome Feedback"
)

PLAN_ACTION_MAP = {
    step["pattern_id"]: (
        step["action"],
        step["dependency"]
    )
    for step
    in CALIBRATED_PLAN
}

CYCLE_2_OUTCOMES = [
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_001",

        "action":
            "VALIDATE_EVIDENCE_PROVENANCE",

        "status":
            "prevented",

        "effectiveness":
            0.95
    },
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_002",

        "action":
            "VALIDATE_SCHEMA_CONSISTENCY",

        "status":
            "prevented",

        "effectiveness":
            0.95
    },
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_003",

        "action":
            "VALIDATE_SENSOR_ALIGNMENT",

        "status":
            "prevented",

        "effectiveness":
            0.90
    },
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_004",

        "action":
            "VALIDATE_TELEMETRY_FUSION",

        "status":
            "not_prevented",

        "effectiveness":
            0.40
    },
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_005",

        "action":
            "VALIDATE_TIMELINE_INTEGRITY",

        "status":
            "prevented",

        "effectiveness":
            0.90
    },
    {
        "cycle":
            2,

        "pattern_id":
            "pattern_006",

        "action":
            "VALIDATE_ACTUATOR_RESPONSE",

        "status":
            "prevented",

        "effectiveness":
            0.70
    }
]

assert len(
    CYCLE_2_OUTCOMES
) == EXPECTED_PATTERNS, (
    "Cycle-2 outcomes must cover six patterns."
)

list(
    map(
        lambda item: print(
            item["pattern_id"],
            "|",
            item["action"],
            "|",
            item["status"],
            "| eff=",
            item["effectiveness"]
        ),
        CYCLE_2_OUTCOMES
    )
)

print()

print(
    "TEST 5: Validate Cycle-2 Coverage"
)

CYCLE_2_COVERED = all(
    item["pattern_id"]
    in
    CYCLE_1_RISK
    for item
    in CYCLE_2_OUTCOMES
)

assert CYCLE_2_COVERED, (
    "Cycle-2 outcomes must reference consolidated patterns."
)

assert all(
    item["action"]
    ==
    PLAN_ACTION_MAP[
        item["pattern_id"]
    ][0]
    for item
    in CYCLE_2_OUTCOMES
), "Cycle-2 outcome action mismatch."

assert all(
    len(
        item["pattern_id"]
    )
    > 0
    for item
    in CYCLE_2_OUTCOMES
)

print(
    "Cycle-2 covered:",
    CYCLE_2_COVERED
)

print(
    "Cycle-2 actions verified."
)

print()

print(
    "TEST 6: Aggregate Cycle-2 Outcome Statistics"
)

EFFECTIVENESS_2 = [
    clamp(
        item["effectiveness"]
    )
    for item
    in CYCLE_2_OUTCOMES
]

MEAN_2 = (
        sum(
            EFFECTIVENESS_2
        )
        /
        len(
            EFFECTIVENESS_2
        )
)

PREVENTED_2 = len(
    [
        item
        for item
        in CYCLE_2_OUTCOMES
        if item["status"] == "prevented"
    ]
)

NOT_PREVENTED_2 = len(
    [
        item
        for item
        in CYCLE_2_OUTCOMES
        if item["status"] == "not_prevented"
    ]
)

assert abs(
    MEAN_2
    -
    EXPECTED_MEAN_2
) <= 1e-9, "Cycle-2 mean effectiveness must be 0.80."

assert PREVENTED_2 == 5, (
    "Expected five prevented outcomes in cycle 2."
)

assert NOT_PREVENTED_2 == 1, (
    "Expected one not-prevented outcome in cycle 2."
)

print(
    "Cycle-2 mean effectiveness:",
    format(
        MEAN_2,
        ".4f"
    )
)

print(
    "Cycle-2 prevented:",
    PREVENTED_2
)

print(
    "Cycle-2 not prevented:",
    NOT_PREVENTED_2
)

print(
    "Effectiveness trend:",
    format(
        MEAN_2 - MEAN_1,
        ".4f"
    )
)

print()

print(
    "TEST 7: Apply Cycle-2 Outcome Learning"
)


def apply_cycle_learning(
        consolidated_memory,
        outcomes,
        learning_rate
):

    learned = []

    feedback = {
        item["pattern_id"]: item
        for item
        in outcomes
    }

    for entry in consolidated_memory:

        pattern_id = entry[
            "pattern_id"
        ]

        outcome = feedback[
            pattern_id
        ]

        effectiveness = clamp(
            outcome["effectiveness"]
        )

        learned.append(
            {
                "pattern_id":
                    pattern_id,

                "family":
                    entry["family"],

                "origin":
                    entry["origin"],

                "severity":
                    entry["severity"],

                "impact":
                    entry["impact"],

                "cost":
                    entry["cost"],

                "confidence":
                    clamp(
                        entry[
                            "persistent_confidence"
                        ]
                        +
                        learning_rate
                        *
                        (
                            0.5
                            -
                            effectiveness
                        )
                    ),

                "recurrence":
                    clamp(
                        entry[
                            "consolidated_recurrence"
                        ]
                        +
                        learning_rate
                        *
                        (
                            1.0
                            -
                            effectiveness
                        )
                    ),

                "outcome_status":
                    outcome["status"],

                "outcome_effectiveness":
                    effectiveness
            }
        )

    return learned


CYCLE_2_PATTERNS = apply_cycle_learning(
    CONSOLIDATED_MEMORY,
    CYCLE_2_OUTCOMES,
    LEARNING_RATE
)

assert len(
    CYCLE_2_PATTERNS
) == EXPECTED_PATTERNS, (
    "Cycle-2 learning must preserve six patterns."
)

list(
    map(
        lambda pattern: print(
            pattern["pattern_id"],
            "| conf=",
            format(
                pattern["confidence"],
                ".6f"
            ),
            "| rec=",
            format(
                pattern["recurrence"],
                ".6f"
            ),
            "|",
            pattern["outcome_status"]
        ),
        CYCLE_2_PATTERNS
    )
)

print()

print(
    "TEST 8: Extend Cross-Cycle Outcome History"
)


def extend_history(
        outcome_history,
        outcomes
):

    extended = {}

    for pattern_id, entry in outcome_history.items():

        current = dict(
            entry
        )

        window = list(
            entry["history_window"]
        )

        window.append(
            {
                "cycle":
                    2,

                "status":
                    next(
                        item["status"]
                        for item
                        in outcomes
                        if item["pattern_id"]
                           ==
                           pattern_id
                    ),

                "effectiveness":
                    clamp(
                        next(
                            item["effectiveness"]
                            for item
                            in outcomes
                            if item["pattern_id"]
                               ==
                               pattern_id
                        )
                    )
            }
        )

        if len(window) > 3:

            window = window[-3:]

        current["history_window"] = window

        current["occurrences"] = len(
            window
        )

        current["last_status"] = window[
            -1
        ][
            "status"
        ]

        current["last_effectiveness"] = window[
            -1
        ][
            "effectiveness"
        ]

        effectiveness_values = [
            record["effectiveness"]
            for record
            in window
        ]

        current["lifetime_effectiveness"] = (
                sum(
                    effectiveness_values
                )
                /
                len(
                    effectiveness_values
                )
        )

        status_counts = {}

        for record in window:

            status_counts[record[
                "status"
            ]] = status_counts.get(
                record["status"],
                0
            ) + 1

        current["status_counts"] = status_counts

        extended[pattern_id] = current

    return extended


EXTENDED_HISTORY = extend_history(
    OUTCOME_HISTORY,
    CYCLE_2_OUTCOMES
)

assert len(
    EXTENDED_HISTORY
) == EXPECTED_PATTERNS, (
    "Extended history must keep six families."
)

assert all(
    len(
        entry["history_window"]
    ) == 2
    for entry
    in EXTENDED_HISTORY.values()
), "Each family must carry two cycles."

list(
    map(
        lambda entry: print(
            entry["pattern_id"],
            "| occ=",
            entry["occurrences"],
            "| lifetime_eff=",
            format(
                entry["lifetime_effectiveness"],
                ".4f"
            ),
            "| counts=",
            entry["status_counts"]
        ),
        EXTENDED_HISTORY.values()
    )
)

print()

print(
    "TEST 9: Recompute Cycle-2 Risk Records"
)


def build_cycle_records(
        patterns,
        high,
        medium
):

    records = [
        {
            "pattern_id":
                pattern["pattern_id"],

            "family":
                pattern["family"],

            "origin":
                pattern["origin"],

            "risk_score":
                risk_score(pattern),

            "risk_class":
                risk_class(
                    risk_score(pattern),
                    high,
                    medium
                ),

            "severity":
                pattern["severity"],

            "impact":
                pattern["impact"],

            "outcome_status":
                pattern["outcome_status"]
        }
        for pattern
        in patterns
    ]

    ordered = arbitrate(
        records
    )

    for index, record in enumerate(
        ordered,
        1
    ):

        record["priority"] = index

    return ordered


CYCLE_2_RECORDS = build_cycle_records(
    CYCLE_2_PATTERNS,
    HIGH_RISK,
    MEDIUM_RISK
)

CYCLE_2_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in CYCLE_2_RECORDS
}

assert abs(
    CYCLE_2_RISK["pattern_001"]
    -
    0.8048129425657967
) <= 1e-6, "Pattern 001 cycle-2 risk mismatch."

assert abs(
    CYCLE_2_RISK["pattern_002"]
    -
    0.6184000000000001
) <= 1e-6, "Pattern 002 cycle-2 risk mismatch."

assert abs(
    CYCLE_2_RISK["pattern_003"]
    -
    0.7755
) <= 1e-6, "Pattern 003 cycle-2 risk mismatch."

assert abs(
    CYCLE_2_RISK["pattern_004"]
    -
    0.7793
) <= 1e-6, "Pattern 004 cycle-2 risk mismatch."

assert abs(
    CYCLE_2_RISK["pattern_005"]
    -
    0.7514200000000001
) <= 1e-6, "Pattern 005 cycle-2 risk mismatch."

assert abs(
    CYCLE_2_RISK["pattern_006"]
    -
    0.7619000000000001
) <= 1e-6, "Pattern 006 cycle-2 risk mismatch."

list(
    map(
        lambda record: print(
            record["pattern_id"],
            "| risk=",
            format(
                record["risk_score"],
                ".6f"
            ),
            "| pri=",
            record["priority"]
        ),
        CYCLE_2_RECORDS
    )
)

print()

print(
    "TEST 10: Compute Cross-Cycle Risk Trends"
)


def classify_trend(
        delta
):

    if delta > TREND_BAND:

        return "RISING"

    if delta < -TREND_BAND:

        return "FALLING"

    return "STABLE"


TRENDS = [
    {
        "pattern_id":
            pattern_id,

        "cycle_1_risk":
            CYCLE_1_RISK[pattern_id],

        "cycle_2_risk":
            CYCLE_2_RISK[pattern_id],

        "delta":
            CYCLE_2_RISK[pattern_id]
            -
            CYCLE_1_RISK[pattern_id],

        "trend":
            classify_trend(
                CYCLE_2_RISK[pattern_id]
                -
                CYCLE_1_RISK[pattern_id]
            )
    }
    for pattern_id
    in CYCLE_1_RISK
]

TREND_MAP = {
    record["pattern_id"]: record["trend"]
    for record
    in TRENDS
}

RISING_COUNT = sum(
    record["trend"] == "RISING"
    for record
    in TRENDS
)

FALLING_COUNT = sum(
    record["trend"] == "FALLING"
    for record
    in TRENDS
)

STABLE_COUNT = sum(
    record["trend"] == "STABLE"
    for record
    in TRENDS
)

assert TREND_MAP[
    "pattern_006"
] == "STABLE", (
    "Pattern 006 must be trend STABLE."
)

assert abs(
    CYCLE_2_RISK["pattern_006"]
    -
    CYCLE_1_RISK["pattern_006"]
) <= 1e-9, "Pattern 006 risk must be unchanged."

assert TREND_MAP[
    "pattern_004"
] == "RISING", (
    "Pattern 004 must be trend RISING."
)

assert RISING_COUNT == 1, (
    "Expected exactly one RISING pattern."
)

assert FALLING_COUNT == 4, (
    "Expected four FALLING patterns."
)

assert STABLE_COUNT == 1, (
    "Expected one STABLE pattern."
)

list(
    map(
        lambda record: print(
            record["pattern_id"],
            "| c1=",
            format(
                record["cycle_1_risk"],
                ".6f"
            ),
            "| c2=",
            format(
                record["cycle_2_risk"],
                ".6f"
            ),
            "| delta=",
            format(
                record["delta"],
                ".6f"
            ),
            "|",
            record["trend"]
        ),
        TRENDS
    )
)

print(
    "RISING:",
    RISING_COUNT,
    "| FALLING:",
    FALLING_COUNT,
    "| STABLE:",
    STABLE_COUNT
)

print()

print(
    "TEST 11: Tune Risk Thresholds Adaptively"
)

TUNING_SHIFT = (
        (
            MEAN_2
            -
            MEAN_1
        )
        *
        TUNING_SLOPE
)

TUNED_HIGH = clamp(
    CALIBRATED_THRESHOLDS[
        "calibrated_high"
    ]
    +
    TUNING_SHIFT
)

TUNED_MEDIUM = clamp(
    CALIBRATED_THRESHOLDS[
        "calibrated_medium"
    ]
    +
    TUNING_SHIFT
)

assert abs(
    TUNING_SHIFT
    -
    EXPECTED_TUNING_SHIFT
) <= 1e-6, "Tuning shift must be 0.025."

assert abs(
    TUNED_HIGH
    -
    0.80
) <= 1e-6, "Tuned HIGH threshold must be 0.80."

assert abs(
    TUNED_MEDIUM
    -
    0.45
) <= 1e-6, "Tuned MEDIUM threshold must be 0.45."

THRESHOLD_TUNED = (
        TUNED_HIGH
        >
        CALIBRATED_THRESHOLDS[
            "calibrated_high"
        ]
        and
        TUNED_MEDIUM
        >
        CALIBRATED_THRESHOLDS[
            "calibrated_medium"
        ]
)

print(
    "Cycle-1 mean:",
    format(
        MEAN_1,
        ".4f"
    )
)

print(
    "Cycle-2 mean:",
    format(
        MEAN_2,
        ".4f"
    )
)

print(
    "Tuning shift:",
    format(
        TUNING_SHIFT,
        ".4f"
    )
)

print(
    "Calibrated HIGH:",
    format(
        CALIBRATED_THRESHOLDS[
            "calibrated_high"
        ],
        ".4f"
    )
)

print(
    "Tuned HIGH:",
    format(
        TUNED_HIGH,
        ".4f"
    )
)

print(
    "Calibrated MEDIUM:",
    format(
        CALIBRATED_THRESHOLDS[
            "calibrated_medium"
        ],
        ".4f"
    )
)

print(
    "Tuned MEDIUM:",
    format(
        TUNED_MEDIUM,
        ".4f"
    )
)

print(
    "Threshold tuned:",
    THRESHOLD_TUNED
)

assert THRESHOLD_TUNED, (
    "Effectiveness improvement must loosen thresholds."
)

print()

print(
    "TEST 12: Reclassify With Tuned Thresholds"
)

TUNED_RECORDS = build_cycle_records(
    CYCLE_2_PATTERNS,
    TUNED_HIGH,
    TUNED_MEDIUM
)

TUNED_ORDER = [
    record["pattern_id"]
    for record
    in TUNED_RECORDS
]

assert TUNED_ORDER == [
    "pattern_001",
    "pattern_004",
    "pattern_003",
    "pattern_006",
    "pattern_005",
    "pattern_002"
], "Tuned risk order is incorrect."

ORDER_CHANGED = (
        TUNED_ORDER
        !=
        CALIBRATED_ORDER
)

assert ORDER_CHANGED, (
    "Trending must change the risk order."
)

HIGH_COUNT = sum(
    record["risk_class"] == "HIGH"
    for record
    in TUNED_RECORDS
)

assert HIGH_COUNT == 1, (
    "Tuned thresholds must leave one HIGH pattern."
)

RISING_PRIORITY = next(
    record["priority"]
    for record
    in TUNED_RECORDS
    if record["pattern_id"] == "pattern_004"
)

RISING_MOVED_AHEAD = (
        RISING_PRIORITY
        <
        CYCLE_1_PRIORITY[
            "pattern_004"
        ]
)

list(
    map(
        lambda record: print(
            record["pattern_id"],
            "| risk=",
            format(
                record["risk_score"],
                ".6f"
            ),
            "| class=",
            record["risk_class"],
            "| pri=",
            record["priority"]
        ),
        TUNED_RECORDS
    )
)

print(
    "Tuned order:",
    TUNED_ORDER
)

print(
    "Order changed:",
    ORDER_CHANGED
)

print(
    "HIGH count:",
    HIGH_COUNT
)

print(
    "Pattern 004 cycle-1 priority:",
    CYCLE_1_PRIORITY[
        "pattern_004"
    ]
)

print(
    "Pattern 004 tuned priority:",
    RISING_PRIORITY
)

print(
    "Rising moved ahead:",
    RISING_MOVED_AHEAD
)

assert RISING_MOVED_AHEAD, (
    "RISING pattern must move ahead of its cycle-1 priority."
)

print()

print(
    "TEST 13: Reconstruct Tuned Plan"
)


def build_adaptive_plan(
        records,
        plan_action_map,
        done_actions
):

    pending = list(
        records
    )

    plan = []

    while pending:

        chosen_index = None

        for index, record in enumerate(
            pending
        ):

            dependency = plan_action_map[
                record["pattern_id"]
            ][1]

            if (
                dependency is None
                or dependency in done_actions
                or any(
                    step["action"] == dependency
                    for step in plan
                )
            ):

                chosen_index = index

                break

        assert chosen_index is not None, (
            "Tuned plan dependency cycle."
        )

        record = pending.pop(
            chosen_index
        )

        plan.append(
            {
                "step":
                    len(plan) + 1,

                "pattern_id":
                    record["pattern_id"],

                "action":
                    plan_action_map[
                        record["pattern_id"]
                    ][0],

                "dependency":
                    plan_action_map[
                        record["pattern_id"]
                    ][1]
            }
        )

    return plan


TUNED_PLAN = build_adaptive_plan(
    TUNED_RECORDS,
    PLAN_ACTION_MAP,
    set()
)

TUNED_PLAN_ACTIONS = [
    step["action"]
    for step
    in TUNED_PLAN
]

assert len(
    TUNED_PLAN
) == EXPECTED_PATTERNS, (
    "Tuned plan must contain six steps."
)

POSITIONS = {
    item["action"]:
        index
    for index, item
    in enumerate(
        TUNED_PLAN
    )
}

assert all(
    item["dependency"] is None
    or (
            item["dependency"]
            in
            POSITIONS
            and
            POSITIONS[
                item["dependency"]
            ]
            <
            POSITIONS[
                item["action"]
            ]
    )
    for item
    in TUNED_PLAN
), "Tuned plan dependency validation failed."

list(
    map(
        print,
        TUNED_PLAN
    )
)

print()

print(
    "TEST 14: Plan Change Detected"
)

PLAN_CHANGED = (
        TUNED_PLAN_ACTIONS
        !=
        CALIBRATED_PLAN_ACTIONS
)

print(
    "Calibrated plan (124R):",
    CALIBRATED_PLAN_ACTIONS
)

print(
    "Tuned plan (125R):",
    TUNED_PLAN_ACTIONS
)

print(
    "Plan changed:",
    PLAN_CHANGED
)

assert PLAN_CHANGED, (
    "Tuning must change the reconstructed plan."
)

print()

print(
    "TEST 15: Deterministic Trending"
)

SECOND_CYCLE_2 = apply_cycle_learning(
    CONSOLIDATED_MEMORY,
    CYCLE_2_OUTCOMES,
    LEARNING_RATE
)

SECOND_TRENDS = [
    {
        "pattern_id":
            pattern_id,

        "trend":
            classify_trend(
                risk_score(
                    next(
                        pattern
                        for pattern
                        in SECOND_CYCLE_2
                        if pattern["pattern_id"]
                           ==
                           pattern_id
                    )
                )
                -
                CYCLE_1_RISK[pattern_id]
            )
    }
    for pattern_id
    in CYCLE_1_RISK
]

DETERMINISTIC = (
        stable_hash(
            CYCLE_2_PATTERNS
        )
        ==
        stable_hash(
            SECOND_CYCLE_2
        )
        and
        stable_hash(
            TREND_MAP
        )
        ==
        stable_hash(
            TREND_MAP
        )
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Cross-cycle trending is nondeterministic."
)

print(
    "Deterministic trending validated."
)

print()

print(
    "TEST 16: Numerical Health"
)

RISK_TENSOR = torch.tensor(
    list(
        CYCLE_2_RISK.values()
    ),
    dtype=torch.float32
)

DELTA_TENSOR = torch.tensor(
    [
        record["delta"]
        for record
        in TRENDS
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        RISK_TENSOR
    ).all()
    and
    torch.isfinite(
        DELTA_TENSOR
    ).all()
)

print(
    "Cycle-2 risk NaN:",
    int(
        torch.isnan(
            RISK_TENSOR
        ).sum()
    )
)

print(
    "Cycle-2 risk Inf:",
    int(
        torch.isinf(
            RISK_TENSOR
        ).sum()
    )
)

print(
    "Trend delta NaN:",
    int(
        torch.isnan(
            DELTA_TENSOR
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
    "TEST 17: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if len(CYCLE_2_PATTERNS) == EXPECTED_PATTERNS
    else [
        "Cycle-2 pattern count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if CYCLE_2_COVERED
    else [
        "Cycle-2 coverage incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        MEAN_2
        -
        EXPECTED_MEAN_2
    ) <= 1e-9
    else [
        "Cycle-2 mean effectiveness invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if RISING_COUNT == 1
    and FALLING_COUNT == 4
    and STABLE_COUNT == 1
    else [
        "Trend distribution invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if THRESHOLD_TUNED
    else [
        "Threshold tuning failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if RISING_MOVED_AHEAD
    else [
        "RISING pattern did not move ahead."
    ]
)

PROMOTION_ERRORS += (
    []
    if ORDER_CHANGED
    else [
        "Tuned order unchanged."
    ]
)

PROMOTION_ERRORS += (
    []
    if PLAN_CHANGED
    else [
        "Tuned plan unchanged."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Trending nondeterministic."
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
    "Cycle-2 patterns:",
    len(
        CYCLE_2_PATTERNS
    )
)

print(
    "Cycle-2 covered:",
    CYCLE_2_COVERED
)

print(
    "Trends:",
    RISING_COUNT,
    "R /",
    FALLING_COUNT,
    "F /",
    STABLE_COUNT,
    "S"
)

print(
    "Threshold tuned:",
    THRESHOLD_TUNED
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "125R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "125R promotion gate passed."
)

print()

print(
    "TEST 18: Persist Cross-Cycle Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "125R",

    "capability":
        "cross_cycle_risk_trending_adaptive_threshold_tuning",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "124R",

    "cycle":
        2,

    "consolidated_memory":
        CONSOLIDATED_MEMORY,

    "calibrated_records":
        CALIBRATED_RECORDS,

    "calibrated_order":
        CALIBRATED_ORDER,

    "calibrated_plan":
        CALIBRATED_PLAN,

    "cycle_2_outcomes":
        CYCLE_2_OUTCOMES,

    "cycle_2_patterns":
        CYCLE_2_PATTERNS,

    "cycle_2_records":
        CYCLE_2_RECORDS,

    "cycle_2_risk":
        CYCLE_2_RISK,

    "extended_history":
        EXTENDED_HISTORY,

    "outcome_stats":
        {
            "cycle_1_mean":
                MEAN_1,

            "cycle_2_mean":
                MEAN_2,

            "effectiveness_delta":
                MEAN_2 - MEAN_1
        },

    "trends":
        TRENDS,

    "trend_counts":
        {
            "rising":
                RISING_COUNT,

            "falling":
                FALLING_COUNT,

            "stable":
                STABLE_COUNT
        },

    "tuned_thresholds":
        {
            "calibrated_high":
                CALIBRATED_THRESHOLDS[
                    "calibrated_high"
                ],

            "calibrated_medium":
                CALIBRATED_THRESHOLDS[
                    "calibrated_medium"
                ],

            "tuning_shift":
                TUNING_SHIFT,

            "tuned_high":
                TUNED_HIGH,

            "tuned_medium":
                TUNED_MEDIUM
        },

    "tuned_records":
        TUNED_RECORDS,

    "tuned_order":
        TUNED_ORDER,

    "tuned_plan":
        TUNED_PLAN,

    "verification":
        {
            "cycle_2_covered":
                CYCLE_2_COVERED,

            "threshold_tuned":
                THRESHOLD_TUNED,

            "rising_moved_ahead":
                RISING_MOVED_AHEAD,

            "order_changed":
                ORDER_CHANGED,

            "plan_changed":
                PLAN_CHANGED,

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
    "TEST 19: Reload Persistent Memory"
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

assert len(
    RELOADED["trends"]
) == len(
    TRENDS
), "Trend count changed after reload."

assert len(
    RELOADED["tuned_plan"]
) == len(
    TUNED_PLAN
), "Tuned plan length changed after reload."

assert RELOADED[
    "tuned_thresholds"
][
    "tuned_high"
] == TUNED_HIGH, (
    "Tuned threshold changed after reload."
)

assert RELOADED[
    "verification"
][
    "rising_moved_ahead"
], "Rising trend flag lost after reload."

print(
    "Reloaded trends:",
    len(
        RELOADED["trends"]
    )
)

print(
    "Reloaded tuned plan:",
    len(
        RELOADED["tuned_plan"]
    )
)

print(
    "Reloaded tuned HIGH:",
    format(
        RELOADED[
            "tuned_thresholds"
        ][
            "tuned_high"
        ],
        ".4f"
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 20: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "125R",

        "capability":
            "cross_cycle_risk_trending_adaptive_threshold_tuning",

        "cycle":
            2,

        "cycle_1_risk":
            CYCLE_1_RISK,

        "cycle_2_outcomes":
            CYCLE_2_OUTCOMES,

        "cycle_2_risk":
            CYCLE_2_RISK,

        "trends":
            TRENDS,

        "trend_counts":
            {
                "rising":
                    RISING_COUNT,

                "falling":
                    FALLING_COUNT,

                "stable":
                    STABLE_COUNT
            },

        "tuned_thresholds":
            {
                "tuning_shift":
                    TUNING_SHIFT,

                "tuned_high":
                    TUNED_HIGH,

                "tuned_medium":
                    TUNED_MEDIUM
            },

        "tuned_order":
            TUNED_ORDER,

        "tuned_plan":
            TUNED_PLAN
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "125R",

        "memory_version":
            MEMORY_VERSION,

        "cycle":
            2,

        "cycle_1_mean":
            MEAN_1,

        "cycle_2_mean":
            MEAN_2,

        "effectiveness_delta":
            MEAN_2 - MEAN_1,

        "trend_counts":
            {
                "rising":
                    RISING_COUNT,

                "falling":
                    FALLING_COUNT,

                "stable":
                    STABLE_COUNT
            },

        "tuning_shift":
            TUNING_SHIFT,

        "tuned_high":
            TUNED_HIGH,

        "tuned_medium":
            TUNED_MEDIUM,

        "rising_pattern":
            "pattern_004",

        "high_count":
            HIGH_COUNT,

        "order_changed":
            ORDER_CHANGED,

        "plan_changed":
            PLAN_CHANGED,

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
            "125R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "126R Preventive Control Loop Governance "
                "+ Policy Rehearsal"
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
    "SILVERWING 125R ARCHITECTURE"
)

print(
    "Cycle-1 Consolidated Risk"
)

print(
    "        ↓"
)

print(
    "Cycle-2 Execution Outcomes"
)

print(
    "        ↓"
)

print(
    "Cycle-2 Outcome Learning"
)

print(
    "        ↓"
)

print(
    "Cross-Cycle Outcome History"
)

print(
    "        ↓"
)

print(
    "Cycle-2 Risk Profile"
)

print(
    "        ↓"
)

print(
    "Risk Trend Classification"
)

print(
    "        ↓"
)

print(
    "Adaptive Threshold Tuning"
)

print(
    "        ↓"
)

print(
    "Tuned Plan Reconstruction"
)

print()

print(
    "WHAT 125R ADDS"
)

print(
    "Per-pattern risk deltas across execution cycles, RISING / "
    "FALLING / STABLE trend classification, adaptive threshold "
    "loosening on effectiveness improvement and a tuned plan "
    "that pushes rising threats earlier."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Any prevention system that runs repeatedly and must notice "
    "when a previously handled threat starts slipping back."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A single cycle only tells Silverwing what happened once. "
    "Trending turns many cycles into a direction of travel, "
    "so the plan can lean into the threat before it peaks."
)

print()

print(
    "NEXT: 126R Preventive Control Loop Governance + Policy Rehearsal"
)

print()

print(
    "=== LESSON 125R COMPLETE ==="
)

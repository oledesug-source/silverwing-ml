# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 124R
# Outcome Memory Consolidation + Risk Calibration
# ============================================================
#
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
# 124R  -> Outcome Memory Consolidation + Risk Calibration
#
# ============================================================
# PURPOSE
# ============================================================
#
# 123R learned from a single execution cycle and rebuilt the
# next plan. 124R does not learn anything new: it consolidates
# what 123R learned into a long-term pattern memory and
# calibrates the risk model against observed effectiveness.
#
# Every learning cycle produces raw lessons. Raw lessons are
# ephemeral. Consolidated memory is the durable form.
#
# 124R consolidates per family:
#
#     outcome history
#          ↓
#     lifetime effectiveness
#          ↓
#     persistent confidence
#          ↓
#     decaying recurrence
#          ↓
#     calibrated risk scores
#
# And it calibrates the risk thresholds themselves:
#
#     observed effectiveness
#          ↓
#     calibration shift
#          ↓
#     calibrated HIGH / MEDIUM boundaries
#          ↓
#     recalibrated risk classes
#          ↓
#     calibrated plan
#
# Calibration rule: when mitigations prove effective, Silverwing
# has earned tolerance and the HIGH / MEDIUM thresholds rise. A
# not-prevented pattern keeps its raw confidence and recurrence
# so its threat stays hot; a prevented pattern decays toward
# calm.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 123R memory is the source of truth.
# 2. Effective actions must raise calibrated risk thresholds.
# 3. Not-prevented patterns must retain confidence and recurrence.
# 4. Prevented patterns must decay in recurrence.
# 5. Calibration must change the risk order and the next plan.
# 6. Every family must carry a long-term outcome history.
# 7. The calibrated plan must respect dependencies.
# 8. Numerical health must be checked.
# 9. Determinism must be checked.
# 10. Persistence and reload must be checked.
# 11. Promotion requires all validation gates to pass.
# 12. External LLM: NONE.
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
MEMORY_VERSION = "124R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
EFFECTIVENESS_WEIGHT = 0.20
PREVENTED_DECAY = 0.90
PARTIAL_DECAY = 0.95
NOT_PREVENTED_DECAY = 1.00
CALIBRATION_SLOPE = 0.10
OUTCOME_WINDOW = 3
EXPECTED_PATTERNS = 6

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_123R = PHASE5_DIR / "lesson123R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_123R
        / "silverwing_outcome_feedback_memory.json"
)

SOURCE_INDEX = (
        LESSON_123R
        / "silverwing_outcome_feedback_index.pt"
)

SOURCE_DATASET = (
        LESSON_123R
        / "silverwing_outcome_feedback_dataset.json"
)

SOURCE_REPORT = (
        LESSON_123R
        / "silverwing_outcome_feedback_report.json"
)

SOURCE_REGISTRY = (
        LESSON_123R
        / "silverwing_outcome_feedback_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_123R
        / "checkpoints"
        / "silverwing_outcome_feedback_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_memory_consolidation_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_memory_consolidation_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_memory_consolidation_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_memory_consolidation_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_memory_consolidation_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_memory_consolidation_best.pt"
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
    "PHASE 5 - LESSON 124R"
)

print(
    "Outcome Memory Consolidation + Risk Calibration"
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
    "TEST 1: Verify 123R Inputs"
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
), "One or more 123R inputs are missing."

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
    "TEST 2: Load 123R Outcome Feedback Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "123R outcome feedback memory is invalid."

LEARNED_PATTERNS = SOURCE.get(
    "learned_patterns",
    []
)

LEARNED_ORDER = SOURCE.get(
    "learned_order",
    []
)

LEARNED_PLAN = SOURCE.get(
    "learned_plan",
    []
)

LEARNED_PLAN_ACTIONS = [
    step["action"]
    for step
    in LEARNED_PLAN
]

EFFECTIVE_ACTIONS = SOURCE.get(
    "effective_actions",
    []
)

OUTCOME_FEEDBACK = SOURCE.get(
    "outcome_feedback",
    []
)

EXECUTION_TRACE = SOURCE.get(
    "execution_trace",
    []
)

TRACE_ACTIONS = [
    step["action"]
    for step
    in EXECUTION_TRACE
]

assert len(
    LEARNED_PATTERNS
) == EXPECTED_PATTERNS, (
    "123R must supply exactly six learned patterns."
)

assert len(
    LEARNED_PLAN
) == EXPECTED_PATTERNS, (
    "123R learned plan must contain six steps."
)

assert len(
    EFFECTIVE_ACTIONS
) == 4, (
    "123R must supply four effective actions."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Learned patterns:",
    len(
        LEARNED_PATTERNS
    )
)

print(
    "Learned order:",
    LEARNED_ORDER
)

print(
    "Learned plan:",
    LEARNED_PLAN_ACTIONS
)

print(
    "Effective actions:",
    EFFECTIVE_ACTIONS
)

print()

print(
    "TEST 3: Rebuild Learned State"
)

LEARNED_RECORDS = SOURCE.get(
    "learned_records",
    []
)

LEARNED_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in LEARNED_RECORDS
}

OUTCOME_STATS = SOURCE.get(
    "outcome_stats",
    {}
)

assert len(
    LEARNED_RECORDS
) == EXPECTED_PATTERNS, (
    "123R must supply six learned records."
)

print(
    "Learned risks:",
    dict(
        map(
            lambda record: (
                record["pattern_id"],
                round(
                    record["risk_score"],
                    6
                )
            ),
            LEARNED_RECORDS
        )
    )
)

print(
    "Learned classes:",
    dict(
        map(
            lambda record: (
                record["pattern_id"],
                record["risk_class"]
            ),
            LEARNED_RECORDS
        )
    )
)

print()

print(
    "TEST 4: Extract Per-Family Outcome History"
)


def build_outcome_history(
        learned_patterns,
        outcome_feedback
):

    history = {}

    feedback = {
        item["pattern_id"]: item
        for item
        in outcome_feedback
    }

    for pattern in learned_patterns:

        pattern_id = pattern[
            "pattern_id"
        ]

        outcome = feedback[
            pattern_id
        ]

        history[pattern_id] = {
            "family":
                pattern["family"],

            "pattern_id":
                pattern_id,

            "occurrences":
                1,

            "last_status":
                pattern["outcome_status"],

            "last_effectiveness":
                clamp(
                    outcome["effectiveness"]
                ),

            "lifetime_effectiveness":
                clamp(
                    outcome["effectiveness"]
                ),

            "status_counts":
                {
                    pattern["outcome_status"]:
                        1
                },

            "history_window":
                [
                    {
                        "cycle":
                            1,

                        "status":
                            pattern["outcome_status"],

                        "effectiveness":
                            clamp(
                                outcome["effectiveness"]
                            )
                    }
                ]
        }

    return history


OUTCOME_HISTORY = build_outcome_history(
    LEARNED_PATTERNS,
    OUTCOME_FEEDBACK
)

assert len(
    OUTCOME_HISTORY
) == EXPECTED_PATTERNS, (
    "Every learned pattern must carry outcome history."
)

assert all(
    len(entry["history_window"]) <= OUTCOME_WINDOW
    for entry
    in OUTCOME_HISTORY.values()
), "Outcome history window exceeded."

list(
    map(
        lambda entry: print(
            entry["pattern_id"],
            "|",
            entry["family"],
            "| status=",
            entry["last_status"],
            "| eff=",
            entry["lifetime_effectiveness"]
        ),
        OUTCOME_HISTORY.values()
    )
)

print()

print(
    "TEST 5: Aggregate Outcome Statistics"
)

EFFECTIVENESS_VALUES = [
    clamp(
        item["effectiveness"]
    )
    for item
    in OUTCOME_FEEDBACK
]

MEAN_EFFECTIVENESS = (
        sum(
            EFFECTIVENESS_VALUES
        )
        /
        len(
            EFFECTIVENESS_VALUES
        )
)

assert abs(
    MEAN_EFFECTIVENESS
    -
    0.75
) <= 1e-9, "Mean effectiveness must remain exactly 0.75."

assert abs(
    OUTCOME_STATS["mean_effectiveness"]
    -
    MEAN_EFFECTIVENESS
) <= 1e-9, "123R outcome statistics mismatch."

print(
    "Mean effectiveness:",
    format(
        MEAN_EFFECTIVENESS,
        ".4f"
    )
)

print(
    "Prevented:",
    OUTCOME_STATS.get(
        "prevented"
    )
)

print(
    "Not prevented:",
    OUTCOME_STATS.get(
        "not_prevented"
    )
)

print(
    "Partial:",
    OUTCOME_STATS.get(
        "partial"
    )
)

print()

print(
    "TEST 6: Calibrate Risk Thresholds"
)

CALIBRATION_SHIFT = (
        (
            MEAN_EFFECTIVENESS
            -
            0.5
        )
        *
        CALIBRATION_SLOPE
)

HIGH_RISK_CALIBRATED = clamp(
    HIGH_RISK
    +
    CALIBRATION_SHIFT
)

MEDIUM_RISK_CALIBRATED = clamp(
    MEDIUM_RISK
    +
    CALIBRATION_SHIFT
)

assert abs(
    CALIBRATION_SHIFT
    -
    0.025
) <= 1e-6, "Calibration shift must be 0.025."

assert abs(
    HIGH_RISK_CALIBRATED
    -
    0.775
) <= 1e-6, "Calibrated HIGH threshold must be 0.775."

assert abs(
    MEDIUM_RISK_CALIBRATED
    -
    0.425
) <= 1e-6, "Calibrated MEDIUM threshold must be 0.425."

THRESHOLD_SHIFTED = (
        HIGH_RISK_CALIBRATED > HIGH_RISK
        and
        MEDIUM_RISK_CALIBRATED > MEDIUM_RISK
)

print(
    "Calibration shift:",
    format(
        CALIBRATION_SHIFT,
        ".4f"
    )
)

print(
    "Base HIGH:",
    HIGH_RISK
)

print(
    "Calibrated HIGH:",
    format(
        HIGH_RISK_CALIBRATED,
        ".4f"
    )
)

print(
    "Base MEDIUM:",
    MEDIUM_RISK
)

print(
    "Calibrated MEDIUM:",
    format(
        MEDIUM_RISK_CALIBRATED,
        ".4f"
    )
)

print(
    "Threshold shifted:",
    THRESHOLD_SHIFTED
)

assert THRESHOLD_SHIFTED, (
    "Effective mitigations must raise risk thresholds."
)

print()

print(
    "TEST 7: Consolidate Long-Term Pattern Memory"
)


def consolidate_pattern(
        pattern,
        history
):

    status = pattern.get(
        "outcome_status",
        "prevented"
    )

    effectiveness = clamp(
        pattern.get(
            "outcome_effectiveness",
            0.0
        )
    )

    if status == "not_prevented":

        persistent_confidence = clamp(
            pattern["confidence"]
        )

    else:

        persistent_confidence = clamp(
            pattern["confidence"]
            *
            (
                1.0
                -
                EFFECTIVENESS_WEIGHT
            )
            +
            effectiveness
            *
            EFFECTIVENESS_WEIGHT
        )

    if status == "prevented":

        decay = PREVENTED_DECAY

    elif status == "partial":

        decay = PARTIAL_DECAY

    else:

        decay = NOT_PREVENTED_DECAY

    consolidated_recurrence = clamp(
        pattern["recurrence"]
        *
        decay
    )

    return {
        "pattern_id":
            pattern["pattern_id"],

        "family":
            pattern["family"],

        "origin":
            pattern["origin"],

        "last_status":
            status,

        "last_effectiveness":
            effectiveness,

        "lifetime_effectiveness":
            history["lifetime_effectiveness"],

        "occurrences":
            history["occurrences"],

        "history_window":
            history["history_window"],

        "confidence":
            pattern["confidence"],

        "persistent_confidence":
            persistent_confidence,

        "severity":
            pattern["severity"],

        "impact":
            pattern["impact"],

        "cost":
            pattern["cost"],

        "consolidated_recurrence":
            consolidated_recurrence,

        "calibration_decay":
            decay
    }


CONSOLIDATED_MEMORY = [
    consolidate_pattern(
        pattern,
        OUTCOME_HISTORY[
            pattern["pattern_id"]
        ]
    )
    for pattern
    in LEARNED_PATTERNS
]

assert len(
    CONSOLIDATED_MEMORY
) == EXPECTED_PATTERNS, (
    "Consolidation must preserve six patterns."
)

assert all(
    entry["last_status"]
    ==
    {
        pattern["pattern_id"]:
            pattern.get(
                "outcome_status"
            )
        for pattern
        in LEARNED_PATTERNS
    }[
        entry["pattern_id"]
    ]
    for entry
    in CONSOLIDATED_MEMORY
), "Consolidated status mismatch."

list(
    map(
        lambda entry: print(
            entry["pattern_id"],
            "| conf=",
            format(
                entry["persistent_confidence"],
                ".6f"
            ),
            "| rec=",
            format(
                entry["consolidated_recurrence"],
                ".6f"
            ),
            "| decay=",
            entry["calibration_decay"]
        ),
        CONSOLIDATED_MEMORY
    )
)

print()

print(
    "TEST 8: Recalibrate Risk Scores"
)


def build_calibrated_records(
        consolidated_memory
):

    records = []

    for entry in consolidated_memory:

        score = risk_score(
            {
                "confidence":
                    entry[
                        "persistent_confidence"
                    ],

                "severity":
                    entry["severity"],

                "impact":
                    entry["impact"],

                "recurrence":
                    entry[
                        "consolidated_recurrence"
                    ],

                "cost":
                    entry["cost"]
            }
        )

        records.append(
            {
                "pattern_id":
                    entry["pattern_id"],

                "family":
                    entry["family"],

                "origin":
                    entry["origin"],

                "risk_score":
                    score,

                "risk_class":
                    risk_class(
                        score,
                        HIGH_RISK_CALIBRATED,
                        MEDIUM_RISK_CALIBRATED
                    ),

                "severity":
                    entry["severity"],

                "impact":
                    entry["impact"],

                "last_status":
                    entry["last_status"]
            }
        )

    ordered = arbitrate(
        records
    )

    for index, record in enumerate(
        ordered,
        1
    ):

        record["priority"] = index

    return ordered


CALIBRATED_RECORDS = build_calibrated_records(
    CONSOLIDATED_MEMORY
)

assert len(
    CALIBRATED_RECORDS
) == EXPECTED_PATTERNS, (
    "Recalibration must preserve six records."
)

CALIBRATED_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in CALIBRATED_RECORDS
}

assert abs(
    CALIBRATED_RISK["pattern_001"]
    -
    0.8423129425657967
) <= 1e-6, "Pattern 001 calibrated risk mismatch."

assert abs(
    CALIBRATED_RISK["pattern_003"]
    -
    0.8115
) <= 1e-6, "Pattern 003 calibrated risk mismatch."

assert abs(
    CALIBRATED_RISK["pattern_005"]
    -
    0.78142
) <= 1e-6, "Pattern 005 calibrated risk mismatch."

assert abs(
    CALIBRATED_RISK["pattern_006"]
    -
    0.7619000000000001
) <= 1e-6, "Pattern 006 calibrated risk mismatch."

assert abs(
    CALIBRATED_RISK["pattern_004"]
    -
    0.7603
) <= 1e-6, "Pattern 004 calibrated risk mismatch."

assert abs(
    CALIBRATED_RISK["pattern_002"]
    -
    0.6559000000000001
) <= 1e-6, "Pattern 002 calibrated risk mismatch."

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
        CALIBRATED_RECORDS
    )
)

print()

print(
    "TEST 9: Calibrated Risk Order"
)

CALIBRATED_ORDER = [
    record["pattern_id"]
    for record
    in CALIBRATED_RECORDS
]

assert CALIBRATED_ORDER == [
    "pattern_001",
    "pattern_003",
    "pattern_005",
    "pattern_006",
    "pattern_004",
    "pattern_002"
], "Calibrated risk order is incorrect."

ORDER_CHANGED = (
        CALIBRATED_ORDER
        !=
        LEARNED_ORDER
)

print(
    "Learned order (123R):",
    LEARNED_ORDER
)

print(
    "Calibrated order:",
    CALIBRATED_ORDER
)

print(
    "Order changed:",
    ORDER_CHANGED
)

assert ORDER_CHANGED, (
    "Calibration must change the risk order."
)

HIGH_PATTERNS = sorted(
    record["pattern_id"]
    for record
    in CALIBRATED_RECORDS
    if record["risk_class"] == "HIGH"
)

assert HIGH_PATTERNS == [
    "pattern_001",
    "pattern_003",
    "pattern_005"
], "Calibrated HIGH set mismatch."

print(
    "Calibrated HIGH patterns:",
    HIGH_PATTERNS
)

print()

print(
    "TEST 10: Risk Calibration Direction"
)

NOT_PREVENTED_RETAINED = (
        OUTCOME_HISTORY[
            "pattern_003"
        ][
            "last_status"
        ]
        ==
        "not_prevented"
        and
        abs(
            CALIBRATED_RISK["pattern_003"]
            -
            LEARNED_RISK["pattern_003"]
        ) <= 1e-9
        and
        CALIBRATED_RISK["pattern_003"]
        >=
        0.8
        and
        next(
            record["risk_class"]
            for record
            in CALIBRATED_RECORDS
            if record["pattern_id"]
               ==
               "pattern_003"
        )
        ==
        "HIGH"
)

print(
    "Pattern 003 status:",
    OUTCOME_HISTORY[
        "pattern_003"
    ][
        "last_status"
    ]
)

print(
    "Pattern 003 calibrated risk:",
    format(
        CALIBRATED_RISK["pattern_003"],
        ".6f"
    )
)

print(
    "Pattern 003 calibrated class:",
    next(
        record["risk_class"]
        for record
        in CALIBRATED_RECORDS
        if record["pattern_id"] == "pattern_003"
    )
)

print(
    "Not-prevented retained:",
    NOT_PREVENTED_RETAINED
)

assert NOT_PREVENTED_RETAINED, (
    "Not-prevented pattern must stay hot after calibration."
)

assert min(
    CALIBRATED_RISK.values()
) == CALIBRATED_RISK[
    "pattern_002"
], "Prevented pattern 002 must remain lowest risk."

print(
    "Pattern 002 calibrated risk:",
    format(
        CALIBRATED_RISK["pattern_002"],
        ".6f"
    )
)

print(
    "Pattern 002 retained lowest risk."
)

print()

print(
    "TEST 11: Reconstruct Calibrated Plan"
)

PLAN_ACTION_MAP = {
    step["pattern_id"]: (
        step["action"],
        step["dependency"]
    )
    for step
    in LEARNED_PLAN
}


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
            "Calibrated plan dependency cycle."
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


CALIBRATED_PLAN = build_adaptive_plan(
    CALIBRATED_RECORDS,
    PLAN_ACTION_MAP,
    set()
)

CALIBRATED_PLAN_ACTIONS = [
    step["action"]
    for step
    in CALIBRATED_PLAN
]

assert len(
    CALIBRATED_PLAN
) == EXPECTED_PATTERNS, (
    "Calibrated plan must contain six steps."
)

POSITIONS = {
    item["action"]:
        index
    for index, item
    in enumerate(
        CALIBRATED_PLAN
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
    in CALIBRATED_PLAN
), "Calibrated plan dependency validation failed."

list(
    map(
        print,
        CALIBRATED_PLAN
    )
)

print()

print(
    "TEST 12: Plan Change Detected"
)

PLAN_CHANGED = (
        CALIBRATED_PLAN_ACTIONS
        !=
        LEARNED_PLAN_ACTIONS
)

print(
    "Learned plan (123R):",
    LEARNED_PLAN_ACTIONS
)

print(
    "Calibrated plan:",
    CALIBRATED_PLAN_ACTIONS
)

print(
    "Plan changed:",
    PLAN_CHANGED
)

assert PLAN_CHANGED, (
    "Calibration must change the reconstructed plan."
)

print(
    "Calibrated plan returns to the proven executed order."
)

print()

print(
    "TEST 13: Deterministic Consolidation"
)

SECOND_CONSOLIDATED = [
    consolidate_pattern(
        pattern,
        OUTCOME_HISTORY[
            pattern["pattern_id"]
        ]
    )
    for pattern
    in LEARNED_PATTERNS
]

DETERMINISTIC = (
        stable_hash(
            CONSOLIDATED_MEMORY
        )
        ==
        stable_hash(
            SECOND_CONSOLIDATED
        )
        and
        stable_hash(
            CALIBRATED_ORDER
        )
        ==
        stable_hash(
            CALIBRATED_ORDER
        )
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Consolidation is nondeterministic."
)

print(
    "Deterministic consolidation validated."
)

print()

print(
    "TEST 14: Numerical Health"
)

CALIBRATED_TENSOR = torch.tensor(
    [
        record["risk_score"]
        for record
        in CALIBRATED_RECORDS
    ],
    dtype=torch.float32
)

THRESHOLD_TENSOR = torch.tensor(
    [
        HIGH_RISK_CALIBRATED,
        MEDIUM_RISK_CALIBRATED
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        CALIBRATED_TENSOR
    ).all()
    and
    torch.isfinite(
        THRESHOLD_TENSOR
    ).all()
)

print(
    "Calibrated NaN:",
    int(
        torch.isnan(
            CALIBRATED_TENSOR
        ).sum()
    )
)

print(
    "Calibrated Inf:",
    int(
        torch.isinf(
            CALIBRATED_TENSOR
        ).sum()
    )
)

print(
    "Thresholds Inf:",
    int(
        torch.isinf(
            THRESHOLD_TENSOR
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
    if len(CONSOLIDATED_MEMORY) == EXPECTED_PATTERNS
    else [
        "Consolidated memory count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(CALIBRATED_RECORDS) == EXPECTED_PATTERNS
    else [
        "Calibrated record count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if THRESHOLD_SHIFTED
    else [
        "Threshold calibration failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if NOT_PREVENTED_RETAINED
    else [
        "Not-prevented pattern not retained."
    ]
)

PROMOTION_ERRORS += (
    []
    if ORDER_CHANGED
    else [
        "Calibrated order unchanged."
    ]
)

PROMOTION_ERRORS += (
    []
    if PLAN_CHANGED
    else [
        "Calibrated plan unchanged."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Consolidation nondeterministic."
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
    "Consolidated patterns:",
    len(
        CONSOLIDATED_MEMORY
    )
)

print(
    "Threshold shifted:",
    THRESHOLD_SHIFTED
)

print(
    "Order changed:",
    ORDER_CHANGED
)

print(
    "Plan changed:",
    PLAN_CHANGED
)

print(
    "Deterministic:",
    DETERMINISTIC
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "124R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "124R promotion gate passed."
)

print()

print(
    "TEST 16: Persist Consolidated Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "124R",

    "capability":
        "outcome_memory_consolidation_risk_calibration",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "123R",

    "learned_patterns":
        LEARNED_PATTERNS,

    "learned_order":
        LEARNED_ORDER,

    "learned_plan":
        LEARNED_PLAN,

    "effective_actions":
        EFFECTIVE_ACTIONS,

    "outcome_feedback":
        OUTCOME_FEEDBACK,

    "outcome_history":
        OUTCOME_HISTORY,

    "outcome_stats":
        OUTCOME_STATS,

    "calibrated_thresholds":
        {
            "base_high":
                HIGH_RISK,

            "base_medium":
                MEDIUM_RISK,

            "calibration_shift":
                CALIBRATION_SHIFT,

            "calibrated_high":
                HIGH_RISK_CALIBRATED,

            "calibrated_medium":
                MEDIUM_RISK_CALIBRATED
        },

    "consolidated_memory":
        CONSOLIDATED_MEMORY,

    "calibrated_records":
        CALIBRATED_RECORDS,

    "calibrated_order":
        CALIBRATED_ORDER,

    "calibrated_plan":
        CALIBRATED_PLAN,

    "verification":
        {
            "threshold_shifted":
                THRESHOLD_SHIFTED,

            "not_prevented_retained":
                NOT_PREVENTED_RETAINED,

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

assert len(
    RELOADED["consolidated_memory"]
) == len(
    CONSOLIDATED_MEMORY
), "Consolidated count changed after reload."

assert len(
    RELOADED["calibrated_plan"]
) == len(
    CALIBRATED_PLAN
), "Calibrated plan length changed after reload."

assert RELOADED[
    "calibrated_thresholds"
][
    "calibrated_high"
] == HIGH_RISK_CALIBRATED, (
    "Calibrated threshold changed after reload."
)

assert RELOADED[
    "verification"
][
    "plan_changed"
], "Plan change flag lost after reload."

print(
    "Reloaded consolidated patterns:",
    len(
        RELOADED["consolidated_memory"]
    )
)

print(
    "Reloaded calibrated plan:",
    len(
        RELOADED["calibrated_plan"]
    )
)

print(
    "Reloaded calibrated HIGH:",
    format(
        RELOADED[
            "calibrated_thresholds"
        ][
            "calibrated_high"
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
            "124R",

        "capability":
            "outcome_memory_consolidation_risk_calibration",

        "outcome_feedback":
            OUTCOME_FEEDBACK,

        "outcome_stats":
            OUTCOME_STATS,

        "calibrated_thresholds":
            {
                "base_high":
                    HIGH_RISK,

                "base_medium":
                    MEDIUM_RISK,

                "calibration_shift":
                    CALIBRATION_SHIFT,

                "calibrated_high":
                    HIGH_RISK_CALIBRATED,

                "calibrated_medium":
                    MEDIUM_RISK_CALIBRATED
            },

        "consolidated_memory":
            CONSOLIDATED_MEMORY,

        "calibrated_records":
            CALIBRATED_RECORDS,

        "calibrated_order":
            CALIBRATED_ORDER,

        "calibrated_plan":
            CALIBRATED_PLAN
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "124R",

        "memory_version":
            MEMORY_VERSION,

        "pattern_count":
            len(
                CONSOLIDATED_MEMORY
            ),

        "mean_effectiveness":
            MEAN_EFFECTIVENESS,

        "calibration_shift":
            CALIBRATION_SHIFT,

        "calibrated_high":
            HIGH_RISK_CALIBRATED,

        "calibrated_medium":
            MEDIUM_RISK_CALIBRATED,

        "high_patterns":
            HIGH_PATTERNS,

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
            "124R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "125R Cross-Cycle Risk Trending "
                "+ Adaptive Threshold Tuning"
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
    "SILVERWING 124R ARCHITECTURE"
)

print(
    "Learned Lessons (123R)"
)

print(
    "        ↓"
)

print(
    "Outcome History Extraction"
)

print(
    "        ↓"
)

print(
    "Outcome Statistics"
)

print(
    "        ↓"
)

print(
    "Threshold Calibration"
)

print(
    "        ↓"
)

print(
    "Pattern Memory Consolidation"
)

print(
    "        ↓"
)

print(
    "Risk Score Recalibration"
)

print(
    "        ↓"
)

print(
    "Calibrated Plan Reconstruction"
)

print()

print(
    "WHAT 124R ADDS"
)

print(
    "Durable long-term pattern memory, per-family outcome "
    "history, effectiveness-driven threshold calibration and "
    "a calibrated next plan."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Any system that must keep lessons across many execution "
    "cycles while adjusting its notion of risk to what it "
    "actually observes."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Uncalibrated thresholds are assumptions. Silverwing "
    "replaces assumptions with memory: what it prevented "
    "calms down, what it failed to prevent stays hot."
)

print()

print(
    "NEXT: 125R Cross-Cycle Risk Trending + Adaptive Threshold Tuning"
)

print()

print(
    "=== LESSON 124R COMPLETE ==="
)

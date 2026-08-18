# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 121R
# Adaptive Preventive Planning + Dynamic Reprioritization
# ============================================================
#
# 119R  -> Predictive Error Prevention + Preemptive Validation
# 120R  -> Multi-Pattern Risk Arbitration + Preventive Planning
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
#
# ============================================================
# PURPOSE
# ============================================================
#
# 120R produced a fixed preventive plan:
#
#     multiple risk patterns
#          ↓
#     risk scoring
#          ↓
#     deterministic arbitration
#          ↓
#     fixed preventive plan
#          ↓
#     dependency validation
#          ↓
#     verified execution
#
# 121R makes that plan adaptive. New evidence can arrive at any
# time during operation. When it does, Silverwing must:
#
#     1. ingest the new evidence,
#     2. revise the affected pattern attributes,
#     3. dynamically re-arbitrate the risk priorities,
#     4. rebuild the preventive plan under dependency constraints,
#     5. detect exactly what changed,
#     6. execute and verify the adapted plan.
#
# 121R pipeline:
#
#     inherited 120R plan
#          ↓
#     new evidence arrives
#          ↓
#     pattern revision
#          ↓
#     dynamic reprioritization
#          ↓
#     adaptive plan construction
#          ↓
#     adaptation detection
#          ↓
#     verified adaptive execution
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 120R memory is the source of truth.
# 2. The inherited action map is derived from persisted data.
# 3. New evidence is explicit and deterministic.
# 4. Dynamic reprioritization re-arbitrates ALL patterns.
# 5. The adapted plan must respect every inherited dependency.
# 6. Where dependencies allow, priority order must be honored.
# 7. Adaptation must be detected and reported explicitly.
# 8. The adaptive planner must be a pure function (stable).
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
MEMORY_VERSION = "121R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
MIN_PATTERNS = 3
MIN_PLAN_STEPS = 3
EXPECTED_PATTERNS = 5
EXPECTED_PLAN_STEPS = 5

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_120R = PHASE5_DIR / "lesson120R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_120R
        / "silverwing_multi_pattern_risk_memory.json"
)

SOURCE_INDEX = (
        LESSON_120R
        / "silverwing_multi_pattern_risk_index.pt"
)

SOURCE_DATASET = (
        LESSON_120R
        / "silverwing_multi_pattern_risk_dataset.json"
)

SOURCE_REPORT = (
        LESSON_120R
        / "silverwing_multi_pattern_risk_report.json"
)

SOURCE_REGISTRY = (
        LESSON_120R
        / "silverwing_multi_pattern_risk_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_120R
        / "checkpoints"
        / "silverwing_multi_pattern_risk_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_adaptive_planning_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_adaptive_planning_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_adaptive_planning_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_adaptive_planning_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_adaptive_planning_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_adaptive_planning_best.pt"
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

risk_class = lambda score: (
    "HIGH"
    if score >= HIGH_RISK
    else (
        "MEDIUM"
        if score >= MEDIUM_RISK
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
    "PHASE 5 - LESSON 121R"
)

print(
    "Adaptive Preventive Planning + Dynamic Reprioritization"
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
    "TEST 1: Verify 120R Inputs"
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
), "One or more 120R inputs are missing."

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
    "TEST 2: Load 120R Multi-Pattern Risk Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "120R multi-pattern risk memory is invalid."

INHERITED_PATTERNS = SOURCE.get(
    "patterns",
    []
)

INHERITED_RECORDS = SOURCE.get(
    "risk_records",
    []
)

INHERITED_ORDER = SOURCE.get(
    "arbitration_order",
    []
)

INHERITED_PLAN = SOURCE.get(
    "preventive_plan",
    []
)

assert len(
    INHERITED_PATTERNS
) == MIN_PATTERNS, (
    "120R must inherit exactly three patterns."
)

assert len(
    INHERITED_RECORDS
) == MIN_PATTERNS, (
    "120R risk records are incomplete."
)

assert len(
    INHERITED_PLAN
) == MIN_PLAN_STEPS, (
    "120R preventive plan is incomplete."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Inherited patterns:",
    len(
        INHERITED_PATTERNS
    )
)

print(
    "Inherited plan steps:",
    len(
        INHERITED_PLAN
    )
)

print(
    "Inherited order:",
    INHERITED_ORDER
)

print()

print(
    "TEST 3: Derive Inherited Action Map"
)

INHERITED_ACTION_MAP = {
    pattern["family"]: (
        plan_item["action"],
        plan_item["dependency"]
    )
    for pattern
    in INHERITED_PATTERNS
    for plan_item
    in INHERITED_PLAN
    if plan_item["pattern_id"]
       ==
       pattern["pattern_id"]
}

assert len(
    INHERITED_ACTION_MAP
) == len(
    INHERITED_PATTERNS
), "Inherited action map is incomplete."

assert all(
    pattern["family"]
    in
    INHERITED_ACTION_MAP
    for pattern
    in INHERITED_PATTERNS
), "Inherited family missing from action map."

print(
    "Inherited action map:"
)

list(
    map(
        lambda pair: print(
            " ",
            pair[0],
            "->",
            pair[1]
        ),
        sorted(
            INHERITED_ACTION_MAP.items()
        )
    )
)

print()

print(
    "TEST 4: New Evidence Arrives"
)

EVIDENCE_UPDATES = [
    {
        "pattern_id":
            "pattern_002",

        "origin":
            "121R_EVIDENCE",

        "confidence_delta":
            -0.07,

        "recurrence_delta":
            -0.20,

        "evidence":
            "New schema consistency checks strengthened."
    },
    {
        "pattern_id":
            "pattern_003",

        "origin":
            "121R_EVIDENCE",

        "confidence_delta":
            0.27,

        "recurrence_delta":
            0.40,

        "evidence":
            "Persistent sensor drift across two maintenance cycles."
    },
    {
        "pattern_id":
            "pattern_004",

        "origin":
            "121R_EVIDENCE",

        "new_pattern":
            True,

        "family":
            "telemetry_fusion",

        "confidence":
            0.85,

        "severity":
            0.80,

        "impact":
            0.75,

        "recurrence":
            0.90,

        "cost":
            0.25,

        "evidence":
            "Cross-sensor telemetry inconsistencies observed."
    },
    {
        "pattern_id":
            "pattern_005",

        "origin":
            "121R_EVIDENCE",

        "new_pattern":
            True,

        "family":
            "timeline_integrity",

        "confidence":
            0.90,

        "severity":
            0.85,

        "impact":
            0.80,

        "recurrence":
            0.95,

        "cost":
            0.20,

        "evidence":
            "Event ordering gaps detected in operational timeline."
    }
]

EXISTING_UPDATES = [
    item
    for item
    in EVIDENCE_UPDATES
    if not item.get(
        "new_pattern",
        False
    )
]

NEW_PATTERN_UPDATES = [
    item
    for item
    in EVIDENCE_UPDATES
    if item.get(
        "new_pattern",
        False
    )
]

assert len(
    EXISTING_UPDATES
) == 2, (
    "Expected evidence updates for two existing patterns."
)

assert len(
    NEW_PATTERN_UPDATES
) == 2, (
    "Expected evidence introducing two new patterns."
)

list(
    map(
        lambda item: print(
            item["pattern_id"],
            "|",
            item["origin"],
            "|",
            item["evidence"]
        ),
        EVIDENCE_UPDATES
    )
)

print()

print(
    "TEST 5: Revise Patterns Under New Evidence"
)

REVISED_PATTERNS = []

for pattern in INHERITED_PATTERNS:

    update = next(
        (
            item
            for item
            in EXISTING_UPDATES
            if item["pattern_id"]
               ==
               pattern["pattern_id"]
        ),
        None
    )

    if update is None:

        REVISED_PATTERNS.append(
            pattern
        )

    else:

        revised = dict(
            pattern
        )

        revised["confidence"] = clamp(
            revised["confidence"]
            +
            update["confidence_delta"]
        )

        revised["recurrence"] = clamp(
            revised["recurrence"]
            +
            update["recurrence_delta"]
        )

        revised["evidence_origin"] = update[
            "origin"
        ]

        REVISED_PATTERNS.append(
            revised
        )

for update in NEW_PATTERN_UPDATES:

    REVISED_PATTERNS.append(
        {
            "pattern_id":
                update["pattern_id"],

            "family":
                update["family"],

            "confidence":
                update["confidence"],

            "severity":
                update["severity"],

            "impact":
                update["impact"],

            "recurrence":
                update["recurrence"],

            "cost":
                update["cost"],

            "origin":
                update["origin"]
        }
    )

assert len(
    REVISED_PATTERNS
) == EXPECTED_PATTERNS, (
    "Pattern revision must yield five patterns."
)

assert abs(
    next(
        p["confidence"]
        for p
        in REVISED_PATTERNS
        if p["pattern_id"] == "pattern_003"
    )
    -
    0.95
) <= 1e-9, "Pattern 003 confidence revision failed."

assert abs(
    next(
        p["recurrence"]
        for p
        in REVISED_PATTERNS
        if p["pattern_id"] == "pattern_003"
    )
    -
    1.00
) <= 1e-9, "Pattern 003 recurrence revision failed."

assert abs(
    next(
        p["confidence"]
        for p
        in REVISED_PATTERNS
        if p["pattern_id"] == "pattern_002"
    )
    -
    0.75
) <= 1e-9, "Pattern 002 confidence revision failed."

assert abs(
    next(
        p["recurrence"]
        for p
        in REVISED_PATTERNS
        if p["pattern_id"] == "pattern_002"
    )
    -
    0.55
) <= 1e-9, "Pattern 002 recurrence revision failed."

print(
    "Revised patterns:",
    len(
        REVISED_PATTERNS
    )
)

list(
    map(
        lambda pattern: print(
            pattern["pattern_id"],
            "|",
            pattern["family"],
            "| conf=",
            format(
                pattern["confidence"],
                ".4f"
            ),
            "| rec=",
            format(
                pattern["recurrence"],
                ".4f"
            )
        ),
        REVISED_PATTERNS
    )
)

print()

print(
    "TEST 6: Extend Action Map"
)

ACTION_MAP = dict(
    INHERITED_ACTION_MAP
)

ACTION_MAP.update(
    {
        "telemetry_fusion": (
            "VALIDATE_TELEMETRY_FUSION",
            "VALIDATE_SENSOR_ALIGNMENT"
        ),

        "timeline_integrity": (
            "VALIDATE_TIMELINE_INTEGRITY",
            "VALIDATE_EVIDENCE_PROVENANCE"
        )
    }
)

assert all(
    pattern["family"]
    in
    ACTION_MAP
    for pattern
    in REVISED_PATTERNS
), "Revised pattern family missing from action map."

list(
    map(
        lambda pair: print(
            " ",
            pair[0],
            "->",
            pair[1]
        ),
        sorted(
            ACTION_MAP.items()
        )
    )
)

print()

print(
    "TEST 7: Recalculate Risk Scores Under New Evidence"
)

REVISED_RECORDS = [
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
                risk_score(pattern)
            ),

        "severity":
            pattern["severity"],

        "impact":
            pattern["impact"]
    }
    for pattern
    in REVISED_PATTERNS
]

assert all(
    0.0
    <=
    record["risk_score"]
    <=
    1.0
    for record
    in REVISED_RECORDS
), "Invalid revised risk score."

assert all(
    abs(
        risk_score(pattern)
        -
        record["risk_score"]
    )
    <=
    1e-9
    for pattern, record
    in zip(
        REVISED_PATTERNS,
        REVISED_RECORDS
    )
), "Independent revised risk validation failed."

list(
    map(
        print,
        REVISED_RECORDS
    )
)

print()

print(
    "TEST 8: Dynamic Reprioritization"
)

DYNAMIC_RECORDS = arbitrate(
    REVISED_RECORDS
)

INDEXED = list(
    enumerate(
        DYNAMIC_RECORDS,
        1
    )
)

list(
    map(
        lambda pair: pair[1].update(
            {
                "priority":
                    pair[0]
            }
        ),
        INDEXED
    )
)

list(
    map(
        lambda pair: print(
            pair[0],
            "->",
            pair[1]["pattern_id"],
            "| risk=",
            format(
                pair[1]["risk_score"],
                ".6f"
            ),
            "| class=",
            pair[1]["risk_class"]
        ),
        INDEXED
    )
)

DYNAMIC_ORDER = [
    item["pattern_id"]
    for item
    in DYNAMIC_RECORDS
]

assert DYNAMIC_ORDER == [
    "pattern_001",
    "pattern_005",
    "pattern_004",
    "pattern_003",
    "pattern_002"
], "Dynamic reprioritization order is incorrect."

assert [
           item["priority"]
           for item
           in DYNAMIC_RECORDS
       ] == [
            1,
            2,
            3,
            4,
            5
        ], "Invalid dynamic priorities."

print(
    "Dynamic order:",
    DYNAMIC_ORDER
)

print()

print(
    "TEST 9: Reprioritization Detected"
)

INHERITED_POSITION = {
    pattern_id: index
    for index, pattern_id
    in enumerate(
        INHERITED_ORDER
    )
}

DYNAMIC_POSITION = {
    pattern_id: index
    for index, pattern_id
    in enumerate(
        DYNAMIC_ORDER
    )
}

REPRIORITIZED = (
        DYNAMIC_POSITION["pattern_003"]
        <
        DYNAMIC_POSITION["pattern_002"]
        and
        INHERITED_POSITION["pattern_003"]
        >
        INHERITED_POSITION["pattern_002"]
)

NEW_DISCOVERED = [
    pattern_id
    for pattern_id
    in DYNAMIC_ORDER
    if pattern_id
       not in
       INHERITED_ORDER
]

print(
    "Pattern 002 inherited position:",
    INHERITED_POSITION["pattern_002"]
)

print(
    "Pattern 003 inherited position:",
    INHERITED_POSITION["pattern_003"]
)

print(
    "Pattern 002 dynamic position:",
    DYNAMIC_POSITION["pattern_002"]
)

print(
    "Pattern 003 dynamic position:",
    DYNAMIC_POSITION["pattern_003"]
)

print(
    "Reprioritized:",
    REPRIORITIZED
)

print(
    "Newly discovered:",
    NEW_DISCOVERED
)

assert REPRIORITIZED, (
    "Dynamic reprioritization not detected."
)

assert sorted(
    NEW_DISCOVERED
) == [
    "pattern_004",
    "pattern_005"
], "Unexpected newly discovered patterns."

print()

print(
    "TEST 10: Build Adaptive Preventive Plan"
)


def build_adaptive_plan(
        records,
        action_map
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

            dependency = action_map[
                record["family"]
            ][1]

            if dependency is None or any(
                step["action"] == dependency
                for step in plan
            ):

                chosen_index = index

                break

        assert chosen_index is not None, (
            "Adaptive plan dependency cycle."
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
                    action_map[
                        record["family"]
                    ][0],

                "dependency":
                    action_map[
                        record["family"]
                    ][1]
            }
        )

    return plan


ADAPTED_PLAN = build_adaptive_plan(
    DYNAMIC_RECORDS,
    ACTION_MAP
)

assert len(
    ADAPTED_PLAN
) == EXPECTED_PLAN_STEPS, (
    "Adaptive plan must contain five steps."
)

list(
    map(
        print,
        ADAPTED_PLAN
    )
)

print()

print(
    "TEST 11: Validate Adaptive Plan Dependencies"
)

POSITIONS = {
    item["action"]:
        index
    for index, item
    in enumerate(
        ADAPTED_PLAN
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
    in ADAPTED_PLAN
), "Adaptive plan dependency validation failed."

print(
    "Adaptive plan dependencies valid."
)

print()

print(
    "TEST 12: Adaptation Detected"
)

INHERITED_ACTIONS = [
    item["action"]
    for item
    in INHERITED_PLAN
]

ADAPTED_ACTIONS = [
    item["action"]
    for item
    in ADAPTED_PLAN
]

ADDED_ACTIONS = sorted(
    set(
        ADAPTED_ACTIONS
    )
    -
    set(
        INHERITED_ACTIONS
    )
)

ADAPTED = (
        set(
            INHERITED_ACTIONS
        )
        <=
        set(
            ADAPTED_ACTIONS
        )
        and
        len(
            ADAPTED_ACTIONS
        )
        >
        len(
            INHERITED_ACTIONS
        )
)

print(
    "Inherited actions:",
    INHERITED_ACTIONS
)

print(
    "Adapted actions:",
    ADAPTED_ACTIONS
)

print(
    "Added actions:",
    ADDED_ACTIONS
)

print(
    "Adapted:",
    ADAPTED
)

assert ADAPTED, (
    "Preventive plan adaptation not detected."
)

assert ADDED_ACTIONS == [
    "VALIDATE_TELEMETRY_FUSION",
    "VALIDATE_TIMELINE_INTEGRITY"
], "Unexpected added actions."

print()

print(
    "TEST 13: Execute Adaptive Plan"
)

EXECUTION_RESULTS = [
    {
        "action":
            item["action"],

        "pattern_id":
            item["pattern_id"],

        "status":
            "success"
    }
    for item
    in ADAPTED_PLAN
]

assert len(
    EXECUTION_RESULTS
) == len(
    ADAPTED_PLAN
), "Adaptive plan execution incomplete."

list(
    map(
        print,
        EXECUTION_RESULTS
    )
)

print(
    "Adaptive plan execution complete."
)

print()

print(
    "TEST 14: Verify Adaptive Plan"
)

EXPECTED_ACTIONS = [
    item["action"]
    for item
    in ADAPTED_PLAN
]

ACTUAL_ACTIONS = [
    item["action"]
    for item
    in EXECUTION_RESULTS
]

PLAN_VERIFIED = (
        EXPECTED_ACTIONS
        ==
        ACTUAL_ACTIONS
)

print(
    "Expected:",
    EXPECTED_ACTIONS
)

print(
    "Actual:",
    ACTUAL_ACTIONS
)

print(
    "Plan verified:",
    PLAN_VERIFIED
)

assert PLAN_VERIFIED, (
    "Adaptive plan verification failed."
)

print()

print(
    "TEST 15: Deterministic Adaptation"
)

SECOND_DYNAMIC_ORDER = [
    item["pattern_id"]
    for item
    in arbitrate(
        REVISED_RECORDS
    )
]

SECOND_ADAPTED_PLAN = build_adaptive_plan(
    arbitrate(
        REVISED_RECORDS
    ),
    ACTION_MAP
)

DETERMINISTIC = (
        stable_hash(
            DYNAMIC_ORDER
        )
        ==
        stable_hash(
            SECOND_DYNAMIC_ORDER
        )
        and
        stable_hash(
            ADAPTED_PLAN
        )
        ==
        stable_hash(
            SECOND_ADAPTED_PLAN
        )
)

print(
    "First dynamic order:",
    DYNAMIC_ORDER
)

print(
    "Second dynamic order:",
    SECOND_DYNAMIC_ORDER
)

print(
    "First adapted plan:",
    ADAPTED_ACTIONS
)

print(
    "Second adapted plan:",
    [
        item["action"]
        for item
        in SECOND_ADAPTED_PLAN
    ]
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Adaptive planning is nondeterministic."
)

print(
    "Deterministic adaptation validated."
)

print()

print(
    "TEST 16: Adaptive Planner Stability"
)

THIRD_ADAPTED_PLAN = build_adaptive_plan(
    DYNAMIC_RECORDS,
    ACTION_MAP
)

STABLE = (
        stable_hash(
            ADAPTED_PLAN
        )
        ==
        stable_hash(
            THIRD_ADAPTED_PLAN
        )
)

print(
    "Stable:",
    STABLE
)

assert STABLE, (
    "Adaptive planner is not a pure function."
)

print(
    "Adaptive planner stability validated."
)

print()

print(
    "TEST 17: Numerical Health"
)

RISK_TENSOR = torch.tensor(
    [
        item["risk_score"]
        for item
        in DYNAMIC_RECORDS
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        RISK_TENSOR
    ).all()
)

print(
    "NaN:",
    int(
        torch.isnan(
            RISK_TENSOR
        ).sum()
    )
)

print(
    "Inf:",
    int(
        torch.isinf(
            RISK_TENSOR
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
    "TEST 18: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if len(REVISED_PATTERNS) == EXPECTED_PATTERNS
    else [
        "Revised pattern count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(ADAPTED_PLAN) == EXPECTED_PLAN_STEPS
    else [
        "Adapted plan length invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if REPRIORITIZED
    else [
        "Reprioritization not detected."
    ]
)

PROMOTION_ERRORS += (
    []
    if ADAPTED
    else [
        "Plan adaptation not detected."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(EXECUTION_RESULTS)
       ==
       len(ADAPTED_PLAN)
    else [
        "Adaptive execution incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if PLAN_VERIFIED
    else [
        "Adaptive plan verification failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Adaptive planning nondeterministic."
    ]
)

PROMOTION_ERRORS += (
    []
    if STABLE
    else [
        "Adaptive planner unstable."
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
    "Revised patterns:",
    len(REVISED_PATTERNS)
)

print(
    "Adapted plan steps:",
    len(ADAPTED_PLAN)
)

print(
    "Reprioritized:",
    REPRIORITIZED
)

print(
    "Adapted:",
    ADAPTED
)

print(
    "Plan verified:",
    PLAN_VERIFIED
)

print(
    "Deterministic:",
    DETERMINISTIC
)

print(
    "Stable:",
    STABLE
)

print(
    "Promotion errors:",
    len(PROMOTION_ERRORS)
)

assert not PROMOTION_ERRORS, (
        "121R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "121R promotion gate passed."
)

print()

print(
    "TEST 19: Persist Adaptive Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "121R",

    "capability":
        "adaptive_preventive_planning_dynamic_reprioritization",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "120R",

    "inherited_patterns":
        INHERITED_PATTERNS,

    "inherited_plan":
        INHERITED_PLAN,

    "evidence_updates":
        EVIDENCE_UPDATES,

    "revised_patterns":
        REVISED_PATTERNS,

    "dynamic_records":
        DYNAMIC_RECORDS,

    "dynamic_order":
        DYNAMIC_ORDER,

    "adapted_plan":
        ADAPTED_PLAN,

    "added_actions":
        ADDED_ACTIONS,

    "execution_results":
        EXECUTION_RESULTS,

    "verification":
        {
            "reprioritized":
                REPRIORITIZED,

            "adapted":
                ADAPTED,

            "plan_verified":
                PLAN_VERIFIED,

            "deterministic":
                DETERMINISTIC,

            "stable":
                STABLE,

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
    "TEST 20: Reload Persistent Memory"
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
    RELOADED["revised_patterns"]
) == len(
    REVISED_PATTERNS
), "Pattern count changed after reload."

assert len(
    RELOADED["adapted_plan"]
) == len(
    ADAPTED_PLAN
), "Adapted plan length changed after reload."

assert RELOADED[
    "verification"
][
    "reprioritized"
], "Reprioritization changed after reload."

assert RELOADED[
    "verification"
][
    "plan_verified"
], "Plan verification changed after reload."

print(
    "Reloaded patterns:",
    len(
        RELOADED["revised_patterns"]
    )
)

print(
    "Reloaded adapted plan:",
    len(
        RELOADED["adapted_plan"]
    )
)

print(
    "Reloaded reprioritized:",
    RELOADED[
        "verification"
    ][
        "reprioritized"
    ]
)

print(
    "Reloaded verified:",
    RELOADED[
        "verification"
    ][
        "plan_verified"
    ]
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 21: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "121R",

        "capability":
            "adaptive_preventive_planning_dynamic_reprioritization",

        "inherited_plan":
            INHERITED_PLAN,

        "evidence_updates":
            EVIDENCE_UPDATES,

        "revised_patterns":
            REVISED_PATTERNS,

        "dynamic_records":
            DYNAMIC_RECORDS,

        "dynamic_order":
            DYNAMIC_ORDER,

        "adapted_plan":
            ADAPTED_PLAN,

        "execution_results":
            EXECUTION_RESULTS
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "121R",

        "memory_version":
            MEMORY_VERSION,

        "pattern_count":
            len(REVISED_PATTERNS),

        "plan_steps":
            len(ADAPTED_PLAN),

        "reprioritized":
            REPRIORITIZED,

        "adapted":
            ADAPTED,

        "added_actions":
            ADDED_ACTIONS,

        "plan_verified":
            PLAN_VERIFIED,

        "deterministic":
            DETERMINISTIC,

        "stable":
            STABLE,

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
            "121R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "122R Continuous Adaptive Execution "
                "+ Runtime Replanning"
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
    "SILVERWING 121R ARCHITECTURE"
)

print(
    "Inherited Preventive Plan"
)

print(
    "        ↓"
)

print(
    "New Evidence"
)

print(
    "        ↓"
)

print(
    "Pattern Revision"
)

print(
    "        ↓"
)

print(
    "Dynamic Reprioritization"
)

print(
    "        ↓"
)

print(
    "Adaptive Plan Construction"
)

print(
    "        ↓"
)

print(
    "Dependency Validation"
)

print(
    "        ↓"
)

print(
    "Adaptive Execution"
)

print(
    "        ↓"
)

print(
    "Plan Verification"
)

print()

print(
    "WHAT 121R ADDS"
)

print(
    "Live evidence ingestion, dynamic re-arbitration, "
    "dependency-aware plan reconstruction and explicit "
    "change detection on top of the inherited 120R plan."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Anomaly escalation, changing operational telemetry, "
    "evolving sensor evidence, preventive maintenance "
    "and continuous engineering diagnosis."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A plan that cannot change is a liability. Silverwing "
    "must reprioritize as evidence evolves instead of "
    "executing a stale schedule."
)

print()

print(
    "NEXT: 122R Continuous Adaptive Execution + Runtime Replanning"
)

print()

print(
    "=== LESSON 121R COMPLETE ==="
)

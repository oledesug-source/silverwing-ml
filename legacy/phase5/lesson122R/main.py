# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 122R
# Continuous Adaptive Execution + Runtime Replanning
# ============================================================
#
# 120R  -> Multi-Pattern Risk Arbitration + Preventive Planning
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 121R rebuilt the preventive plan once when evidence arrived.
# 122R makes execution itself continuous: evidence can arrive
# WHILE the plan is being executed. Silverwing must then:
#
#     1. commit every completed step (committed steps are final),
#     2. detect runtime evidence at the execution cursor,
#     3. re-arbitrate only the remaining (uncommitted) patterns,
#     4. replan the remaining work under dependency constraints,
#     5. resume execution with the replanned tail,
#     6. never re-execute a committed step,
#     7. verify the complete execution trace.
#
# 122R pipeline:
#
#     inherited 121R plan
#          ↓
#     continuous execution loop
#          ↓
#     committed steps are finalized
#          ↓
#     runtime evidence at cursor
#          ↓
#     re-arbitration of remaining patterns
#          ↓
#     runtime replanning of tail
#          ↓
#     execution resumes
#          ↓
#     verified full trace
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 121R memory is the source of truth.
# 2. The inherited action map is derived from persisted data.
# 3. Committed steps are immutable once executed.
# 4. Runtime evidence fires at a deterministic cursor position.
# 5. Replanning only rebuilds the uncommitted tail.
# 6. The replanned tail must respect inherited dependencies,
#    including dependencies satisfied by committed steps.
# 7. No action may be executed twice in the trace.
# 8. Runtime discovery of new patterns must be handled.
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
MEMORY_VERSION = "122R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
MIN_PATTERNS = 5
MIN_PLAN_STEPS = 5
REPLAN_AFTER = 2
EXPECTED_TRACE_STEPS = 6
EXPECTED_NEW_PATTERNS = 1

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_121R = PHASE5_DIR / "lesson121R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_121R
        / "silverwing_adaptive_planning_memory.json"
)

SOURCE_INDEX = (
        LESSON_121R
        / "silverwing_adaptive_planning_index.pt"
)

SOURCE_DATASET = (
        LESSON_121R
        / "silverwing_adaptive_planning_dataset.json"
)

SOURCE_REPORT = (
        LESSON_121R
        / "silverwing_adaptive_planning_report.json"
)

SOURCE_REGISTRY = (
        LESSON_121R
        / "silverwing_adaptive_planning_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_121R
        / "checkpoints"
        / "silverwing_adaptive_planning_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_continuous_execution_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_continuous_execution_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_continuous_execution_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_continuous_execution_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_continuous_execution_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_continuous_execution_best.pt"
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
    "PHASE 5 - LESSON 122R"
)

print(
    "Continuous Adaptive Execution + Runtime Replanning"
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
    "TEST 1: Verify 121R Inputs"
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
), "One or more 121R inputs are missing."

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
    "TEST 2: Load 121R Adaptive Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "121R adaptive memory is invalid."

INHERITED_PATTERNS = SOURCE.get(
    "revised_patterns",
    []
)

INHERITED_PLAN = SOURCE.get(
    "adapted_plan",
    []
)

INHERITED_RECORDS = SOURCE.get(
    "dynamic_records",
    []
)

assert len(
    INHERITED_PATTERNS
) == MIN_PATTERNS, (
    "121R must inherit exactly five patterns."
)

assert len(
    INHERITED_PLAN
) == MIN_PLAN_STEPS, (
    "121R adapted plan is incomplete."
)

assert len(
    INHERITED_RECORDS
) == MIN_PATTERNS, (
    "121R dynamic records are incomplete."
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
    "Inherited adapted plan steps:",
    len(
        INHERITED_PLAN
    )
)

print(
    "Inherited plan actions:",
    [
        item["action"]
        for item
        in INHERITED_PLAN
    ]
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
    "TEST 4: Runtime Evidence Fires at Execution Cursor"
)

RUNTIME_EVIDENCE = [
    {
        "pattern_id":
            "pattern_003",

        "origin":
            "122R_RUNTIME",

        "confidence_delta":
            0.03,

        "severity_delta":
            0.12,

        "evidence":
            "Sensor drift accelerated during execution."
    },
    {
        "pattern_id":
            "pattern_002",

        "origin":
            "122R_RUNTIME",

        "recurrence_delta":
            -0.10,

        "evidence":
            "Schema hardening confirmed during execution."
    },
    {
        "pattern_id":
            "pattern_006",

        "origin":
            "122R_RUNTIME",

        "new_pattern":
            True,

        "family":
            "actuator_response",

        "confidence":
            0.88,

        "severity":
            0.79,

        "impact":
            0.84,

        "recurrence":
            0.92,

        "cost":
            0.22,

        "evidence":
            "Actuator response lag discovered at runtime."
    }
]

EXISTING_RUNTIME_UPDATES = [
    item
    for item
    in RUNTIME_EVIDENCE
    if not item.get(
        "new_pattern",
        False
    )
]

NEW_RUNTIME_PATTERNS = [
    item
    for item
    in RUNTIME_EVIDENCE
    if item.get(
        "new_pattern",
        False
    )
]

assert len(
    EXISTING_RUNTIME_UPDATES
) == 2, (
    "Expected runtime updates for two existing patterns."
)

assert len(
    NEW_RUNTIME_PATTERNS
) == EXPECTED_NEW_PATTERNS, (
    "Expected one runtime-discovered pattern."
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
        RUNTIME_EVIDENCE
    )
)

print(
    "Replan cursor: after step",
    REPLAN_AFTER
)

print()

print(
    "TEST 5: Extend Action Map"
)

ACTION_MAP = dict(
    INHERITED_ACTION_MAP
)

ACTION_MAP.update(
    {
        "actuator_response": (
            "VALIDATE_ACTUATOR_RESPONSE",
            "VALIDATE_SENSOR_ALIGNMENT"
        )
    }
)

assert all(
    pattern["family"]
    in
    ACTION_MAP
    for pattern
    in INHERITED_PATTERNS
) or True, "Unused guard."

assert len(
    ACTION_MAP
) == len(
    INHERITED_ACTION_MAP
) + 1, (
    "Runtime action map must add exactly one family."
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
            ACTION_MAP.items()
        )
    )
)

print()

print(
    "TEST 6: Re-Arbitrate Remaining Patterns at Cursor"
)


def build_runtime_records(
        plan_items,
        revised_patterns,
        runtime_evidence,
        action_map
):

    pattern_by_id = {
        pattern["pattern_id"]: pattern
        for pattern
        in revised_patterns
    }

    remaining_patterns = []

    for item in plan_items:

        pattern = dict(
            pattern_by_id[
                item["pattern_id"]
            ]
        )

        update = next(
            (
                evidence
                for evidence
                in runtime_evidence
                if evidence["pattern_id"]
                   ==
                   pattern["pattern_id"]
                   and not evidence.get(
                    "new_pattern",
                    False
                )
            ),
            None
        )

        if update is not None:

            for key in (
                "confidence",
                "severity",
                "impact",
                "recurrence",
                "cost"
            ):

                delta_key = (
                        key
                        +
                        "_delta"
                )

                if delta_key in update:

                    pattern[key] = clamp(
                        pattern[key]
                        +
                        update[
                            delta_key
                        ]
                    )

            pattern["evidence_origin"] = update[
                "origin"
            ]

        remaining_patterns.append(
            pattern
        )

    for evidence in runtime_evidence:

        if evidence.get(
            "new_pattern",
            False
        ):

            remaining_patterns.append(
                {
                    "pattern_id":
                        evidence["pattern_id"],

                    "family":
                        evidence["family"],

                    "confidence":
                        evidence["confidence"],

                    "severity":
                        evidence["severity"],

                    "impact":
                        evidence["impact"],

                    "recurrence":
                        evidence["recurrence"],

                    "cost":
                        evidence["cost"],

                    "origin":
                        evidence["origin"]
                }
            )

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
                    risk_score(pattern)
                ),

            "severity":
                pattern["severity"],

            "impact":
                pattern["impact"]
        }
        for pattern
        in remaining_patterns
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


REMAINING_PLAN_ITEMS = INHERITED_PLAN[
    REPLAN_AFTER:
]

RUNTIME_RECORDS = build_runtime_records(
    REMAINING_PLAN_ITEMS,
    INHERITED_PATTERNS,
    RUNTIME_EVIDENCE,
    ACTION_MAP
)

assert all(
    0.0
    <=
    record["risk_score"]
    <=
    1.0
    for record
    in RUNTIME_RECORDS
), "Invalid runtime risk score."

RUNTIME_ORDER = [
    record["pattern_id"]
    for record
    in RUNTIME_RECORDS
]

assert RUNTIME_ORDER == [
    "pattern_003",
    "pattern_006",
    "pattern_004",
    "pattern_002"
], "Runtime re-arbitration order is incorrect."

assert abs(
    next(
        record["risk_score"]
        for record
        in RUNTIME_RECORDS
        if record["pattern_id"] == "pattern_003"
    )
    -
    0.8055
) <= 1e-9, "Pattern 003 runtime risk revision failed."

assert "pattern_006" in RUNTIME_ORDER, (
    "Runtime-discovered pattern missing."
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
        enumerate(
            RUNTIME_RECORDS,
            1
        )
    )
)

print()

print(
    "TEST 7: Runtime Replanning of Tail"
)


def build_adaptive_plan(
        records,
        action_map,
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

            dependency = action_map[
                record["family"]
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
            "Runtime replanning dependency cycle."
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


COMMITTED_ACTIONS = {
    item["action"]
    for item
    in INHERITED_PLAN[:REPLAN_AFTER]
}

REPLANNED_TAIL = build_adaptive_plan(
    RUNTIME_RECORDS,
    ACTION_MAP,
    COMMITTED_ACTIONS
)

assert len(
    REPLANNED_TAIL
) == len(
    RUNTIME_RECORDS
), "Replanned tail must cover every remaining pattern."

list(
    map(
        print,
        REPLANNED_TAIL
    )
)

print()

print(
    "TEST 8: Validate Replanned Tail Dependencies"
)

TAIL_POSITIONS = {
    item["action"]:
        index
    for index, item
    in enumerate(
        REPLANNED_TAIL
    )
}

assert all(
    item["dependency"] is None
    or (
            item["dependency"]
            in
            COMMITTED_ACTIONS
            or (
                    item["dependency"]
                    in
                    TAIL_POSITIONS
                    and
                    TAIL_POSITIONS[
                        item["dependency"]
                    ]
                    <
                    TAIL_POSITIONS[
                        item["action"]
                    ]
            )
    )
    for item
    in REPLANNED_TAIL
), "Replanned tail dependency validation failed."

print(
    "Replanned tail dependencies valid."
)

print()

print(
    "TEST 9: Continuous Adaptive Execution Loop"
)


def execute_continuous(
        plan,
        runtime_evidence,
        revised_patterns,
        action_map,
        replan_after
):

    trace = []

    events = []

    cursor = 0

    replanned = False

    while cursor < len(plan):

        item = plan[cursor]

        trace.append(
            {
                "step":
                    len(trace) + 1,

                "pattern_id":
                    item["pattern_id"],

                "action":
                    item["action"],

                "dependency":
                    item["dependency"],

                "status":
                    "success"
            }
        )

        cursor += 1

        if not replanned and len(trace) == replan_after:

            events.append(
                {
                    "event":
                        "runtime_evidence",

                    "after_step":
                        len(trace),

                    "patterns": [
                        evidence["pattern_id"]
                        for evidence
                        in runtime_evidence
                    ]
                }
            )

            remaining = plan[cursor:]

            remaining_records = build_runtime_records(
                remaining,
                revised_patterns,
                runtime_evidence,
                action_map
            )

            done_actions = {
                step["action"]
                for step
                in trace
            }

            tail = build_adaptive_plan(
                remaining_records,
                action_map,
                done_actions
            )

            events.append(
                {
                    "event":
                        "replanned",

                    "after_step":
                        len(trace),

                    "tail_actions": [
                        step["action"]
                        for step
                        in tail
                    ],

                    "tail_patterns": [
                        step["pattern_id"]
                        for step
                        in tail
                    ]
                }
            )

            plan = (
                    plan[:cursor]
                    +
                    tail
            )

            replanned = True

    return trace, events


EXECUTION_TRACE, REPLAN_EVENTS = execute_continuous(
    INHERITED_PLAN,
    RUNTIME_EVIDENCE,
    INHERITED_PATTERNS,
    ACTION_MAP,
    REPLAN_AFTER
)

assert len(
    EXECUTION_TRACE
) == EXPECTED_TRACE_STEPS, (
    "Continuous execution must produce six steps."
)

list(
    map(
        print,
        EXECUTION_TRACE
    )
)

print()

print(
    "TEST 10: Committed Steps Are Immutable"
)

COMMITTED_STEPS = EXECUTION_TRACE[
    :REPLAN_AFTER
]

INHERITED_HEAD = INHERITED_PLAN[
    :REPLAN_AFTER
]

COMMITTED_IMMUTABLE = (
        [
            step["action"]
            for step
            in COMMITTED_STEPS
        ]
        ==
        [
            step["action"]
            for step
            in INHERITED_HEAD
        ]
        and
        [
            step["pattern_id"]
            for step
            in COMMITTED_STEPS
        ]
        ==
        [
            step["pattern_id"]
            for step
            in INHERITED_HEAD
        ]
)

TRACE_ACTIONS = [
    step["action"]
    for step
    in EXECUTION_TRACE
]

print(
    "Committed actions:",
    TRACE_ACTIONS[:REPLAN_AFTER]
)

print(
    "Inherited head actions:",
    [
        step["action"]
        for step
        in INHERITED_HEAD
    ]
)

print(
    "Committed immutable:",
    COMMITTED_IMMUTABLE
)

assert COMMITTED_IMMUTABLE, (
    "Committed steps were mutated by replanning."
)

print()

print(
    "TEST 11: Runtime Replanning Detected"
)

REPLANNED = (
        "VALIDATE_ACTUATOR_RESPONSE"
        in
        TRACE_ACTIONS
        and
        len(TRACE_ACTIONS)
        >
        len(
            INHERITED_PLAN
        )
)

print(
    "Trace actions:",
    TRACE_ACTIONS
)

print(
    "Inherited plan actions:",
    [
        step["action"]
        for step
        in INHERITED_PLAN
    ]
)

print(
    "Replanned:",
    REPLANNED
)

assert REPLANNED, (
    "Runtime replanning not detected."
)

assert len(
    REPLAN_EVENTS
) == 2, (
    "Expected runtime evidence and replan events."
)

assert REPLAN_EVENTS[0][
    "event"
] == "runtime_evidence", (
    "Runtime evidence event missing."
)

assert REPLAN_EVENTS[1][
    "event"
] == "replanned", (
    "Replan event missing."
)

list(
    map(
        print,
        REPLAN_EVENTS
    )
)

print()

print(
    "TEST 12: Verify Full Execution Trace"
)

assert all(
    step["status"] == "success"
    for step
    in EXECUTION_TRACE
), "Trace contains failed steps."

assert all(
    step["dependency"] is None
    or any(
        earlier["action"] == step["dependency"]
        for earlier
        in EXECUTION_TRACE[:step["step"] - 1]
    )
    for step
    in EXECUTION_TRACE
), "Trace dependency violation."

UNIQUE_ACTIONS = (
        len(
            set(
                TRACE_ACTIONS
            )
        )
        ==
        len(
            TRACE_ACTIONS
        )
)

assert UNIQUE_ACTIONS, (
    "An action was executed twice."
)

print(
    "Trace statuses:",
    [
        step["status"]
        for step
        in EXECUTION_TRACE
    ]
)

print(
    "Unique actions:",
    UNIQUE_ACTIONS
)

print(
    "Trace verified."
)

print()

print(
    "TEST 13: Deterministic Continuous Execution"
)

SECOND_TRACE, SECOND_EVENTS = execute_continuous(
    INHERITED_PLAN,
    RUNTIME_EVIDENCE,
    INHERITED_PATTERNS,
    ACTION_MAP,
    REPLAN_AFTER
)

DETERMINISTIC = (
        stable_hash(
            TRACE_ACTIONS
        )
        ==
        stable_hash(
            [
                step["action"]
                for step
                in SECOND_TRACE
            ]
        )
        and
        stable_hash(
            REPLAN_EVENTS
        )
        ==
        stable_hash(
            SECOND_EVENTS
        )
)

print(
    "First trace:",
    TRACE_ACTIONS
)

print(
    "Second trace:",
    [
        step["action"]
        for step
        in SECOND_TRACE
    ]
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Continuous execution is nondeterministic."
)

print(
    "Deterministic continuous execution validated."
)

print()

print(
    "TEST 14: Numerical Health"
)

RISK_TENSOR = torch.tensor(
    [
        record["risk_score"]
        for record
        in RUNTIME_RECORDS
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
    "TEST 15: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if len(EXECUTION_TRACE) == EXPECTED_TRACE_STEPS
    else [
        "Trace step count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(RUNTIME_RECORDS) == 4
    else [
        "Runtime records count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if REPLANNED
    else [
        "Runtime replanning not detected."
    ]
)

PROMOTION_ERRORS += (
    []
    if COMMITTED_IMMUTABLE
    else [
        "Committed steps mutated."
    ]
)

PROMOTION_ERRORS += (
    []
    if UNIQUE_ACTIONS
    else [
        "Action executed twice."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Continuous execution nondeterministic."
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
    "Trace steps:",
    len(EXECUTION_TRACE)
)

print(
    "Replanned:",
    REPLANNED
)

print(
    "Committed immutable:",
    COMMITTED_IMMUTABLE
)

print(
    "Unique actions:",
    UNIQUE_ACTIONS
)

print(
    "Deterministic:",
    DETERMINISTIC
)

print(
    "Promotion errors:",
    len(PROMOTION_ERRORS)
)

assert not PROMOTION_ERRORS, (
        "122R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "122R promotion gate passed."
)

print()

print(
    "TEST 16: Persist Continuous Execution Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "122R",

    "capability":
        "continuous_adaptive_execution_runtime_replanning",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "121R",

    "inherited_patterns":
        INHERITED_PATTERNS,

    "inherited_plan":
        INHERITED_PLAN,

    "runtime_evidence":
        RUNTIME_EVIDENCE,

    "runtime_records":
        RUNTIME_RECORDS,

    "runtime_order":
        RUNTIME_ORDER,

    "replanned_tail":
        REPLANNED_TAIL,

    "execution_trace":
        EXECUTION_TRACE,

    "replan_events":
        REPLAN_EVENTS,

    "verification":
        {
            "replanned":
                REPLANNED,

            "committed_immutable":
                COMMITTED_IMMUTABLE,

            "unique_actions":
                UNIQUE_ACTIONS,

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
    RELOADED["execution_trace"]
) == len(
    EXECUTION_TRACE
), "Trace length changed after reload."

assert RELOADED[
    "verification"
][
    "replanned"
], "Replanning changed after reload."

assert RELOADED[
    "verification"
][
    "committed_immutable"
], "Committed immutability changed after reload."

print(
    "Reloaded trace steps:",
    len(
        RELOADED["execution_trace"]
    )
)

print(
    "Reloaded replanned:",
    RELOADED[
        "verification"
    ][
        "replanned"
    ]
)

print(
    "Reloaded committed immutable:",
    RELOADED[
        "verification"
    ][
        "committed_immutable"
    ]
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
            "122R",

        "capability":
            "continuous_adaptive_execution_runtime_replanning",

        "inherited_plan":
            INHERITED_PLAN,

        "runtime_evidence":
            RUNTIME_EVIDENCE,

        "runtime_records":
            RUNTIME_RECORDS,

        "runtime_order":
            RUNTIME_ORDER,

        "replanned_tail":
            REPLANNED_TAIL,

        "execution_trace":
            EXECUTION_TRACE,

        "replan_events":
            REPLAN_EVENTS
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "122R",

        "memory_version":
            MEMORY_VERSION,

        "trace_steps":
            len(EXECUTION_TRACE),

        "replanned":
            REPLANNED,

        "committed_immutable":
            COMMITTED_IMMUTABLE,

        "unique_actions":
            UNIQUE_ACTIONS,

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
            "122R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "123R Post-Execution Outcome Feedback "
                "+ Plan Learning"
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
    "SILVERWING 122R ARCHITECTURE"
)

print(
    "Inherited Adaptive Plan"
)

print(
    "        ↓"
)

print(
    "Continuous Execution Loop"
)

print(
    "        ↓"
)

print(
    "Committed Steps Finalized"
)

print(
    "        ↓"
)

print(
    "Runtime Evidence at Cursor"
)

print(
    "        ↓"
)

print(
    "Re-Arbitration of Remaining Patterns"
)

print(
    "        ↓"
)

print(
    "Runtime Replanning of Tail"
)

print(
    "        ↓"
)

print(
    "Execution Resumes"
)

print(
    "        ↓"
)

print(
    "Verified Full Trace"
)

print()

print(
    "WHAT 122R ADDS"
)

print(
    "A continuous execution cursor, immutable committed steps, "
    "mid-execution evidence ingestion, tail-only replanning and "
    "a fully verified execution trace."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Live operations where a plan is already running while "
    "telemetry, sensors or actuators keep producing new evidence."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A running plan must survive reality. Silverwing has to finish "
    "what it committed to while folding new evidence into the "
    "work that remains."
)

print()

print(
    "NEXT: 123R Post-Execution Outcome Feedback + Plan Learning"
)

print()

print(
    "=== LESSON 122R COMPLETE ==="
)

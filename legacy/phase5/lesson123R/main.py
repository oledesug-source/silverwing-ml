# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 123R
# Post-Execution Outcome Feedback + Plan Learning
# ============================================================
#
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 122R executed an adaptive plan and produced a verified trace.
# 123R closes the loop: after execution finishes, Silverwing
# observes what actually happened and learns from it.
#
# For every executed step Silverwing collects an outcome:
#
#     executed action
#          ↓
#     observed outcome
#          ↓
#     effectiveness score
#          ↓
#     pattern attributes revised
#          ↓
#     risk scores recalibrated
#          ↓
#     prevention rules learned
#          ↓
#     next plan reconstructed
#
# Learning rule: a prevented step lowers expected recurrence and
# confidence in that threat; a step that failed to prevent the
# failure keeps the pattern hot. Risk then flows toward reality.
#
# 123R pipeline:
#
#     inherited 122R trace
#          ↓
#     outcome feedback collection
#          ↓
#     outcome coverage validation
#          ↓
#     aggregate outcome statistics
#          ↓
#     pattern learning
#          ↓
#     risk recalibration
#          ↓
#     prevention rule learning
#          ↓
#     learned plan reconstruction
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 122R memory is the source of truth.
# 2. The action map is derived from the persisted execution trace.
# 3. Every executed action must have an outcome (full coverage).
# 4. Outcomes are explicit and deterministic.
# 5. Prevented steps reduce recurrence; failed steps keep it hot.
# 6. Learned patterns must re-arbitrate under dependency rules.
# 7. Learned prevention rules must flag ineffective actions.
# 8. The learned plan must differ from the executed plan.
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
MEMORY_VERSION = "123R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
LEARNING_RATE = 0.30
EFFECTIVE_THRESHOLD = 0.75
MIN_TRACE_STEPS = 6
EXPECTED_PATTERNS = 6
EXPECTED_TRACE_ACTIONS = [
    "VALIDATE_EVIDENCE_PROVENANCE",
    "VALIDATE_TIMELINE_INTEGRITY",
    "VALIDATE_SCHEMA_CONSISTENCY",
    "VALIDATE_SENSOR_ALIGNMENT",
    "VALIDATE_ACTUATOR_RESPONSE",
    "VALIDATE_TELEMETRY_FUSION"
]

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_122R = PHASE5_DIR / "lesson122R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_122R
        / "silverwing_continuous_execution_memory.json"
)

SOURCE_INDEX = (
        LESSON_122R
        / "silverwing_continuous_execution_index.pt"
)

SOURCE_DATASET = (
        LESSON_122R
        / "silverwing_continuous_execution_dataset.json"
)

SOURCE_REPORT = (
        LESSON_122R
        / "silverwing_continuous_execution_report.json"
)

SOURCE_REGISTRY = (
        LESSON_122R
        / "silverwing_continuous_execution_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_122R
        / "checkpoints"
        / "silverwing_continuous_execution_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_outcome_feedback_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_outcome_feedback_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_outcome_feedback_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_outcome_feedback_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_outcome_feedback_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_outcome_feedback_best.pt"
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
    "PHASE 5 - LESSON 123R"
)

print(
    "Post-Execution Outcome Feedback + Plan Learning"
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
    "TEST 1: Verify 122R Inputs"
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
), "One or more 122R inputs are missing."

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
    "TEST 2: Load 122R Continuous Execution Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "122R continuous execution memory is invalid."

INHERITED_PATTERNS = SOURCE.get(
    "inherited_patterns",
    []
)

RUNTIME_EVIDENCE = SOURCE.get(
    "runtime_evidence",
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
    EXECUTION_TRACE
) == MIN_TRACE_STEPS, (
    "122R trace must contain six steps."
)

assert TRACE_ACTIONS == EXPECTED_TRACE_ACTIONS, (
    "122R trace actions are unexpected."
)

assert len(
    INHERITED_PATTERNS
) == 5, (
    "122R must inherit exactly five patterns."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Trace steps:",
    len(
        EXECUTION_TRACE
    )
)

print(
    "Trace actions:",
    TRACE_ACTIONS
)

print(
    "Inherited patterns:",
    len(
        INHERITED_PATTERNS
    )
)

print()

print(
    "TEST 3: Derive Action Map from Execution Trace"
)

PATTERN_TO_FAMILY = {
    pattern["pattern_id"]: pattern["family"]
    for pattern
    in INHERITED_PATTERNS
}

for evidence in RUNTIME_EVIDENCE:

    if evidence.get(
        "new_pattern",
        False
    ):

        PATTERN_TO_FAMILY[evidence[
            "pattern_id"
        ]] = evidence[
            "family"
        ]

ACTION_MAP = {}

for step in EXECUTION_TRACE:

    ACTION_MAP[
        PATTERN_TO_FAMILY[
            step["pattern_id"]
        ]
    ] = (
        step["action"],
        step["dependency"]
    )

assert len(
    ACTION_MAP
) == 6, (
    "Action map must contain six families."
)

assert len(
    PATTERN_TO_FAMILY
) == 6, (
    "Pattern family map must contain six patterns."
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
    "TEST 4: Rebuild As-Of-Execution Patterns"
)


def build_execution_patterns(
        revised_patterns,
        runtime_evidence
):

    execution_patterns = []

    for pattern in revised_patterns:

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

        if update is None:

            execution_patterns.append(
                pattern
            )

        else:

            revised = dict(
                pattern
            )

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

                    revised[key] = clamp(
                        revised[key]
                        +
                        update[
                            delta_key
                        ]
                    )

            revised["evidence_origin"] = update[
                "origin"
            ]

            execution_patterns.append(
                revised
            )

    for evidence in runtime_evidence:

        if evidence.get(
            "new_pattern",
            False
        ):

            execution_patterns.append(
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

    return execution_patterns


EXECUTION_PATTERNS = build_execution_patterns(
    INHERITED_PATTERNS,
    RUNTIME_EVIDENCE
)

assert len(
    EXECUTION_PATTERNS
) == EXPECTED_PATTERNS, (
    "Execution pattern set must contain six patterns."
)

assert all(
    pattern["family"]
    in
    ACTION_MAP
    for pattern
    in EXECUTION_PATTERNS
), "Execution pattern family missing from action map."

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
        EXECUTION_PATTERNS
    )
)

print()

print(
    "TEST 5: Collect Post-Execution Outcome Feedback"
)

OUTCOME_FEEDBACK = [
    {
        "pattern_id":
            "pattern_001",

        "action":
            "VALIDATE_EVIDENCE_PROVENANCE",

        "status":
            "prevented",

        "effectiveness":
            0.92,

        "outcome":
            "Evidence provenance verified clean."
    },
    {
        "pattern_id":
            "pattern_005",

        "action":
            "VALIDATE_TIMELINE_INTEGRITY",

        "status":
            "prevented",

        "effectiveness":
            0.88,

        "outcome":
            "Timeline gaps closed before impact."
    },
    {
        "pattern_id":
            "pattern_002",

        "action":
            "VALIDATE_SCHEMA_CONSISTENCY",

        "status":
            "prevented",

        "effectiveness":
            0.85,

        "outcome":
            "Schema hardening held under load."
    },
    {
        "pattern_id":
            "pattern_003",

        "action":
            "VALIDATE_SENSOR_ALIGNMENT",

        "status":
            "not_prevented",

        "effectiveness":
            0.35,

        "outcome":
            "Sensor drift persisted after validation."
    },
    {
        "pattern_id":
            "pattern_006",

        "action":
            "VALIDATE_ACTUATOR_RESPONSE",

        "status":
            "prevented",

        "effectiveness":
            0.90,

        "outcome":
            "Actuator lag corrected in time."
    },
    {
        "pattern_id":
            "pattern_004",

        "action":
            "VALIDATE_TELEMETRY_FUSION",

        "status":
            "partial",

        "effectiveness":
            0.60,

        "outcome":
            "Telemetry anomalies partially resolved."
    }
]

assert len(
    OUTCOME_FEEDBACK
) == len(
    EXECUTION_TRACE
), "Outcome feedback must cover the full trace."

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
        OUTCOME_FEEDBACK
    )
)

print()

print(
    "TEST 6: Outcome Coverage Validation"
)

OUTCOME_ACTIONS = sorted(
    item["action"]
    for item
    in OUTCOME_FEEDBACK
)

OUTCOME_COVERED = (
        OUTCOME_ACTIONS
        ==
        sorted(
            TRACE_ACTIONS
        )
)

print(
    "Outcome actions:",
    OUTCOME_ACTIONS
)

print(
    "Trace actions:",
    sorted(
        TRACE_ACTIONS
    )
)

print(
    "Outcome covered:",
    OUTCOME_COVERED
)

assert OUTCOME_COVERED, (
    "Not every executed action has an outcome."
)

assert all(
    item["pattern_id"]
    in
    [
        step["pattern_id"]
        for step
        in EXECUTION_TRACE
    ]
    for item
    in OUTCOME_FEEDBACK
), "Outcome references an unexecuted pattern."

print()

print(
    "TEST 7: Aggregate Outcome Statistics"
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

PREVENTED_COUNT = len(
    [
        item
        for item
        in OUTCOME_FEEDBACK
        if item["status"] == "prevented"
    ]
)

NOT_PREVENTED_COUNT = len(
    [
        item
        for item
        in OUTCOME_FEEDBACK
        if item["status"] == "not_prevented"
    ]
)

PARTIAL_COUNT = len(
    [
        item
        for item
        in OUTCOME_FEEDBACK
        if item["status"] == "partial"
    ]
)

assert abs(
    MEAN_EFFECTIVENESS
    -
    0.75
) <= 1e-9, "Mean effectiveness must be exactly 0.75."

assert PREVENTED_COUNT == 4, (
    "Expected four prevented outcomes."
)

assert NOT_PREVENTED_COUNT == 1, (
    "Expected one not-prevented outcome."
)

assert PARTIAL_COUNT == 1, (
    "Expected one partial outcome."
)

print(
    "Mean effectiveness:",
    format(
        MEAN_EFFECTIVENESS,
        ".4f"
    )
)

print(
    "Prevented:",
    PREVENTED_COUNT
)

print(
    "Not prevented:",
    NOT_PREVENTED_COUNT
)

print(
    "Partial:",
    PARTIAL_COUNT
)

print()

print(
    "TEST 8: Apply Outcome Learning"
)


def learn_from_outcomes(
        execution_patterns,
        outcome_feedback,
        learning_rate
):

    learned = []

    for pattern in execution_patterns:

        outcome = next(
            (
                item
                for item
                in outcome_feedback
                if item["pattern_id"]
                   ==
                   pattern["pattern_id"]
            ),
            None
        )

        if outcome is None:

            learned.append(
                pattern
            )

        else:

            updated = dict(
                pattern
            )

            effectiveness = clamp(
                outcome["effectiveness"]
            )

            updated["confidence"] = clamp(
                updated["confidence"]
                +
                learning_rate
                *
                (
                    0.5
                    -
                    effectiveness
                )
            )

            updated["recurrence"] = clamp(
                updated["recurrence"]
                +
                learning_rate
                *
                (
                    1.0
                    -
                    effectiveness
                )
            )

            updated["outcome_status"] = outcome[
                "status"
            ]

            updated["outcome_effectiveness"] = (
                effectiveness
            )

            learned.append(
                updated
            )

    return learned


LEARNED_PATTERNS = learn_from_outcomes(
    EXECUTION_PATTERNS,
    OUTCOME_FEEDBACK,
    LEARNING_RATE
)

assert len(
    LEARNED_PATTERNS
) == EXPECTED_PATTERNS, (
    "Learning must preserve six patterns."
)

list(
    map(
        lambda pattern: print(
            pattern["pattern_id"],
            "| conf=",
            format(
                pattern["confidence"],
                ".4f"
            ),
            "| rec=",
            format(
                pattern["recurrence"],
                ".4f"
            ),
            "|",
            pattern.get(
                "outcome_status",
                "-"
            )
        ),
        LEARNED_PATTERNS
    )
)

print()

print(
    "TEST 9: Recalibrate Risk Records"
)


def build_records(
        patterns
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
                    risk_score(pattern)
                ),

            "severity":
                pattern["severity"],

            "impact":
                pattern["impact"]
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


PRE_LEARNING_RECORDS = build_records(
    EXECUTION_PATTERNS
)

LEARNED_RECORDS = build_records(
    LEARNED_PATTERNS
)

PRE_LEARNING_ORDER = [
    record["pattern_id"]
    for record
    in PRE_LEARNING_RECORDS
]

LEARNED_ORDER = [
    record["pattern_id"]
    for record
    in LEARNED_RECORDS
]

assert PRE_LEARNING_ORDER == [
    "pattern_001",
    "pattern_005",
    "pattern_003",
    "pattern_006",
    "pattern_004",
    "pattern_002"
], "Pre-learning risk order is incorrect."

assert LEARNED_ORDER == [
    "pattern_001",
    "pattern_003",
    "pattern_005",
    "pattern_004",
    "pattern_006",
    "pattern_002"
], "Learned risk order is incorrect."

print(
    "Pre-learning order:",
    PRE_LEARNING_ORDER
)

print(
    "Learned order:",
    LEARNED_ORDER
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
            LEARNED_RECORDS,
            1
        )
    )
)

print()

print(
    "TEST 10: Learning Direction Validation"
)

PRE_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in PRE_LEARNING_RECORDS
}

LEARNED_RISK = {
    record["pattern_id"]: record["risk_score"]
    for record
    in LEARNED_RECORDS
}

PRE_RANK = {
    pattern_id: index + 1
    for index, pattern_id
    in enumerate(
        PRE_LEARNING_ORDER
    )
}

LEARNED_RANK = {
    pattern_id: index + 1
    for index, pattern_id
    in enumerate(
        LEARNED_ORDER
    )
}

NOT_PREVENTED_PERSISTS = (
        LEARNED_RISK["pattern_003"]
        >=
        PRE_RISK["pattern_003"]
        - 1e-9
        and
        LEARNED_RANK["pattern_003"] == 2
        and
        PRE_RANK["pattern_003"] == 3
)

PREVENTED_DROPS = all(
    LEARNED_RISK[pattern_id]
    <
    PRE_RISK[pattern_id]
    - 1e-9
    for pattern_id
    in [
        "pattern_001",
        "pattern_005",
        "pattern_006"
    ]
)

print(
    "Pattern 003 pre risk:",
    format(
        PRE_RISK["pattern_003"],
        ".6f"
    )
)

print(
    "Pattern 003 learned risk:",
    format(
        LEARNED_RISK["pattern_003"],
        ".6f"
    )
)

print(
    "Pattern 003 pre rank:",
    PRE_RANK["pattern_003"]
)

print(
    "Pattern 003 learned rank:",
    LEARNED_RANK["pattern_003"]
)

print(
    "Not-prevented persists:",
    NOT_PREVENTED_PERSISTS
)

print(
    "Prevented drops:",
    PREVENTED_DROPS
)

assert NOT_PREVENTED_PERSISTS, (
    "Not-prevented pattern must retain top risk."
)

assert PREVENTED_DROPS, (
    "Prevented patterns must drop in risk."
)

print()

print(
    "TEST 11: Learn Prevention Rules"
)

PREVENTION_RULES = [
    {
        "action":
            item["action"],

        "pattern_id":
            item["pattern_id"],

        "effectiveness":
            clamp(
                item["effectiveness"]
            ),

        "effective":
            clamp(
                item["effectiveness"]
            )
            >=
            EFFECTIVE_THRESHOLD
    }
    for item
    in OUTCOME_FEEDBACK
]

EFFECTIVE_ACTIONS = sorted(
    rule["action"]
    for rule
    in PREVENTION_RULES
    if rule["effective"]
)

assert len(
    EFFECTIVE_ACTIONS
) == 4, (
    "Expected exactly four effective actions."
)

assert "VALIDATE_SENSOR_ALIGNMENT" not in (
    EFFECTIVE_ACTIONS
), "Failed action must not be flagged effective."

assert "VALIDATE_TELEMETRY_FUSION" not in (
    EFFECTIVE_ACTIONS
), "Partial action must not be flagged effective."

list(
    map(
        print,
        PREVENTION_RULES
    )
)

print(
    "Effective actions:",
    EFFECTIVE_ACTIONS
)

print()

print(
    "TEST 12: Reconstruct Learned Plan"
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
            "Learned plan dependency cycle."
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


LEARNED_PLAN = build_adaptive_plan(
    LEARNED_RECORDS,
    ACTION_MAP,
    set()
)

LEARNED_PLAN_ACTIONS = [
    step["action"]
    for step
    in LEARNED_PLAN
]

assert len(
    LEARNED_PLAN
) == EXPECTED_PATTERNS, (
    "Learned plan must contain six steps."
)

POSITIONS = {
    item["action"]:
        index
    for index, item
    in enumerate(
        LEARNED_PLAN
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
    in LEARNED_PLAN
), "Learned plan dependency validation failed."

list(
    map(
        print,
        LEARNED_PLAN
    )
)

print()

print(
    "TEST 13: Plan Learning Detected"
)

PLAN_LEARNED = (
        LEARNED_PLAN_ACTIONS
        !=
        TRACE_ACTIONS
)

TAIL_SWAPPED = (
        LEARNED_PLAN_ACTIONS[-2:]
        ==
        [
            "VALIDATE_TELEMETRY_FUSION",
            "VALIDATE_ACTUATOR_RESPONSE"
        ]
        and
        TRACE_ACTIONS[-2:]
        ==
        [
            "VALIDATE_ACTUATOR_RESPONSE",
            "VALIDATE_TELEMETRY_FUSION"
        ]
)

print(
    "Executed trace actions:",
    TRACE_ACTIONS
)

print(
    "Learned plan actions:",
    LEARNED_PLAN_ACTIONS
)

print(
    "Plan learned:",
    PLAN_LEARNED
)

print(
    "Tail swapped:",
    TAIL_SWAPPED
)

assert PLAN_LEARNED, (
    "Learned plan must differ from executed trace."
)

assert TAIL_SWAPPED, (
    "Lower-effectiveness tail action must move earlier."
)

print()

print(
    "TEST 14: Deterministic Learning"
)

SECOND_LEARNED = learn_from_outcomes(
    EXECUTION_PATTERNS,
    OUTCOME_FEEDBACK,
    LEARNING_RATE
)

DETERMINISTIC = (
        stable_hash(
            LEARNED_PATTERNS
        )
        ==
        stable_hash(
            SECOND_LEARNED
        )
        and
        stable_hash(
            LEARNED_ORDER
        )
        ==
        stable_hash(
            LEARNED_ORDER
        )
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Outcome learning is nondeterministic."
)

print(
    "Deterministic learning validated."
)

print()

print(
    "TEST 15: Numerical Health"
)

LEARNED_TENSOR = torch.tensor(
    [
        record["risk_score"]
        for record
        in LEARNED_RECORDS
    ],
    dtype=torch.float32
)

EFFECT_TENSOR = torch.tensor(
    EFFECTIVENESS_VALUES,
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        LEARNED_TENSOR
    ).all()
    and
    torch.isfinite(
        EFFECT_TENSOR
    ).all()
)

print(
    "Learned NaN:",
    int(
        torch.isnan(
            LEARNED_TENSOR
        ).sum()
    )
)

print(
    "Learned Inf:",
    int(
        torch.isinf(
            LEARNED_TENSOR
        ).sum()
    )
)

print(
    "Effectiveness NaN:",
    int(
        torch.isnan(
            EFFECT_TENSOR
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
    "TEST 16: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if len(LEARNED_PATTERNS) == EXPECTED_PATTERNS
    else [
        "Learned pattern count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if OUTCOME_COVERED
    else [
        "Outcome coverage incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        MEAN_EFFECTIVENESS
        -
        0.75
    ) <= 1e-9
    else [
        "Mean effectiveness invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if NOT_PREVENTED_PERSISTS
    else [
        "Not-prevented risk did not persist."
    ]
)

PROMOTION_ERRORS += (
    []
    if PREVENTED_DROPS
    else [
        "Prevented risks did not drop."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(EFFECTIVE_ACTIONS) == 4
    else [
        "Prevention rule learning failed."
    ]
)

PROMOTION_ERRORS += (
    []
    if PLAN_LEARNED
    else [
        "Learned plan unchanged."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Outcome learning nondeterministic."
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
    "Learned patterns:",
    len(LEARNED_PATTERNS)
)

print(
    "Outcome covered:",
    OUTCOME_COVERED
)

print(
    "Mean effectiveness:",
    format(
        MEAN_EFFECTIVENESS,
        ".4f"
    )
)

print(
    "Plan learned:",
    PLAN_LEARNED
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
        "123R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "123R promotion gate passed."
)

print()

print(
    "TEST 17: Persist Outcome Learning Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "123R",

    "capability":
        "post_execution_outcome_feedback_plan_learning",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "122R",

    "inherited_patterns":
        INHERITED_PATTERNS,

    "execution_trace":
        EXECUTION_TRACE,

    "execution_patterns":
        EXECUTION_PATTERNS,

    "outcome_feedback":
        OUTCOME_FEEDBACK,

    "outcome_stats":
        {
            "mean_effectiveness":
                MEAN_EFFECTIVENESS,

            "prevented":
                PREVENTED_COUNT,

            "not_prevented":
                NOT_PREVENTED_COUNT,

            "partial":
                PARTIAL_COUNT
        },

    "learned_patterns":
        LEARNED_PATTERNS,

    "pre_learning_records":
        PRE_LEARNING_RECORDS,

    "learned_records":
        LEARNED_RECORDS,

    "pre_learning_order":
        PRE_LEARNING_ORDER,

    "learned_order":
        LEARNED_ORDER,

    "prevention_rules":
        PREVENTION_RULES,

    "effective_actions":
        EFFECTIVE_ACTIONS,

    "learned_plan":
        LEARNED_PLAN,

    "verification":
        {
            "outcome_covered":
                OUTCOME_COVERED,

            "not_prevented_persists":
                NOT_PREVENTED_PERSISTS,

            "prevented_drops":
                PREVENTED_DROPS,

            "plan_learned":
                PLAN_LEARNED,

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
    "TEST 18: Reload Persistent Memory"
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
    RELOADED["learned_patterns"]
) == len(
    LEARNED_PATTERNS
), "Learned pattern count changed after reload."

assert len(
    RELOADED["learned_plan"]
) == len(
    LEARNED_PLAN
), "Learned plan length changed after reload."

assert RELOADED[
    "verification"
][
    "plan_learned"
], "Plan learning changed after reload."

assert RELOADED[
    "verification"
][
    "outcome_covered"
], "Outcome coverage changed after reload."

print(
    "Reloaded learned patterns:",
    len(
        RELOADED["learned_patterns"]
    )
)

print(
    "Reloaded learned plan:",
    len(
        RELOADED["learned_plan"]
    )
)

print(
    "Reloaded plan learned:",
    RELOADED[
        "verification"
    ][
        "plan_learned"
    ]
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 19: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "123R",

        "capability":
            "post_execution_outcome_feedback_plan_learning",

        "execution_trace":
            EXECUTION_TRACE,

        "outcome_feedback":
            OUTCOME_FEEDBACK,

        "outcome_stats":
            {
                "mean_effectiveness":
                    MEAN_EFFECTIVENESS,

                "prevented":
                    PREVENTED_COUNT,

                "not_prevented":
                    NOT_PREVENTED_COUNT,

                "partial":
                    PARTIAL_COUNT
            },

        "learned_patterns":
            LEARNED_PATTERNS,

        "learned_records":
            LEARNED_RECORDS,

        "learned_order":
            LEARNED_ORDER,

        "prevention_rules":
            PREVENTION_RULES,

        "effective_actions":
            EFFECTIVE_ACTIONS,

        "learned_plan":
            LEARNED_PLAN
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "123R",

        "memory_version":
            MEMORY_VERSION,

        "pattern_count":
            len(LEARNED_PATTERNS),

        "mean_effectiveness":
            MEAN_EFFECTIVENESS,

        "prevented":
            PREVENTED_COUNT,

        "not_prevented":
            NOT_PREVENTED_COUNT,

        "plan_learned":
            PLAN_LEARNED,

        "effective_actions":
            EFFECTIVE_ACTIONS,

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
            "123R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "124R Outcome Memory Consolidation "
                "+ Risk Calibration"
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
    "SILVERWING 123R ARCHITECTURE"
)

print(
    "Executed Plan Trace"
)

print(
    "        ↓"
)

print(
    "Outcome Feedback Collection"
)

print(
    "        ↓"
)

print(
    "Outcome Coverage Validation"
)

print(
    "        ↓"
)

print(
    "Aggregate Outcome Statistics"
)

print(
    "        ↓"
)

print(
    "Pattern Learning"
)

print(
    "        ↓"
)

print(
    "Risk Recalibration"
)

print(
    "        ↓"
)

print(
    "Prevention Rule Learning"
)

print(
    "        ↓"
)

print(
    "Learned Plan Reconstruction"
)

print()

print(
    "WHAT 123R ADDS"
)

print(
    "Post-execution outcome observation, effectiveness scoring, "
    "risk recalibration toward observed reality, prevention-rule "
    "learning and reconstruction of the next plan."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Any deployed prevention loop that must measure whether its "
    "mitigations actually worked and adapt to what it observes."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "An unmeasured plan cannot improve. Silverwing turns executed "
    "actions into lessons that sharpen the next decision."
)

print()

print(
    "NEXT: 124R Outcome Memory Consolidation + Risk Calibration"
)

print()

print(
    "=== LESSON 123R COMPLETE ==="
)

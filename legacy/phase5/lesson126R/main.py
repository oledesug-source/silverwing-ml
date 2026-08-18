# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 126R
# Preventive Control Loop Governance + Policy Rehearsal
# ============================================================
#
# 121R  -> Adaptive Preventive Planning + Dynamic Reprioritization
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
# 124R  -> Outcome Memory Consolidation + Risk Calibration
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
#
# ============================================================
# PURPOSE
# ============================================================
#
# By 125R the control loop learns, recalibrates and re-plans.
# 126R asks: who watches the watcher? An adaptive loop without
# governance can chase noise. 126R adds a governance layer
# that reviews every proposed plan against explicit policy
# before it is allowed to execute.
#
# Governance:
#
#     incoming tuned plan
#          ↓
#     policy frame (P1..P6)
#          ↓
#     per-policy check
#          ↓
#     APPROVED / VETOED verdict
#
# Policy rehearsal:
#
#     approved plan steps
#          ↓
#     predicted effectiveness (history)
#          ↓
#     per-step residual risk
#          ↓
#     coverage / capacity / budget
#          ↓
#     feasibility verdict
#
# The planner proposes; governance disposes. A policy-violating
# candidate must be vetoed to prove the control is real.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 125R memory is the source of truth.
# 2. Governance reviews the tuned plan against explicit policy.
# 3. Coverage targets the critical set: HIGH union RISING.
# 4. Rehearsal predicts effectiveness from lifetime history.
# 5. Residual risk is the unmitigated portion per step.
# 6. A compliant plan is APPROVED; a violating plan is VETOED.
# 7. Every check leaves an audit trail entry.
# 8. Determinism must be checked.
# 9. Numerical health must be checked.
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
MEMORY_VERSION = "126R.1"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
MITIGATION_FLOOR = 0.75
RESIDUAL_BUDGET = 1.10
EXPECTED_PATTERNS = 6
EXPECTED_MEAN_REHEARSAL = 0.775

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_125R = PHASE5_DIR / "lesson125R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_memory.json"
)

SOURCE_INDEX = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_index.pt"
)

SOURCE_DATASET = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_dataset.json"
)

SOURCE_REPORT = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_report.json"
)

SOURCE_REGISTRY = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_125R
        / "checkpoints"
        / "silverwing_cross_cycle_trending_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_control_loop_governance_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_control_loop_governance_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_control_loop_governance_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_control_loop_governance_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_control_loop_governance_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_control_loop_governance_best.pt"
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
    "PHASE 5 - LESSON 126R"
)

print(
    "Preventive Control Loop Governance + Policy Rehearsal"
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

print(
    "126R -> Preventive Control Loop Governance + Policy Rehearsal"
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
    "TEST 1: Verify 125R Inputs"
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
), "One or more 125R inputs are missing."

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
    "TEST 2: Load 125R Trending Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "125R trending memory is invalid."

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

TUNED_PLAN_ACTIONS = [
    step["action"]
    for step
    in TUNED_PLAN
]

TRENDS = SOURCE.get(
    "trends",
    []
)

EXTENDED_HISTORY = SOURCE.get(
    "extended_history",
    {}
)

TUNED_THRESHOLDS = SOURCE.get(
    "tuned_thresholds",
    {}
)

assert len(
    TUNED_RECORDS
) == EXPECTED_PATTERNS, (
    "125R must supply exactly six tuned records."
)

assert len(
    TUNED_PLAN
) == EXPECTED_PATTERNS, (
    "125R tuned plan must contain six steps."
)

assert len(
    EXTENDED_HISTORY
) == EXPECTED_PATTERNS, (
    "125R extended history must contain six families."
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

print(
    "Tuned plan:",
    TUNED_PLAN_ACTIONS
)

print(
    "Tuned HIGH:",
    TUNED_THRESHOLDS[
        "tuned_high"
    ]
)

print()

print(
    "TEST 3: Rebuild Governed State"
)

RECORD_MAP = {
    record["pattern_id"]: record
    for record
    in TUNED_RECORDS
}

TREND_MAP = {
    record["pattern_id"]: record["trend"]
    for record
    in TRENDS
}

assert len(
    TREND_MAP
) == EXPECTED_PATTERNS, (
    "Trend map must cover six patterns."
)

print(
    "Tuned risks:",
    dict(
        map(
            lambda pair: (
                pair[0],
                round(
                    pair[1]["risk_score"],
                    6
                )
            ),
            RECORD_MAP.items()
        )
    )
)

print(
    "Trend map:",
    TREND_MAP
)

print(
    "Lifetime effectiveness:",
    dict(
        map(
            lambda pair: (
                pair[0],
                round(
                    pair[1]["lifetime_effectiveness"],
                    4
                )
            ),
            EXTENDED_HISTORY.items()
        )
    )
)

print()

print(
    "TEST 4: Define Governance Policy Frame"
)

POLICY_FRAME = [
    {
        "policy_id":
            "P1",

        "name":
            "CRITICAL_COVERAGE",

        "description":
            "Every HIGH or RISING pattern must be planned."
    },
    {
        "policy_id":
            "P2",

        "name":
            "DEPENDENCY_ORDER",

        "description":
            "No step may run before its dependency."
    },
    {
        "policy_id":
            "P3",

        "name":
            "ANCHOR_IMMUTABILITY",

        "description":
            "The priority-1 pattern must open the plan."
    },
    {
        "policy_id":
            "P4",

        "name":
            "ACTION_UNIQUENESS",

        "description":
            "Every action may appear exactly once."
    },
    {
        "policy_id":
            "P5",

        "name":
            "MITIGATION_CAPACITY",

        "description":
            "Rehearsed mean effectiveness must meet the floor."
    },
    {
        "policy_id":
            "P6",

        "name":
            "RESIDUAL_TOLERANCE",

        "description":
            "Total residual risk must stay within budget."
    }
]

assert len(
    POLICY_FRAME
) == 6, (
    "Governance frame must define six policies."
)

list(
    map(
        lambda policy: print(
            policy["policy_id"],
            "|",
            policy["name"],
            "|",
            policy["description"]
        ),
        POLICY_FRAME
    )
)

print()

print(
    "TEST 5: Identify Critical Set"
)

HIGH_PATTERNS = {
    record["pattern_id"]
    for record
    in TUNED_RECORDS
    if record["risk_class"] == "HIGH"
}

RISING_PATTERNS = {
    pattern_id
    for pattern_id, trend
    in TREND_MAP.items()
    if trend == "RISING"
}

CRITICAL_SET = (
        HIGH_PATTERNS
        |
        RISING_PATTERNS
)

assert HIGH_PATTERNS == {
    "pattern_001"
}, "Expected pattern_001 as the sole HIGH pattern."

assert RISING_PATTERNS == {
    "pattern_004"
}, "Expected pattern_004 as the sole RISING pattern."

assert CRITICAL_SET == {
    "pattern_001",
    "pattern_004"
}, "Critical set mismatch."

print(
    "HIGH patterns:",
    sorted(
        HIGH_PATTERNS
    )
)

print(
    "RISING patterns:",
    sorted(
        RISING_PATTERNS
    )
)

print(
    "Critical set:",
    sorted(
        CRITICAL_SET
    )
)

print()

print(
    "TEST 6: Rehearse Approved Plan"
)


def rehearse_plan(
        plan_steps,
        record_map,
        history
):

    rehearsals = []

    for step in plan_steps:

        pattern_id = step[
            "pattern_id"
        ]

        predicted = clamp(
            history[
                pattern_id
            ][
                "lifetime_effectiveness"
            ]
        )

        rehearsals.append(
            {
                "step":
                    step["step"],

                "pattern_id":
                    pattern_id,

                "action":
                    step["action"],

                "risk_score":
                    record_map[
                        pattern_id
                    ][
                        "risk_score"
                    ],

                "predicted_effectiveness":
                    predicted,

                "residual_risk":
                    record_map[
                        pattern_id
                    ][
                        "risk_score"
                    ]
                    *
                    (
                        1.0
                        -
                        predicted
                    )
            }
        )

    total_residual = sum(
        item["residual_risk"]
        for item
        in rehearsals
    )

    mean_effectiveness = (
            sum(
                item["predicted_effectiveness"]
                for item
                in rehearsals
            )
            /
            len(
                rehearsals
            )
    )

    return (
        rehearsals,
        total_residual,
        mean_effectiveness
    )


APPROVED_REHEARSALS, APPROVED_RESIDUAL, APPROVED_MEAN = (
    rehearse_plan(
        TUNED_PLAN,
        RECORD_MAP,
        EXTENDED_HISTORY
    )
)

assert len(
    APPROVED_REHEARSALS
) == EXPECTED_PATTERNS, (
    "Rehearsal must cover six steps."
)

list(
    map(
        lambda item: print(
            item["step"],
            "|",
            item["pattern_id"],
            "|",
            item["action"],
            "| pred_eff=",
            format(
                item["predicted_effectiveness"],
                ".4f"
            ),
            "| residual=",
            format(
                item["residual_risk"],
                ".6f"
            )
        ),
        APPROVED_REHEARSALS
    )
)

print()

print(
    "TEST 7: Validate Rehearsal Metrics"
)

assert abs(
    APPROVED_MEAN
    -
    EXPECTED_MEAN_REHEARSAL
) <= 1e-9, "Rehearsed mean effectiveness must be 0.775."

assert abs(
    APPROVED_RESIDUAL
    -
    1.0296515412667769
) <= 1e-6, "Approved plan residual risk mismatch."

PLANNED_PATTERNS = {
    step["pattern_id"]
    for step
    in TUNED_PLAN
}

COVERAGE_RATIO = (
        len(
            CRITICAL_SET
            &
            PLANNED_PATTERNS
        )
        /
        len(
            CRITICAL_SET
        )
)

assert abs(
    COVERAGE_RATIO
    -
    1.0
) <= 1e-9, "Approved plan must fully cover the critical set."

print(
    "Mean predicted effectiveness:",
    format(
        APPROVED_MEAN,
        ".4f"
    )
)

print(
    "Total residual risk:",
    format(
        APPROVED_RESIDUAL,
        ".6f"
    )
)

print(
    "Coverage ratio:",
    COVERAGE_RATIO
)

print()

print(
    "TEST 8: Run Governance Checks on Approved Plan"
)


def check_policies(
        plan_steps,
        record_map,
        history,
        critical_set,
        rehearsed,
        total_residual,
        mean_effectiveness
):

    results = []

    plan_ids = [
        step["pattern_id"]
        for step
        in plan_steps
    ]

    actions = [
        step["action"]
        for step
        in plan_steps
    ]

    positions = {
        item["action"]:
            index
        for index, item
        in enumerate(
            plan_steps
        )
    }

    p1_passed = critical_set.issubset(
        set(
            plan_ids
        )
    )

    p2_passed = all(
        step["dependency"] is None
        or (
                step["dependency"]
                in
                positions
                and
                positions[
                    step["dependency"]
                ]
                <
                positions[
                    step["action"]
                ]
        )
        for step
        in plan_steps
    )

    anchor_id = next(
        record["pattern_id"]
        for record
        in sorted(
            record_map.values(),
            key=lambda item: item["priority"]
        )
    )

    p3_passed = (
            len(plan_steps) > 0
            and
            plan_steps[0]["pattern_id"]
            ==
            anchor_id
    )

    p4_passed = (
            len(actions)
            ==
            len(
                set(
                    actions
                )
            )
    )

    p5_passed = (
            mean_effectiveness
            >=
            MITIGATION_FLOOR
    )

    p6_passed = (
            total_residual
            <=
            RESIDUAL_BUDGET
    )

    passed_flags = [
        p1_passed,
        p2_passed,
        p3_passed,
        p4_passed,
        p5_passed,
        p6_passed
    ]

    for policy, passed in zip(
        POLICY_FRAME,
        passed_flags
    ):

        results.append(
            {
                "policy_id":
                    policy["policy_id"],

                "name":
                    policy["name"],

                "passed":
                    passed
            }
        )

    return results


APPROVED_CHECKS = check_policies(
    TUNED_PLAN,
    RECORD_MAP,
    EXTENDED_HISTORY,
    CRITICAL_SET,
    APPROVED_REHEARSALS,
    APPROVED_RESIDUAL,
    APPROVED_MEAN
)

list(
    map(
        lambda check: print(
            check["policy_id"],
            "|",
            check["name"],
            "|",
            "PASS"
            if check["passed"]
            else "FAIL"
        ),
        APPROVED_CHECKS
    )
)

assert all(
    check["passed"]
    for check
    in APPROVED_CHECKS
), "Approved plan must pass every governance check."

print()

print(
    "TEST 9: Governance Approval Verdict"
)

APPROVED_VERDICT = (
    "APPROVED"
    if all(
        check["passed"]
        for check
        in APPROVED_CHECKS
    )
    else "VETOED"
)

assert APPROVED_VERDICT == "APPROVED", (
    "Compliant plan must be approved."
)

print(
    "Verdict:",
    APPROVED_VERDICT
)

print()

print(
    "TEST 10: Construct Policy-Violating Candidate"
)

VIOLATING_CANDIDATE = [
    {
        "step":
            1,

        "pattern_id":
            "pattern_002",

        "action":
            "VALIDATE_SCHEMA_CONSISTENCY",

        "dependency":
            "VALIDATE_EVIDENCE_PROVENANCE"
    },
    {
        "step":
            2,

        "pattern_id":
            "pattern_001",

        "action":
            "VALIDATE_EVIDENCE_PROVENANCE",

        "dependency":
            None
    },
    {
        "step":
            3,

        "pattern_id":
            "pattern_005",

        "action":
            "VALIDATE_TIMELINE_INTEGRITY",

        "dependency":
            "VALIDATE_EVIDENCE_PROVENANCE"
    },
    {
        "step":
            4,

        "pattern_id":
            "pattern_003",

        "action":
            "VALIDATE_SENSOR_ALIGNMENT",

        "dependency":
            "VALIDATE_SCHEMA_CONSISTENCY"
    },
    {
        "step":
            5,

        "pattern_id":
            "pattern_006",

        "action":
            "VALIDATE_ACTUATOR_RESPONSE",

        "dependency":
            "VALIDATE_SENSOR_ALIGNMENT"
    }
]

assert len(
    VIOLATING_CANDIDATE
) == 5, (
    "Violating candidate must drop one pattern."
)

assert "pattern_004" not in {
    step["pattern_id"]
    for step
    in VIOLATING_CANDIDATE
}, "Violating candidate must drop the RISING pattern."

print(
    "Violating candidate:",
    [
        step["pattern_id"]
        for step
        in VIOLATING_CANDIDATE
    ]
)

print(
    "Candidate defect: schema before evidence, telemetry dropped."
)

print()

print(
    "TEST 11: Rehearse and Check Violating Candidate"
)

CANDIDATE_REHEARSALS, CANDIDATE_RESIDUAL, CANDIDATE_MEAN = (
    rehearse_plan(
        VIOLATING_CANDIDATE,
        RECORD_MAP,
        EXTENDED_HISTORY
    )
)

assert len(
    CANDIDATE_REHEARSALS
) == 5, (
    "Candidate rehearsal must cover five steps."
)

CANDIDATE_PLANNED = {
    step["pattern_id"]
    for step
    in VIOLATING_CANDIDATE
}

CANDIDATE_COVERAGE = (
        len(
            CRITICAL_SET
            &
            CANDIDATE_PLANNED
        )
        /
        len(
            CRITICAL_SET
        )
)

assert abs(
    CANDIDATE_COVERAGE
    -
    0.5
) <= 1e-9, "Candidate coverage must be exactly 0.5."

CANDIDATE_CHECKS = check_policies(
    VIOLATING_CANDIDATE,
    RECORD_MAP,
    EXTENDED_HISTORY,
    CRITICAL_SET,
    CANDIDATE_REHEARSALS,
    CANDIDATE_RESIDUAL,
    CANDIDATE_MEAN
)

list(
    map(
        lambda check: print(
            check["policy_id"],
            "|",
            check["name"],
            "|",
            "PASS"
            if check["passed"]
            else "FAIL"
        ),
        CANDIDATE_CHECKS
    )
)

print(
    "Candidate coverage ratio:",
    CANDIDATE_COVERAGE
)

print(
    "Candidate mean effectiveness:",
    format(
        CANDIDATE_MEAN,
        ".4f"
    )
)

print(
    "Candidate total residual:",
    format(
        CANDIDATE_RESIDUAL,
        ".6f"
    )
)

FAILED_POLICIES = [
    check["policy_id"]
    for check
    in CANDIDATE_CHECKS
    if not check["passed"]
]

assert len(
    FAILED_POLICIES
) >= 3, (
    "Violating candidate must fail at least three policies."
)

assert "P1" in FAILED_POLICIES, (
    "Candidate must fail coverage."
)

assert "P2" in FAILED_POLICIES, (
    "Candidate must fail dependency order."
)

assert "P3" in FAILED_POLICIES, (
    "Candidate must fail anchor immutability."
)

print()

print(
    "TEST 12: Governance Veto Verdict"
)

CANDIDATE_VERDICT = (
    "APPROVED"
    if all(
        check["passed"]
        for check
        in CANDIDATE_CHECKS
    )
    else "VETOED"
)

assert CANDIDATE_VERDICT == "VETOED", (
    "Violating plan must be vetoed."
)

print(
    "Verdict:",
    CANDIDATE_VERDICT
)

print()

print(
    "TEST 13: Compare Verdicts"
)

VERDICTS_DIFFER = (
        APPROVED_VERDICT
        ==
        "APPROVED"
        and
        CANDIDATE_VERDICT
        ==
        "VETOED"
)

assert VERDICTS_DIFFER, (
    "Governance must distinguish compliant and violating plans."
)

print(
    "Approved plan:",
    APPROVED_VERDICT
)

print(
    "Violating candidate:",
    CANDIDATE_VERDICT
)

print(
    "Governance discriminates:",
    VERDICTS_DIFFER
)

print()

print(
    "TEST 14: Deterministic Governance"
)

SECOND_REHEARSALS, SECOND_RESIDUAL, SECOND_MEAN = rehearse_plan(
    TUNED_PLAN,
    RECORD_MAP,
    EXTENDED_HISTORY
)

SECOND_CHECKS = check_policies(
    TUNED_PLAN,
    RECORD_MAP,
    EXTENDED_HISTORY,
    CRITICAL_SET,
    SECOND_REHEARSALS,
    SECOND_RESIDUAL,
    SECOND_MEAN
)

DETERMINISTIC = (
        stable_hash(
            APPROVED_REHEARSALS
        )
        ==
        stable_hash(
            SECOND_REHEARSALS
        )
        and
        stable_hash(
            APPROVED_CHECKS
        )
        ==
        stable_hash(
            SECOND_CHECKS
        )
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Governance review is nondeterministic."
)

print(
    "Deterministic governance validated."
)

print()

print(
    "TEST 15: Numerical Health"
)

RESIDUAL_TENSOR = torch.tensor(
    [
        item["residual_risk"]
        for item
        in APPROVED_REHEARSALS
    ],
    dtype=torch.float32
)

CANDIDATE_TENSOR = torch.tensor(
    [
        item["residual_risk"]
        for item
        in CANDIDATE_REHEARSALS
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        RESIDUAL_TENSOR
    ).all()
    and
    torch.isfinite(
        CANDIDATE_TENSOR
    ).all()
)

print(
    "Residual NaN:",
    int(
        torch.isnan(
            RESIDUAL_TENSOR
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
    "Candidate NaN:",
    int(
        torch.isnan(
            CANDIDATE_TENSOR
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
    if len(APPROVED_REHEARSALS) == EXPECTED_PATTERNS
    else [
        "Rehearsal step count invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        APPROVED_MEAN
        -
        EXPECTED_MEAN_REHEARSAL
    ) <= 1e-9
    else [
        "Rehearsed mean effectiveness invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        COVERAGE_RATIO
        -
        1.0
    ) <= 1e-9
    else [
        "Approved plan coverage incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        check["passed"]
        for check
        in APPROVED_CHECKS
    )
    else [
        "Compliant plan failed governance."
    ]
)

PROMOTION_ERRORS += (
    []
    if APPROVED_VERDICT == "APPROVED"
    else [
        "Approved verdict wrong."
    ]
)

PROMOTION_ERRORS += (
    []
    if CANDIDATE_VERDICT == "VETOED"
    else [
        "Veto verdict wrong."
    ]
)

PROMOTION_ERRORS += (
    []
    if VERDICTS_DIFFER
    else [
        "Governance cannot discriminate."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Governance nondeterministic."
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
    "Rehearsed steps:",
    len(
        APPROVED_REHEARSALS
    )
)

print(
    "Mean effectiveness:",
    format(
        APPROVED_MEAN,
        ".4f"
    )
)

print(
    "Coverage ratio:",
    COVERAGE_RATIO
)

print(
    "Approved verdict:",
    APPROVED_VERDICT
)

print(
    "Candidate verdict:",
    CANDIDATE_VERDICT
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "126R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "126R promotion gate passed."
)

print()

print(
    "TEST 17: Persist Governance Memory"
)

AUDIT_TRAIL = [
    {
        "lesson":
            "126R",

        "policy_id":
            check["policy_id"],

        "name":
            check["name"],

        "plan":
            "approved",

        "verdict":
            check["passed"]
    }
    for check
    in APPROVED_CHECKS
]

AUDIT_TRAIL += [
    {
        "lesson":
            "126R",

        "policy_id":
            check["policy_id"],

        "name":
            check["name"],

        "plan":
            "candidate",

        "verdict":
            check["passed"]
    }
    for check
    in CANDIDATE_CHECKS
]

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "126R",

    "capability":
        "preventive_control_loop_governance_policy_rehearsal",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "125R",

    "tuned_records":
        TUNED_RECORDS,

    "tuned_order":
        TUNED_ORDER,

    "tuned_plan":
        TUNED_PLAN,

    "trend_map":
        TREND_MAP,

    "extended_history":
        EXTENDED_HISTORY,

    "policy_frame":
        POLICY_FRAME,

    "critical_set":
        sorted(
            CRITICAL_SET
        ),

    "mitigation_floor":
        MITIGATION_FLOOR,

    "residual_budget":
        RESIDUAL_BUDGET,

    "approved_rehearsal":
        {
            "steps":
                APPROVED_REHEARSALS,

            "total_residual":
                APPROVED_RESIDUAL,

            "mean_effectiveness":
                APPROVED_MEAN,

            "coverage_ratio":
                COVERAGE_RATIO
        },

    "candidate_rehearsal":
        {
            "steps":
                CANDIDATE_REHEARSALS,

            "total_residual":
                CANDIDATE_RESIDUAL,

            "mean_effectiveness":
                CANDIDATE_MEAN,

            "coverage_ratio":
                CANDIDATE_COVERAGE
        },

    "governance_verdicts":
        {
            "approved_plan":
                APPROVED_VERDICT,

            "violating_candidate":
                CANDIDATE_VERDICT
        },

    "audit_trail":
        AUDIT_TRAIL,

    "verification":
        {
            "approved":
                APPROVED_VERDICT == "APPROVED",

            "vetoed":
                CANDIDATE_VERDICT == "VETOED",

            "discriminates":
                VERDICTS_DIFFER,

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
    RELOADED["approved_rehearsal"]["steps"]
) == len(
    APPROVED_REHEARSALS
), "Rehearsal length changed after reload."

assert len(
    RELOADED["audit_trail"]
) == len(
    AUDIT_TRAIL
), "Audit trail changed after reload."

assert RELOADED[
    "governance_verdicts"
][
    "approved_plan"
] == "APPROVED", "Approved verdict changed after reload."

assert RELOADED[
    "governance_verdicts"
][
    "violating_candidate"
] == "VETOED", "Veto verdict changed after reload."

print(
    "Reloaded rehearsal steps:",
    len(
        RELOADED["approved_rehearsal"]["steps"]
    )
)

print(
    "Reloaded audit entries:",
    len(
        RELOADED["audit_trail"]
    )
)

print(
    "Reloaded approved verdict:",
    RELOADED[
        "governance_verdicts"
    ][
        "approved_plan"
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
            "126R",

        "capability":
            "preventive_control_loop_governance_policy_rehearsal",

        "policy_frame":
            POLICY_FRAME,

        "critical_set":
            sorted(
                CRITICAL_SET
            ),

        "approved_rehearsal":
            {
                "steps":
                    APPROVED_REHEARSALS,

                "total_residual":
                    APPROVED_RESIDUAL,

                "mean_effectiveness":
                    APPROVED_MEAN,

                "coverage_ratio":
                    COVERAGE_RATIO
            },

        "candidate_rehearsal":
            {
                "steps":
                    CANDIDATE_REHEARSALS,

                "total_residual":
                    CANDIDATE_RESIDUAL,

                "mean_effectiveness":
                    CANDIDATE_MEAN,

                "coverage_ratio":
                    CANDIDATE_COVERAGE
            },

        "governance_verdicts":
            {
                "approved_plan":
                    APPROVED_VERDICT,

                "violating_candidate":
                    CANDIDATE_VERDICT
            },

        "audit_trail":
            AUDIT_TRAIL
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "126R",

        "memory_version":
            MEMORY_VERSION,

        "policy_count":
            len(
                POLICY_FRAME
            ),

        "critical_count":
            len(
                CRITICAL_SET
            ),

        "approved_mean_effectiveness":
            APPROVED_MEAN,

        "approved_total_residual":
            APPROVED_RESIDUAL,

        "approved_coverage":
            COVERAGE_RATIO,

        "approved_verdict":
            APPROVED_VERDICT,

        "candidate_verdict":
            CANDIDATE_VERDICT,

        "failed_candidate_policies":
            FAILED_POLICIES,

        "audit_entries":
            len(
                AUDIT_TRAIL
            ),

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
            "126R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "127R Multi-Layer Defense Simulation "
                "+ Adaptive Control"
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
    "SILVERWING 126R ARCHITECTURE"
)

print(
    "Incoming Tuned Plan (125R)"
)

print(
    "        ↓"
)

print(
    "Governance Policy Frame"
)

print(
    "        ↓"
)

print(
    "Critical Set Identification"
)

print(
    "        ↓"
)

print(
    "Policy Rehearsal"
)

print(
    "        ↓"
)

print(
    "Per-Policy Checks"
)

print(
    "        ↓"
)

print(
    "APPROVED / VETOED Verdict"
)

print(
    "        ↓"
)

print(
    "Audit Trail"
)

print()

print(
    "WHAT 126R ADDS"
)

print(
    "Explicit governance over the adaptive loop, critical-set "
    "coverage, rehearsal-based feasibility, per-policy verdicts "
    "and a veto mechanism that rejects non-compliant plans."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Any adaptive prevention loop whose autonomy must stay "
    "bounded by explicit safety and policy invariants."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A loop that can change its own plan must be checked before "
    "it acts. Governance turns adaptive freedom into audited "
    "freedom: the planner proposes, the policy disposes."
)

print()

print(
    "NEXT: 127R Multi-Layer Defense Simulation + Adaptive Control"
)

print()

print(
    "=== LESSON 126R COMPLETE ==="
)

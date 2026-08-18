# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 132R
# Autonomous Defense Self-Tuning + Online Control Refinement
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# 131R produced the consolidated scorecard: a per-pattern
# reading of risk, trend, anomaly, tier, control action and
# controlled stop rate. 132R asks the next question: what
# happens when a pattern's EFFECTIVENESS degrades after the
# defense is already deployed?
#
# Prevention is static: 127R set ESCALATE on the anomaly set and
# HOLD on the rest. 132R makes defense ALIVE. When pattern_004's
# defense weakens, its penetration rises above tolerance. The
# controller observes the violation, then refines the
# reinforcement applied to that one pattern, cycle by cycle,
# until penetration returns under the ceiling. No manual
# escalation, no human tuning: the loop tunes itself from
# observations alone.
#
# Online control refinement:
#
#     controlled stop rate (defense)
#               ↓
#     stress degradation
#               ↓
#     penetration = residual^boost
#               ↓
#     tolerance violation observed
#               ↓
#     log-domain boost refinement
#               ↓
#     converged reinforcement
#
# The self-tuning loop is deterministic: penetration is a closed
# function of boost, and boost follows a fixed control law, so a
# given stress always converges to the same reinforcement.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 131R memory is the source of truth.
# 2. Controlled stop rate defines per-pattern defense strength.
# 3. A pattern is stressed by degrading its defense.
# 4. Penetration = (1 - defense) ^ boost.
# 5. Boost is the reinforcement applied to a pattern.
# 6. Tolerance is the penetration ceiling (127R threshold 0.12).
# 7. The controller observes penetration each cycle.
# 8. Boost is refined in the log domain (deterministic).
# 9. Only out-of-compliance patterns are tuned.
# 10. Self-tuning must converge within the cycle budget.
# 11. Tuning must be monotone (no oscillation).
# 12. Tuning must reduce exposure below the static baseline.
# 13. Determinism must be checked.
# 14. Numerical health must be checked.
# 15. Persistence and reload must be checked.
# 16. Promotion requires all validation gates to pass.
# 17. External LLM: NONE.
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
MEMORY_VERSION = "132R.1"
EXPECTED_PATTERNS = 6
TOLERANCE = 0.12
STRESS_PATTERN = "pattern_004"
STRESS_DEGRADATION = 0.85
LEARNING_RATE = 0.75
MAX_CYCLES = 8

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_131R = PHASE5_DIR / "lesson131R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_131R
        / "silverwing_resilience_audit_memory.json"
)

SOURCE_INDEX = (
        LESSON_131R
        / "silverwing_resilience_audit_index.pt"
)

SOURCE_DATASET = (
        LESSON_131R
        / "silverwing_resilience_audit_dataset.json"
)

SOURCE_REPORT = (
        LESSON_131R
        / "silverwing_resilience_audit_report.json"
)

SOURCE_REGISTRY = (
        LESSON_131R
        / "silverwing_resilience_audit_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_131R
        / "checkpoints"
        / "silverwing_resilience_audit_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_self_tuning_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_self_tuning_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_self_tuning_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_self_tuning_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_self_tuning_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_self_tuning_best.pt"
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
    "PHASE 5 - LESSON 132R"
)

print(
    "Autonomous Defense Self-Tuning"
)

print(
    "+ Online Control Refinement"
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
    "TEST 1: Verify 131R Inputs"
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
    "TEST 2: Load 131R Consolidated Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "131R audit memory is invalid."

SCORECARD = SOURCE.get(
    "scorecard",
    {}
)

assert isinstance(
    SCORECARD,
    dict
), "131R scorecard is invalid."

assert len(
    SCORECARD
) == EXPECTED_PATTERNS, (
    "131R scorecard must cover six patterns."
)

TUNED_ORDER = SOURCE.get(
    "tuned_order",
    []
)

CONTROL_ACTIONS = SOURCE.get(
    "control_actions",
    {}
)

SYSTEM_METRICS = SOURCE.get(
    "system_metrics",
    {}
)

RESILIENCE_SCORE = SOURCE.get(
    "resilience_score"
)

RESILIENCE_GRADE = SOURCE.get(
    "resilience_grade"
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Resilience grade:",
    RESILIENCE_GRADE
)

print(
    "Collective exposure:",
    format(
        SYSTEM_METRICS.get(
            "collective_exposure",
            0.0
        ),
        ".4f"
    )
)

print()

print(
    "TEST 3: Extract Control Baseline"
)

DEFENSE = {}

for pattern_id in TUNED_ORDER:

    record = SCORECARD[
        pattern_id
    ]

    DEFENSE[
        pattern_id
    ] = record[
        "controlled_stop_rate"
    ]

BASE_PENETRATION = {
    pattern_id: (
        1.0
        -
        DEFENSE[
            pattern_id
        ]
    )
    for pattern_id
    in TUNED_ORDER
}

assert abs(
    BASE_PENETRATION["pattern_004"]
    -
    0.104590
) <= 1e-4, (
    "pattern_004 base penetration mismatch."
)

assert all(
    BASE_PENETRATION[
        pattern_id
    ]
    <=
    TOLERANCE
    for pattern_id
    in TUNED_ORDER
), "Baseline must be compliant before stress."

print(
    "Tolerance:",
    TOLERANCE
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "|",
        CONTROL_ACTIONS[
            pattern_id
        ],
        "| defense=",
        format(
            DEFENSE[
                pattern_id
            ],
            ".4f"
        ),
        "| penetration=",
        format(
            BASE_PENETRATION[
                pattern_id
            ],
            ".4f"
        )
    )

print()

print(
    "TEST 4: Define Self-Tuning Controller"
)

assert (
        CONTROL_ACTIONS[STRESS_PATTERN]
        ==
        "ESCALATE"
), "Stressed pattern must be an anomaly under escalation."

assert (
        BASE_PENETRATION[STRESS_PATTERN]
        <
        TOLERANCE
), "Stressed pattern must start compliant."


def refine_control(
        defense,
        tolerance,
        learning_rate,
        max_cycles
):

    base_penetration = (
            1.0
            -
            defense
    )

    ln_base = math.log(
        base_penetration
    )

    boost = 1.0

    penetration = (
            base_penetration
            **
            boost
    )

    history = [
        {
            "cycle": 1,
            "boost": round(
                boost,
                4
            ),
            "penetration": round(
                penetration,
                4
            )
        }
    ]

    for cycle in range(
            2,
            max_cycles + 1
    ):

        boost = (
                boost
                -
                learning_rate
                *
                math.log(
                    penetration
                    /
                    tolerance
                )
                /
                ln_base
        )

        penetration = (
                base_penetration
                **
                boost
        )

        history.append(
            {
                "cycle": cycle,
                "boost": round(
                    boost,
                    4
                ),
                "penetration": round(
                    penetration,
                    4
                )
            }
        )

    return boost, penetration, history


print(
    "Penetration model: (1 - defense) ^ boost"
)

print(
    "Control law: log-domain refinement"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Cycle budget:",
    MAX_CYCLES
)

print()

print(
    "TEST 5: Inject Stress on Stressed Pattern"
)

STRESSED_DEFENSE = (
        DEFENSE[STRESS_PATTERN]
        *
        STRESS_DEGRADATION
)

STATIC_PENETRATION = (
        1.0
        -
        STRESSED_DEFENSE
)

assert abs(
    STATIC_PENETRATION
    -
    0.238902
) <= 1e-4, (
    "Stressed penetration mismatch."
)

assert (
        STATIC_PENETRATION
        >
        TOLERANCE
), "Stress must violate the tolerance ceiling."

STATIC_EXPOSURE = sum(
    (
        STATIC_PENETRATION
        if pattern_id == STRESS_PATTERN
        else BASE_PENETRATION[
            pattern_id
        ]
    )
    for pattern_id
    in TUNED_ORDER
) / EXPECTED_PATTERNS

assert abs(
    STATIC_EXPOSURE
    -
    0.117851
) <= 1e-4, (
    "Static stressed exposure mismatch."
)

print(
    "Stressed defense:",
    format(
        STRESSED_DEFENSE,
        ".4f"
    )
)

print(
    "Static penetration:",
    format(
        STATIC_PENETRATION,
        ".4f"
    ),
    "(above tolerance)"
)

print(
    "Static exposure:",
    format(
        STATIC_EXPOSURE,
        ".4f"
    )
)

print()

print(
    "TEST 6: Run Online Refinement"
)

TUNED_BOOST, TUNED_PENETRATION, TUNING_HISTORY = refine_control(
    STRESSED_DEFENSE,
    TOLERANCE,
    LEARNING_RATE,
    MAX_CYCLES
)

assert TUNING_HISTORY[0][
    "penetration"
] == 0.2389, (
    "First observation mismatch."
)

assert abs(
    TUNING_HISTORY[1]["boost"]
    -
    1.3607
) <= 1e-4, (
    "Cycle 2 boost mismatch."
)

BOOST_SEQUENCE = [
    entry["boost"]
    for entry
    in TUNING_HISTORY
]

PENETRATION_SEQUENCE = [
    entry["penetration"]
    for entry
    in TUNING_HISTORY
]

assert BOOST_SEQUENCE == sorted(
    BOOST_SEQUENCE
), "Boost refinement must be monotone."

assert PENETRATION_SEQUENCE == sorted(
    PENETRATION_SEQUENCE,
    reverse=True
), "Penetration must fall every cycle."

for entry in TUNING_HISTORY:

    print(
        "Cycle",
        entry["cycle"],
        "| boost=",
        format(
            entry["boost"],
            ".4f"
        ),
        "| penetration=",
        format(
            entry["penetration"],
            ".4f"
        )
    )

print()

print(
    "TEST 7: Verify Convergence"
)

assert (
        abs(
            TUNED_PENETRATION
            -
            TOLERANCE
        )
        <=
        1e-4
), "Self-tuning must converge to tolerance."

assert (
        1.45
        <
        TUNED_BOOST
        <
        1.51
), "Converged boost must land in the reinforcement band."

print(
    "Converged boost:",
    format(
        TUNED_BOOST,
        ".4f"
    )
)

print(
    "Converged penetration:",
    format(
        TUNED_PENETRATION,
        ".4f"
    )
)

print(
    "Closed-loop error:",
    format(
        abs(
            TUNED_PENETRATION
            -
            TOLERANCE
        ),
        ".6f"
    )
)

print()

print(
    "TEST 8: Verify Autonomous Scope"
)

REFINED_CONTROL = {}

for pattern_id in TUNED_ORDER:

    if pattern_id == STRESS_PATTERN:

        REFINED_CONTROL[
            pattern_id
        ] = {
            "control_action":
                CONTROL_ACTIONS[
                    pattern_id
                ],
            "defense":
                STRESSED_DEFENSE,
            "boost":
                TUNED_BOOST,
            "penetration":
                TUNED_PENETRATION,
            "compliant":
                TUNED_PENETRATION
                <=
                TOLERANCE
                +
                1e-4
        }

    else:

        REFINED_CONTROL[
            pattern_id
        ] = {
            "control_action":
                CONTROL_ACTIONS[
                    pattern_id
                ],
            "defense":
                DEFENSE[
                    pattern_id
                ],
            "boost":
                1.0,
            "penetration":
                BASE_PENETRATION[
                    pattern_id
                ],
            "compliant":
                BASE_PENETRATION[
                    pattern_id
                ]
                <=
                TOLERANCE
        }

assert set(
    REFINED_CONTROL
) == set(
    TUNED_ORDER
), "Refined control must cover all patterns."

assert all(
    REFINED_CONTROL[
        pattern_id
    ][
        "compliant"
    ]
    for pattern_id
    in TUNED_ORDER
), "Every pattern must end compliant."

assert all(
    REFINED_CONTROL[
        pattern_id
    ][
        "boost"
    ]
    ==
    1.0
    for pattern_id
    in TUNED_ORDER
    if pattern_id != STRESS_PATTERN
), "Only the stressed pattern may be tuned."

print(
    "Tuned patterns:",
    [
        pattern_id
        for pattern_id
        in TUNED_ORDER
        if REFINED_CONTROL[
            pattern_id
        ][
            "boost"
        ]
        !=
        1.0
    ]
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "|",
        REFINED_CONTROL[
            pattern_id
        ][
            "control_action"
        ],
        "| boost=",
        format(
            REFINED_CONTROL[
                pattern_id
            ][
                "boost"
            ],
            ".4f"
        ),
        "| compliant=",
        REFINED_CONTROL[
            pattern_id
        ][
            "compliant"
        ]
    )

print()

print(
    "TEST 9: Compare Static Baseline"
)

TUNED_EXPOSURE = sum(
    REFINED_CONTROL[
        pattern_id
    ][
        "penetration"
    ]
    for pattern_id
    in TUNED_ORDER
) / EXPECTED_PATTERNS

assert abs(
    TUNED_EXPOSURE
    -
    0.098035
) <= 1e-4, (
    "Tuned exposure mismatch."
)

assert (
        TUNED_EXPOSURE
        <
        STATIC_EXPOSURE
), "Self-tuning must beat the static baseline."

EXPOSURE_REDUCTION = (
        STATIC_EXPOSURE
        -
        TUNED_EXPOSURE
)

print(
    "Static exposure:",
    format(
        STATIC_EXPOSURE,
        ".4f"
    )
)

print(
    "Tuned exposure:",
    format(
        TUNED_EXPOSURE,
        ".4f"
    )
)

print(
    "Reduction:",
    format(
        EXPOSURE_REDUCTION,
        ".4f"
    )
)

print()

print(
    "TEST 10: Stability"
)

STABLE_BOOST, STABLE_PENETRATION, STABLE_HISTORY = refine_control(
    STRESSED_DEFENSE,
    TOLERANCE,
    LEARNING_RATE,
    MAX_CYCLES
)

assert abs(
    STABLE_PENETRATION
    -
    TUNED_PENETRATION
) <= 1e-4, (
    "Re-run must reproduce the converged state."
)

assert (
        STABLE_PENETRATION
        <=
        TOLERANCE
        +
        1e-4
), "Converged state must remain compliant."

assert STABLE_HISTORY == TUNING_HISTORY, (
    "Second pass must not oscillate."
)

print(
    "Converged penetration:",
    format(
        STABLE_PENETRATION,
        ".4f"
    )
)

print(
    "Stable within tolerance:",
    STABLE_PENETRATION
    <=
    TOLERANCE
    +
    1e-4
)

print()

print(
    "TEST 11: Determinism"
)

assert abs(
    STABLE_BOOST
    -
    TUNED_BOOST
) <= 1e-9, (
    "Self-tuning is nondeterministic."
)

print(
    "Deterministic:",
    abs(
        STABLE_BOOST
        -
        TUNED_BOOST
    )
    <=
    1e-9
)

print(
    "Deterministic refinement validated."
)

print()

print(
    "TEST 12: Numerical Health"
)

BOOST_TENSOR = torch.tensor(
    BOOST_SEQUENCE,
    dtype=torch.float32
)

PENETRATION_TENSOR = torch.tensor(
    [
        REFINED_CONTROL[
            pattern_id
        ][
            "penetration"
        ]
        for pattern_id
        in TUNED_ORDER
    ],
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
    "TEST 13: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if abs(
        TUNED_PENETRATION
        -
        TOLERANCE
    ) <= 1e-4
    else [
        "Self-tuning did not converge to tolerance."
    ]
)

PROMOTION_ERRORS += (
    []
    if (
            1.45
            <
            TUNED_BOOST
            <
            1.51
    )
    else [
        "Converged boost outside the reinforcement band."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        REFINED_CONTROL[
            pattern_id
        ][
            "compliant"
        ]
        for pattern_id
        in TUNED_ORDER
    )
    else [
        "Not every pattern is compliant after tuning."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        REFINED_CONTROL[
            pattern_id
        ][
            "boost"
        ]
        ==
        1.0
        for pattern_id
        in TUNED_ORDER
        if pattern_id != STRESS_PATTERN
    )
    else [
        "Self-tuning touched an already-compliant pattern."
    ]
)

PROMOTION_ERRORS += (
    []
    if TUNED_EXPOSURE < STATIC_EXPOSURE
    else [
        "Self-tuning did not improve system exposure."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        STABLE_PENETRATION
        -
        TUNED_PENETRATION
    ) <= 1e-4
    else [
        "Self-tuning oscillated on the second pass."
    ]
)

PROMOTION_ERRORS += (
    []
    if abs(
        STABLE_BOOST
        -
        TUNED_BOOST
    ) <= 1e-9
    else [
        "Self-tuning is nondeterministic."
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
    "Converged boost:",
    format(
        TUNED_BOOST,
        ".4f"
    )
)

print(
    "Converged penetration:",
    format(
        TUNED_PENETRATION,
        ".4f"
    )
)

print(
    "Tuned exposure:",
    format(
        TUNED_EXPOSURE,
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
        "132R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "132R promotion gate passed."
)

print()

print(
    "TEST 14: Persist Self-Tuning Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "132R",

    "capability":
        "autonomous_defense_self_tuning_online_control_refinement",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "131R",

    "control_frame":
        {
            "tolerance":
                TOLERANCE,

            "stress_pattern":
                STRESS_PATTERN,

            "stress_degradation":
                STRESS_DEGRADATION,

            "learning_rate":
                LEARNING_RATE,

            "max_cycles":
                MAX_CYCLES,

            "penetration_model":
                "(1 - defense) ^ boost"
        },

    "baseline":
        {
            "defense":
                DEFENSE,

            "base_penetration":
                BASE_PENETRATION,

            "control_actions":
                CONTROL_ACTIONS
        },

    "stress_scenario":
        {
            "stressed_defense":
                STRESSED_DEFENSE,

            "static_penetration":
                STATIC_PENETRATION,

            "static_exposure":
                STATIC_EXPOSURE
        },

    "tuning_history":
        TUNING_HISTORY,

    "refined_control":
        REFINED_CONTROL,

    "convergence":
        {
            "converged_boost":
                TUNED_BOOST,

            "converged_penetration":
                TUNED_PENETRATION,

            "tuned_exposure":
                TUNED_EXPOSURE,

            "exposure_reduction":
                EXPOSURE_REDUCTION,

            "cycles":
                MAX_CYCLES
        },

    "verification":
        {
            "converged":
                abs(
                    TUNED_PENETRATION
                    -
                    TOLERANCE
                )
                <=
                1e-4,

            "autonomous_scope":
                all(
                    REFINED_CONTROL[
                        pattern_id
                    ][
                        "boost"
                    ]
                    ==
                    1.0
                    for pattern_id
                    in TUNED_ORDER
                    if pattern_id != STRESS_PATTERN
                ),

            "monotone":
                BOOST_SEQUENCE
                ==
                sorted(
                    BOOST_SEQUENCE
                ),

            "stable":
                abs(
                    STABLE_PENETRATION
                    -
                    TUNED_PENETRATION
                )
                <=
                1e-4,

            "deterministic":
                abs(
                    STABLE_BOOST
                    -
                    TUNED_BOOST
                )
                <=
                1e-9,

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
    "TEST 15: Reload Persistent Memory"
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
        RELOADED["convergence"]["converged_boost"]
        ==
        TUNED_BOOST
), "Converged boost changed after reload."

assert (
        RELOADED["tuning_history"]
        ==
        TUNING_HISTORY
), "Tuning history changed after reload."

assert (
        RELOADED["refined_control"]
        ==
        REFINED_CONTROL
), "Refined control changed after reload."

print(
    "Reloaded converged boost:",
    format(
        RELOADED[
            "convergence"
        ][
            "converged_boost"
        ],
        ".4f"
    )
)

print(
    "Reloaded tuned exposure:",
    format(
        RELOADED[
            "convergence"
        ][
            "tuned_exposure"
        ],
        ".4f"
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 16: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "132R",

        "capability":
            "autonomous_defense_self_tuning_online_control_refinement",

        "control_frame":
            {
                "tolerance":
                    TOLERANCE,

                "stress_pattern":
                    STRESS_PATTERN,

                "stress_degradation":
                    STRESS_DEGRADATION,

                "learning_rate":
                    LEARNING_RATE,

                "max_cycles":
                    MAX_CYCLES
            },

        "static_exposure":
            STATIC_EXPOSURE,

        "tuned_exposure":
            TUNED_EXPOSURE,

        "exposure_reduction":
            EXPOSURE_REDUCTION,

        "tuned_boosts":
            {
                pattern_id:
                    REFINED_CONTROL[
                        pattern_id
                    ][
                        "boost"
                    ]
                for pattern_id
                in TUNED_ORDER
            },

        "converged_penetrations":
            {
                pattern_id:
                    REFINED_CONTROL[
                        pattern_id
                    ][
                        "penetration"
                    ]
                for pattern_id
                in TUNED_ORDER
            }
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "132R",

        "memory_version":
            MEMORY_VERSION,

        "stress_pattern":
            STRESS_PATTERN,

        "static_penetration":
            STATIC_PENETRATION,

        "converged_boost":
            TUNED_BOOST,

        "converged_penetration":
            TUNED_PENETRATION,

        "static_exposure":
            STATIC_EXPOSURE,

        "tuned_exposure":
            TUNED_EXPOSURE,

        "exposure_reduction":
            EXPOSURE_REDUCTION,

        "tuned_patterns":
            [
                pattern_id
                for pattern_id
                in TUNED_ORDER
                if REFINED_CONTROL[
                    pattern_id
                ][
                    "boost"
                ]
                !=
                1.0
            ],

        "all_compliant":
            all(
                REFINED_CONTROL[
                    pattern_id
                ][
                    "compliant"
                ]
                for pattern_id
                in TUNED_ORDER
            ),

        "monotone":
            BOOST_SEQUENCE
            ==
            sorted(
                BOOST_SEQUENCE
            ),

        "deterministic":
            abs(
                STABLE_BOOST
                -
                TUNED_BOOST
            )
            <=
            1e-9,

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
            "132R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "133R End-to-End Self-Improving Control Ledger "
                "+ Autonomous Policy Memory"
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
    "SILVERWING 132R ARCHITECTURE"
)

print(
    "Controlled Stop Rate (Defense)"
)

print(
    "   ↓"
)

print(
    "Stress Degradation"
)

print(
    "   ↓"
)

print(
    "Penetration = Residual^Boost"
)

print(
    "   ↓"
)

print(
    "Tolerance Violation Observed"
)

print(
    "   ↓"
)

print(
    "Log-Domain Boost Refinement"
)

print(
    "   ↓"
)

print(
    "Converged Reinforcement"
)

print(
    "   ↓"
)

print(
    "Exposure Reduced vs Static Baseline"
)

print()

print(
    "WHAT 132R ADDS"
)

print(
    "A control loop that tunes its own defense from "
    "observations: when a pattern weakens, reinforcement is "
    "refined cycle by cycle until penetration returns under "
    "tolerance. No manual escalation, no human tuning."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Live defense where effectiveness decays over time and "
    "escalation decisions must be made from data, not from "
    "a fixed precomputed policy."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A static defense rots as the environment shifts. "
    "Autonomous self-tuning keeps every pattern compliant "
    "while touching only what it must, and does so "
    "deterministically and provably."
)

print()

print(
    "NEXT: 133R End-to-End Self-Improving Control Ledger "
    "+ Autonomous Policy Memory"
)

print()

print(
    "=== LESSON 132R COMPLETE ==="
)

# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 127R
# Multi-Layer Defense Simulation + Adaptive Control
# ============================================================
#
# 122R  -> Continuous Adaptive Execution + Runtime Replanning
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
# 124R  -> Outcome Memory Consolidation + Risk Calibration
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
#
# ============================================================
# PURPOSE
# ============================================================
#
# By 126R a governance layer decides whether a plan may run.
# 127R asks: what happens when the plan runs? Instead of a
# single gate, defense is layered. Each pattern is attacked
# through a chain of layers; each layer has a probability of
# stopping the pattern before it breaks through.
#
# Multi-layer defense:
#
#     pattern
#         ↓
#     L1 DETECT
#         ↓
#     L2 INTERCEPT
#         ↓
#     L3 VERIFY
#         ↓
#     L4 REPAIR
#         ↓
#     penetration?
#
# Adaptive control:
#
#     per-pattern penetration
#         ↓
#     threshold compare
#         ↓
#     HOLD / ESCALATE
#         ↓
#     reinforcement boost
#         ↓
#     re-simulate (closed loop)
#
# Weak layers get reinforced until the expected penetration of
# every pattern falls below tolerance. The loop closes.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 126R memory is the source of truth.
# 2. Defense is modelled as a chain of independent layers.
# 3. Layer effectiveness derives from lifetime effectiveness.
# 4. Penetration is the product of layer failure probabilities.
# 5. Simulation is expected-value based: fully deterministic.
# 6. Control: HOLD when penetration is tolerable, ESCALATE else.
# 7. Escalation reinforces layers and re-simulates.
# 8. The loop must drive every pattern below tolerance.
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
MEMORY_VERSION = "127R.1"
EXPECTED_PATTERNS = 6
EXPECTED_LAYERS = 4
PENETRATION_THRESHOLD = 0.12
REINFORCEMENT_BOOST = 1.30
LAYER_FACTOR_BASE = 0.5
MEAN_STOP_FLOOR = 0.80
CONTROLLED_STOP_FLOOR = 0.88

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_126R = PHASE5_DIR / "lesson126R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_126R
        / "silverwing_control_loop_governance_memory.json"
)

SOURCE_INDEX = (
        LESSON_126R
        / "silverwing_control_loop_governance_index.pt"
)

SOURCE_DATASET = (
        LESSON_126R
        / "silverwing_control_loop_governance_dataset.json"
)

SOURCE_REPORT = (
        LESSON_126R
        / "silverwing_control_loop_governance_report.json"
)

SOURCE_REGISTRY = (
        LESSON_126R
        / "silverwing_control_loop_governance_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_126R
        / "checkpoints"
        / "silverwing_control_loop_governance_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_multi_layer_defense_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_multi_layer_defense_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_multi_layer_defense_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_multi_layer_defense_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_multi_layer_defense_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_multi_layer_defense_best.pt"
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
    "PHASE 5 - LESSON 127R"
)

print(
    "Multi-Layer Defense Simulation + Adaptive Control"
)

print()

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

print(
    "127R -> Multi-Layer Defense Simulation + Adaptive Control"
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
    "TEST 1: Verify 126R Inputs"
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
), "One or more 126R inputs are missing."

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
    "TEST 2: Load 126R Governance Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "126R governance memory is invalid."

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

EXTENDED_HISTORY = SOURCE.get(
    "extended_history",
    {}
)

AUDIT_TRAIL = SOURCE.get(
    "audit_trail",
    []
)

APPROVED_REHEARSAL = SOURCE.get(
    "approved_rehearsal",
    {}
)

assert len(
    TUNED_RECORDS
) == EXPECTED_PATTERNS, (
    "126R must supply exactly six tuned records."
)

assert len(
    EXTENDED_HISTORY
) == EXPECTED_PATTERNS, (
    "126R must carry six extended histories."
)

assert len(
    POLICY_FRAME
) == 6, (
    "126R must carry six governance policies."
)

assert len(
    AUDIT_TRAIL
) == 12, (
    "126R must carry twelve audit entries."
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
    "Critical set:",
    CRITICAL_SET
)

print()

print(
    "TEST 3: Rebuild Defense State"
)

RECORD_MAP = {
    record["pattern_id"]: record
    for record
    in TUNED_RECORDS
}

assert (
        sorted(
            RECORD_MAP.keys()
        )
        ==
        sorted(
            EXTENDED_HISTORY.keys()
        )
), "Record map and history must align."

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
    "TEST 4: Define Defense Layer Frame"
)

LAYER_FRAME = [
    {
        "layer_id":
            "L1",

        "name":
            "DETECT",

        "base_effectiveness":
            0.55
    },
    {
        "layer_id":
            "L2",

        "name":
            "INTERCEPT",

        "base_effectiveness":
            0.45
    },
    {
        "layer_id":
            "L3",

        "name":
            "VERIFY",

        "base_effectiveness":
            0.40
    },
    {
        "layer_id":
            "L4",

        "name":
            "REPAIR",

        "base_effectiveness":
            0.35
    }
]

assert len(
    LAYER_FRAME
) == EXPECTED_LAYERS, (
    "Defense frame must define four layers."
)

assert (
        abs(
            sum(
                layer["base_effectiveness"]
                for layer
                in LAYER_FRAME
            )
            -
            1.75
        )
        <=
        1e-9
), "Layer base effectiveness sum must be 1.75."

list(
    map(
        lambda layer: print(
            layer["layer_id"],
            "|",
            layer["name"],
            "| base=",
            layer["base_effectiveness"]
        ),
        LAYER_FRAME
    )
)

print()

print(
    "TEST 5: Compute Layer Effectiveness Matrix"
)

PATTERN_FACTOR = {
    pattern_id:
        (
                LAYER_FACTOR_BASE
                +
                LAYER_FACTOR_BASE
                *
                EXTENDED_HISTORY[
                    pattern_id
                ][
                    "lifetime_effectiveness"
                ]
        )
    for pattern_id
    in RECORD_MAP
}

LAYER_EFFECTIVENESS = {
    pattern_id:
        [
            clamp(
                layer["base_effectiveness"]
                *
                PATTERN_FACTOR[
                    pattern_id
                ]
            )
            for layer
            in LAYER_FRAME
        ]
    for pattern_id
    in RECORD_MAP
}

assert all(
    len(
        LAYER_EFFECTIVENESS[
            pattern_id
        ]
    ) == EXPECTED_LAYERS
    for pattern_id
    in RECORD_MAP
), "Effectiveness matrix must cover four layers per pattern."

print(
    "Pattern factor:",
    dict(
        map(
            lambda pair: (
                pair[0],
                round(
                    pair[1],
                    4
                )
            ),
            PATTERN_FACTOR.items()
        )
    )
)

print(
    "Layer effectiveness:"
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            [
                format(
                    eff,
                    ".4f"
                )
                for eff
                in LAYER_EFFECTIVENESS[
                    pattern_id
                ]
            ]
        ),
        sorted(
            RECORD_MAP.keys()
        )
    )
)

print()

print(
    "TEST 6: Multi-Layer Penetration Simulation"
)


def simulate_defense(
        record_map,
        history,
        layer_frame,
        controlled,
        boost_patterns=None
):

    if boost_patterns is None:

        boost_patterns = set()

    results = []

    for pattern_id in record_map:

        factor = (
                LAYER_FACTOR_BASE
                +
                LAYER_FACTOR_BASE
                *
                history[
                    pattern_id
                ][
                    "lifetime_effectiveness"
                ]
        )

        multiplier = (
            REINFORCEMENT_BOOST
            if (
                    controlled
                    and
                    pattern_id
                    in
                    boost_patterns
            )
            else 1.0
        )

        effs = [
            clamp(
                layer["base_effectiveness"]
                *
                factor
                *
                multiplier
            )
            for layer
            in layer_frame
        ]

        stop_profile = []

        remaining = 1.0

        for eff in effs:

            stop_profile.append(
                remaining
                *
                eff
            )

            remaining *= (
                1.0
                -
                eff
            )

        penetration = remaining

        expected_depth = sum(
            (index + 1) * prob
            for index, prob
            in enumerate(
                stop_profile
            )
        )

        expected_depth += (
                EXPECTED_LAYERS
                *
                penetration
        )

        results.append(
            {
                "pattern_id":
                    pattern_id,

                "risk_score":
                    record_map[
                        pattern_id
                    ][
                        "risk_score"
                    ],

                "layer_effectiveness":
                    effs,

                "stop_profile":
                    stop_profile,

                "penetration":
                    penetration,

                "stop_rate":
                    1.0
                    -
                    penetration,

                "expected_depth":
                    expected_depth
            }
        )

    return results


BASELINE = simulate_defense(
    RECORD_MAP,
    EXTENDED_HISTORY,
    LAYER_FRAME,
    False
)

assert len(
    BASELINE
) == EXPECTED_PATTERNS, (
    "Baseline simulation must cover six patterns."
)

list(
    map(
        lambda item: print(
            item["pattern_id"],
            "| pen=",
            format(
                item["penetration"],
                ".6f"
            ),
            "| stop=",
            format(
                item["stop_rate"],
                ".4f"
            ),
            "| depth=",
            format(
                item["expected_depth"],
                ".4f"
            )
        ),
        BASELINE
    )
)

print()

print(
    "TEST 7: Validate Baseline Simulation Metrics"
)

BASELINE_MAP = {
    item["pattern_id"]: item
    for item
    in BASELINE
}

assert abs(
    BASELINE_MAP["pattern_001"]["penetration"]
    -
    0.1071025
) <= 1e-5, "pattern_001 baseline penetration mismatch."

assert abs(
    BASELINE_MAP["pattern_004"]["penetration"]
    -
    0.2009344
) <= 1e-5, "pattern_004 baseline penetration mismatch."

assert abs(
    BASELINE_MAP["pattern_006"]["penetration"]
    -
    0.1317281
) <= 1e-5, "pattern_006 baseline penetration mismatch."

BASELINE_MEAN_STOP = sum(
    item["stop_rate"]
    for item
    in BASELINE
) / len(
    BASELINE
)

assert (
        BASELINE_MEAN_STOP
        >=
        MEAN_STOP_FLOOR
), "Baseline mean stop rate below floor."

print(
    "Mean baseline stop rate:",
    format(
        BASELINE_MEAN_STOP,
        ".4f"
    )
)

print()

print(
    "TEST 8: Classify Adaptive Control Actions"
)

CONTROL_ACTIONS = {}

for item in BASELINE:

    CONTROL_ACTIONS[
        item["pattern_id"]
    ] = (
        "ESCALATE"
        if item["penetration"]
           > PENETRATION_THRESHOLD
        else "HOLD"
    )

ESCALATE_SET = {
    pattern_id
    for pattern_id, action
    in CONTROL_ACTIONS.items()
    if action == "ESCALATE"
}

HOLD_SET = {
    pattern_id
    for pattern_id, action
    in CONTROL_ACTIONS.items()
    if action == "HOLD"
}

assert ESCALATE_SET == {
    "pattern_003",
    "pattern_004",
    "pattern_006"
}, "Escalation set mismatch."

assert HOLD_SET == {
    "pattern_001",
    "pattern_002",
    "pattern_005"
}, "Hold set mismatch."

assert len(
    ESCALATE_SET
) + len(
    HOLD_SET
) == EXPECTED_PATTERNS, (
    "Control actions must partition the pattern set."
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "|",
            CONTROL_ACTIONS[
                pattern_id
            ],
            "| pen=",
            format(
                BASELINE_MAP[
                    pattern_id
                ][
                    "penetration"
                ],
                ".6f"
            )
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 9: Apply Reinforcement (Adaptive Control)"
)

CONTROLLED = simulate_defense(
    RECORD_MAP,
    EXTENDED_HISTORY,
    LAYER_FRAME,
    True,
    ESCALATE_SET
)

CONTROLLED_MAP = {
    item["pattern_id"]: item
    for item
    in CONTROLLED
}

assert len(
    CONTROLLED
) == EXPECTED_PATTERNS, (
    "Controlled simulation must cover six patterns."
)

list(
    map(
        lambda item: print(
            item["pattern_id"],
            "| pen=",
            format(
                item["penetration"],
                ".6f"
            ),
            "| stop=",
            format(
                item["stop_rate"],
                ".4f"
            )
        ),
        CONTROLLED
    )
)

print()

print(
    "TEST 10: Verify Closed-Loop Penetration"
)

for item in CONTROLLED:

    assert (
            item["penetration"]
            <
            PENETRATION_THRESHOLD
    ), (
        "Controlled penetration must fall below tolerance: "
        +
        item["pattern_id"]
    )

    if item["pattern_id"] in ESCALATE_SET:

        assert (
                item["penetration"]
                <
                BASELINE_MAP[
                    item["pattern_id"]
                ][
                    "penetration"
                ]
        ), (
            "Escalation must reduce penetration."
        )

    else:

        assert abs(
            item["penetration"]
            -
            BASELINE_MAP[
                item["pattern_id"]
            ][
                "penetration"
            ]
        ) <= 1e-9, (
            "Hold patterns must keep baseline penetration."
        )

CONTROLLED_MEAN_STOP = sum(
    item["stop_rate"]
    for item
    in CONTROLLED
) / len(
    CONTROLLED
)

assert (
        CONTROLLED_MEAN_STOP
        >=
        CONTROLLED_STOP_FLOOR
), "Controlled mean stop rate below floor."

print(
    "All controlled penetrations below tolerance."
)

print(
    "Mean controlled stop rate:",
    format(
        CONTROLLED_MEAN_STOP,
        ".4f"
    )
)

print(
    "Closed-loop control verified."
)

print()

print(
    "TEST 11: Determinism"
)

SECOND_BASELINE = simulate_defense(
    RECORD_MAP,
    EXTENDED_HISTORY,
    LAYER_FRAME,
    False
)

DETERMINISTIC = (
        stable_hash(
            BASELINE
        )
        ==
        stable_hash(
            SECOND_BASELINE
        )
)

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Defense simulation is nondeterministic."
)

print(
    "Deterministic simulation validated."
)

print()

print(
    "TEST 12: Numerical Health"
)

BASELINE_TENSOR = torch.tensor(
    [
        item["penetration"]
        for item
        in BASELINE
    ],
    dtype=torch.float32
)

CONTROLLED_TENSOR = torch.tensor(
    [
        item["penetration"]
        for item
        in CONTROLLED
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        BASELINE_TENSOR
    ).all()
    and
    torch.isfinite(
        CONTROLLED_TENSOR
    ).all()
)

print(
    "Baseline NaN:",
    int(
        torch.isnan(
            BASELINE_TENSOR
        ).sum()
    )
)

print(
    "Baseline Inf:",
    int(
        torch.isinf(
            BASELINE_TENSOR
        ).sum()
    )
)

print(
    "Controlled NaN:",
    int(
        torch.isnan(
            CONTROLLED_TENSOR
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
    if len(BASELINE) == EXPECTED_PATTERNS
    else [
        "Baseline coverage invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(CONTROLLED) == EXPECTED_PATTERNS
    else [
        "Controlled coverage invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if ESCALATE_SET == {
        "pattern_003",
        "pattern_004",
        "pattern_006"
    }
    else [
        "Escalation set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if HOLD_SET == {
        "pattern_001",
        "pattern_002",
        "pattern_005"
    }
    else [
        "Hold set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        item["penetration"]
        <
        PENETRATION_THRESHOLD
        for item
        in CONTROLLED
    )
    else [
        "Closed-loop control incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if BASELINE_MEAN_STOP >= MEAN_STOP_FLOOR
    else [
        "Baseline stop rate below floor."
    ]
)

PROMOTION_ERRORS += (
    []
    if CONTROLLED_MEAN_STOP >= CONTROLLED_STOP_FLOOR
    else [
        "Controlled stop rate below floor."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Simulation nondeterministic."
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
    "Baseline patterns:",
    len(
        BASELINE
    )
)

print(
    "Escalated patterns:",
    sorted(
        ESCALATE_SET
    )
)

print(
    "Held patterns:",
    sorted(
        HOLD_SET
    )
)

print(
    "Mean baseline stop rate:",
    format(
        BASELINE_MEAN_STOP,
        ".4f"
    )
)

print(
    "Mean controlled stop rate:",
    format(
        CONTROLLED_MEAN_STOP,
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
        "127R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "127R promotion gate passed."
)

print()

print(
    "TEST 14: Persist Defense Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "127R",

    "capability":
        "multi_layer_defense_simulation_adaptive_control",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "126R",

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

    "extended_history":
        EXTENDED_HISTORY,

    "layer_frame":
        LAYER_FRAME,

    "penetration_threshold":
        PENETRATION_THRESHOLD,

    "reinforcement_boost":
        REINFORCEMENT_BOOST,

    "baseline_simulation":
        BASELINE,

    "controlled_simulation":
        CONTROLLED,

    "control_actions":
        CONTROL_ACTIONS,

    "baseline_mean_stop_rate":
        BASELINE_MEAN_STOP,

    "controlled_mean_stop_rate":
        CONTROLLED_MEAN_STOP,

    "closed_loop":
        all(
            item["penetration"]
            <
            PENETRATION_THRESHOLD
            for item
            in CONTROLLED
        ),

    "verification":
        {
            "escaped":
                sorted(
                    ESCALATE_SET
                ),

            "held":
                sorted(
                    HOLD_SET
                ),

            "closed_loop":
                all(
                    item["penetration"]
                    <
                    PENETRATION_THRESHOLD
                    for item
                    in CONTROLLED
                ),

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

assert len(
    RELOADED["baseline_simulation"]
) == len(
    BASELINE
), "Baseline changed after reload."

assert len(
    RELOADED["controlled_simulation"]
) == len(
    CONTROLLED
), "Controlled simulation changed after reload."

assert (
        RELOADED["control_actions"]
        ==
        CONTROL_ACTIONS
), "Control actions changed after reload."

assert RELOADED[
    "closed_loop"
], "Closed-loop flag changed after reload."

print(
    "Reloaded baseline patterns:",
    len(
        RELOADED["baseline_simulation"]
    )
)

print(
    "Reloaded control actions:",
    dict(
        map(
            lambda pair: (
                pair[0],
                pair[1][0]
            ),
            RELOADED[
                "control_actions"
            ].items()
        )
    )
)

print(
    "Reloaded closed loop:",
    RELOADED[
        "closed_loop"
    ]
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
            "127R",

        "capability":
            "multi_layer_defense_simulation_adaptive_control",

        "layer_frame":
            LAYER_FRAME,

        "baseline_simulation":
            BASELINE,

        "controlled_simulation":
            CONTROLLED,

        "control_actions":
            CONTROL_ACTIONS,

        "baseline_mean_stop_rate":
            BASELINE_MEAN_STOP,

        "controlled_mean_stop_rate":
            CONTROLLED_MEAN_STOP,

        "closed_loop":
            all(
                item["penetration"]
                <
                PENETRATION_THRESHOLD
                for item
                in CONTROLLED
            )
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "127R",

        "memory_version":
            MEMORY_VERSION,

        "layer_count":
            EXPECTED_LAYERS,

        "penetration_threshold":
            PENETRATION_THRESHOLD,

        "reinforcement_boost":
            REINFORCEMENT_BOOST,

        "baseline_mean_stop_rate":
            BASELINE_MEAN_STOP,

        "controlled_mean_stop_rate":
            CONTROLLED_MEAN_STOP,

        "escalated_patterns":
            sorted(
                ESCALATE_SET
            ),

        "held_patterns":
            sorted(
                HOLD_SET
            ),

        "closed_loop":
            all(
                item["penetration"]
                <
                PENETRATION_THRESHOLD
                for item
                in CONTROLLED
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
            "127R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "128R Uncertainty-Aware Preventive Execution "
                "+ Probabilistic Guardrails"
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
    "SILVERWING 127R ARCHITECTURE"
)

print(
    "Pattern"
)

print(
    "   ↓"
)

print(
    "L1 DETECT"
)

print(
    "   ↓"
)

print(
    "L2 INTERCEPT"
)

print(
    "   ↓"
)

print(
    "L3 VERIFY"
)

print(
    "   ↓"
)

print(
    "L4 REPAIR"
)

print(
    "   ↓"
)

print(
    "Penetration Simulation"
)

print(
    "   ↓"
)

print(
    "Threshold Classification"
)

print(
    "   ↓"
)

print(
    "HOLD / ESCALATE"
)

print(
    "   ↓"
)

print(
    "Reinforcement Boost"
)

print(
    "   ↓"
)

print(
    "Re-Simulation (Closed Loop)"
)

print()

print(
    "WHAT 127R ADDS"
)

print(
    "A layered defense model where the loop is not a single "
    "gate but a chain, expected-value penetration accounting, "
    "automatic escalation of weak layers and a closed-loop "
    "control law that forces every pattern below tolerance."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems where a single validation gate is not enough and "
    "defense must be staged across multiple independent layers."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "One gate can be bypassed in a single step. A layered "
    "defense degrades gracefully, and adaptive reinforcement "
    "keeps the weakest link from being the winning link."
)

print()

print(
    "NEXT: 128R Uncertainty-Aware Preventive Execution "
    "+ Probabilistic Guardrails"
)

print()

print(
    "=== LESSON 127R COMPLETE ==="
)

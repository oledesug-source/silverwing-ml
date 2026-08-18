# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 131R
# Collective Defense Consolidation + System-Level Resilience
# Audit
# ============================================================
#
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
# 128R  -> Uncertainty-Aware Preventive Execution
#         + Probabilistic Guardrails
# 129R  -> Anomaly-First Adaptive Scheduling
#         + Critical Path Defense
# 130R  -> Self-Healing Recovery Orchestration
#         + Failure Absorption
# 131R  -> Collective Defense Consolidation
#         + System-Level Resilience Audit
#
# ============================================================
# PURPOSE
# ============================================================
#
# Each lesson since 125R built one defensive capability:
# trending, governance, layers, guardrails, scheduling,
# recovery. 131R is the consolidation: every capability is
# folded into a single per-pattern defense scorecard, the
# system-level metrics are computed, and a resilience audit
# asks whether the collective defense is actually sound.
#
# The audit is a battery of checks - exposure, coverage,
# sequencing, closed-loop gain, absorption, recovery safety,
# governance, determinism - each producing PASS or FAIL.
# The resilience grade is the fraction of checks that pass.
#
# Collective defense consolidation:
#
#     125R..130R memories
#              ↓
#     per-pattern defense scorecard
#              ↓
#     system-level metrics
#              ↓
#     resilience audit (A1..A8)
#              ↓
#     resilience grade
#
# The audit is deterministic: the same memories always
# produce the same grade.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 130R memory is the primary source of truth.
# 2. Auxiliary memories supply trend, guardrail, layer data.
# 3. Chain integrity must be verified across memories.
# 4. A single scorecard consolidates all per-pattern state.
# 5. System-level metrics aggregate the scorecard.
# 6. Each audit check is an explicit boolean gate.
# 7. The resilience grade is the pass fraction.
# 8. Promotion requires the full audit to pass.
# 9. Determinism must be checked.
# 10. Numerical health must be checked.
# 11. Persistence and reload must be checked.
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
MEMORY_VERSION = "131R.1"
EXPECTED_PATTERNS = 6
EXPOSURE_LIMIT = 0.75
RESILIENCE_THRESHOLD = 0.875

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent

LESSON_125R = PHASE5_DIR / "lesson125R"
LESSON_126R = PHASE5_DIR / "lesson126R"
LESSON_127R = PHASE5_DIR / "lesson127R"
LESSON_128R = PHASE5_DIR / "lesson128R"
LESSON_129R = PHASE5_DIR / "lesson129R"
LESSON_130R = PHASE5_DIR / "lesson130R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_130R = (
        LESSON_130R
        / "silverwing_recovery_orchestration_memory.json"
)

SOURCE_129R = (
        LESSON_129R
        / "silverwing_anomaly_scheduling_memory.json"
)

SOURCE_128R = (
        LESSON_128R
        / "silverwing_uncertainty_guardrails_memory.json"
)

SOURCE_127R = (
        LESSON_127R
        / "silverwing_multi_layer_defense_memory.json"
)

SOURCE_126R = (
        LESSON_126R
        / "silverwing_control_loop_governance_memory.json"
)

SOURCE_125R = (
        LESSON_125R
        / "silverwing_cross_cycle_trending_memory.json"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_resilience_audit_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_resilience_audit_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_resilience_audit_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_resilience_audit_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_resilience_audit_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_resilience_audit_best.pt"
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
    "PHASE 5 - LESSON 131R"
)

print(
    "Collective Defense Consolidation"
)

print(
    "+ System-Level Resilience Audit"
)

print()

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

print(
    "131R -> Collective Defense Consolidation"
)

print(
    "        + System-Level Resilience Audit"
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
    SOURCE_125R,
    SOURCE_126R,
    SOURCE_127R,
    SOURCE_128R,
    SOURCE_129R,
    SOURCE_130R
]

assert all(
    path.exists()
    for path in REQUIRED_FILES
), "One or more source inputs are missing."

print(
    "FOUND:",
    SOURCE_125R
)

print(
    "FOUND:",
    SOURCE_126R
)

print(
    "FOUND:",
    SOURCE_127R
)

print(
    "FOUND:",
    SOURCE_128R
)

print(
    "FOUND:",
    SOURCE_129R
)

print(
    "FOUND:",
    SOURCE_130R
)

print()

print(
    "TEST 2: Load 130R Source Memory"
)

M130 = read_json(
    SOURCE_130R
)

assert isinstance(
    M130,
    dict
), "130R memory is invalid."

ANOMALY_SCORES = M130.get(
    "anomaly_scores",
    {}
)

ANOMALY_SCHEDULE = M130.get(
    "anomaly_schedule",
    []
)

DEPENDENCIES = M130.get(
    "dependencies",
    {}
)

ACTION_MAP = M130.get(
    "action_map",
    {}
)

CHILDREN = M130.get(
    "children",
    {}
)

ANOMALY_SET = M130.get(
    "anomaly_set",
    []
)

CRITICAL_PATH = M130.get(
    "critical_path",
    {}
)

TUNED_RECORDS = M130.get(
    "tuned_records",
    []
)

TUNED_ORDER = M130.get(
    "tuned_order",
    []
)

RECOVERY_FRAME = M130.get(
    "recovery_frame",
    {}
)

BASE_PENETRATION = M130.get(
    "base_penetration",
    {}
)

ABSORPTION_CAPACITY = M130.get(
    "absorption_capacity",
    0.0
)

BREACH_SCENARIO = M130.get(
    "breach_scenario",
    {}
)

ROOT_CASCADE = M130.get(
    "root_cascade_scenario",
    {}
)

M130_VERIFICATION = M130.get(
    "verification",
    {}
)

assert len(
    ANOMALY_SCORES
) == EXPECTED_PATTERNS, (
    "130R must supply six anomaly scores."
)

assert len(
    TUNED_RECORDS
) == EXPECTED_PATTERNS, (
    "130R must supply six tuned records."
)

print(
    "130R memory:",
    M130.get(
        "memory_version"
    )
)

print(
    "Critical path:",
    CRITICAL_PATH
)

print()

print(
    "TEST 3: Load Cross-Lesson Auxiliary Memories"
)

M129 = read_json(
    SOURCE_129R
)

M128 = read_json(
    SOURCE_128R
)

M127 = read_json(
    SOURCE_127R
)

M126 = read_json(
    SOURCE_126R
)

M125 = read_json(
    SOURCE_125R
)

DEFENSE_TIERS = M129.get(
    "defense_tiers",
    {}
)

DRIFT_TIERS = M129.get(
    "drift",
    {}
).get(
    "tiers",
    {}
)

BASELINE_BANDS = {
    pattern_id: band["band_width"]
    for pattern_id, band
    in M128.get(
        "baseline_bands",
        {}
    ).items()
}

CONTROLLED_GATE = M128.get(
    "controlled_gate",
    {}
)

GUARDRAIL_FRAME = M128.get(
    "guardrail_frame",
    {}
)

LAYER_FRAME = M127.get(
    "layer_frame",
    []
)

CONTROL_ACTIONS = M127.get(
    "control_actions",
    {}
)

BASELINE_STOP = M127.get(
    "baseline_mean_stop_rate",
    0.0
)

CONTROLLED_STOP = M127.get(
    "controlled_mean_stop_rate",
    0.0
)

APPROVED_REHEARSAL = M126.get(
    "approved_rehearsal",
    {}
)

RESIDUAL_BUDGET = M126.get(
    "residual_budget",
    1.1
)

POLICY_FRAME = M126.get(
    "policy_frame",
    []
)

TRENDS = {
    entry["pattern_id"]: entry["trend"]
    for entry
    in M125.get(
        "trends",
        []
    )
}

CONTROLLED_SIMULATIONS = {
    entry["pattern_id"]: entry
    for entry
    in M127.get(
        "controlled_simulation",
        []
    )
}

BASELINE_SIMULATIONS = {
    entry["pattern_id"]: entry
    for entry
    in M127.get(
        "baseline_simulation",
        []
    )
}

assert len(
    DEFENSE_TIERS
) == EXPECTED_PATTERNS, (
    "129R must supply six defense tiers."
)

assert len(
    BASELINE_BANDS
) == EXPECTED_PATTERNS, (
    "128R must supply six bands."
)

assert len(
    TRENDS
) == EXPECTED_PATTERNS, (
    "125R must supply six trends."
)

print(
    "129R memory:",
    M129.get(
        "memory_version"
    )
)

print(
    "128R memory:",
    M128.get(
        "memory_version"
    )
)

print(
    "127R memory:",
    M127.get(
        "memory_version"
    )
)

print(
    "126R memory:",
    M126.get(
        "memory_version"
    )
)

print(
    "125R memory:",
    M125.get(
        "memory_version"
    )
)

print()

print(
    "TEST 4: Verify Chain Integrity"
)

RISK_SCORES = {
    record["pattern_id"]: record["risk_score"]
    for record
    in TUNED_RECORDS
}

CHAIN_INTEGRITY = True

for memory, label in [
    (M129, "129R"),
    (M128, "128R"),
    (M127, "127R"),
    (M126, "126R")
]:

    for record in memory.get(
        "tuned_records",
        []
    ):

        if (
                RISK_SCORES[
                    record["pattern_id"]
                ]
                !=
                record["risk_score"]
        ):

            CHAIN_INTEGRITY = False

            print(
                "MISMATCH",
                label,
                record["pattern_id"]
            )

assert CHAIN_INTEGRITY, (
    "Risk scores drifted across the lesson chain."
)

print(
    "Risk scores consistent across 126R..130R."
)

print()

print(
    "TEST 5: Build Collective Defense Scorecard"
)

SCORECARD = {}

for record in TUNED_RECORDS:

    pattern_id = record["pattern_id"]

    SCORECARD[
        pattern_id
    ] = {
        "pattern_id":
            pattern_id,

        "family":
            record["family"],

        "risk_score":
            record["risk_score"],

        "risk_class":
            record["risk_class"],

        "severity":
            record["severity"],

        "impact":
            record["impact"],

        "trend":
            TRENDS[
                pattern_id
            ],

        "anomaly_score":
            ANOMALY_SCORES[
                pattern_id
            ],

        "classification":
            "ANOMALY"
            if pattern_id in ANOMALY_SET
            else "NORMAL",

        "defense_tier":
            DEFENSE_TIERS[
                pattern_id
            ],

        "control_action":
            CONTROL_ACTIONS[
                pattern_id
            ],

        "band_width":
            BASELINE_BANDS[
                pattern_id
            ],

        "base_stop_rate":
            BASELINE_SIMULATIONS[
                pattern_id
            ][
                "stop_rate"
            ],

        "controlled_stop_rate":
            CONTROLLED_SIMULATIONS[
                pattern_id
            ][
                "stop_rate"
            ],

        "base_penetration":
            BASE_PENETRATION[
                pattern_id
            ],

        "on_critical_path":
            pattern_id
            in CRITICAL_PATH[
                "patterns"
            ]
    }

assert all(
    SCORECARD[
        pattern_id
    ][
        "family"
    ]
    for pattern_id
    in TUNED_ORDER
), "Every pattern must have a family."

for pattern_id in TUNED_ORDER:

    entry = SCORECARD[
        pattern_id
    ]

    print(
        pattern_id,
        "|",
        entry["risk_class"],
        "|",
        entry["trend"],
        "|",
        entry["classification"],
        "|",
        entry["defense_tier"],
        "|",
        entry["control_action"],
        "| stop",
        format(
            entry["base_stop_rate"],
            ".3f"
        ),
        "->",
        format(
            entry["controlled_stop_rate"],
            ".3f"
        )
    )

print()

print(
    "TEST 6: Compute System-Level Metrics"
)

CRITICAL_PATTERNS = CRITICAL_PATH[
    "patterns"
]

COLLECTIVE_EXPOSURE = (
        sum(
            ANOMALY_SCORES.values()
        )
        /
        len(
            ANOMALY_SCORES
        )
)

CRITICAL_EXPOSURE = (
        sum(
            ANOMALY_SCORES[
                pattern_id
            ]
            for pattern_id
            in CRITICAL_PATTERNS
        )
        /
        len(
            CRITICAL_PATTERNS
        )
)

CRITICAL_COVERAGE = (
        len(
            [
                pattern_id
                for pattern_id
                in CRITICAL_PATTERNS
                if pattern_id in ANOMALY_SCHEDULE
            ]
        )
        /
        len(
            CRITICAL_PATTERNS
        )
)

CLOSED_LOOP_GAIN = (
        CONTROLLED_STOP
        -
        BASELINE_STOP
)

ABSORBED = BREACH_SCENARIO.get(
    "absorbed",
    0.0
)

ABSORPTION_MARGIN = (
        ABSORPTION_CAPACITY
        -
        ABSORBED
)

assert abs(
    COLLECTIVE_EXPOSURE
    -
    0.708797
) <= 1e-4, "Collective exposure mismatch."

assert abs(
    CRITICAL_EXPOSURE
    -
    0.718394
) <= 1e-4, "Critical exposure mismatch."

assert CRITICAL_COVERAGE == 1.0, (
    "Critical coverage must be complete."
)

assert abs(
    CLOSED_LOOP_GAIN
    -
    0.044084
) <= 1e-4, "Closed-loop gain mismatch."

assert abs(
    ABSORPTION_MARGIN
    -
    0.634145
) <= 1e-4, "Absorption margin mismatch."

print(
    "Collective exposure:",
    format(
        COLLECTIVE_EXPOSURE,
        ".4f"
    )
)

print(
    "Critical exposure:",
    format(
        CRITICAL_EXPOSURE,
        ".4f"
    )
)

print(
    "Critical coverage:",
    format(
        CRITICAL_COVERAGE,
        ".4f"
    )
)

print(
    "Closed-loop gain:",
    format(
        CLOSED_LOOP_GAIN,
        ".4f"
    )
)

print(
    "Absorption margin:",
    format(
        ABSORPTION_MARGIN,
        ".4f"
    )
)

print()

print(
    "TEST 7: Audit A1 - Collective Exposure"
)

A1_PASS = (
        CRITICAL_EXPOSURE
        <
        EXPOSURE_LIMIT
)

assert A1_PASS, "A1 exposure audit failed."

print(
    "Critical exposure:",
    format(
        CRITICAL_EXPOSURE,
        ".4f"
    ),
    "limit:",
    EXPOSURE_LIMIT
)

print(
    "A1 PASS"
)

print()

print(
    "TEST 8: Audit A2 - Critical Coverage"
)

A2_PASS = (
        CRITICAL_COVERAGE
        >=
        1.0
)

assert A2_PASS, "A2 coverage audit failed."

print(
    "A2 PASS"
)

print()

print(
    "TEST 9: Audit A3 - Anomaly-First Sequencing"
)

ANOMALY_FIRST = True

for pattern_id in ANOMALY_SET:

    if pattern_id not in ANOMALY_SCHEDULE:

        ANOMALY_FIRST = False

positions = {
    pattern_id: index
    for index, pattern_id
    in enumerate(
        ANOMALY_SCHEDULE
    )
}

NON_ESSENTIAL_NORMALS = [
    pattern_id
    for pattern_id
    in TUNED_ORDER
    if pattern_id not in ANOMALY_SET
    and pattern_id not in CRITICAL_PATTERNS
]

for anomaly_id in ANOMALY_SET:

    for normal_id in NON_ESSENTIAL_NORMALS:

        if (
                positions[
                    anomaly_id
                ]
                >
                positions[
                    normal_id
                ]
        ):

            ANOMALY_FIRST = False

A3_PASS = ANOMALY_FIRST

assert A3_PASS, "A3 sequencing audit failed."

print(
    "Non-essential normals:",
    NON_ESSENTIAL_NORMALS
)

print(
    "A3 PASS"
)

print()

print(
    "TEST 10: Audit A4 - Closed-Loop Gain"
)

A4_PASS = (
        CLOSED_LOOP_GAIN
        >
        0.0
)

assert A4_PASS, "A4 closed-loop audit failed."

print(
    "Baseline mean stop:",
    format(
        BASELINE_STOP,
        ".4f"
    )
)

print(
    "Controlled mean stop:",
    format(
        CONTROLLED_STOP,
        ".4f"
    )
)

print(
    "Gain:",
    format(
        CLOSED_LOOP_GAIN,
        ".4f"
    )
)

print(
    "A4 PASS"
)

print()

print(
    "TEST 11: Audit A5 - Absorption Margin"
)

A5_PASS = (
        ABSORPTION_MARGIN
        >
        0.0
)

assert A5_PASS, "A5 absorption audit failed."

print(
    "Absorption capacity:",
    format(
        ABSORPTION_CAPACITY,
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
    "Margin:",
    format(
        ABSORPTION_MARGIN,
        ".4f"
    )
)

print(
    "A5 PASS"
)

print()

print(
    "TEST 12: Audit A6 - Recovery Safety"
)

A6_PASS = bool(
    M130_VERIFICATION.get(
        "all_residuals_safe",
        False
    )
)

assert A6_PASS, "A6 recovery audit failed."

print(
    "All residuals safe:",
    A6_PASS
)

print(
    "A6 PASS"
)

print()

print(
    "TEST 13: Audit A7 - Governance Compliance"
)

REHEARSAL_RESIDUAL = APPROVED_REHEARSAL.get(
    "total_residual",
    1e9
)

REHEARSAL_COVERAGE = APPROVED_REHEARSAL.get(
    "coverage_ratio",
    0.0
)

A7_PASS = (
        REHEARSAL_RESIDUAL
        <=
        RESIDUAL_BUDGET
        and
        REHEARSAL_COVERAGE
        >=
        1.0
)

assert A7_PASS, "A7 governance audit failed."

print(
    "Rehearsal residual:",
    format(
        REHEARSAL_RESIDUAL,
        ".4f"
    ),
    "budget:",
    RESIDUAL_BUDGET
)

print(
    "Rehearsal coverage:",
    format(
        REHEARSAL_COVERAGE,
        ".4f"
    )
)

print(
    "A7 PASS"
)

print()

print(
    "TEST 14: Audit A8 - Chain Determinism"
)

CHAIN_DETERMINISM = [
    M130_VERIFICATION.get(
        "deterministic",
        False
    ),
    M129.get(
        "verification",
        {}
    ).get(
        "deterministic",
        False
    ),
    M128.get(
        "verification",
        {}
    ).get(
        "deterministic",
        False
    ),
    M127.get(
        "verification",
        {}
    ).get(
        "deterministic",
        False
    ),
    M126.get(
        "verification",
        {}
    ).get(
        "deterministic",
        False
    )
]

A8_PASS = all(
    CHAIN_DETERMINISM
)

assert A8_PASS, "A8 determinism audit failed."

print(
    "Chain determinism flags:",
    CHAIN_DETERMINISM
)

print(
    "A8 PASS"
)

print()

print(
    "TEST 15: Resilience Composite Grade"
)

AUDIT_CHECKS = {
    "A1_COLLECTIVE_EXPOSURE":
        A1_PASS,

    "A2_CRITICAL_COVERAGE":
        A2_PASS,

    "A3_ANOMALY_FIRST":
        A3_PASS,

    "A4_CLOSED_LOOP_GAIN":
        A4_PASS,

    "A5_ABSORPTION_MARGIN":
        A5_PASS,

    "A6_RECOVERY_SAFETY":
        A6_PASS,

    "A7_GOVERNANCE_COMPLIANCE":
        A7_PASS,

    "A8_CHAIN_DETERMINISM":
        A8_PASS
}

PASSED = sum(
    AUDIT_CHECKS.values()
)

TOTAL_CHECKS = len(
    AUDIT_CHECKS
)

RESILIENCE_SCORE = (
        PASSED
        /
        TOTAL_CHECKS
)

RESILIENCE_GRADE = (
    "RESILIENT"
    if RESILIENCE_SCORE >= RESILIENCE_THRESHOLD
    else "VULNERABLE"
)

assert RESILIENCE_SCORE == 1.0, (
    "Not every audit check passed."
)

assert RESILIENCE_GRADE == "RESILIENT", (
    "Resilience grade must be RESILIENT."
)

print(
    "Audit checks:",
    AUDIT_CHECKS
)

print(
    "Passed:",
    PASSED,
    "/",
    TOTAL_CHECKS
)

print(
    "Resilience score:",
    format(
        RESILIENCE_SCORE,
        ".4f"
    )
)

print(
    "Resilience grade:",
    RESILIENCE_GRADE
)

print()

print(
    "TEST 16: Determinism"
)

SECOND_COLLECTIVE_EXPOSURE = (
        sum(
            ANOMALY_SCORES.values()
        )
        /
        len(
            ANOMALY_SCORES
        )
)

DETERMINISTIC = (
        COLLECTIVE_EXPOSURE
        ==
        SECOND_COLLECTIVE_EXPOSURE
)

assert DETERMINISTIC, (
    "Consolidation is nondeterministic."
)

print(
    "Deterministic:",
    DETERMINISTIC
)

print()

print(
    "TEST 17: Numerical Health"
)

SCORECARD_TENSOR = torch.tensor(
    [
        SCORECARD[
            pattern_id
        ][
            "anomaly_score"
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

GAIN_TENSOR = torch.tensor(
    [
        CLOSED_LOOP_GAIN,
        ABSORPTION_MARGIN,
        RESILIENCE_SCORE
    ],
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        SCORECARD_TENSOR
    ).all()
    and
    torch.isfinite(
        GAIN_TENSOR
    ).all()
)

assert NUMERICALLY_HEALTHY, (
    "Numerical health failed."
)

print(
    "Scorecard NaN:",
    int(
        torch.isnan(
            SCORECARD_TENSOR
        ).sum()
    )
)

print(
    "Metrics Inf:",
    int(
        torch.isinf(
            GAIN_TENSOR
        ).sum()
    )
)

print(
    "Numerically healthy:",
    NUMERICALLY_HEALTHY
)

print()

print(
    "TEST 18: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if CHAIN_INTEGRITY
    else [
        "Chain integrity violated."
    ]
)

PROMOTION_ERRORS += (
    []
    if RESILIENCE_GRADE == "RESILIENT"
    else [
        "System not resilient."
    ]
)

PROMOTION_ERRORS += (
    []
    if RESILIENCE_SCORE == 1.0
    else [
        "Audit checks incomplete."
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
    "Resilience grade:",
    RESILIENCE_GRADE
)

print(
    "Resilience score:",
    format(
        RESILIENCE_SCORE,
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
        "131R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "131R promotion gate passed."
)

print()

print(
    "TEST 19: Persist Audit Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "131R",

    "capability":
        "collective_defense_consolidation_system_resilience_audit",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "130R",

    "tuned_records":
        TUNED_RECORDS,

    "tuned_order":
        TUNED_ORDER,

    "anomaly_scores":
        ANOMALY_SCORES,

    "anomaly_set":
        ANOMALY_SET,

    "anomaly_schedule":
        ANOMALY_SCHEDULE,

    "critical_path":
        CRITICAL_PATH,

    "defense_tiers":
        DEFENSE_TIERS,

    "drift_tiers":
        DRIFT_TIERS,

    "control_actions":
        CONTROL_ACTIONS,

    "baseline_bands":
        BASELINE_BANDS,

    "guardrail_frame":
        GUARDRAIL_FRAME,

    "controlled_gate":
        CONTROLLED_GATE,

    "layer_frame":
        LAYER_FRAME,

    "trends":
        TRENDS,

    "recovery_frame":
        RECOVERY_FRAME,

    "base_penetration":
        BASE_PENETRATION,

    "absorption_capacity":
        ABSORPTION_CAPACITY,

    "breach_scenario":
        BREACH_SCENARIO,

    "root_cascade_scenario":
        ROOT_CASCADE,

    "chain_integrity":
        CHAIN_INTEGRITY,

    "scorecard":
        SCORECARD,

    "system_metrics":
        {
            "collective_exposure":
                COLLECTIVE_EXPOSURE,

            "critical_exposure":
                CRITICAL_EXPOSURE,

            "critical_coverage":
                CRITICAL_COVERAGE,

            "closed_loop_gain":
                CLOSED_LOOP_GAIN,

            "absorption_margin":
                ABSORPTION_MARGIN,

            "baseline_mean_stop_rate":
                BASELINE_STOP,

            "controlled_mean_stop_rate":
                CONTROLLED_STOP,

            "rehearsal_residual":
                REHEARSAL_RESIDUAL,

            "rehearsal_coverage":
                REHEARSAL_COVERAGE
        },

    "resilience_audit":
        {
            "exposure_limit":
                EXPOSURE_LIMIT,

            "checks":
                AUDIT_CHECKS,

            "passed":
                PASSED,

            "total":
                TOTAL_CHECKS,

            "score":
                RESILIENCE_SCORE,

            "grade":
                RESILIENCE_GRADE
        },

    "verification":
        {
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

assert (
        RELOADED["resilience_audit"]["grade"]
        ==
        "RESILIENT"
), "Resilience grade changed after reload."

assert (
        RELOADED["resilience_audit"]["score"]
        ==
        RESILIENCE_SCORE
), "Resilience score changed after reload."

assert (
        RELOADED["scorecard"]
        ==
        SCORECARD
), "Scorecard changed after reload."

assert (
        RELOADED["system_metrics"]
        ==
        MEMORY["system_metrics"]
), "System metrics changed after reload."

print(
    "Reloaded grade:",
    RELOADED[
        "resilience_audit"
    ][
        "grade"
    ]
)

print(
    "Reloaded score:",
    format(
        RELOADED[
            "resilience_audit"
        ][
            "score"
        ],
        ".4f"
    )
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
            "131R",

        "capability":
            "collective_defense_consolidation_system_resilience_audit",

        "scorecard":
            SCORECARD,

        "system_metrics":
            MEMORY["system_metrics"],

        "resilience_audit":
            MEMORY["resilience_audit"],

        "audit_checks":
            AUDIT_CHECKS,

        "resilience_grade":
            RESILIENCE_GRADE
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "131R",

        "memory_version":
            MEMORY_VERSION,

        "collective_exposure":
            COLLECTIVE_EXPOSURE,

        "critical_exposure":
            CRITICAL_EXPOSURE,

        "critical_coverage":
            CRITICAL_COVERAGE,

        "closed_loop_gain":
            CLOSED_LOOP_GAIN,

        "absorption_margin":
            ABSORPTION_MARGIN,

        "resilience_score":
            RESILIENCE_SCORE,

        "resilience_grade":
            RESILIENCE_GRADE,

        "checks_passed":
            PASSED,

        "checks_total":
            TOTAL_CHECKS,

        "chain_integrity":
            CHAIN_INTEGRITY,

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
            "131R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "132R Autonomous Defense Self-Tuning + "
                "Online Control Refinement"
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
    "SILVERWING 131R ARCHITECTURE"
)

print(
    "125R..130R Memories"
)

print(
    "   ↓"
)

print(
    "Chain Integrity"
)

print(
    "   ↓"
)

print(
    "Per-Pattern Defense Scorecard"
)

print(
    "   ↓"
)

print(
    "System-Level Metrics"
)

print(
    "   ↓"
)

print(
    "Resilience Audit A1..A8"
)

print(
    "   ↓"
)

print(
    "Resilience Grade"
)

print()

print(
    "WHAT 131R ADDS"
)

print(
    "A single consolidated view of every defensive capability "
    "built since 125R, and a deterministic system-level audit "
    "that turns six lessons of evidence into one resilience "
    "grade."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems where defense is built in pieces and leadership "
    "needs a single, honest, reproducible health report."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A system is only as resilient as its weakest integrated "
    "link. Consolidation surfaces gaps; the audit grades the "
    "whole, not the parts."
)

print()

print(
    "NEXT: 132R Autonomous Defense Self-Tuning + "
    "Online Control Refinement"
)

print()

print(
    "=== LESSON 131R COMPLETE ==="
)

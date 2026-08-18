# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 128R
# Uncertainty-Aware Preventive Execution + Probabilistic Guardrails
# ============================================================
#
# 123R  -> Post-Execution Outcome Feedback + Plan Learning
# 124R  -> Outcome Memory Consolidation + Risk Calibration
# 125R  -> Cross-Cycle Risk Trending + Adaptive Threshold Tuning
# 126R  -> Preventive Control Loop Governance + Policy Rehearsal
# 127R  -> Multi-Layer Defense Simulation + Adaptive Control
# 128R  -> Uncertainty-Aware Preventive Execution
#         + Probabilistic Guardrails
#
# ============================================================
# PURPOSE
# ============================================================
#
# By 127R defense is layered and adaptively reinforced, but the
# simulation reports point estimates. 128R asks: what if the
# layer effectiveness is not exact? Effectiveness carries
# uncertainty. A guardrail that ignores uncertainty can let a
# borderline pattern through on a false sense of confidence.
#
# 128R adds uncertainty awareness:
#
#     layer effectiveness (point)
#              ↓
#     ± uncertainty (relative)
#              ↓
#     Monte Carlo penetration distribution
#              ↓
#     trip probability vs guardrail threshold
#              ↓
#     TRIP / CLEAR
#              ↓
#     preventive execution gating
#
# Probabilistic guardrails:
#
#     P(penetration > guardrail threshold) > trip threshold
#          → TRIP (block execution)
#     otherwise
#          → CLEAR (allow execution)
#
# A pattern that cleared the point estimate may still trip the
# probabilistic guardrail. Adaptive control must shrink the
# uncertainty band until every pattern clears.
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 127R memory is the source of truth.
# 2. Effectiveness carries relative uncertainty per layer.
# 3. Penetration is sampled by seeded Monte Carlo.
# 4. Guardrail trips when trip probability exceeds threshold.
# 5. Preventive execution gates on the guardrail verdict.
# 6. Adaptive control must clear every guardrail.
# 7. Conservative and optimistic bounds frame the band.
# 8. Monte Carlo must be deterministic (fixed seed per run).
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
MEMORY_VERSION = "128R.1"
EXPECTED_PATTERNS = 6
UNCERTAINTY = 0.10
GUARDRAIL_THRESHOLD = 0.18
TRIP_PROB_THRESHOLD = 0.25
MC_TRIALS = 5000

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_127R = PHASE5_DIR / "lesson127R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_MEMORY = (
        LESSON_127R
        / "silverwing_multi_layer_defense_memory.json"
)

SOURCE_INDEX = (
        LESSON_127R
        / "silverwing_multi_layer_defense_index.pt"
)

SOURCE_DATASET = (
        LESSON_127R
        / "silverwing_multi_layer_defense_dataset.json"
)

SOURCE_REPORT = (
        LESSON_127R
        / "silverwing_multi_layer_defense_report.json"
)

SOURCE_REGISTRY = (
        LESSON_127R
        / "silverwing_multi_layer_defense_registry.json"
)

SOURCE_CHECKPOINT = (
        LESSON_127R
        / "checkpoints"
        / "silverwing_multi_layer_defense_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_uncertainty_guardrails_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_uncertainty_guardrails_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_uncertainty_guardrails_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_uncertainty_guardrails_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_uncertainty_guardrails_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_uncertainty_guardrails_best.pt"
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
    "PHASE 5 - LESSON 128R"
)

print(
    "Uncertainty-Aware Preventive Execution"
)

print(
    "+ Probabilistic Guardrails"
)

print()

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

print(
    "128R -> Uncertainty-Aware Preventive Execution"
)

print(
    "        + Probabilistic Guardrails"
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
    "TEST 1: Verify 127R Inputs"
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
), "One or more 127R inputs are missing."

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
    "TEST 2: Load 127R Defense Memory"
)

SOURCE = read_json(
    SOURCE_MEMORY
)

assert isinstance(
    SOURCE,
    dict
), "127R defense memory is invalid."

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

CONTROL_ACTIONS = SOURCE.get(
    "control_actions",
    {}
)

BASELINE_SIM = SOURCE.get(
    "baseline_simulation",
    []
)

CONTROLLED_SIM = SOURCE.get(
    "controlled_simulation",
    []
)

assert len(
    TUNED_RECORDS
) == EXPECTED_PATTERNS, (
    "127R must supply exactly six tuned records."
)

assert len(
    BASELINE_SIM
) == EXPECTED_PATTERNS, (
    "127R baseline simulation must cover six patterns."
)

assert len(
    CONTROLLED_SIM
) == EXPECTED_PATTERNS, (
    "127R controlled simulation must cover six patterns."
)

print(
    "Memory version:",
    SOURCE.get(
        "memory_version"
    )
)

print(
    "Control actions:",
    CONTROL_ACTIONS
)

print(
    "Closed loop:",
    SOURCE.get(
        "closed_loop"
    )
)

print()

print(
    "TEST 3: Rebuild Uncertainty State"
)

BASELINE_EFFS = {
    item["pattern_id"]:
        item["layer_effectiveness"]
    for item
    in BASELINE_SIM
}

CONTROLLED_EFFS = {
    item["pattern_id"]:
        item["layer_effectiveness"]
    for item
    in CONTROLLED_SIM
}

assert (
        sorted(
            BASELINE_EFFS.keys()
        )
        ==
        sorted(
            CONTROLLED_EFFS.keys()
        )
), "Baseline and controlled must cover the same patterns."

assert all(
    len(
        BASELINE_EFFS[
            pattern_id
        ]
    ) == 4
    for pattern_id
    in BASELINE_EFFS
), "Each pattern must carry four layer effectivenesses."

print(
    "Baseline layer effectiveness:"
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
                in BASELINE_EFFS[
                    pattern_id
                ]
            ]
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 4: Define Probabilistic Guardrail Frame"
)

GUARDRAIL_FRAME = {
    "uncertainty":
        UNCERTAINTY,

    "guardrail_threshold":
        GUARDRAIL_THRESHOLD,

    "trip_probability_threshold":
        TRIP_PROB_THRESHOLD,

    "monte_carlo_trials":
        MC_TRIALS
}

assert (
        0.0
        <
        UNCERTAINTY
        <
        1.0
), "Uncertainty must be a relative fraction."

assert (
        0.0
        <
        GUARDRAIL_THRESHOLD
        <
        1.0
), "Guardrail threshold must be a probability."

assert (
        0.0
        <
        TRIP_PROB_THRESHOLD
        <
        1.0
), "Trip probability threshold must be a probability."

list(
    map(
        lambda item: print(
            item[0],
            ":",
            item[1]
        ),
        GUARDRAIL_FRAME.items()
    )
)

print()

print(
    "TEST 5: Monte Carlo Penetration Distribution"
)


def mc_penetration(
        effectiveness,
        trials,
        uncertainty
):

    generator = torch.Generator().manual_seed(
        SEED
    )

    outcomes = torch.ones(
        trials,
        dtype=torch.float32
    )

    for eff in effectiveness:

        loc = torch.full(
            (trials,),
            float(eff),
            dtype=torch.float32
        )

        scale = torch.full(
            (trials,),
            float(eff) * uncertainty,
            dtype=torch.float32
        )

        samples = torch.clamp(
            torch.normal(
                loc,
                scale,
                generator=generator
            ),
            0.0,
            1.0
        )

        outcomes *= (
            1.0
            -
            samples
        )

    return outcomes


def guardrail_profile(
        effectiveness,
        trials,
        uncertainty,
        guardrail_threshold
):

    outcomes = mc_penetration(
        effectiveness,
        trials,
        uncertainty
    )

    mean_penetration = float(
        outcomes.mean()
    )

    std_penetration = float(
        outcomes.std()
    )

    p95 = float(
        torch.quantile(
            outcomes,
            0.95
        )
    )

    trip_probability = float(
        (
            outcomes
            >
            guardrail_threshold
        ).float().mean()
    )

    return {
        "mean_penetration":
            mean_penetration,

        "std_penetration":
            std_penetration,

        "p95_penetration":
            p95,

        "trip_probability":
            trip_probability
    }


BASELINE_PROFILES = {}

for pattern_id in BASELINE_EFFS:

    BASELINE_PROFILES[
        pattern_id
    ] = guardrail_profile(
        BASELINE_EFFS[
            pattern_id
        ],
        MC_TRIALS,
        UNCERTAINTY,
        GUARDRAIL_THRESHOLD
    )

CONTROLLED_PROFILES = {}

for pattern_id in CONTROLLED_EFFS:

    CONTROLLED_PROFILES[
        pattern_id
    ] = guardrail_profile(
        CONTROLLED_EFFS[
            pattern_id
        ],
        MC_TRIALS,
        UNCERTAINTY,
        GUARDRAIL_THRESHOLD
    )

assert len(
    BASELINE_PROFILES
) == EXPECTED_PATTERNS, (
    "Baseline profiles must cover six patterns."
)

assert len(
    CONTROLLED_PROFILES
) == EXPECTED_PATTERNS, (
    "Controlled profiles must cover six patterns."
)

print(
    "Baseline profiles:"
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "| mean=",
            format(
                BASELINE_PROFILES[
                    pattern_id
                ][
                    "mean_penetration"
                ],
                ".4f"
            ),
            "| p95=",
            format(
                BASELINE_PROFILES[
                    pattern_id
                ][
                    "p95_penetration"
                ],
                ".4f"
            ),
            "| trip=",
            format(
                BASELINE_PROFILES[
                    pattern_id
                ][
                    "trip_probability"
                ],
                ".4f"
            )
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 6: Validate Baseline Uncertainty Profile"
)

assert (
        BASELINE_PROFILES[
            "pattern_003"
        ][
            "trip_probability"
        ]
        >
        TRIP_PROB_THRESHOLD
), "pattern_003 must exceed the trip threshold."

assert (
        BASELINE_PROFILES[
            "pattern_004"
        ][
            "trip_probability"
        ]
        >
        TRIP_PROB_THRESHOLD
), "pattern_004 must exceed the trip threshold."

assert (
        BASELINE_PROFILES[
            "pattern_001"
        ][
            "trip_probability"
        ]
        <
        TRIP_PROB_THRESHOLD
), "pattern_001 must stay below the trip threshold."

assert (
        BASELINE_PROFILES[
            "pattern_004"
        ][
            "mean_penetration"
        ]
        >
        BASELINE_PROFILES[
            "pattern_006"
        ][
            "mean_penetration"
        ]
), "pattern_004 must dominate pattern_006 in mean penetration."

assert (
        BASELINE_PROFILES[
            "pattern_004"
        ][
            "trip_probability"
        ]
        >
        BASELINE_PROFILES[
            "pattern_006"
        ][
            "trip_probability"
        ]
), "pattern_004 must dominate pattern_006 in trip probability."

print(
    "pattern_004 trip probability:",
    format(
        BASELINE_PROFILES[
            "pattern_004"
        ][
            "trip_probability"
        ],
        ".4f"
    )
)

print(
    "pattern_006 trip probability:",
    format(
        BASELINE_PROFILES[
            "pattern_006"
        ][
            "trip_probability"
        ],
        ".4f"
    )
)

print()

print(
    "TEST 7: Guardrail Classification"
)

BASELINE_GUARDRAILS = {}

for pattern_id in BASELINE_PROFILES:

    BASELINE_GUARDRAILS[
        pattern_id
    ] = (
        "TRIP"
        if BASELINE_PROFILES[
            pattern_id
        ][
            "trip_probability"
        ]
           > TRIP_PROB_THRESHOLD
        else "CLEAR"
    )

CONTROLLED_GUARDRAILS = {}

for pattern_id in CONTROLLED_PROFILES:

    CONTROLLED_GUARDRAILS[
        pattern_id
    ] = (
        "TRIP"
        if CONTROLLED_PROFILES[
            pattern_id
        ][
            "trip_probability"
        ]
           > TRIP_PROB_THRESHOLD
        else "CLEAR"
    )

BASELINE_TRIP = {
    pattern_id
    for pattern_id, status
    in BASELINE_GUARDRAILS.items()
    if status == "TRIP"
}

assert BASELINE_TRIP == {
    "pattern_003",
    "pattern_004"
}, "Baseline trip set mismatch."

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "|",
            BASELINE_GUARDRAILS[
                pattern_id
            ],
            "| trip=",
            format(
                BASELINE_PROFILES[
                    pattern_id
                ][
                    "trip_probability"
                ],
                ".4f"
            )
        ),
        TUNED_ORDER
    )
)

print()

print(
    "TEST 8: Preventive Execution Gating"
)

def execution_gate(
        guardrails
):

    blocked = sorted(
        pattern_id
        for pattern_id, status
        in guardrails.items()
        if status == "TRIP"
    )

    executed = sorted(
        pattern_id
        for pattern_id, status
        in guardrails.items()
        if status == "CLEAR"
    )

    return {
        "blocked":
            blocked,

        "executed":
            executed
    }


BASELINE_GATE = execution_gate(
    BASELINE_GUARDRAILS
)

assert BASELINE_GATE["blocked"] == [
    "pattern_003",
    "pattern_004"
], "Baseline gate must block the trip set."

assert BASELINE_GATE["executed"] == [
    "pattern_001",
    "pattern_002",
    "pattern_005",
    "pattern_006"
], "Baseline gate must execute the clear set."

print(
    "Baseline blocked:",
    BASELINE_GATE[
        "blocked"
    ]
)

print(
    "Baseline executed:",
    BASELINE_GATE[
        "executed"
    ]
)

print()

print(
    "TEST 9: Controlled Guardrail Verification"
)

assert all(
    status == "CLEAR"
    for status
    in CONTROLLED_GUARDRAILS.values()
), "Controlled defense must clear every guardrail."

CONTROLLED_GATE = execution_gate(
    CONTROLLED_GUARDRAILS
)

assert CONTROLLED_GATE["blocked"] == [], (
    "Controlled gate must block nothing."
)

assert len(
    CONTROLLED_GATE["executed"]
) == EXPECTED_PATTERNS, (
    "Controlled gate must execute all six patterns."
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "|",
            CONTROLLED_GUARDRAILS[
                pattern_id
            ],
            "| trip=",
            format(
                CONTROLLED_PROFILES[
                    pattern_id
                ][
                    "trip_probability"
                ],
                ".4f"
            )
        ),
        TUNED_ORDER
    )
)

print(
    "Controlled blocked:",
    CONTROLLED_GATE[
        "blocked"
    ]
)

print(
    "Controlled executed:",
    CONTROLLED_GATE[
        "executed"
    ]
)

print()

print(
    "TEST 10: Uncertainty Bandwidth Analysis"
)


def uncertainty_band(
        effectiveness,
        uncertainty
):

    conservative = 1.0

    optimistic = 1.0

    for eff in effectiveness:

        conservative *= (
            1.0
            -
            clamp(
                eff
                *
                (
                    1.0
                    -
                    uncertainty
                )
            )
        )

        optimistic *= (
            1.0
            -
            clamp(
                eff
                *
                (
                    1.0
                    +
                    uncertainty
                )
            )
        )

    return {
        "conservative":
            conservative,

        "optimistic":
            optimistic,

        "band_width":
            conservative
            -
            optimistic
    }


BASELINE_BANDS = {}

for pattern_id in BASELINE_EFFS:

    BASELINE_BANDS[
        pattern_id
    ] = uncertainty_band(
        BASELINE_EFFS[
            pattern_id
        ],
        UNCERTAINTY
    )

CONTROLLED_BANDS = {}

for pattern_id in CONTROLLED_EFFS:

    CONTROLLED_BANDS[
        pattern_id
    ] = uncertainty_band(
        CONTROLLED_EFFS[
            pattern_id
        ],
        UNCERTAINTY
    )

assert all(
    band["band_width"] > 0.0
    for band
    in BASELINE_BANDS.values()
), "Baseline bands must be strictly positive."

assert all(
    band["band_width"] > 0.0
    for band
    in CONTROLLED_BANDS.values()
), "Controlled bands must be strictly positive."

WIDEST_BASELINE = max(
    BASELINE_BANDS,
    key=lambda pattern_id:
        BASELINE_BANDS[
            pattern_id
        ][
            "band_width"
        ]
)

assert WIDEST_BASELINE == "pattern_004", (
    "pattern_004 must carry the widest baseline band."
)

assert (
        CONTROLLED_BANDS[
            "pattern_004"
        ][
            "band_width"
        ]
        <
        BASELINE_BANDS[
            "pattern_004"
        ][
            "band_width"
        ]
), "Adaptive control must shrink the pattern_004 band."

print(
    "Baseline bands:"
)

list(
    map(
        lambda pattern_id: print(
            pattern_id,
            "| cons=",
            format(
                BASELINE_BANDS[
                    pattern_id
                ][
                    "conservative"
                ],
                ".4f"
            ),
            "| opt=",
            format(
                BASELINE_BANDS[
                    pattern_id
                ][
                    "optimistic"
                ],
                ".4f"
            ),
            "| band=",
            format(
                BASELINE_BANDS[
                    pattern_id
                ][
                    "band_width"
                ],
                ".4f"
            )
        ),
        TUNED_ORDER
    )
)

print(
    "Controlled pattern_004 band:",
    format(
        CONTROLLED_BANDS[
            "pattern_004"
        ][
            "band_width"
        ],
        ".4f"
    )
)

print()

print(
    "TEST 11: Determinism"
)

DETERMINISTIC = True

for pattern_id in BASELINE_EFFS:

    first = guardrail_profile(
        BASELINE_EFFS[
            pattern_id
        ],
        MC_TRIALS,
        UNCERTAINTY,
        GUARDRAIL_THRESHOLD
    )

    second = guardrail_profile(
        BASELINE_EFFS[
            pattern_id
        ],
        MC_TRIALS,
        UNCERTAINTY,
        GUARDRAIL_THRESHOLD
    )

    if stable_hash(
        first
    ) != stable_hash(
        second
    ):

        DETERMINISTIC = False

print(
    "Deterministic:",
    DETERMINISTIC
)

assert DETERMINISTIC, (
    "Monte Carlo guardrails are nondeterministic."
)

print(
    "Deterministic guardrails validated."
)

print()

print(
    "TEST 12: Numerical Health"
)

BASELINE_TENSOR = torch.tensor(
    [
        BASELINE_PROFILES[
            pattern_id
        ][
            "mean_penetration"
        ]
        for pattern_id
        in TUNED_ORDER
    ],
    dtype=torch.float32
)

CONTROLLED_TENSOR = torch.tensor(
    [
        CONTROLLED_PROFILES[
            pattern_id
        ][
            "mean_penetration"
        ]
        for pattern_id
        in TUNED_ORDER
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
    if len(
        BASELINE_PROFILES
    ) == EXPECTED_PATTERNS
    else [
        "Baseline profiles incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(
        CONTROLLED_PROFILES
    ) == EXPECTED_PATTERNS
    else [
        "Controlled profiles incomplete."
    ]
)

PROMOTION_ERRORS += (
    []
    if BASELINE_TRIP == {
        "pattern_003",
        "pattern_004"
    }
    else [
        "Baseline trip set invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if CONTROLLED_GATE["blocked"] == []
    else [
        "Controlled gate still blocks patterns."
    ]
)

PROMOTION_ERRORS += (
    []
    if len(
        CONTROLLED_GATE["executed"]
    ) == EXPECTED_PATTERNS
    else [
        "Controlled gate must execute all patterns."
    ]
)

PROMOTION_ERRORS += (
    []
    if WIDEST_BASELINE == "pattern_004"
    else [
        "Widest baseline band invalid."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Guardrails nondeterministic."
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
    "Baseline trip set:",
    sorted(
        BASELINE_TRIP
    )
)

print(
    "Controlled blocked:",
    CONTROLLED_GATE[
        "blocked"
    ]
)

print(
    "Widest baseline band:",
    WIDEST_BASELINE
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "128R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "128R promotion gate passed."
)

print()

print(
    "TEST 14: Persist Uncertainty Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "128R",

    "capability":
        "uncertainty_aware_preventive_execution_probabilistic_guardrails",

    "created_at":
        datetime.now().isoformat(),

    "source_lesson":
        "127R",

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

    "control_actions":
        CONTROL_ACTIONS,

    "guardrail_frame":
        GUARDRAIL_FRAME,

    "baseline_profiles":
        BASELINE_PROFILES,

    "controlled_profiles":
        CONTROLLED_PROFILES,

    "baseline_guardrails":
        BASELINE_GUARDRAILS,

    "controlled_guardrails":
        CONTROLLED_GUARDRAILS,

    "baseline_gate":
        BASELINE_GATE,

    "controlled_gate":
        CONTROLLED_GATE,

    "baseline_bands":
        BASELINE_BANDS,

    "controlled_bands":
        CONTROLLED_BANDS,

    "verification":
        {
            "baseline_trip":
                sorted(
                    BASELINE_TRIP
                ),

            "controlled_blocked":
                CONTROLLED_GATE[
                    "blocked"
                ],

            "widest_band":
                WIDEST_BASELINE,

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

assert (
        RELOADED["baseline_guardrails"]
        ==
        BASELINE_GUARDRAILS
), "Baseline guardrails changed after reload."

assert (
        RELOADED["controlled_guardrails"]
        ==
        CONTROLLED_GUARDRAILS
), "Controlled guardrails changed after reload."

assert (
        RELOADED["controlled_gate"]["blocked"]
        ==
        []
), "Controlled gate changed after reload."

assert (
        RELOADED["baseline_bands"]
        ==
        BASELINE_BANDS
), "Baseline bands changed after reload."

print(
    "Reloaded baseline guardrails:",
    dict(
        map(
            lambda pair: (
                pair[0],
                pair[1][0]
            ),
            RELOADED[
                "baseline_guardrails"
            ].items()
        )
    )
)

print(
    "Reloaded controlled gate executed:",
    len(
        RELOADED[
            "controlled_gate"
        ][
            "executed"
        ]
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
            "128R",

        "capability":
            "uncertainty_aware_preventive_execution_probabilistic_guardrails",

        "guardrail_frame":
            GUARDRAIL_FRAME,

        "baseline_profiles":
            BASELINE_PROFILES,

        "controlled_profiles":
            CONTROLLED_PROFILES,

        "baseline_guardrails":
            BASELINE_GUARDRAILS,

        "controlled_guardrails":
            CONTROLLED_GUARDRAILS,

        "baseline_gate":
            BASELINE_GATE,

        "controlled_gate":
            CONTROLLED_GATE,

        "baseline_bands":
            BASELINE_BANDS,

        "controlled_bands":
            CONTROLLED_BANDS
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "128R",

        "memory_version":
            MEMORY_VERSION,

        "uncertainty":
            UNCERTAINTY,

        "guardrail_threshold":
            GUARDRAIL_THRESHOLD,

        "trip_probability_threshold":
            TRIP_PROB_THRESHOLD,

        "monte_carlo_trials":
            MC_TRIALS,

        "baseline_trip_set":
            sorted(
                BASELINE_TRIP
            ),

        "controlled_blocked":
            CONTROLLED_GATE[
                "blocked"
            ],

        "controlled_executed":
            CONTROLLED_GATE[
                "executed"
            ],

        "widest_baseline_band":
            WIDEST_BASELINE,

        "baseline_pattern_004_trip_probability":
            BASELINE_PROFILES[
                "pattern_004"
            ][
                "trip_probability"
            ],

        "controlled_pattern_004_trip_probability":
            CONTROLLED_PROFILES[
                "pattern_004"
            ][
                "trip_probability"
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
            "128R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "129R Anomaly-First Adaptive Scheduling "
                "+ Critical Path Defense"
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
    "SILVERWING 128R ARCHITECTURE"
)

print(
    "Layer Effectiveness (point)"
)

print(
    "   ↓"
)

print(
    "± Relative Uncertainty"
)

print(
    "   ↓"
)

print(
    "Seeded Monte Carlo"
)

print(
    "   ↓"
)

print(
    "Penetration Distribution"
)

print(
    "   ↓"
)

print(
    "Trip Probability"
)

print(
    "   ↓"
)

print(
    "P(pen > guardrail) > trip threshold?"
)

print(
    "   ↓"
)

print(
    "TRIP / CLEAR"
)

print(
    "   ↓"
)

print(
    "Preventive Execution Gate"
)

print(
    "   ↓"
)

print(
    "Adaptive Control Shrinks the Band"
)

print()

print(
    "WHAT 128R ADDS"
)

print(
    "Uncertainty as a first-class signal: effectiveness is no "
    "longer a point estimate. Monte Carlo trip probabilities "
    "decide whether a pattern may run, and preventive execution "
    "gates on the verdict."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems whose model confidence is not perfect and where "
    "acting on an optimistic point estimate can be catastrophic."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "A pattern at the edge of the threshold is not safe just "
    "because its mean looks acceptable. Probabilistic guardrails "
    "expose that lie and refuse to execute until the band shrinks."
)

print()

print(
    "NEXT: 129R Anomaly-First Adaptive Scheduling "
    "+ Critical Path Defense"
)

print()

print(
    "=== LESSON 128R COMPLETE ==="
)

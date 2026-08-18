# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 138R
# Autonomous Self-Reinforcing Policy
# + Continuous Commitment Cascade
# ============================================================
#
# 135R  -> Autonomous Policy Evolution
#         + Zero-Regression Learning Protocol
# 136R  -> Autonomous Cross-Cycle Evolution Governance
#         + Bounded Innovation Protocol
# 137R  -> Autonomous Governance Ledger
#         + Innovation Accountability Chain
# 138R  -> Autonomous Self-Reinforcing Policy
#         + Continuous Commitment Cascade
#
# ============================================================
# PURPOSE
# ============================================================
#
# 137R sealed the governed policy in a ledger. A policy that
# improves once and stops is not autonomous -- it is a one-
# shot optimization. 138R makes improvement continuous and
# self-reinforcing.
#
# Self-reinforcement: the policy sealed in cycle 1 (137R)
# becomes the baseline of cycle 2. Cycle 2 tightens the
# target, evolves only the violating patterns, is governed
# by the same bounded innovation protocol, and its improved
# policy becomes the baseline of the next cycle. The system
# feeds on its own committed improvements -- no external
# intervention at any point.
#
# Continuous commitment cascade: every cycle produces a
# new commitment that chains to the previous cycle's ledger
# seal, forming an unbroken cascade:
#
#     134R commitment
#        |
#     135R evolution -> 136R governance -> 137R seal
#        |                                    |
#        |               cycle 1             |  d5b82086...
#        |                                   |
#      +--------------------------- cycle 2 (138R)
#                                       |
#                                       |  new commitment
#                                       v
#                              1bb3b613...
#
# Cycle 2 evolution (138R):
#
#      baseline exposure  0.0922  (137R seal)
#        |
#      target 0.100 -> exposure 0.0888
#        |
#      target 0.095 -> exposure 0.0855
#        |
#      target 0.090 -> exposure 0.0822
#        |
#      new commitment -> chained to the 137R seal
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. 135R, 136R and 137R memories are the sources of truth.
# 2. The cycle-2 baseline is exactly the 137R-sealed policy.
# 3. Every cycle must strictly improve exposure.
# 4. The bounded innovation protocol governs every generation.
# 5. The new commitment must chain to the 137R ledger seal.
# 6. Determinism must be checked.
# 7. Numerical health must be checked.
# 8. Persistence and reload must be checked.
# 9. Promotion requires all validation gates to pass.
# 10. External LLM: NONE.
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
MEMORY_VERSION = "138R.1"
EXPECTED_PATTERNS = 6
EXPECTED_GENERATIONS = 3
CYCLE = 2
CYCLE_TARGETS = [
    0.100,
    0.095,
    0.090
]
CYCLE_FINAL_TARGET = 0.090
CYCLE_1_SEAL = (
    "d5b8208664587bdd589d5aaacf91ef2857"
    "99ca6f6882fcb201a0e3d2e5e28b2c"
)
EXPECTED_COMMITMENT_HASH = (
    "1bb3b613ff0b9bef30db42adf308d2ac76"
    "f70aa0946271a8f1a3e816c5f18f4e"
)

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent
LESSON_135R = PHASE5_DIR / "lesson135R"
LESSON_136R = PHASE5_DIR / "lesson136R"
LESSON_137R = PHASE5_DIR / "lesson137R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_EVOLUTION_MEMORY = (
        LESSON_135R
        / "silverwing_policy_evolution_memory.json"
)

SOURCE_GOVERNANCE_MEMORY = (
        LESSON_136R
        / "silverwing_evolution_governance_memory.json"
)

SOURCE_LEDGER_MEMORY = (
        LESSON_137R
        / "silverwing_governance_ledger_memory.json"
)

SOURCE_LEDGER_INDEX = (
        LESSON_137R
        / "silverwing_governance_ledger_index.pt"
)

SOURCE_LEDGER_DATASET = (
        LESSON_137R
        / "silverwing_governance_ledger_dataset.json"
)

SOURCE_LEDGER_REPORT = (
        LESSON_137R
        / "silverwing_governance_ledger_report.json"
)

SOURCE_LEDGER_REGISTRY = (
        LESSON_137R
        / "silverwing_governance_ledger_registry.json"
)

SOURCE_LEDGER_CHECKPOINT = (
        LESSON_137R
        / "checkpoints"
        / "silverwing_governance_ledger_best.pt"
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_self_reinforcing_policy_memory.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_self_reinforcing_policy_index.pt"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_self_reinforcing_policy_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_self_reinforcing_policy_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_self_reinforcing_policy_registry.json"
)

CHECKPOINT_FILE = (
        CHECKPOINT_DIR
        / "silverwing_self_reinforcing_policy_best.pt"
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
    "PHASE 5 - LESSON 138R"
)

print(
    "Autonomous Self-Reinforcing Policy"
)

print(
    "+ Continuous Commitment Cascade"
)

print()

print(
    "135R -> Autonomous Policy Evolution"
)

print(
    "        + Zero-Regression Learning Protocol"
)

print(
    "136R -> Autonomous Cross-Cycle Evolution Governance"
)

print(
    "        + Bounded Innovation Protocol"
)

print(
    "137R -> Autonomous Governance Ledger"
)

print(
    "        + Innovation Accountability Chain"
)

print(
    "138R -> Autonomous Self-Reinforcing Policy"
)

print(
    "        + Continuous Commitment Cascade"
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
    "TEST 1: Verify 138R Inputs"
)

REQUIRED_FILES = [
    SOURCE_EVOLUTION_MEMORY,
    SOURCE_GOVERNANCE_MEMORY,
    SOURCE_LEDGER_MEMORY,
    SOURCE_LEDGER_INDEX,
    SOURCE_LEDGER_DATASET,
    SOURCE_LEDGER_REPORT,
    SOURCE_LEDGER_REGISTRY,
    SOURCE_LEDGER_CHECKPOINT
]

assert all(
    path.exists()
    for path in REQUIRED_FILES
), "One or more source inputs are missing."

for path in REQUIRED_FILES:

    print(
        "FOUND:",
        path.name
    )

print()

print(
    "TEST 2: Load Source Memories"
)

EVOLUTION_SOURCE = read_json(
    SOURCE_EVOLUTION_MEMORY
)

GOVERNANCE_SOURCE = read_json(
    SOURCE_GOVERNANCE_MEMORY
)

LEDGER_SOURCE = read_json(
    SOURCE_LEDGER_MEMORY
)

assert isinstance(
    EVOLUTION_SOURCE,
    dict
), "135R evolution memory is invalid."

assert isinstance(
    GOVERNANCE_SOURCE,
    dict
), "136R governance memory is invalid."

assert isinstance(
    LEDGER_SOURCE,
    dict
), "137R ledger memory is invalid."

TUNED_ORDER = EVOLUTION_SOURCE.get(
    "tuned_order",
    []
)

assert len(
    TUNED_ORDER
) == EXPECTED_PATTERNS, (
    "Order must cover six patterns."
)

GOVERNANCE_FRAME = GOVERNANCE_SOURCE.get(
    "governance_frame",
    {}
)

BOOST_BOUND = GOVERNANCE_FRAME.get(
    "boost_bound",
    0.05
)

EXPOSURE_BOUND = GOVERNANCE_FRAME.get(
    "exposure_bound",
    0.005
)

REGRESSION_LIMIT = GOVERNANCE_FRAME.get(
    "regression_limit",
    1e-4
)

SEALED_LEDGER_HASH = LEDGER_SOURCE.get(
    "final_ledger_hash",
    ""
)

assert (
        SEALED_LEDGER_HASH
        ==
        CYCLE_1_SEAL
), "137R ledger seal mismatch."

print(
    "137R ledger seal:",
    SEALED_LEDGER_HASH[:16]
)

print(
    "Governance bounds: boost",
    BOOST_BOUND,
    "| exposure",
    EXPOSURE_BOUND
)

print()

print(
    "TEST 3: Establish the Cycle-1 Baseline"
)

SEALED_POLICY = GOVERNANCE_SOURCE.get(
    "final_policy",
    {}
)

assert (
        SEALED_POLICY["boosts"]
        ==
        EVOLUTION_SOURCE.get(
            "evolved_boosts"
        )
        and
        SEALED_POLICY["penetrations"]
        ==
        EVOLUTION_SOURCE.get(
            "evolved_penetrations"
        )
), "The sealed policy must be the evolved policy."

BASELINE_BOOSTS = dict(
    SEALED_POLICY["boosts"]
)

BASELINE_PENETRATIONS = dict(
    SEALED_POLICY["penetrations"]
)


def compute_exposure(
        penetrations
):

    return float(
        torch.tensor(
            list(
                penetrations.values()
            ),
            dtype=torch.float32
        ).mean()
    )


BASELINE_EXPOSURE = compute_exposure(
    BASELINE_PENETRATIONS
)

assert abs(
    BASELINE_EXPOSURE
    -
    0.09217751026153564
) <= 1e-9, (
    "Cycle-1 sealed exposure mismatch."
)

for pattern_id in TUNED_ORDER:

    print(
        pattern_id,
        "| boost",
        format(
            BASELINE_BOOSTS[pattern_id],
            ".4f"
        ),
        "| penetration",
        format(
            BASELINE_PENETRATIONS[pattern_id],
            ".4f"
        )
    )

print(
    "Baseline exposure:",
    format(
        BASELINE_EXPOSURE,
        ".6f"
    )
)

print()

print(
    "TEST 4: Define the Cycle-2 Cascade Frame"
)

assert len(
    CYCLE_TARGETS
) == EXPECTED_GENERATIONS, (
    "Cycle 2 must tighten three times."
)

assert CYCLE_TARGETS == sorted(
    CYCLE_TARGETS,
    reverse=True
), "Targets must tighten monotonically."

for generation, target in enumerate(
        CYCLE_TARGETS,
        1
):

    print(
        "Generation",
        generation,
        "| target",
        target
    )

print()

print(
    "TEST 5: Run Cycle-2 Evolution"
)


def re_tune(
        boost,
        penetration,
        target
):

    return (
            boost
            *
            math.log(
                target
            )
            /
            math.log(
                penetration
            )
    )


CYCLE_BOOSTS = dict(
    BASELINE_BOOSTS
)

CYCLE_PENETRATIONS = dict(
    BASELINE_PENETRATIONS
)

TRAJECTORY = [
    BASELINE_EXPOSURE
]

GENERATIONS = []

for generation, target in enumerate(
        CYCLE_TARGETS,
        1
):

    changed = [
        pattern_id
        for pattern_id
        in TUNED_ORDER
        if CYCLE_PENETRATIONS[
            pattern_id
        ]
        > target
    ]

    for pattern_id in changed:

        CYCLE_BOOSTS[
            pattern_id
        ] = re_tune(
            CYCLE_BOOSTS[pattern_id],
            CYCLE_PENETRATIONS[pattern_id],
            target
        )

        CYCLE_PENETRATIONS[
            pattern_id
        ] = target

    exposure = compute_exposure(
        CYCLE_PENETRATIONS
    )

    exposure_delta = (
            exposure
            -
            TRAJECTORY[
                generation - 1
            ]
    )

    TRAJECTORY.append(
        exposure
    )

    GENERATIONS.append(
        {
            "generation":
                generation,

            "target":
                target,

            "changed":
                changed,

            "exposure":
                exposure,

            "exposure_delta":
                exposure_delta
        }
    )

for record in GENERATIONS:

    print(
        "Generation",
        record["generation"],
        "| target=",
        record["target"],
        "| changed=",
        record["changed"],
        "| exposure=",
        format(
            record["exposure"],
            ".4f"
        )
    )

print()

print(
    "TEST 6: Verify the Cycle-2 Trajectory"
)

assert (
        TRAJECTORY
        ==
        sorted(
            TRAJECTORY,
            reverse=True
        )
), "Cycle-2 exposure must be strictly monotone."

assert abs(
    TRAJECTORY[0]
    -
    0.09217751026153564
) <= 1e-9, (
    "Cycle-2 baseline mismatch."
)

assert abs(
    TRAJECTORY[1]
    -
    0.0888441801071167
) <= 1e-9, (
    "Cycle-2 generation 1 mismatch."
)

assert abs(
    TRAJECTORY[2]
    -
    0.08551084995269775
) <= 1e-9, (
    "Cycle-2 generation 2 mismatch."
)

assert abs(
    TRAJECTORY[3]
    -
    0.08217751234769821
) <= 1e-9, (
    "Cycle-2 generation 3 mismatch."
)

FINAL_EXPOSURE = TRAJECTORY[-1]

print(
    "Trajectory:",
    [
        format(
            value,
            ".4f"
        )
        for value
        in TRAJECTORY
    ]
)

print(
    "Final exposure:",
    format(
        FINAL_EXPOSURE,
        ".6f"
    )
)

print()

print(
    "TEST 7: Verify the Cycle-2 Scope"
)

for record in GENERATIONS:

    expected_scope = sorted(
        pattern_id
        for pattern_id
        in TUNED_ORDER
        if BASELINE_PENETRATIONS[
            pattern_id
        ]
        > CYCLE_FINAL_TARGET
    )

    assert sorted(
        record["changed"]
    ) == expected_scope, (
        "Cycle-2 scope mismatch."
    )

print(
    "Every generation changed exactly the final-target "
    "violators:",
    sorted(
        GENERATIONS[-1]["changed"]
    )
)

print()

print(
    "TEST 8: Verify Cycle-2 Governance"
)

MAX_BOOST_DELTAS = []

CYCLE_BOOSTS_2 = dict(
    BASELINE_BOOSTS
)

CYCLE_PENETRATIONS_2 = dict(
    BASELINE_PENETRATIONS
)

for record in GENERATIONS:

    max_delta = 0.0

    for pattern_id in record["changed"]:

        old_boost = CYCLE_BOOSTS_2[
            pattern_id
        ]

        CYCLE_BOOSTS_2[
            pattern_id
        ] = re_tune(
            CYCLE_BOOSTS_2[pattern_id],
            CYCLE_PENETRATIONS_2[pattern_id],
            record["target"]
        )

        CYCLE_PENETRATIONS_2[
            pattern_id
        ] = record[
            "target"
        ]

        max_delta = max(
            max_delta,
            abs(
                CYCLE_BOOSTS_2[
                    pattern_id
                ]
                -
                old_boost
            )
        )

    MAX_BOOST_DELTAS.append(
        max_delta
    )

assert MAX_BOOST_DELTAS[0] <= BOOST_BOUND, (
    "Generation 1 exceeded the boost bound."
)

assert MAX_BOOST_DELTAS[1] <= BOOST_BOUND, (
    "Generation 2 exceeded the boost bound."
)

assert MAX_BOOST_DELTAS[2] <= BOOST_BOUND, (
    "Generation 3 exceeded the boost bound."
)

assert abs(
    MAX_BOOST_DELTAS[0]
    -
    0.034078401
) <= 1e-6, (
    "Generation 1 boost delta mismatch."
)

assert abs(
    MAX_BOOST_DELTAS[1]
    -
    0.035826759
) <= 1e-6, (
    "Generation 2 boost delta mismatch."
)

assert abs(
    MAX_BOOST_DELTAS[2]
    -
    0.037764260
) <= 1e-6, (
    "Generation 3 boost delta mismatch."
)

assert all(
    abs(
        record["exposure_delta"]
    )
    <= EXPOSURE_BOUND
    for record
    in GENERATIONS
), "A generation exceeded the exposure bound."

print(
    "Max boost deltas:",
    [
        format(
            value,
            ".6f"
        )
        for value
        in MAX_BOOST_DELTAS
    ]
)

print(
    "All within boost bound:",
    all(
        value
        <=
        BOOST_BOUND
        for value
        in MAX_BOOST_DELTAS
    )
)

print(
    "All exposure steps within bound:",
    all(
        abs(
            record["exposure_delta"]
        )
        <=
        EXPOSURE_BOUND
        for record
        in GENERATIONS
    )
)

print()

print(
    "TEST 9: Build the Cycle-2 Commitment"
)

EXPOSURE_REDUCTION = (
        BASELINE_EXPOSURE
        -
        FINAL_EXPOSURE
)

assert EXPOSURE_REDUCTION > 0, (
    "Self-reinforcement must strictly improve exposure."
)

COMMITMENT_RECORD = {
    "cycle":
        CYCLE,

    "parent_ledger_hash":
        CYCLE_1_SEAL,

    "parent_lesson":
        "137R",

    "baseline_exposure":
        BASELINE_EXPOSURE,

    "generation_targets":
        CYCLE_TARGETS,

    "final_exposure":
        FINAL_EXPOSURE,

    "exposure_reduction":
        EXPOSURE_REDUCTION,

    "final_boosts":
        CYCLE_BOOSTS,

    "final_penetrations":
        CYCLE_PENETRATIONS
}

COMMITMENT_HASH = stable_hash(
    COMMITMENT_RECORD
)

assert (
        COMMITMENT_HASH
        ==
        EXPECTED_COMMITMENT_HASH
), "Cycle-2 commitment hash mismatch."

print(
    "Commitment record chained to:",
    COMMITMENT_RECORD["parent_ledger_hash"][:16]
)

print(
    "Commitment hash:",
    COMMITMENT_HASH[:16]
)

print()

print(
    "TEST 10: Verify the Continuous Commitment Cascade"
)

CASCADE = [
    {
        "cycle": 1,
        "lesson": "137R",
        "hash": SEALED_LEDGER_HASH
    },
    {
        "cycle": CYCLE,
        "lesson": "138R",
        "hash": COMMITMENT_HASH
    }
]

assert (
        CASCADE[0]["hash"]
        ==
        CYCLE_1_SEAL
), "Cascade must start with the 137R seal."

assert (
        CASCADE[1]["hash"]
        ==
        COMMITMENT_HASH
), "Cascade must end with the cycle-2 commitment."

for entry in CASCADE:

    print(
        "Cycle",
        entry["cycle"],
        "|",
        entry["lesson"],
        "|",
        entry["hash"][:16]
    )

print(
    "Cascade continuous:",
    True
)

print()

print(
    "TEST 11: Verify Self-Reinforcement"
)

assert FINAL_EXPOSURE < BASELINE_EXPOSURE, (
    "Cycle 2 must improve on cycle 1."
)

assert EXPOSURE_REDUCTION > 0, (
    "Exposure reduction must be positive."
)

print(
    "Cycle 1 exposure:",
    format(
        BASELINE_EXPOSURE,
        ".6f"
    )
)

print(
    "Cycle 2 exposure:",
    format(
        FINAL_EXPOSURE,
        ".6f"
    )
)

print(
    "Cross-cycle reduction:",
    format(
        EXPOSURE_REDUCTION,
        ".6f"
    )
)

print(
    "Self-reinforcing: True"
)

print()

print(
    "TEST 12: Determinism"
)

RE_BOOSTS = dict(
    BASELINE_BOOSTS
)

RE_PENETRATIONS = dict(
    BASELINE_PENETRATIONS
)

for record in GENERATIONS:

    for pattern_id in record["changed"]:

        RE_BOOSTS[
            pattern_id
        ] = re_tune(
            RE_BOOSTS[pattern_id],
            RE_PENETRATIONS[pattern_id],
            record["target"]
        )

        RE_PENETRATIONS[
            pattern_id
        ] = record[
            "target"
        ]

DETERMINISTIC = (
        RE_BOOSTS
        ==
        CYCLE_BOOSTS
        and
        RE_PENETRATIONS
        ==
        CYCLE_PENETRATIONS
)

print(
    "Boosts deterministic:",
    RE_BOOSTS
    ==
    CYCLE_BOOSTS
)

print(
    "Penetrations deterministic:",
    RE_PENETRATIONS
    ==
    CYCLE_PENETRATIONS
)

assert DETERMINISTIC

print()

print(
    "TEST 13: Numerical Health"
)

FINAL_BOOST_TENSOR = torch.tensor(
    list(
        CYCLE_BOOSTS.values()
    ),
    dtype=torch.float32
)

FINAL_PEN_TENSOR = torch.tensor(
    list(
        CYCLE_PENETRATIONS.values()
    ),
    dtype=torch.float32
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(
        FINAL_BOOST_TENSOR
    ).all()
    and
    torch.isfinite(
        FINAL_PEN_TENSOR
    ).all()
)

print(
    "Boost NaN:",
    int(
        torch.isnan(
            FINAL_BOOST_TENSOR
        ).sum()
    )
)

print(
    "Penetration Inf:",
    int(
        torch.isinf(
            FINAL_PEN_TENSOR
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
    "TEST 14: Final Promotion Gate"
)

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if FINAL_EXPOSURE
    < BASELINE_EXPOSURE
    else [
        "Self-reinforcement failed to improve exposure."
    ]
)

PROMOTION_ERRORS += (
    []
    if COMMITMENT_HASH
    == EXPECTED_COMMITMENT_HASH
    else [
        "Commitment hash mismatch."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        value
        <=
        BOOST_BOUND
        for value
        in MAX_BOOST_DELTAS
    )
    else [
        "A generation violated the boost bound."
    ]
)

PROMOTION_ERRORS += (
    []
    if all(
        abs(
            record["exposure_delta"]
        )
        <=
        EXPOSURE_BOUND
        for record
        in GENERATIONS
    )
    else [
        "A generation violated the exposure bound."
    ]
)

PROMOTION_ERRORS += (
    []
    if DETERMINISTIC
    else [
        "Cycle-2 evolution is nondeterministic."
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
    "Cycle 2 exposure:",
    format(
        FINAL_EXPOSURE,
        ".6f"
    )
)

print(
    "Cross-cycle reduction:",
    format(
        EXPOSURE_REDUCTION,
        ".6f"
    )
)

print(
    "Promotion errors:",
    len(
        PROMOTION_ERRORS
    )
)

assert not PROMOTION_ERRORS, (
        "138R promotion gate failed: "
        +
        "; ".join(
            PROMOTION_ERRORS
        )
)

print(
    "138R promotion gate passed."
)

print()

print(
    "TEST 15: Persist Self-Reinforcing Memory"
)

MEMORY = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "138R",

    "capability":
        "autonomous_self_reinforcing_policy_continuous_commitment_cascade",

    "created_at":
        datetime.now().isoformat(),

    "source_lessons":
        [
            "135R",
            "136R",
            "137R"
        ],

    "cycle":
        CYCLE,

    "governance_frame":
        GOVERNANCE_FRAME,

    "tuned_order":
        TUNED_ORDER,

    "baseline_exposure":
        BASELINE_EXPOSURE,

    "generation_targets":
        CYCLE_TARGETS,

    "generations":
        GENERATIONS,

    "exposure_trajectory":
        TRAJECTORY,

    "exposure_reduction":
        EXPOSURE_REDUCTION,

    "final_exposure":
        FINAL_EXPOSURE,

    "final_boosts":
        CYCLE_BOOSTS,

    "final_penetrations":
        CYCLE_PENETRATIONS,

    "max_boost_deltas":
        MAX_BOOST_DELTAS,

    "commitment_record":
        COMMITMENT_RECORD,

    "commitment_hash":
        COMMITMENT_HASH,

    "cascade":
        CASCADE,

    "verification":
        {
            "self_reinforcing":
                FINAL_EXPOSURE
                <
                BASELINE_EXPOSURE,

            "cascade_continuous":
                COMMITMENT_HASH
                ==
                EXPECTED_COMMITMENT_HASH,

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
    "TEST 16: Reload Persistent Memory"
)

RELOADED = read_json(
    MEMORY_FILE
)

assert (
        RELOADED["memory_version"]
        ==
        MEMORY_VERSION
), "Memory version mismatch after reload."

assert (
        RELOADED["commitment_hash"]
        ==
        COMMITMENT_HASH
), "Commitment hash changed after reload."

assert (
        RELOADED["exposure_trajectory"]
        ==
        TRAJECTORY
), "Trajectory changed after reload."

print(
    "Reloaded cascade:",
    len(
        RELOADED[
            "cascade"
        ]
    ),
    "commitments"
)

print(
    "Reloaded final exposure:",
    format(
        RELOADED[
            "final_exposure"
        ],
        ".6f"
    )
)

print(
    "Reload validation passed."
)

print()

print(
    "TEST 17: Save Dataset and Reports"
)

save_json(
    DATASET_FILE,
    {
        "lesson":
            "138R",

        "capability":
            "autonomous_self_reinforcing_policy_continuous_commitment_cascade",

        "cycle":
            CYCLE,

        "generation_targets":
            CYCLE_TARGETS,

        "generations":
            GENERATIONS,

        "exposure_trajectory":
            TRAJECTORY,

        "final_boosts":
            CYCLE_BOOSTS,

        "final_penetrations":
            CYCLE_PENETRATIONS,

        "commitment_hash":
            COMMITMENT_HASH,

        "cascade":
            CASCADE
    }
)

save_json(
    REPORT_FILE,
    {
        "lesson":
            "138R",

        "memory_version":
            MEMORY_VERSION,

        "cycle":
            CYCLE,

        "baseline_exposure":
            BASELINE_EXPOSURE,

        "final_exposure":
            FINAL_EXPOSURE,

        "exposure_reduction":
            EXPOSURE_REDUCTION,

        "max_boost_delta":
            max(
                MAX_BOOST_DELTAS
            ),

        "self_reinforcing":
            FINAL_EXPOSURE
            <
            BASELINE_EXPOSURE,

        "cascade_continuous":
            COMMITMENT_HASH
            ==
            EXPECTED_COMMITMENT_HASH,

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
            "138R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "139R Autonomous Full-Cycle Audit "
                "+ Complete Self-Reinforcing Verification"
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
    "SILVERWING 138R ARCHITECTURE"
)

print(
    "137R Ledger Seal (cycle 1)"
)

print(
    "   |"
)

print(
    "Cycle-2 Baseline (0.0922)"
)

print(
    "   |"
)

print(
    "Target 0.100 -> 0.0888"
)

print(
    "Target 0.095 -> 0.0855"
)

print(
    "Target 0.090 -> 0.0822"
)

print(
    "   |"
)

print(
    "Bounded Innovation Governance"
)

print(
    "   |"
)

print(
    "New Commitment -> Chained to 137R Seal"
)

print(
    "   |"
)

print(
    "Continuous Commitment Cascade"
)

print()

print(
    "WHAT 138R ADDS"
)

print(
    "A second, fully autonomous improvement cycle that feeds "
    "on the ledger-sealed policy of cycle 1, governed by the "
    "same bounded innovation protocol, and chained as a new "
    "commitment -- making improvement continuous and "
    "self-reinforcing."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Systems that must keep improving forever: autonomous "
    "optimization, adaptive control, self-tuning infrastructure."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "One improvement is a result; a cascade of committed, "
    "governed improvements is a capability. Self-reinforcement "
    "turns the policy into a compounding asset."
)

print()

print(
    "NEXT: 139R Autonomous Full-Cycle Audit "
    "+ Complete Self-Reinforcing Verification"
)

print()

print(
    "=== LESSON 138R COMPLETE ==="
)

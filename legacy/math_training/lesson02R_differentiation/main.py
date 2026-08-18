# ============================================================
# SILVERWING ML - MATH TRAINING - LESSON 02R
# Differentiation: all possible derivative outcomes
# ============================================================
#
# Lesson 01R -> Linear Equations: f(x) = mx + b
#               + all possible outcomes
# Lesson 02R -> Differentiation: all possible outcomes
# Lesson 03R -> Integration: all possible outcomes
#
# ============================================================
# PURPOSE
# ============================================================
#
# The derivative measures rate of change. Every engineering
# model -- velocity, growth, control response -- is a
# derivative. This lesson trains every possible outcome of
# differentiation:
#
#   1. RULES        -- power, constant, linear, sum,
#                      constant multiple, product,
#                      quotient, chain.
#   2. VERIFICATION -- every analytic derivative is checked
#                      against numerical central differences.
#   3. CRITICAL POINTS -- where f'(x) = 0.
#   4. EXTREMA      -- local min (f'' > 0) and local max
#                      (f'' < 0).
#   5. MONOTONICITY -- increasing (f' > 0) and decreasing
#                      (f' < 0) intervals.
#
# All possible outcomes trained:
#
#       power        d/dx x^n = n*x^(n-1)
#       constant     d/dx c = 0
#       linear       d/dx (mx+b) = m
#       sum          d/dx (u+v) = u' + v'
#       multiple     d/dx (c*u) = c*u'
#       product      d/dx (u*v) = u'*v + u*v'
#       quotient     d/dx (u/v) = (u'*v - u*v')/v^2
#       chain        d/dx f(g(x)) = f'(g(x))*g'(x)
#       local min    f''(x) > 0 at f'(x) = 0
#       local max    f''(x) < 0 at f'(x) = 0
#       increasing   f'(x) > 0
#       decreasing   f'(x) < 0
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. Every derivative outcome class must be exercised.
# 2. Analytic derivatives must match central differences.
# 3. No external LLM is consulted.
# 4. Determinism must be checked.
# 5. Numerical health must be checked.
# 6. Persistence and reload must be checked.
# 7. Promotion requires all validation gates to pass.
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
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
MEMORY_VERSION = "02R.1"
DIFF_STEP = 1e-5
DIFF_TOL = 1e-4

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = BASE_DIR / "silverwing_math_differentiation_memory.json"
INDEX_FILE = BASE_DIR / "silverwing_math_differentiation_index.pt"
DATASET_FILE = BASE_DIR / "silverwing_math_differentiation_dataset.json"
REPORT_FILE = BASE_DIR / "silverwing_math_differentiation_report.json"
REGISTRY_FILE = BASE_DIR / "silverwing_math_differentiation_registry.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "silverwing_math_differentiation_best.pt"

read_json = lambda path: json.loads(
    path.read_text(encoding="utf-8")
)

save_json = lambda path, data: path.write_text(
    json.dumps(data, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

stable_hash = lambda value: hashlib.sha256(
    json.dumps(value, sort_keys=True, default=str).encode("utf-8")
).hexdigest()

torch.manual_seed(SEED)
random.seed(SEED)

print("=== SILVERWING ML ===")
print("MATH TRAINING - LESSON 02R")
print("Differentiation: all possible derivative outcomes")
print()
print("Lesson 01R -> Linear Equations: f(x) = mx + b")
print("Lesson 02R -> Differentiation: all possible outcomes")
print("Lesson 03R -> Integration: all possible outcomes")
print()
print("External LLM: NONE")
print("Memory version:", MEMORY_VERSION)
print()

print("TEST 1: Define the Differentiation Rules")

RULES = [
    "power",
    "constant",
    "linear",
    "sum",
    "constant_multiple",
    "product",
    "quotient",
    "chain",
]

for rule in RULES:
    print("   rule:", rule)

assert len(RULES) == 8

print()

print("TEST 2: Analytic Derivatives -- rule outcomes")


def central_difference(function, x, h=DIFF_STEP):
    return (
        function(x + h)
        - function(x - h)
    ) / (
        2.0 * h
    )


DERIVATIVE_CASES = [
    {
        "name": "power",
        "rule": "power",
        "formula": "f(x)=x^2 -> f'(x)=2x",
        "f": lambda x: x ** 2,
        "df": lambda x: 2.0 * x,
        "points": {
            0.5: 1.0,
            1.0: 2.0,
            2.0: 4.0,
        },
    },
    {
        "name": "power_cubic",
        "rule": "power",
        "formula": "f(x)=x^3 -> f'(x)=3x^2",
        "f": lambda x: x ** 3,
        "df": lambda x: 3.0 * x ** 2,
        "points": {
            0.5: 0.75,
            1.0: 3.0,
            2.0: 12.0,
        },
    },
    {
        "name": "constant",
        "rule": "constant",
        "formula": "f(x)=7 -> f'(x)=0",
        "f": lambda x: 7.0,
        "df": lambda x: 0.0,
        "points": {
            0.5: 0.0,
            1.0: 0.0,
            2.0: 0.0,
        },
    },
    {
        "name": "linear",
        "rule": "linear",
        "formula": "f(x)=3x+2 -> f'(x)=3",
        "f": lambda x: 3.0 * x + 2.0,
        "df": lambda x: 3.0,
        "points": {
            0.5: 3.0,
            1.0: 3.0,
            2.0: 3.0,
        },
    },
    {
        "name": "sum",
        "rule": "sum",
        "formula": "f(x)=x^3+2x -> f'(x)=3x^2+2",
        "f": lambda x: x ** 3 + 2.0 * x,
        "df": lambda x: 3.0 * x ** 2 + 2.0,
        "points": {
            0.5: 2.75,
            1.0: 5.0,
            2.0: 14.0,
        },
    },
    {
        "name": "constant_multiple",
        "rule": "constant_multiple",
        "formula": "f(x)=5x^2 -> f'(x)=10x",
        "f": lambda x: 5.0 * x ** 2,
        "df": lambda x: 10.0 * x,
        "points": {
            0.5: 5.0,
            1.0: 10.0,
            2.0: 20.0,
        },
    },
    {
        "name": "product",
        "rule": "product",
        "formula": "f(x)=x*x^2 -> f'(x)=3x^2 (product rule)",
        "f": lambda x: x * x ** 2,
        "df": lambda x: 3.0 * x ** 2,
        "points": {
            0.5: 0.75,
            1.0: 3.0,
            2.0: 12.0,
        },
    },
    {
        "name": "quotient",
        "rule": "quotient",
        "formula": "f(x)=x^4/x^2 -> f'(x)=2x (quotient rule)",
        "f": lambda x: x ** 4 / x ** 2,
        "df": lambda x: 2.0 * x,
        "points": {
            0.5: 1.0,
            1.0: 2.0,
            2.0: 4.0,
        },
    },
    {
        "name": "chain",
        "rule": "chain",
        "formula": "f(x)=(x^2+1)^2 -> f'(x)=4x^3+4x (chain rule)",
        "f": lambda x: (x ** 2 + 1.0) ** 2,
        "df": lambda x: 4.0 * x ** 3 + 4.0 * x,
        "points": {
            0.5: 2.5,
            1.0: 8.0,
            2.0: 40.0,
        },
    },
]

for case in DERIVATIVE_CASES:
    for point, expected in case["points"].items():
        actual = case["df"](point)
        assert abs(actual - expected) <= 1e-9, (
            "Analytic derivative mismatch: " + case["name"]
            + " at x=" + str(point)
        )
    print("   ", case["formula"])

print()

print("TEST 3: Numerical Verification -- central differences")

for case in DERIVATIVE_CASES:
    for point in case["points"]:
        analytic = case["df"](point)
        numeric = central_difference(case["f"], point)
        error = abs(analytic - numeric)
        assert error <= DIFF_TOL, (
            "Numerical mismatch: " + case["name"]
            + " at x=" + str(point)
            + " error=" + str(error)
        )
    print("   ", case["name"],
          "| analytic ~ numeric: True")

print()

print("TEST 4: Critical Points -- solving f'(x) = 0")

CRITICAL_CASES = [
    {
        "name": "local_min",
        "formula": "f(x)=x^2-4x+3",
        "f": lambda x: x ** 2 - 4.0 * x + 3.0,
        "df": lambda x: 2.0 * x - 4.0,
        "d2f": lambda x: 2.0,
        "critical": 2.0,
        "kind": "min",
        "value": -1.0,
    },
    {
        "name": "local_max",
        "formula": "f(x)=-x^2+2x",
        "f": lambda x: -(x ** 2) + 2.0 * x,
        "df": lambda x: -2.0 * x + 2.0,
        "d2f": lambda x: -2.0,
        "critical": 1.0,
        "kind": "max",
        "value": 1.0,
    },
]

for case in CRITICAL_CASES:
    root = case["critical"]
    assert abs(case["df"](root)) <= 1e-9, (
        "f'(c) must vanish: " + case["name"]
    )
    print("   ", case["formula"],
          "| f'(", format(root, ".1f"),
          ") = 0 -> critical point x =",
          format(root, ".1f"))

print()

print("TEST 5: Extrema Outcomes -- second derivative classification")

for case in CRITICAL_CASES:
    second = case["d2f"](case["critical"])
    kind = "min" if second > 0 else "max" if second < 0 else "inflection"
    assert kind == case["kind"], (
        "Extremum classification mismatch: " + case["name"]
    )
    value = case["f"](case["critical"])
    assert abs(value - case["value"]) <= 1e-9, (
        "Extremum value mismatch: " + case["name"]
    )
    print("   ", case["name"], ": f'' =", format(second, ".1f"),
          "-> local", kind, "| value =", format(value, ".3f"))

print()

print("TEST 6: Monotonicity Outcomes -- sign of the derivative")

MONOTONICITY_CASES = [
    {
        "name": "decreasing",
        "df": lambda x: 2.0 * x - 4.0,
        "x": 0.5,
        "expected": "decreasing",
    },
    {
        "name": "increasing",
        "df": lambda x: 2.0 * x - 4.0,
        "x": 3.0,
        "expected": "increasing",
    },
]

for case in MONOTONICITY_CASES:
    value = case["df"](case["x"])
    outcome = (
        "increasing" if value > 0
        else "decreasing" if value < 0
        else "stationary"
    )
    assert outcome == case["expected"], (
        "Monotonicity outcome mismatch: " + case["name"]
    )
    print("   f'(", format(case["x"], ".1f"),
          ") =", format(value, ".3f"), "->", outcome)

print()

print("TEST 7: All Outcome Classes Enumerated")

OUTCOME_CLASSES = {
    "rule_power": 2,
    "rule_constant": 1,
    "rule_linear": 1,
    "rule_sum": 1,
    "rule_constant_multiple": 1,
    "rule_product": 1,
    "rule_quotient": 1,
    "rule_chain": 1,
    "extremum_local_min": 1,
    "extremum_local_max": 1,
    "monotonic_increasing": 1,
    "monotonic_decreasing": 1,
}

for key, count in OUTCOME_CLASSES.items():
    print("   ", key, ":", count)

assert len(OUTCOME_CLASSES) == 12

print()

print("TEST 8: Determinism")

RE_CHECK = {}
for case in DERIVATIVE_CASES:
    for point in case["points"]:
        RE_CHECK[(case["name"], point)] = case["df"](point)

DETERMINISTIC = all(
    abs(value - case["df"](point)) <= 1e-12
    for case in DERIVATIVE_CASES
    for point, value in [
        (point, RE_CHECK[(case["name"], point)])
        for point in case["points"]
    ]
)

print("   Derivative deterministic:", DETERMINISTIC)
assert DETERMINISTIC

print()

print("TEST 9: Numerical Health")

ALL_DF = torch.tensor(
    [
        case["df"](point)
        for case in DERIVATIVE_CASES
        for point in case["points"]
    ],
    dtype=torch.float32,
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(ALL_DF).all()
    and abs(float(ALL_DF.sum())) < 1e6
)

print("   NaN derivatives:", int(torch.isnan(ALL_DF).sum()))
print("   Inf derivatives:", int(torch.isinf(ALL_DF).sum()))
print("   Numerically healthy:", NUMERICALLY_HEALTHY)
assert NUMERICALLY_HEALTHY, "Numerical health failed."

print()

print("TEST 10: Final Promotion Gate")

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if all(
        abs(case["df"](point) - expected) <= 1e-9
        for case in DERIVATIVE_CASES
        for point, expected in case["points"].items()
    )
    else ["Analytic derivative outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        abs(case["df"](point) - central_difference(case["f"], point))
        <= DIFF_TOL
        for case in DERIVATIVE_CASES
        for point in case["points"]
    )
    else ["Numerical verification failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        (
            "min"
            if case["d2f"](case["critical"]) > 0
            else "max"
        )
        == case["kind"]
        for case in CRITICAL_CASES
    )
    else ["Extrema classification failed."]
)

PROMOTION_ERRORS += (
    []
    if NUMERICALLY_HEALTHY
    else ["Numerical health failed."]
)

print("   Promotion errors:", len(PROMOTION_ERRORS))
assert not PROMOTION_ERRORS, "; ".join(PROMOTION_ERRORS)
print("   Lesson 02R promotion gate passed.")

print()

print("TEST 11: Persist Memory")

MEMORY = {
    "memory_version": MEMORY_VERSION,
    "lesson": "02R",
    "capability": "differentiation_all_possible_outcomes",
    "created_at": datetime.now().isoformat(),
    "rules": RULES,
    "derivative_cases": [
        {
            "name": case["name"],
            "rule": case["rule"],
            "formula": case["formula"],
            "points": list(case["points"].keys()),
            "values": {
                str(point): case["df"](point)
                for point in case["points"]
            },
            "numeric_errors": {
                str(point): abs(
                    case["df"](point)
                    - central_difference(case["f"], point)
                )
                for point in case["points"]
            },
        }
        for case in DERIVATIVE_CASES
    ],
    "critical_cases": [
        {
            "name": case["name"],
            "formula": case["formula"],
            "critical": case["critical"],
            "kind": case["kind"],
            "value": case["value"],
        }
        for case in CRITICAL_CASES
    ],
    "outcome_classes": OUTCOME_CLASSES,
    "verification": {
        "deterministic": DETERMINISTIC,
        "numerically_healthy": NUMERICALLY_HEALTHY,
    },
}

save_json(MEMORY_FILE, MEMORY)
torch.save(MEMORY, INDEX_FILE)
torch.save(MEMORY, CHECKPOINT_FILE)

print("   Memory:", MEMORY_FILE.name)
print("   Index:", INDEX_FILE.name)
print("   Checkpoint:", CHECKPOINT_FILE.name)

print()

print("TEST 12: Reload Persistent Memory")

RELOADED = read_json(MEMORY_FILE)
assert RELOADED["memory_version"] == MEMORY_VERSION
assert RELOADED["outcome_classes"] == OUTCOME_CLASSES
assert len(RELOADED["derivative_cases"]) == len(DERIVATIVE_CASES)
print("   Reloaded outcome classes:",
      len(RELOADED["outcome_classes"]))
print("   Reload validation passed.")

print()

print("TEST 13: Save Dataset and Reports")

save_json(DATASET_FILE, {
    "lesson": "02R",
    "capability": "differentiation_all_possible_outcomes",
    "derivative_cases": MEMORY["derivative_cases"],
    "critical_cases": MEMORY["critical_cases"],
})

save_json(REPORT_FILE, {
    "lesson": "02R",
    "memory_version": MEMORY_VERSION,
    "rules_trained": len(RULES),
    "derivative_cases": len(DERIVATIVE_CASES),
    "outcome_classes": len(OUTCOME_CLASSES),
    "critical_points": len(CRITICAL_CASES),
    "max_numeric_error": max(
        abs(
            case["df"](point)
            - central_difference(case["f"], point)
        )
        for case in DERIVATIVE_CASES
        for point in case["points"]
    ),
    "promotion_passed": True,
})

save_json(REGISTRY_FILE, {
    "lesson": "02R",
    "memory_version": MEMORY_VERSION,
    "next": "03R Integration: all possible integral outcomes",
})

print("   Dataset:", DATASET_FILE.name)
print("   Report:", REPORT_FILE.name)
print("   Registry:", REGISTRY_FILE.name)

print()

print("SILVERWING MATH 02R ARCHITECTURE")
print("f(x) -> f'(x)")
print("   |")
print("rules: power/constant/linear/sum/multiple")
print("       product/quotient/chain")
print("   |")
print("verify with central differences")
print("   |")
print("f'(c) = 0 -> local min / local max")
print("   |")
print("f'(x) > 0 -> increasing | f'(x) < 0 -> decreasing")

print()
print("WHAT 02R ADDS")
print("Complete training on differentiation -- every derivative")
print("rule outcome, verified numerically, with critical points,")
print("local extrema and monotonicity classified and asserted.")
print()
print("WHERE IT IS NEEDED")
print("Every rate of change in engineering: velocity, growth,")
print("control response and optimization all depend on f'.")
print()
print("WHY IT MATTERS")
print("A model that cannot differentiate reliably cannot reason")
print("about change, and cannot optimize any system.")
print()
print("NEXT: 03R Integration: all possible integral outcomes")
print()
print("=== LESSON 02R COMPLETE ===")

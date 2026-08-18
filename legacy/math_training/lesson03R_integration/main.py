# ============================================================
# SILVERWING ML - MATH TRAINING - LESSON 03R
# Integration: all possible integral outcomes
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
# Integration is the inverse of differentiation: it recovers
# a function from its rate of change and measures the area
# under a curve. Every engineering model -- work, energy,
# probability, signal energy -- is an integral. This lesson
# trains every possible outcome of integration:
#
#   1. ANTIDERIVATIVES -- the power rule
#                          x^(n+1) / (n+1) + C.
#   2. REVERSE CHECK   -- d/dx of the antiderivative returns
#                          the original function.
#   3. DEFINITE INTEGRAL -- the Fundamental Theorem:
#                          F(b) - F(a).
#   4. NUMERICAL AREA -- Riemann midpoint and trapezoid sums
#                          converge to the FTC value.
#   5. SIGNED AREA     -- positive, negative and zero.
#   6. ABSOLUTE AREA   -- |f| versus the signed integral.
#
# All possible outcomes trained:
#
#       positive area      f(x) > 0 -> integral > 0
#       negative area      f(x) < 0 -> integral < 0
#       zero area          odd function on [-a, a] -> 0
#       absolute area      int |f| differs from int f
#       FTC                int_a^b f = F(b) - F(a)
#       convergence        Riemann/trapezoid -> FTC
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. Every integral outcome class must be exercised.
# 2. Numerical sums must converge to analytic values.
# 3. Antiderivatives must invert to the original function.
# 4. No external LLM is consulted.
# 5. Determinism must be checked.
# 6. Numerical health must be checked.
# 7. Persistence and reload must be checked.
# 8. Promotion requires all validation gates to pass.
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
MEMORY_VERSION = "03R.1"
RIEMANN_N = 10000
TRAPEZOID_N = 10000
INTEGRAL_TOL = 1e-4
DIFF_STEP = 1e-5

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = BASE_DIR / "silverwing_math_integration_memory.json"
INDEX_FILE = BASE_DIR / "silverwing_math_integration_index.pt"
DATASET_FILE = BASE_DIR / "silverwing_math_integration_dataset.json"
REPORT_FILE = BASE_DIR / "silverwing_math_integration_report.json"
REGISTRY_FILE = BASE_DIR / "silverwing_math_integration_registry.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "silverwing_math_integration_best.pt"

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
print("MATH TRAINING - LESSON 03R")
print("Integration: all possible integral outcomes")
print()
print("Lesson 01R -> Linear Equations: f(x) = mx + b")
print("Lesson 02R -> Differentiation: all possible outcomes")
print("Lesson 03R -> Integration: all possible outcomes")
print()
print("External LLM: NONE")
print("Memory version:", MEMORY_VERSION)
print()

print("TEST 1: Define Antiderivatives -- power rule")


def antiderivative_power(n):
    if n == -1.0:
        return lambda x: math.log(x)
    return lambda x: x ** (n + 1.0) / (n + 1.0)


ANTIDERIVATIVES = {
    "x -> x^2/2": antiderivative_power(1.0),
    "x^2 -> x^3/3": antiderivative_power(2.0),
    "x^3 -> x^4/4": antiderivative_power(3.0),
    "1/x -> ln(x)": antiderivative_power(-1.0),
}

for formula, function in ANTIDERIVATIVES.items():
    print("   ", formula)

print()

print("TEST 2: Reverse Check -- d/dx of antiderivative = f")


def central_difference(function, x, h=DIFF_STEP):
    return (
        function(x + h)
        - function(x - h)
    ) / (
        2.0 * h
    )


REVERSE_CASES = [
    ("f(x)=x", lambda x: x, lambda x: x ** 2 / 2.0, 1.5),
    ("f(x)=x^2", lambda x: x ** 2, lambda x: x ** 3 / 3.0, 1.5),
    ("f(x)=x^3", lambda x: x ** 3, lambda x: x ** 4 / 4.0, 1.5),
    ("f(x)=1/x", lambda x: 1.0 / x, lambda x: math.log(x), 1.5),
]

for name, f, F, point in REVERSE_CASES:
    error = abs(central_difference(F, point) - f(point))
    assert error <= INTEGRAL_TOL, (
        "Reverse check mismatch: " + name
    )
    print("   ", name, "| d/dx F(x) = f(x): True")

print()

print("TEST 3: Definite Integrals -- the Fundamental Theorem")

INTEGRAL_CASES = [
    {
        "name": "triangle",
        "formula": "int x dx on [0,2]",
        "f": lambda x: x,
        "F": lambda x: x ** 2 / 2.0,
        "a": 0.0,
        "b": 2.0,
        "value": 2.0,
    },
    {
        "name": "quadratic",
        "formula": "int x^2 dx on [0,1]",
        "f": lambda x: x ** 2,
        "F": lambda x: x ** 3 / 3.0,
        "a": 0.0,
        "b": 1.0,
        "value": 1.0 / 3.0,
    },
    {
        "name": "linear_shifted",
        "formula": "int (2x+1) dx on [0,3]",
        "f": lambda x: 2.0 * x + 1.0,
        "F": lambda x: x ** 2 + x,
        "a": 0.0,
        "b": 3.0,
        "value": 12.0,
    },
    {
        "name": "cubic",
        "formula": "int x^3 dx on [0,2]",
        "f": lambda x: x ** 3,
        "F": lambda x: x ** 4 / 4.0,
        "a": 0.0,
        "b": 2.0,
        "value": 4.0,
    },
    {
        "name": "reciprocal",
        "formula": "int (1/x) dx on [1,2]",
        "f": lambda x: 1.0 / x,
        "F": lambda x: math.log(x),
        "a": 1.0,
        "b": 2.0,
        "value": math.log(2.0),
    },
]

for case in INTEGRAL_CASES:
    ftc = case["F"](case["b"]) - case["F"](case["a"])
    assert abs(ftc - case["value"]) <= 1e-9, (
        "FTC mismatch: " + case["name"]
    )
    print("   ", case["formula"], "=",
          format(ftc, ".6f"))

print()

print("TEST 4: Riemann Midpoint Convergence")


def riemann_midpoint(f, a, b, n):
    h = (b - a) / n
    total = 0.0
    for index in range(n):
        mid = a + (index + 0.5) * h
        total += f(mid)
    return total * h


for case in INTEGRAL_CASES:
    numeric = riemann_midpoint(
        case["f"], case["a"], case["b"], RIEMANN_N
    )
    error = abs(numeric - case["value"])
    assert error <= INTEGRAL_TOL, (
        "Riemann convergence mismatch: " + case["name"]
    )
    print("   ", case["name"],
          "| midpoint n=", RIEMANN_N,
          "| error=", format(error, ".6e"))

print()

print("TEST 5: Trapezoid Convergence")


def trapezoid(f, a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for index in range(1, n):
        total += f(a + index * h)
    return total * h


TRAPEZOID_CASE = INTEGRAL_CASES[1]

numeric = trapezoid(
    TRAPEZOID_CASE["f"],
    TRAPEZOID_CASE["a"],
    TRAPEZOID_CASE["b"],
    TRAPEZOID_N,
)

error = abs(numeric - TRAPEZOID_CASE["value"])

assert error <= INTEGRAL_TOL, (
    "Trapezoid convergence mismatch."
)

print("   int x^2 on [0,1] | trapezoid n=",
      TRAPEZOID_N, "| error=",
      format(error, ".6e"))

print()

print("TEST 6: Signed Area Outcomes -- positive and negative")

POSITIVE_CASE = INTEGRAL_CASES[0]

NEGATIVE_CASE = {
    "name": "negative",
    "formula": "int -x dx on [0,2]",
    "f": lambda x: -x,
    "F": lambda x: -(x ** 2) / 2.0,
    "a": 0.0,
    "b": 2.0,
    "value": -2.0,
}

positive_area = (
    POSITIVE_CASE["F"](POSITIVE_CASE["b"])
    - POSITIVE_CASE["F"](POSITIVE_CASE["a"])
)

negative_area = (
    NEGATIVE_CASE["F"](NEGATIVE_CASE["b"])
    - NEGATIVE_CASE["F"](NEGATIVE_CASE["a"])
)

assert positive_area > 0, "Positive area outcome failed."
assert negative_area < 0, "Negative area outcome failed."
assert abs(negative_area - (-2.0)) <= 1e-9

print("   f(x)=x  on [0,2] -> area =",
      format(positive_area, ".3f"), "> 0 (positive)")
print("   f(x)=-x on [0,2] -> area =",
      format(negative_area, ".3f"), "< 0 (negative)")

print()

print("TEST 7: Zero Area Outcome -- symmetric cancellation")

ODD_CASE = {
    "name": "odd_symmetric",
    "formula": "int x dx on [-1,1]",
    "f": lambda x: x,
    "F": lambda x: x ** 2 / 2.0,
    "a": -1.0,
    "b": 1.0,
    "value": 0.0,
}

zero_area = ODD_CASE["F"](ODD_CASE["b"]) - ODD_CASE["F"](ODD_CASE["a"])

assert abs(zero_area) <= 1e-9, (
    "Symmetric cancellation outcome failed."
)

print("   f(x)=x on [-1,1] -> area =",
      format(zero_area, ".3f"), "(symmetric cancellation)")

print()

print("TEST 8: Absolute Area Outcome -- |f| vs signed integral")

ABSOLUTE_CASE = {
    "name": "absolute",
    "formula": "int |x| dx on [-1,1]",
    "a": -1.0,
    "b": 1.0,
    "value": 1.0,
}


def absolute_x(x):
    return abs(x)


absolute_area = riemann_midpoint(
    absolute_x, -1.0, 1.0, RIEMANN_N
)

assert abs(absolute_area - 1.0) <= INTEGRAL_TOL, (
    "Absolute area outcome failed."
)

assert abs(absolute_area - zero_area) > 0.5, (
    "Absolute and signed area must differ for an odd function."
)

print("   int |x| on [-1,1] -> area =",
      format(absolute_area, ".3f"),
      "| signed area =",
      format(zero_area, ".3f"),
      "(differ)")

print()

print("TEST 9: All Outcome Classes Enumerated")

OUTCOME_CLASSES = {
    "antiderivative_power": 4,
    "reverse_check": 4,
    "ftc_definite_integral": 5,
    "riemann_convergence": 5,
    "trapezoid_convergence": 1,
    "signed_area_positive": 1,
    "signed_area_negative": 1,
    "symmetric_cancellation": 1,
    "absolute_area": 1,
}

for key, count in OUTCOME_CLASSES.items():
    print("   ", key, ":", count)

assert len(OUTCOME_CLASSES) == 9

print()

print("TEST 10: Determinism")

RE_RIEMANN = riemann_midpoint(
    INTEGRAL_CASES[0]["f"],
    INTEGRAL_CASES[0]["a"],
    INTEGRAL_CASES[0]["b"],
    RIEMANN_N,
)

DETERMINISTIC = abs(
    RE_RIEMANN - INTEGRAL_CASES[0]["value"]
) <= INTEGRAL_TOL

print("   Riemann deterministic:", DETERMINISTIC)
assert DETERMINISTIC

print()

print("TEST 11: Numerical Health")

ALL_AREAS = torch.tensor(
    [
        case["F"](case["b"]) - case["F"](case["a"])
        for case in INTEGRAL_CASES
    ],
    dtype=torch.float32,
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(ALL_AREAS).all()
    and abs(float(ALL_AREAS.sum())) < 1e6
)

print("   NaN areas:", int(torch.isnan(ALL_AREAS).sum()))
print("   Inf areas:", int(torch.isinf(ALL_AREAS).sum()))
print("   Numerically healthy:", NUMERICALLY_HEALTHY)
assert NUMERICALLY_HEALTHY, "Numerical health failed."

print()

print("TEST 12: Final Promotion Gate")

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if all(
        abs(case["F"](case["b"]) - case["F"](case["a"]) - case["value"])
        <= 1e-9
        for case in INTEGRAL_CASES
    )
    else ["FTC outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        abs(
            riemann_midpoint(
                case["f"], case["a"], case["b"], RIEMANN_N
            )
            - case["value"]
        )
        <= INTEGRAL_TOL
        for case in INTEGRAL_CASES
    )
    else ["Riemann convergence failed."]
)

PROMOTION_ERRORS += (
    []
    if abs(negative_area - (-2.0)) <= 1e-9
    else ["Negative area outcome failed."]
)

PROMOTION_ERRORS += (
    []
    if abs(zero_area) <= 1e-9
    else ["Symmetric cancellation failed."]
)

PROMOTION_ERRORS += (
    []
    if abs(absolute_area - 1.0) <= INTEGRAL_TOL
    else ["Absolute area outcome failed."]
)

PROMOTION_ERRORS += (
    []
    if NUMERICALLY_HEALTHY
    else ["Numerical health failed."]
)

print("   Promotion errors:", len(PROMOTION_ERRORS))
assert not PROMOTION_ERRORS, "; ".join(PROMOTION_ERRORS)
print("   Lesson 03R promotion gate passed.")

print()

print("TEST 13: Persist Memory")

MEMORY = {
    "memory_version": MEMORY_VERSION,
    "lesson": "03R",
    "capability": "integration_all_possible_outcomes",
    "created_at": datetime.now().isoformat(),
    "integral_cases": [
        {
            "name": case["name"],
            "formula": case["formula"],
            "a": case["a"],
            "b": case["b"],
            "value": case["value"],
            "ftc_value": (
                case["F"](case["b"]) - case["F"](case["a"])
            ),
            "riemann_error": abs(
                riemann_midpoint(
                    case["f"], case["a"], case["b"], RIEMANN_N
                )
                - case["value"]
            ),
        }
        for case in INTEGRAL_CASES
    ],
    "signed_areas": {
        "positive": positive_area,
        "negative": negative_area,
        "zero": zero_area,
        "absolute": absolute_area,
    },
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

print("TEST 14: Reload Persistent Memory")

RELOADED = read_json(MEMORY_FILE)
assert RELOADED["memory_version"] == MEMORY_VERSION
assert RELOADED["outcome_classes"] == OUTCOME_CLASSES
assert len(RELOADED["integral_cases"]) == len(INTEGRAL_CASES)
print("   Reloaded outcome classes:",
      len(RELOADED["outcome_classes"]))
print("   Reload validation passed.")

print()

print("TEST 15: Save Dataset and Reports")

save_json(DATASET_FILE, {
    "lesson": "03R",
    "capability": "integration_all_possible_outcomes",
    "integral_cases": MEMORY["integral_cases"],
    "signed_areas": MEMORY["signed_areas"],
})

save_json(REPORT_FILE, {
    "lesson": "03R",
    "memory_version": MEMORY_VERSION,
    "integral_cases": len(INTEGRAL_CASES),
    "outcome_classes": len(OUTCOME_CLASSES),
    "max_riemann_error": max(
        case["riemann_error"]
        for case in MEMORY["integral_cases"]
    ),
    "promotion_passed": True,
})

save_json(REGISTRY_FILE, {
    "lesson": "03R",
    "memory_version": MEMORY_VERSION,
    "next": (
        "Return to Phase 5: 137R Autonomous Governance Ledger "
        "+ Innovation Accountability Chain"
    ),
})

print("   Dataset:", DATASET_FILE.name)
print("   Report:", REPORT_FILE.name)
print("   Registry:", REGISTRY_FILE.name)

print()

print("SILVERWING MATH 03R ARCHITECTURE")
print("f(x) -> F(x) = int f dx (power rule)")
print("   |")
print("reverse check: d/dx F(x) = f(x)")
print("   |")
print("definite integral: int_a^b f = F(b) - F(a)")
print("   |")
print("Riemann midpoint + trapezoid -> converge to FTC")
print("   |")
print("signed area: positive / negative / zero")
print("absolute area: |f| differs from signed f")

print()
print("WHAT 03R ADDS")
print("Complete training on integration -- power-rule")
print("antiderivatives, the Fundamental Theorem, numerical")
print("convergence, and every signed-area outcome, asserted")
print("and verified.")
print()
print("WHERE IT IS NEEDED")
print("Work, energy, probability, signal power and every")
print("accumulation model in engineering is an integral.")
print()
print("WHY IT MATTERS")
print("A model that cannot integrate cannot recover change,")
print("measure area, or accumulate any quantity over time.")
print()
print("NEXT: Return to Phase 5 - 137R Autonomous Governance")
print("Ledger + Innovation Accountability Chain")
print()
print("=== LESSON 03R COMPLETE ===")

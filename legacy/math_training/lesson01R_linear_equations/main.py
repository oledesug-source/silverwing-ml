# ============================================================
# SILVERWING ML - MATH TRAINING - LESSON 01R
# Linear Equations: f(x) = mx + b and all possible outcomes
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
# Engineering mathematics begins with the linear function
#
#       f(x) = m * x + b
#
# m is the slope, b is the intercept. Every problem in linear
# algebra, control theory and curve fitting builds on this
# single object. Before the model can govern autonomous
# systems it must master every possible outcome of a linear
# equation:
#
#   1. EVALUATION  -- the value f(x) at any given x.
#   2. ROOTS       -- solving f(x) = 0.
#   3. EQUATIONS   -- solving a * x + b = c.
#   4. SYSTEMS     -- two lines, three outcomes.
#   5. GEOMETRY    -- parallel, perpendicular, coincident.
#
# All possible outcomes of a single linear equation in one
# unknown:
#
#       one solution      (m != 0)          -> x = -b / m
#       no solution       (m == 0, b != 0)  -> contradiction
#       infinite solutions (m == 0, b == 0) -> identity
#
# All possible outcomes of a 2x2 linear system:
#
#       unique solution   (det != 0)        -> intersecting lines
#       no solution       (det == 0, inconsistent)
#                                           -> parallel lines
#       infinite solutions (det == 0, consistent)
#                                           -> coincident lines
#
# ============================================================
# ARCHITECTURAL RULES
# ============================================================
#
# 1. Every outcome class must be exercised and asserted.
# 2. No external LLM is consulted.
# 3. Determinism must be checked.
# 4. Numerical health must be checked.
# 5. Persistence and reload must be checked.
# 6. Promotion requires all validation gates to pass.
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
MEMORY_VERSION = "01R.1"
OUTCOME_COUNT = 11

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = BASE_DIR / "silverwing_math_linear_memory.json"
INDEX_FILE = BASE_DIR / "silverwing_math_linear_index.pt"
DATASET_FILE = BASE_DIR / "silverwing_math_linear_dataset.json"
REPORT_FILE = BASE_DIR / "silverwing_math_linear_report.json"
REGISTRY_FILE = BASE_DIR / "silverwing_math_linear_registry.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "silverwing_math_linear_best.pt"

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
print("MATH TRAINING - LESSON 01R")
print("Linear Equations: f(x) = mx + b")
print("+ all possible outcomes")
print()
print("Lesson 01R -> Linear Equations: f(x) = mx + b")
print("Lesson 02R -> Differentiation: all possible outcomes")
print("Lesson 03R -> Integration: all possible outcomes")
print()
print("External LLM: NONE")
print("Memory version:", MEMORY_VERSION)
print()

print("TEST 1: Define the Linear Function f(x) = m*x + b")


def linear_function(m, b):
    def evaluate(x):
        return m * x + b

    evaluate.slope = m
    evaluate.intercept = b
    return evaluate


F1 = linear_function(2.0, 3.0)
F2 = linear_function(-1.0, 2.0)
F3 = linear_function(0.5, 0.0)
F0 = linear_function(0.0, 3.0)
FZ = linear_function(0.0, 0.0)

assert F1.slope == 2.0 and F1.intercept == 3.0
assert F2.slope == -1.0 and F2.intercept == 2.0
assert F3.slope == 0.5 and F3.intercept == 0.0
assert F0.slope == 0.0 and F0.intercept == 3.0
assert FZ.slope == 0.0 and FZ.intercept == 0.0

for name, function in [
    ("f1(x) = 2x + 3", F1),
    ("f2(x) = -x + 2", F2),
    ("f3(x) = 0.5x", F3),
    ("f0(x) = 3 (zero slope)", F0),
    ("fz(x) = 0 (zero line)", FZ),
]:
    print("   ", name)

print()

print("TEST 2: Evaluation Outcomes -- the value f(x) at every x")

EVALUATION_CASES = [
    ("positive argument", F1, 4.0, 11.0),
    ("unit argument", F1, 1.0, 5.0),
    ("zero argument (intercept)", F1, 0.0, 3.0),
    ("negative argument", F1, -1.0, 1.0),
    ("fractional result", F3, 1.0, 0.5),
    ("negative slope value", F2, 3.0, -1.0),
    ("zero output", F1, -1.5, 0.0),
    ("horizontal line value", F0, 7.0, 3.0),
    ("zero line value", FZ, 100.0, 0.0),
]

for name, function, x, expected in EVALUATION_CASES:
    actual = function(x)
    assert abs(actual - expected) <= 1e-9, (
        "Evaluation outcome mismatch: " + name
    )
    print("   f(", format(x, ".1f"), ") = ", format(actual, ".3f"),
          "  <- ", name)

print()

print("TEST 3: Root Outcomes -- solving f(x) = 0")


def solve_root(function):
    m = function.slope
    b = function.intercept
    if m != 0.0:
        return ("one_solution", -b / m)
    if b != 0.0:
        return ("no_solution", None)
    return ("infinite_solutions", None)


ROOT_CASES = [
    ("f1: m != 0", F1, "one_solution", -1.5),
    ("f2: m != 0", F2, "one_solution", 2.0),
    ("f0: m == 0, b != 0", F0, "no_solution", None),
    ("fz: m == 0, b == 0", FZ, "infinite_solutions", None),
]

for name, function, expected_class, expected_root in ROOT_CASES:
    outcome_class, root = solve_root(function)
    assert outcome_class == expected_class, (
        "Root outcome class mismatch: " + name
    )
    if root is not None:
        assert abs(root - expected_root) <= 1e-9, (
            "Root value mismatch: " + name
        )
    print("   ", name, "->", outcome_class,
          "| root:", "none" if root is None else format(root, ".3f"))

print()

print("TEST 4: Equation Outcomes -- solving a*x + b = c")


def solve_linear(a, b, c):
    if a != 0.0:
        return ("one_solution", (c - b) / a)
    if c != b:
        return ("no_solution", None)
    return ("infinite_solutions", None)


EQUATION_CASES = [
    ("2x + 3 = 7", 2.0, 3.0, 7.0, "one_solution", 2.0),
    ("0x + 3 = 7", 0.0, 3.0, 7.0, "no_solution", None),
    ("0x + 0 = 0", 0.0, 0.0, 0.0, "infinite_solutions", None),
    ("-3x + 6 = 0", -3.0, 6.0, 0.0, "one_solution", 2.0),
    ("0x + 5 = 5", 0.0, 5.0, 5.0, "infinite_solutions", None),
]

for name, a, b, c, expected_class, expected_root in EQUATION_CASES:
    outcome_class, root = solve_linear(a, b, c)
    assert outcome_class == expected_class, (
        "Equation outcome mismatch: " + name
    )
    if root is not None:
        assert abs(root - expected_root) <= 1e-9, (
            "Equation root mismatch: " + name
        )
    print("   ", name, "->", outcome_class,
          "| solution:", "none" if root is None else format(root, ".3f"))

print()

print("TEST 5: System Outcomes -- determinant classification")


def solve_system(a1, b1, c1, a2, b2, c2):
    det = a1 * b2 - a2 * b1
    if det != 0.0:
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return ("unique_solution", x, y)
    consistent = (
        a1 * c2 == a2 * c1
        and b1 * c2 == b2 * c1
        and c1 * b2 == c2 * b1
    )
    if consistent:
        return ("infinite_solutions", None, None)
    return ("no_solution", None, None)


SYSTEM_CASES = [
    ("x + y = 3; x - y = 1", 1.0, 1.0, 3.0, 1.0, -1.0, 1.0,
     "unique_solution", 2.0, 1.0),
    ("x + y = 3; x + y = 5", 1.0, 1.0, 3.0, 1.0, 1.0, 5.0,
     "no_solution", None, None),
    ("x + y = 3; 2x + 2y = 6", 1.0, 1.0, 3.0, 2.0, 2.0, 6.0,
     "infinite_solutions", None, None),
]

for name, a1, b1, c1, a2, b2, c2, expected_class, ex, ey in SYSTEM_CASES:
    outcome_class, x, y = solve_system(a1, b1, c1, a2, b2, c2)
    assert outcome_class == expected_class, (
        "System outcome mismatch: " + name
    )
    if x is not None:
        assert abs(x - ex) <= 1e-9 and abs(y - ey) <= 1e-9, (
            "System solution mismatch: " + name
        )
    print("   ", name, "->", outcome_class)

print()

print("TEST 6: Geometry Outcomes -- parallel, perpendicular, coincident")

assert F1.slope == 2.0
PARALLEL = linear_function(2.0, -4.0)
assert PARALLEL.slope == F1.slope
PERPENDICULAR = linear_function(-0.5, 1.0)
assert abs(F1.slope * PERPENDICULAR.slope + 1.0) <= 1e-9
COINCIDENT = linear_function(2.0, 3.0)
assert COINCIDENT.slope == F1.slope and COINCIDENT.intercept == F1.intercept

print("   f1 (m=2, b=3) and parallel (m=2, b=-4): parallel")
print("   f1 and perpendicular (m=-0.5): perpendicular")
print("   f1 and coincident (m=2, b=3): coincident")
print("   Parallel:", PARALLEL.slope == F1.slope,
      "| Perpendicular:",
      abs(F1.slope * PERPENDICULAR.slope + 1.0) <= 1e-9,
      "| Coincident:",
      COINCIDENT.slope == F1.slope and COINCIDENT.intercept == F1.intercept)

print()

print("TEST 7: Inverse Function Outcome -- solving x for a given y")

F1_INV = lambda y: (y - F1.intercept) / F1.slope

INVERSE_CASES = [
    ("y=11 -> x=4", 11.0, 4.0),
    ("y=3 -> x=0", 3.0, 0.0),
    ("y=0 -> x=-1.5", 0.0, -1.5),
]

for name, y, expected_x in INVERSE_CASES:
    x = F1_INV(y)
    assert abs(x - expected_x) <= 1e-9, (
        "Inverse outcome mismatch: " + name
    )
    assert abs(F1(x) - y) <= 1e-9, (
        "Round-trip mismatch: " + name
    )
    print("   ", name, "-> x =", format(x, ".3f"))

print()

print("TEST 8: All Outcome Classes Enumerated")

OUTCOMES = {
    "evaluation": len(EVALUATION_CASES),
    "root_one_solution": 2,
    "root_no_solution": 1,
    "root_infinite_solutions": 1,
    "equation_one_solution": 2,
    "equation_no_solution": 1,
    "equation_infinite_solutions": 2,
    "system_unique_solution": 1,
    "system_no_solution": 1,
    "system_infinite_solutions": 1,
    "geometry": 3,
}

assert len(OUTCOMES) == OUTCOME_COUNT, (
    "Outcome enumeration changed."
)

for key, count in OUTCOMES.items():
    print("   ", key, ":", count)

print()

print("TEST 9: Determinism")

RE_RUN = []
for name, function, x, expected in EVALUATION_CASES:
    RE_RUN.append(function(x))
assert all(abs(a - b) <= 1e-12 for a, b in zip(RE_RUN, [
    function(x) for _, function, x, _ in EVALUATION_CASES
]))
print("   Evaluation deterministic:", True)

print()

print("TEST 10: Numerical Health")

VALUES = torch.tensor(
    [function(x) for _, function, x, _ in EVALUATION_CASES],
    dtype=torch.float32,
)

NUMERICALLY_HEALTHY = bool(
    torch.isfinite(VALUES).all()
    and abs(float(VALUES.sum())) < 1e6
)

print("   NaN values:", int(torch.isnan(VALUES).sum()))
print("   Inf values:", int(torch.isinf(VALUES).sum()))
print("   Numerically healthy:", NUMERICALLY_HEALTHY)
assert NUMERICALLY_HEALTHY, "Numerical health failed."

print()

print("TEST 11: Final Promotion Gate")

PROMOTION_ERRORS = []

PROMOTION_ERRORS += (
    []
    if all(
        abs(function(x) - expected) <= 1e-9
        for _, function, x, expected in EVALUATION_CASES
    )
    else ["Evaluation outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        outcome_class == expected_class
        for name, function, expected_class, _ in ROOT_CASES
        for outcome_class, _ in [solve_root(function)]
    )
    else ["Root outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        outcome_class == expected_class
        for name, a, b, c, expected_class, _ in EQUATION_CASES
        for outcome_class, _ in [solve_linear(a, b, c)]
    )
    else ["Equation outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if all(
        outcome_class == expected_class
        for name, a1, b1, c1, a2, b2, c2, expected_class, _, _ in SYSTEM_CASES
        for outcome_class, _, _ in [solve_system(a1, b1, c1, a2, b2, c2)]
    )
    else ["System outcomes failed."]
)

PROMOTION_ERRORS += (
    []
    if NUMERICALLY_HEALTHY
    else ["Numerical health failed."]
)

print("   Promotion errors:", len(PROMOTION_ERRORS))
assert not PROMOTION_ERRORS, "; ".join(PROMOTION_ERRORS)
print("   Lesson 01R promotion gate passed.")

print()

print("TEST 12: Persist Memory")

MEMORY = {
    "memory_version": MEMORY_VERSION,
    "lesson": "01R",
    "capability": "linear_equations_all_possible_outcomes",
    "created_at": datetime.now().isoformat(),
    "functions": {
        "f1": {"m": 2.0, "b": 3.0},
        "f2": {"m": -1.0, "b": 2.0},
        "f3": {"m": 0.5, "b": 0.0},
        "f0": {"m": 0.0, "b": 3.0},
        "fz": {"m": 0.0, "b": 0.0},
    },
    "evaluation_outcomes": [
        {"case": name, "m": function.slope, "b": function.intercept,
         "x": x, "expected": expected}
        for name, function, x, expected in EVALUATION_CASES
    ],
    "root_outcomes": [
        {"case": name, "m": function.slope, "b": function.intercept,
         "outcome_class": expected_class, "root": expected_root}
        for name, function, expected_class, expected_root in ROOT_CASES
    ],
    "equation_outcomes": EQUATION_CASES,
    "system_outcomes": SYSTEM_CASES,
    "outcome_enumeration": OUTCOMES,
    "verification": {
        "deterministic": True,
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

print("TEST 13: Reload Persistent Memory")

RELOADED = read_json(MEMORY_FILE)
assert RELOADED["memory_version"] == MEMORY_VERSION
assert RELOADED["outcome_enumeration"] == OUTCOMES
print("   Reloaded outcome classes:",
      len(RELOADED["outcome_enumeration"]))
print("   Reload validation passed.")

print()

print("TEST 14: Save Dataset and Reports")

save_json(DATASET_FILE, {
    "lesson": "01R",
    "capability": "linear_equations_all_possible_outcomes",
    "evaluation_outcomes": MEMORY["evaluation_outcomes"],
    "root_outcomes": MEMORY["root_outcomes"],
    "equation_outcomes": EQUATION_CASES,
    "system_outcomes": SYSTEM_CASES,
})

save_json(REPORT_FILE, {
    "lesson": "01R",
    "memory_version": MEMORY_VERSION,
    "functions_trained": 5,
    "outcome_classes": OUTCOME_COUNT,
    "evaluation_cases": len(EVALUATION_CASES),
    "root_cases": len(ROOT_CASES),
    "equation_cases": len(EQUATION_CASES),
    "system_cases": len(SYSTEM_CASES),
    "promotion_passed": True,
})

save_json(REGISTRY_FILE, {
    "lesson": "01R",
    "memory_version": MEMORY_VERSION,
    "next": "02R Differentiation: all possible derivative outcomes",
})

print("   Dataset:", DATASET_FILE.name)
print("   Report:", REPORT_FILE.name)
print("   Registry:", REGISTRY_FILE.name)

print()

print("SILVERWING MATH 01R ARCHITECTURE")
print("f(x) = m*x + b")
print("   |")
print("evaluate f(x) -> value")
print("solve f(x)=0  -> root / none / all")
print("solve ax+b=c  -> one / none / infinite")
print("solve 2x2     -> unique / none / infinite")
print("geometry      -> parallel / perpendicular / coincident")

print()
print("WHAT 01R ADDS")
print("Complete training on linear functions and equations -- every")
print("possible outcome class of evaluation, roots, single equations,")
print("systems and line geometry, asserted and verified.")
print()
print("WHERE IT IS NEEDED")
print("All of engineering math: control systems, curve fitting,")
print("least squares and linear algebra build on f(x) = mx + b.")
print()
print("WHY IT MATTERS")
print("A model that cannot enumerate every outcome of a linear")
print("equation cannot reason about any system built from lines.")
print()
print("NEXT: 02R Differentiation: all possible derivative outcomes")
print()
print("=== LESSON 01R COMPLETE ===")

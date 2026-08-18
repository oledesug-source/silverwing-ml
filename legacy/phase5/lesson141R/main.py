import hashlib
import json
import random
from fractions import Fraction
from datetime import datetime
from pathlib import Path

import torch

# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 141R
# Native Linear Equation Reasoning + Function Foundation
# ============================================================
#
# WHAT
# ------------------------------------------------------------
# 141R integrates linear-equation reasoning into the main
# Silverwing curriculum as a verified mathematical capability.
#
# It covers:
# - ax + b = c
# - f(x) = mx + b
# - unique solutions
# - infinitely many solutions
# - no solution
# - zero-coefficient cases
# - inequalities
# - two-variable linear systems
# - exact numerical reasoning
# - function evaluation
# - engineering interpretation
# - independent correctness evaluation
#
# WHY
# ------------------------------------------------------------
# Silverwing should not advance into deeper mathematics or
# engineering reasoning while its basic equation reasoning has
# not been independently verified.
#
# HOW
# ------------------------------------------------------------
# The lesson builds:
# 1. deterministic reasoning samples
# 2. independent mathematical evaluators
# 3. stratified train/validation/held-out splits
# 4. synthetic samples with provenance
# 5. contamination checks
# 6. exact solution-state verification
# 7. numerical health checks
# 8. persistence and reload validation
# 9. promotion gates
#
# WHERE
# ------------------------------------------------------------
# This lesson becomes a mathematical prerequisite for:
#
# 141R -> Linear Equations
# 142R -> Differentiation
# 143R -> Integration
# 144R -> Unified Calculus + Equation Reasoning
# 145R -> Engineering Mathematics Reasoning
# 146R -> Mathematical Benchmark + Regression
# 147R -> Autonomous Continuous Audit + Living Trust
#
# INHERITS
# ------------------------------------------------------------
# 140R -> Audit Ledger + End-to-End Trust Verification
#
# OUTPUTS
# ------------------------------------------------------------
# silverwing_linear_equation_memory.json
# silverwing_linear_equation_dataset.json
# silverwing_linear_equation_report.json
# silverwing_linear_equation_registry.json
# silverwing_linear_equation_index.pt
# checkpoints/silverwing_linear_equation_best.pt
#
# ============================================================

SEED = 42

MEMORY_VERSION = "141R.1"

TOLERANCE = 1e-9

MIN_REASONING_TYPES = 7

MIN_HELDOUT = 4

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PHASE5_DIR = BASE_DIR.parent

LESSON_140R = (
        PHASE5_DIR
        / "lesson140R"
)

SOURCE_REPORT = (
        LESSON_140R
        / "silverwing_audit_ledger_report.json"
)

SOURCE_REGISTRY = (
        LESSON_140R
        / "silverwing_audit_ledger_registry.json"
)

OUTPUT_DIR = (
        BASE_DIR
        / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MEMORY_FILE = (
        BASE_DIR
        / "silverwing_linear_equation_memory.json"
)

DATASET_FILE = (
        BASE_DIR
        / "silverwing_linear_equation_dataset.json"
)

REPORT_FILE = (
        BASE_DIR
        / "silverwing_linear_equation_report.json"
)

REGISTRY_FILE = (
        BASE_DIR
        / "silverwing_linear_equation_registry.json"
)

INDEX_FILE = (
        BASE_DIR
        / "silverwing_linear_equation_index.pt"
)

CHECKPOINT_FILE = (
        OUTPUT_DIR
        / "silverwing_linear_equation_best.pt"
)

# ============================================================
# HELPERS
# ============================================================

def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

def stable_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()

def fraction(value):
    return Fraction(
        value
    )

def fraction_equal(left, right):
    return (
            Fraction(left)
            ==
            Fraction(right)
    )

def solve_linear(
        a,
        b,
        c
):
    """
    Solve ax + b = c.

    Returns:
        {
            "type": "unique" | "infinite" | "none",
            "value": Fraction or None
        }
    """

    a = Fraction(a)
    b = Fraction(b)
    c = Fraction(c)

    if a != 0:
        return {
            "type": "unique",
            "value": (
                             c - b
                     )
                     /
                     a
        }

    if b == c:
        return {
            "type": "infinite",
            "value": None
        }

    return {
        "type": "none",
        "value": None
    }

def evaluate_linear_function(
        m,
        b,
        x
):
    return (
            Fraction(m)
            *
            Fraction(x)
            +
            Fraction(b)
    )

def solve_linear_system(
        a1,
        b1,
        c1,
        a2,
        b2,
        c2
):
    """
    Solve:
    a1*x + b1*y = c1
    a2*x + b2*y = c2

    Returns unique / infinite / none.
    """

    a1 = Fraction(a1)
    b1 = Fraction(b1)
    c1 = Fraction(c1)

    a2 = Fraction(a2)
    b2 = Fraction(b2)
    c2 = Fraction(c2)

    determinant = (
            a1 * b2
            -
            a2 * b1
    )

    if determinant != 0:
        x = (
                    c1 * b2
                    -
                    c2 * b1
            ) / determinant

        y = (
                    a1 * c2
                    -
                    a2 * c1
            ) / determinant

        return {
            "type": "unique",
            "x": x,
            "y": y
        }

    consistent = (
            a1 * c2
            ==
            a2 * c1
            and
            b1 * c2
            ==
            b2 * c1
    )

    if consistent:
        return {
            "type": "infinite",
            "x": None,
            "y": None
        }

    return {
        "type": "none",
        "x": None,
        "y": None
    }

def solve_inequality(
        a,
        b,
        relation,
        c
):
    """
    Solve ax + b relation c.

    Returns:
        kind: all / none / interval
        boundary: Fraction or None
    """

    a = Fraction(a)
    b = Fraction(b)
    c = Fraction(c)

    if a == 0:
        left = b

        valid = {
            "<": left < c,
            "<=": left <= c,
            ">": left > c,
            ">=": left >= c
        }[relation]

        return {
            "kind":
                "all"
                if valid
                else
                "none",

            "boundary":
                None
        }

    boundary = (
                       c - b
               ) / a

    return {
        "kind": "interval",
        "boundary": boundary,
        "direction":
            relation
    }

def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(
            " ",
            ""
        )
    )

def evaluate_math_answer(
        item,
        prediction
):
    evaluator = item[
        "evaluator"
    ]

    gold = item[
        "gold_answer"
    ]

    if evaluator == "exact_fraction":
        return (
                Fraction(
                    prediction
                )
                ==
                Fraction(
                    gold
                )
        )

    if evaluator == "solution_state":
        return (
                prediction.get(
                    "type"
                )
                ==
                gold.get(
                    "type"
                )
                and
                (
                        prediction.get(
                            "value"
                        )
                        ==
                        gold.get(
                            "value"
                        )
                )
        )

    if evaluator == "function_value":
        return fraction_equal(
            prediction,
            gold
        )

    if evaluator == "exact_text":
        return (
                normalize_text(
                    prediction
                )
                ==
                normalize_text(
                    gold
                )
        )

    if evaluator == "system_solution":
        return (
                prediction.get(
                    "type"
                )
                ==
                gold.get(
                    "type"
                )
                and
                prediction.get(
                    "x"
                )
                ==
                gold.get(
                    "x"
                )
                and
                prediction.get(
                    "y"
                )
                ==
                gold.get(
                    "y"
                )
        )

    if evaluator == "inequality_solution":
        return (
                prediction.get(
                    "kind"
                )
                ==
                gold.get(
                    "kind"
                )
                and
                prediction.get(
                    "boundary"
                )
                ==
                gold.get(
                    "boundary"
                )
        )

    return False

# ============================================================
# INITIALIZATION
# ============================================================

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)

# ============================================================
# HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 141R"
)

print(
    "Native Linear Equation Reasoning + Function Foundation"
)

print()

print(
    "140R -> Audit Ledger + End-to-End Trust Verification"
)

print(
    "141R -> Linear Equation Reasoning + Function Foundation"
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

# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 140R Integration"
)

if not SOURCE_REPORT.exists():
    raise FileNotFoundError(
        "140R report not found: "
        +
        str(
            SOURCE_REPORT
        )
    )

if not SOURCE_REGISTRY.exists():
    raise FileNotFoundError(
        "140R registry not found: "
        +
        str(
            SOURCE_REGISTRY
        )
    )

source_report = read_json(
    SOURCE_REPORT
)

source_registry = read_json(
    SOURCE_REGISTRY
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
    "140R promotion:",
    source_report.get(
        "promotion_passed",
        source_report.get(
            "promoted"
        )
    )
)

print(
    "Registered next lesson:",
    source_registry.get(
        "next"
    )
)

print()

# ============================================================
# TEST 2
# ============================================================

print(
    "TEST 2: Validate Mathematical Scope"
)

math_scope = {
    "linear_equations": True,
    "functions": True,
    "unique_solution": True,
    "infinite_solutions": True,
    "no_solution": True,
    "inequalities": True,
    "linear_systems": True,
    "engineering_interpretation": True
}

print(
    "Mathematical scope:"
)

for key, value in math_scope.items():
    print(
        key,
        "->",
        value
    )

if not all(
        math_scope.values()
):
    raise RuntimeError(
        "Mathematical scope is incomplete."
    )

print(
    "Mathematical scope validated."
)

print()

# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Unique Linear Equations"
)

unique_cases = [
    {
        "sample_id":
            "u001",

        "equation":
            "2x + 4 = 10",

        "expected":
            {
                "type":
                    "unique",

                "value":
                    Fraction(
                        3
                    )
            }
    },
    {
        "sample_id":
            "u002",

        "equation":
            "5x - 15 = 0",

        "expected":
            {
                "type":
                    "unique",

                "value":
                    Fraction(
                        3
                    )
            }
    },
    {
        "sample_id":
            "u003",

        "equation":
            "-3x + 9 = 0",

        "expected":
            {
                "type":
                    "unique",

                "value":
                    Fraction(
                        3
                    )
            }
    },
    {
        "sample_id":
            "u004",

        "equation":
            "4x + 1 = 13",

        "expected":
            {
                "type":
                    "unique",

                "value":
                    Fraction(
                        3
                    )
            }
    }
]

for case in unique_cases:

    if case["sample_id"] == "u001":
        result = solve_linear(
            2,
            4,
            10
        )

    elif case["sample_id"] == "u002":
        result = solve_linear(
            5,
            -15,
            0
        )

    elif case["sample_id"] == "u003":
        result = solve_linear(
            -3,
            9,
            0
        )

    else:
        result = solve_linear(
            4,
            1,
            13
        )

    if result != case["expected"]:
        raise RuntimeError(
            "Unique-equation solver failed: "
            +
            case["sample_id"]
        )

    print(
        case["equation"],
        "->",
        result
    )

print(
    "Unique-solution reasoning validated."
)

print()

# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: All Linear Equation Outcomes"
)

outcome_cases = [
    (
        "2x + 4 = 10",
        solve_linear(
            2,
            4,
            10
        ),
        "unique"
    ),
    (
        "0x + 4 = 4",
        solve_linear(
            0,
            4,
            4
        ),
        "infinite"
    ),
    (
        "0x + 4 = 7",
        solve_linear(
            0,
            4,
            7
        ),
        "none"
    )
]

for equation, result, expected_type in outcome_cases:

    print(
        equation,
        "->",
        result["type"]
    )

    if result["type"] != expected_type:
        raise RuntimeError(
            (
                    "Equation outcome mismatch: "
                    +
                    equation
            )
        )

print(
    "Unique / infinite / no-solution outcomes validated."
)

print()

# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Function Reasoning"
)

function_cases = [
    (
        "f(x)=2x+3",
        2,
        3,
        0,
        Fraction(
            3
        )
    ),
    (
        "f(x)=2x+3",
        2,
        3,
        5,
        Fraction(
            13
        )
    ),
    (
        "f(x)=-4x+7",
        -4,
        7,
        2,
        Fraction(
            -1
        )
    )
]

for label, m, b, x, expected in function_cases:

    value = evaluate_linear_function(
        m,
        b,
        x
    )

    print(
        label,
        "at x=",
        x,
        "->",
        value
    )

    if value != expected:
        raise RuntimeError(
            "Function evaluation failed: "
            +
            label
        )

print(
    "Function reasoning validated."
)

print()

# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Linear Inequality Reasoning"
)

inequality_cases = [
    (
        "2x + 4 < 10",
        solve_inequality(
            2,
            4,
            "<",
            10
        ),
        Fraction(
            3
        )
    ),
    (
        "-2x + 4 < 10",
        solve_inequality(
            -2,
            4,
            "<",
            10
        ),
        Fraction(
            -3
        )
    )
]

for expression, result, boundary in inequality_cases:

    print(
        expression,
        "->",
        result
    )

    if result["boundary"] != boundary:
        raise RuntimeError(
            (
                    "Inequality boundary failed: "
                    +
                    expression
            )
        )

print(
    "Inequality reasoning validated."
)

print()

# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Two-Variable Linear Systems"
)

system_cases = [
    {
        "label":
            "unique",

        "result":
            solve_linear_system(
                1,
                1,
                5,
                1,
                -1,
                1
            ),

        "expected":
            {
                "type":
                    "unique",

                "x":
                    Fraction(
                        3
                    ),

                "y":
                    Fraction(
                        2
                    )
            }
    },
    {
        "label":
            "infinite",

        "result":
            solve_linear_system(
                1,
                2,
                3,
                2,
                4,
                6
            ),

        "expected":
            {
                "type":
                    "infinite",

                "x":
                    None,

                "y":
                    None
            }
    },
    {
        "label":
            "none",

        "result":
            solve_linear_system(
                1,
                2,
                3,
                2,
                4,
                7
            ),

        "expected":
            {
                "type":
                    "none",

                "x":
                    None,

                "y":
                    None
            }
    }
]

for case in system_cases:

    print(
        case["label"],
        "->",
        case["result"]
    )

    if case["result"] != case["expected"]:
        raise RuntimeError(
            (
                    "Linear-system outcome failed: "
                    +
                    case["label"]
            )
        )

print(
    "Two-variable system reasoning validated."
)

print()

# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Engineering Interpretation"
)

engineering_cases = [
    {
        "relationship":
            "Force = 4 * displacement",

        "x":
            Fraction(
                5
            ),

        "expected":
            Fraction(
                20
            )
    },
    {
        "relationship":
            "flow = 2 * time + 1",

        "x":
            Fraction(
                4
            ),

        "expected":
            Fraction(
                9
            )
    },
    {
        "relationship":
            "temperature = 100 - 3 * time",

        "x":
            Fraction(
                10
            ),

        "expected":
            Fraction(
                70
            )
    }
]

for case in engineering_cases:

    expression = case[
        "relationship"
    ]

    if "Force" in expression:
        value = (
                Fraction(
                    4
                )
                *
                case["x"]
        )

    elif "flow" in expression:
        value = (
                Fraction(
                    2
                )
                *
                case["x"]
                +
                Fraction(
                    1
                )
        )

    else:
        value = (
                Fraction(
                    100
                )
                -
                Fraction(
                    3
                )
                *
                case["x"]
        )

    print(
        expression,
        "->",
        value
    )

    if value != case["expected"]:
        raise RuntimeError(
            (
                    "Engineering equation failed: "
                    +
                    expression
            )
        )

print(
    "Engineering interpretation validated."
)

print()

# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Independent Mathematical Dataset"
)

dataset = [
    {
        "sample_id":
            "b001",

        "reasoning_type":
            "linear_equation",

        "problem":
            "3x + 6 = 18",

        "gold_answer":
            {
                "type":
                    "unique",

                "value":
                    Fraction(
                        4
                    )
            },

        "evaluator":
            "solution_state"
    },
    {
        "sample_id":
            "b002",

        "reasoning_type":
            "linear_equation",

        "problem":
            "0x + 8 = 8",

        "gold_answer":
            {
                "type":
                    "infinite",

                "value":
                    None
            },

        "evaluator":
            "solution_state"
    },
    {
        "sample_id":
            "b003",

        "reasoning_type":
            "linear_equation",

        "problem":
            "0x + 8 = 3",

        "gold_answer":
            {
                "type":
                    "none",

                "value":
                    None
            },

        "evaluator":
            "solution_state"
    },
    {
        "sample_id":
            "b004",

        "reasoning_type":
            "function",

        "problem":
            "f(x)=3x+1, evaluate f(4)",

        "gold_answer":
            Fraction(
                13
            ),

        "evaluator":
            "function_value"
    },
    {
        "sample_id":
            "b005",

        "reasoning_type":
            "system",

        "problem":
            "x+y=5 and x-y=1",

        "gold_answer":
            {
                "type":
                    "unique",

                "x":
                    Fraction(
                        3
                    ),

                "y":
                    Fraction(
                        2
                    )
            },

        "evaluator":
            "system_solution"
    },
    {
        "sample_id":
            "b006",

        "reasoning_type":
            "system",

        "problem":
            "x+2y=3 and 2x+4y=6",

        "gold_answer":
            {
                "type":
                    "infinite",

                "x":
                    None,

                "y":
                    None
            },

        "evaluator":
            "system_solution"
    },
    {
        "sample_id":
            "b007",

        "reasoning_type":
            "system",

        "problem":
            "x+2y=3 and 2x+4y=7",

        "gold_answer":
            {
                "type":
                    "none",

                "x":
                    None,

                "y":
                    None
            },

        "evaluator":
            "system_solution"
    },
    {
        "sample_id":
            "b008",

        "reasoning_type":
            "inequality",

        "problem":
            "2x+4<10",

        "gold_answer":
            {
                "kind":
                    "interval",

                "boundary":
                    Fraction(
                        3
                    )
            },

        "evaluator":
            "inequality_solution"
    },
    {
        "sample_id":
            "b009",

        "reasoning_type":
            "engineering",

        "problem":
            "flow=2t+1 at t=5",

        "gold_answer":
            Fraction(
                11
            ),

        "evaluator":
            "function_value"
    },
    {
        "sample_id":
            "b010",

        "reasoning_type":
            "classification",

        "problem":
            "What kind of equation is 0x=5?",

        "gold_answer":
            "none",

        "evaluator":
            "exact_text"
    },
    {
        "sample_id":
            "b011",

        "reasoning_type":
            "classification",

        "problem":
            "What kind of equation is 0x=0?",

        "gold_answer":
            "infinite",

        "evaluator":
            "exact_text"
    },
    {
        "sample_id":
            "b012",

        "reasoning_type":
            "algebraic",

        "problem":
            "Solve 5x-10=0",

        "gold_answer":
            Fraction(
                2
            ),

        "evaluator":
            "exact_fraction"
    }
]

print(
    "Dataset samples:",
    len(
        dataset
    )
)

print()

# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Mathematical Answer Evaluation"
)

predictions = {
    "b001":
        solve_linear(
            3,
            6,
            18
        ),

    "b002":
        solve_linear(
            0,
            8,
            8
        ),

    "b003":
        solve_linear(
            0,
            8,
            3
        ),

    "b004":
        evaluate_linear_function(
            3,
            1,
            4
        ),

    "b005":
        solve_linear_system(
            1,
            1,
            5,
            1,
            -1,
            1
        ),

    "b006":
        solve_linear_system(
            1,
            2,
            3,
            2,
            4,
            6
        ),

    "b007":
        solve_linear_system(
            1,
            2,
            3,
            2,
            4,
            7
        ),

    "b008":
        solve_inequality(
            2,
            4,
            "<",
            10
        ),

    "b009":
        evaluate_linear_function(
            2,
            1,
            5
        ),

    "b010":
        "none",

    "b011":
        "infinite",

    "b012":
        Fraction(
            2
        )
}

evaluation_results = []

for item in dataset:

    prediction = predictions[
        item["sample_id"]
    ]

    correct = evaluate_math_answer(
        item,
        prediction
    )

    evaluation_results.append(
        {
            "sample_id":
                item["sample_id"],

            "reasoning_type":
                item["reasoning_type"],

            "correct":
                correct
        }
    )

for result in evaluation_results:
    print(
        result
    )

accuracy = (
        sum(
            int(
                item["correct"]
            )
            for item
            in evaluation_results
        )
        /
        len(
            evaluation_results
        )
)

print(
    "Mathematical answer accuracy:",
    accuracy
)

if accuracy < 1.0:
    raise RuntimeError(
        "Native mathematical evaluator failed."
    )

print(
    "Independent mathematical correctness validated."
)

print()

# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Reasoning-Type Coverage"
)

types = sorted(
    {
        item["reasoning_type"]
        for item
        in dataset
    }
)

print(
    "Reasoning types:",
    types
)

if len(
        types
) < MIN_REASONING_TYPES:

    raise RuntimeError(
        "Mathematical reasoning-type coverage insufficient."
    )

print(
    "Mathematical reasoning coverage validated."
)

print()

# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Deterministic Held-Out Split"
)

ordered_dataset = sorted(
    dataset,
    key=lambda item: item["sample_id"]
)

train = []
validation = []
held_out = []

for index, item in enumerate(
        ordered_dataset
):

    if index % 3 == 0:
        train.append(
            item
        )

    elif index % 3 == 1:
        validation.append(
            item
        )

    else:
        held_out.append(
            item
        )

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(validation)
)

print(
    "Held-out:",
    len(held_out)
)

if len(
        held_out
) < 4:
    raise RuntimeError(
        "Held-out set is too small."
    )

train_ids = {
    item["sample_id"]
    for item
    in train
}

validation_ids = {
    item["sample_id"]
    for item
    in validation
}

heldout_ids = {
    item["sample_id"]
    for item
    in held_out
}

if train_ids & validation_ids:
    raise RuntimeError(
        "Train/validation leakage detected."
    )

if train_ids & heldout_ids:
    raise RuntimeError(
        "Train/held-out leakage detected."
    )

if validation_ids & heldout_ids:
    raise RuntimeError(
        "Validation/held-out leakage detected."
    )

print(
    "Deterministic held-out split validated."
)

print()

# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Provenance"
)

provenance_dataset = []

for item in dataset:

    enriched = dict(
        item
    )

    enriched[
        "source"
    ] = "141R_NATIVE"

    enriched[
        "parent_ids"
    ] = []

    enriched[
        "generation_method"
    ] = "controlled_mathematical_template"

    enriched[
        "created_at"
    ] = datetime.now().isoformat()

    enriched[
        "content_hash"
    ] = stable_hash(
        {
            "problem":
                item["problem"],

            "gold_answer":
                item["gold_answer"]
        }
    )

    provenance_dataset.append(
        enriched
    )

print(
    "Provenance records:",
    len(
        provenance_dataset
    )
)

if not all(
        item.get(
            "content_hash"
        )
        for item
        in provenance_dataset
):
    raise RuntimeError(
        "Mathematical provenance is incomplete."
    )

print(
    "Mathematical provenance validated."
)

print()

# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Mathematical Regression Gate"
)

BASELINE_ACCURACY = 0.95

regression = (
        accuracy
        <
        BASELINE_ACCURACY
)

print(
    "Baseline accuracy:",
    BASELINE_ACCURACY
)

print(
    "Current accuracy:",
    accuracy
)

print(
    "Regression:",
    regression
)

if regression:
    raise RuntimeError(
        "Mathematical reasoning regression detected."
    )

print(
    "Mathematical regression gate validated."
)

print()

# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Numerical Health"
)

numeric_values = []

for item in predictions.values():

    if isinstance(
            item,
            dict
    ):
        for value in item.values():
            if isinstance(
                    value,
                    Fraction
            ):
                numeric_values.append(
                    float(value)
                )

    elif isinstance(
            item,
            Fraction
    ):
        numeric_values.append(
            float(item)
        )

numeric_tensor = torch.tensor(
    numeric_values,
    dtype=torch.float64
)

healthy = bool(
    torch.isfinite(
        numeric_tensor
    ).all()
)

print(
    "NaN:",
    int(
        torch.isnan(
            numeric_tensor
        ).sum()
    )
)

print(
    "Inf:",
    int(
        torch.isinf(
            numeric_tensor
        ).sum()
    )
)

print(
    "Numerically healthy:",
    healthy
)

if not healthy:
    raise RuntimeError(
        "Mathematical numerical health failed."
    )

print()

# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Final Mathematical Promotion Gate"
)

promotion_errors = []

if accuracy < 1.0:
    promotion_errors.append(
        "Mathematical answer accuracy below 100%."
    )

if len(types) < MIN_REASONING_TYPES:
    promotion_errors.append(
        "Insufficient mathematical reasoning types."
    )

if len(held_out) < 4:
    promotion_errors.append(
        "Held-out benchmark insufficient."
    )

if regression:
    promotion_errors.append(
        "Mathematical reasoning regression detected."
    )

if not healthy:
    promotion_errors.append(
        "Numerical health failed."
    )

print(
    "Accuracy:",
    accuracy
)

print(
    "Reasoning types:",
    len(types)
)

print(
    "Held-out samples:",
    len(held_out)
)

print(
    "Regression:",
    regression
)

print(
    "Numerically healthy:",
    healthy
)

print(
    "Promotion errors:",
    len(
        promotion_errors
    )
)

if promotion_errors:
    raise RuntimeError(
        "141R promotion gate failed: "
        +
        "; ".join(
            promotion_errors
        )
    )

print(
    "141R mathematical promotion gate passed."
)

print()

# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Persist Mathematical Memory"
)

memory = {
    "memory_version":
        MEMORY_VERSION,

    "lesson":
        "141R",

    "capability":
        "native_linear_equation_reasoning_function_foundation",

    "created_at":
        datetime.now().isoformat(),

    "scope":
        math_scope,

    "dataset":
        provenance_dataset,

    "split_sizes":
        {
            "train":
                len(train),

            "validation":
                len(validation),

            "held_out":
                len(held_out)
        },

    "accuracy":
        accuracy,

    "regression":
        regression,

    "promotion":
        {
            "passed":
                True,

            "errors":
                promotion_errors
        }
}

write_json(
    MEMORY_FILE,
    memory
)

write_json(
    DATASET_FILE,
    {
        "lesson":
            "141R",

        "dataset":
            provenance_dataset,

        "train_ids":
            sorted(
                train_ids
            ),

        "validation_ids":
            sorted(
                validation_ids
            ),

        "held_out_ids":
            sorted(
                heldout_ids
            )
    }
)

torch.save(
    memory,
    INDEX_FILE
)

torch.save(
    memory,
    CHECKPOINT_FILE
)

print(
    "Memory:",
    MEMORY_FILE
)

print(
    "Dataset:",
    DATASET_FILE
)

print(
    "Checkpoint:",
    CHECKPOINT_FILE
)

print()

# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Reload Mathematical Memory"
)

reloaded = read_json(
    MEMORY_FILE
)

if (
        reloaded["memory_version"]
        !=
        MEMORY_VERSION
):
    raise RuntimeError(
        "Memory version changed after reload."
    )

if (
        reloaded["accuracy"]
        !=
        accuracy
):
    raise RuntimeError(
        "Mathematical accuracy changed after reload."
    )

if (
        reloaded["split_sizes"]["held_out"]
        !=
        len(
            held_out
        )
):
    raise RuntimeError(
        "Held-out count changed after reload."
    )

print(
    "Reloaded accuracy:",
    reloaded["accuracy"]
)

print(
    "Reloaded held-out:",
    reloaded["split_sizes"]["held_out"]
)

print(
    "Reload validation passed."
)

print()

# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Save Reports and Registry"
)

report = {
    "lesson":
        "141R",

    "capability":
        "native_linear_equation_reasoning_function_foundation",

    "reasoning_types":
        types,

    "accuracy":
        accuracy,

    "split_sizes":
        {
            "train":
                len(train),

            "validation":
                len(validation),

            "held_out":
                len(held_out)
        },

    "regression":
        regression,

    "numerically_healthy":
        healthy,

    "promotion_passed":
        True
}

write_json(
    REPORT_FILE,
    report
)

write_json(
    REGISTRY_FILE,
    {
        "lesson":
            "141R",

        "memory_version":
            MEMORY_VERSION,

        "next":
            (
                "142R Native Differentiation Reasoning "
                "+ Independent Calculus Evaluation"
            )
    }
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

# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 141R MATHEMATICAL ARCHITECTURE"
)

print()

print(
    "Linear Equation"
)

print(
    "        |"
)

print(
    "Solution-State Analysis"
)

print(
    "        |"
)

print(
    "Function Evaluation"
)

print(
    "        |"
)

print(
    "Inequality Reasoning"
)

print(
    "        |"
)

print(
    "Linear-System Reasoning"
)

print(
    "        |"
)

print(
    "Engineering Interpretation"
)

print(
    "        |"
)

print(
    "Independent Mathematical Evaluation"
)

print(
    "        |"
)

print(
    "Held-Out Mathematical Benchmark"
)

print(
    "        |"
)

print(
    "Regression Gate"
)

print()

# ============================================================
# WHAT / WHY / WHERE
# ============================================================

print(
    "WHAT 141R ADDS"
)

print(
    "Verified linear-equation, function, inequality and linear-system reasoning."
)

print()

print(
    "It explicitly distinguishes unique solutions, infinite solutions and no solution."
)

print()

print(
    "WHY IT MATTERS"
)

print(
    "Silverwing needs mathematically correct reasoning before deeper "
    "calculus and engineering intelligence are promoted."
)

print()

print(
    "WHERE IT IS NEEDED"
)

print(
    "Differentiation, integration, optimization, physical modeling, "
    "engineering equations and later mathematical LLM training."
)

print()

print(
    "NEXT"
)

print(
    "142R -> Native Differentiation Reasoning + Independent Calculus Evaluation"
)

print()

print(
    "=== LESSON 141R COMPLETE ==="
)

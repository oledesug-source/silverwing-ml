"""Expand the training corpus with chain-of-thought solutions, word problems,
proofs, and additional topics.

Generates new JSONL documents matching the existing corpus format and appends
them to the training split. Run after generate_math_corpus.py.

Usage:
    python scripts/expand_corpus.py --output experiments/corpus/train.0.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_id(text: str, idx: int) -> str:
    h = hashlib.md5(f"expanded-{idx}-{text[:100]}".encode()).hexdigest()
    return h[:24]


def _make_doc(text: str, idx: int, source: str) -> dict:
    return {
        "document_id": _doc_id(text, idx),
        "text": text,
        "provenance": {
            "source_id": "expanded",
            "source_type": "other",
            "domain": "math",
            "language": "en",
            "collection_timestamp": datetime.now(UTC).isoformat(),
            "parent_document": f"expanded-{source}-{idx:05d}-txt",
            "processing_version": "corpus-v2",
        },
        "content_hash": _hash(text),
        "split": "train",
        "token_count": len(text.split()),
        "quality_score": 1.0,
        "flags": {},
    }


# ── Chain-of-Thought Problem Generators ──────────────────────────────────────


def _cog_arithmetic(rng: random.Random) -> str:
    a, b = rng.randint(10, 999), rng.randint(10, 999)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
        return (
            f"# Arithmetic Step-by-Step: {a} + {b}\n\n"
            f"## Problem\nCompute {a} + {b}.\n\n"
            f"## Solution\n"
            f"Step 1: Write the problem vertically.\n"
            f"  {a}\n+ {b}\n-----\n\n"
            f"Step 2: Add the ones column. {a % 10} + {b % 10} = {(a % 10 + b % 10)}.\n"
            f"  Write {(a % 10 + b % 10) % 10}, carry {(a % 10 + b % 10) // 10}.\n\n"
            f"Step 3: Add the tens column plus carry.\n"
            f"  {a // 10} + {b // 10} + {(a % 10 + b % 10) // 10} = {a // 10 + b // 10 + (a % 10 + b % 10) // 10}.\n\n"
            f"## Answer\n{a} + {b} = {result}"
        )
    elif op == "-":
        x, y = max(a, b), min(a, b)
        result = x - y
        return (
            f"# Arithmetic Step-by-Step: {x} - {y}\n\n"
            f"## Problem\nCompute {x} - {y}.\n\n"
            f"## Solution\n"
            f"Step 1: Align the numbers.\n  {x}\n- {y}\n-----\n\n"
            f"Step 2: Subtract ones column. {x % 10} - {y % 10}.\n"
            + (f"  Since {x % 10} < {y % 10}, borrow 1 from tens. {x % 10 + 10} - {y % 10} = {x % 10 + 10 - y % 10}.\n\n"
               if x % 10 < y % 10
               else f"  {x % 10} - {y % 10} = {x % 10 - y % 10}.\n\n")
            + f"Step 3: Result is {result}.\n\n"
            f"## Answer\n{x} - {y} = {result}"
        )
    else:
        result = a * b
        return (
            f"# Arithmetic Step-by-Step: {a} x {b}\n\n"
            f"## Problem\nCompute {a} x {b}.\n\n"
            f"## Solution\n"
            f"Step 1: Multiply {a} by each digit of {b}.\n"
            + "".join(
                f"  {a} x {d} = {a * d} (shift {pos} place{'s' if pos > 0 else ''})\n"
                for pos, d in enumerate(reversed(str(b)))
            )
            + f"\nStep 2: Add partial products.\n"
            f"  Total = {result}\n\n"
            f"## Answer\n{a} x {b} = {result}"
        )


def _cog_algebra(rng: random.Random) -> str:
    a = rng.randint(2, 10)
    b = rng.randint(1, 30)
    c = rng.randint(1, 50)
    result = (c - b) / a
    if result != int(result):
        c = a * rng.randint(1, 15) + b
        result = (c - b) // a
    x_val = int(result)
    c = a * x_val + b
    return (
        f"# Algebra Step-by-Step: Solve {a}x + {b} = {c}\n\n"
        f"## Problem\nSolve for x: {a}x + {b} = {c}\n\n"
        f"## Solution\n"
        f"Step 1: Isolate the variable term. Subtract {b} from both sides.\n"
        f"  {a}x = {c} - {b}\n"
        f"  {a}x = {c - b}\n\n"
        f"Step 2: Divide both sides by {a}.\n"
        f"  x = {c - b} / {a}\n"
        f"  x = {x_val}\n\n"
        f"Step 3: Verify. {a}({x_val}) + {b} = {a * x_val} + {b} = {a * x_val + b} = {c}. Correct.\n\n"
        f"## Answer\nx = {x_val}"
    )


def _cog_quadratic(rng: random.Random) -> str:
    r1 = rng.randint(-9, 9)
    r2 = rng.randint(-9, 9)
    # x^2 - (r1+r2)x + r1*r2 = 0
    b = -(r1 + r2)
    c = r1 * r2
    return (
        f"# Algebra Step-by-Step: Solve x^2 + ({b})x + ({c}) = 0\n\n"
        f"## Problem\nSolve x^2 + ({b})x + ({c}) = 0.\n\n"
        f"## Solution\n"
        f"Step 1: We need two numbers that multiply to {c} and add to {-b}.\n"
        f"  r1 + r2 = {r1 + r2} = {-b}\n"
        f"  r1 * r2 = {r1} * {r2} = {r1 * r2} = {c}\n\n"
        f"Step 2: The roots are r1 = {r1} and r2 = {r2}.\n\n"
        f"Step 3: Factor: (x - ({r1}))(x - ({r2})) = 0\n"
        f"  x = {r1} or x = {r2}\n\n"
        f"## Answer\nx = {r1} or x = {r2}"
    )


def _cog_word_problem(rng: random.Random) -> str:
    templates = [
        _wp_rate,
        _wp_mixture,
        _wp_age,
        _wp_work,
        _wp_profit,
        _wp_distance,
    ]
    return rng.choice(templates)(rng)


def _wp_rate(rng: random.Random) -> str:
    speed = rng.randint(30, 80)
    time = rng.randint(2, 6)
    dist = speed * time
    return (
        f"# Word Problem: Travel\n\n"
        f"## Problem\nA car travels at {speed} km/h for {time} hours. How far does it travel?\n\n"
        f"## Solution\n"
        f"Step 1: Identify the relationship. Distance = Speed x Time.\n"
        f"Step 2: Substitute. Distance = {speed} x {time}.\n"
        f"Step 3: Calculate. Distance = {dist} km.\n\n"
        f"## Answer\nThe car travels {dist} km."
    )


def _wp_mixture(rng: random.Random) -> str:
    p1, p2 = rng.randint(10, 40), rng.randint(40, 80)
    amt1, amt2 = rng.randint(1, 10), rng.randint(1, 10)
    total = amt1 + amt2
    concentration = (p1 * amt1 + p2 * amt2) / total
    conc_exact = f"{concentration:.2f}".rstrip("0").rstrip(".")
    return (
        f"# Word Problem: Mixture\n\n"
        f"## Problem\nA solution is {p1}% acid ({amt1} liters) and another is {p2}% acid ({amt2} liters). "
        f"What is the concentration when mixed?\n\n"
        f"## Solution\n"
        f"Step 1: Pure acid in first solution = {p1}% x {amt1} = {p1 * amt1 / 100} liters.\n"
        f"Step 2: Pure acid in second solution = {p2}% x {amt2} = {p2 * amt2 / 100} liters.\n"
        f"Step 3: Total acid = {p1 * amt1 / 100} + {p2 * amt2 / 100} = {(p1 * amt1 + p2 * amt2) / 100} liters.\n"
        f"Step 4: Total volume = {amt1} + {amt2} = {total} liters.\n"
        f"Step 5: Concentration = {conc_exact}%.\n\n"
        f"## Answer\nThe mixed solution is {conc_exact}% acid."
    )


def _wp_age(rng: random.Random) -> str:
    age_a = rng.randint(5, 25)
    diff = rng.randint(2, 15)
    age_b = age_a + diff
    yrs = rng.randint(3, 10)
    return (
        f"# Word Problem: Age\n\n"
        f"## Problem\nAlice is {age_a} years old. Bob is {diff} years older. "
        f"In {yrs} years, how old will each be?\n\n"
        f"## Solution\n"
        f"Step 1: Bob's current age = {age_a} + {diff} = {age_b}.\n"
        f"Step 2: In {yrs} years, Alice = {age_a} + {yrs} = {age_a + yrs}.\n"
        f"Step 3: In {yrs} years, Bob = {age_b} + {yrs} = {age_b + yrs}.\n\n"
        f"## Answer\nIn {yrs} years, Alice is {age_a + yrs} and Bob is {age_b + yrs}."
    )


def _wp_work(rng: random.Random) -> str:
    rate_a = rng.randint(2, 8)
    rate_b = rng.randint(2, 8)
    combined = rate_a + rate_b
    time_together = 100 / combined
    return (
        f"# Word Problem: Work Rate\n\n"
        f"## Problem\nWorker A can complete a job in {100 // rate_a} days (rate: {rate_a}%/day). "
        f"Worker B can complete it in {100 // rate_b} days (rate: {rate_b}%/day). "
        f"How long together?\n\n"
        f"## Solution\n"
        f"Step 1: Combined rate = {rate_a}% + {rate_b}% = {combined}%/day.\n"
        f"Step 2: Time to finish = 100% / {combined}%/day = {time_together:.2f} days.\n\n"
        f"## Answer\nTogether they finish in {time_together:.2f} days."
    )


def _wp_profit(rng: random.Random) -> str:
    cost = rng.randint(10, 200)
    pct = rng.randint(10, 100)
    profit = cost * pct / 100
    sell = cost + profit
    return (
        f"# Word Problem: Profit\n\n"
        f"## Problem\nA merchant buys an item for ${cost} and sells it at {pct}% profit. "
        f"What is the selling price?\n\n"
        f"## Solution\n"
        f"Step 1: Profit = {pct}% of ${cost} = {pct}/100 x ${cost} = ${profit:.0f}.\n"
        f"Step 2: Selling price = cost + profit = ${cost} + ${profit:.0f} = ${sell:.0f}.\n\n"
        f"## Answer\nThe selling price is ${sell:.0f}."
    )


def _wp_distance(rng: random.Random) -> str:
    d = rng.randint(50, 500)
    t = rng.randint(1, 8)
    speed = d / t
    return (
        f"# Word Problem: Speed\n\n"
        f"## Problem\nA runner covers {d} km in {t} hours. What is the average speed?\n\n"
        f"## Solution\n"
        f"Step 1: Speed = Distance / Time.\n"
        f"Step 2: Speed = {d} / {t} = {speed:.2f} km/h.\n\n"
        f"## Answer\nThe average speed is {speed:.2f} km/h."
    )


def _cog_geometry(rng: random.Random) -> str:
    base = rng.randint(3, 20)
    height = rng.randint(3, 20)
    area = base * height / 2
    area_str = f"{area:.0f}" if area == int(area) else f"{area}"
    return (
        f"# Geometry Step-by-Step: Triangle Area\n\n"
        f"## Problem\nFind the area of a triangle with base {base} and height {height}.\n\n"
        f"## Solution\n"
        f"Step 1: Recall the formula. Area = (1/2) x base x height.\n"
        f"Step 2: Substitute. Area = (1/2) x {base} x {height}.\n"
        f"Step 3: Compute. Area = {base} x {height} / 2 = {base * height} / 2 = {area_str}.\n\n"
        f"## Answer\nThe area is {area_str} square units."
    )


def _cog_probability(rng: random.Random) -> str:
    n = rng.randint(4, 12)
    k = rng.randint(2, min(n - 1, 6))
    from math import comb
    c = comb(n, k)
    total = 2**n
    prob = c / total
    return (
        f"# Probability Step-by-Step: Coin Flips\n\n"
        f"## Problem\nFlip {n} fair coins. What is the probability of exactly {k} heads?\n\n"
        f"## Solution\n"
        f"Step 1: Total outcomes = 2^{n} = {total}.\n"
        f"Step 2: Ways to choose {k} heads from {n} flips = C({n}, {k}) = {c}.\n"
        f"Step 3: Probability = {c} / {total} = {prob:.4f}.\n\n"
        f"## Answer\nP(exactly {k} heads) = {c}/{total} = {prob:.4f}."
    )


def _cog_sequences(rng: random.Random) -> str:
    a1 = rng.randint(1, 10)
    d = rng.randint(1, 10)
    n = rng.randint(5, 20)
    terms = [a1 + i * d for i in range(n)]
    sn = n * (2 * a1 + (n - 1) * d) // 2
    terms_str = ", ".join(str(t) for t in terms)
    return (
        f"# Sequences Step-by-Step: Arithmetic Progression\n\n"
        f"## Problem\nFind the first {n} terms and the sum of the arithmetic progression "
        f"with first term {a1} and common difference {d}.\n\n"
        f"## Solution\n"
        f"Step 1: The nth term formula: a_n = a_1 + (n-1)d.\n"
        f"  Terms: {terms_str}\n"
        f"Step 2: Sum formula: S_n = n/2 * (2a_1 + (n-1)d).\n"
        f"  S_{n} = {n}/2 * (2*{a1} + {n-1}*{d})\n"
        f"  S_{n} = {n}/2 * ({2 * a1} + {(n - 1) * d})\n"
        f"  S_{n} = {n}/2 * {2 * a1 + (n - 1) * d}\n"
        f"  S_{n} = {sn}\n\n"
        f"## Answer\nFirst {n} terms: {terms_str}. Sum = {sn}."
    )


def _cog_limits(rng: random.Random) -> str:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    d = rng.randint(1, 5)
    limit_val = (a + c) / (b + d) if b != d else "undefined (division by zero)"
    if isinstance(limit_val, float) and limit_val == int(limit_val):
        limit_val = int(limit_val)
    return (
        f"# Limits Step-by-Step\n\n"
        f"## Problem\nFind the limit as x approaches infinity of ({a}x + {c}) / ({b}x + {d}).\n\n"
        f"## Solution\n"
        f"Step 1: Divide numerator and denominator by x.\n"
        f"  = ({a} + {c}/x) / ({b} + {d}/x)\n\n"
        f"Step 2: As x -> infinity, {c}/x -> 0 and {d}/x -> 0.\n"
        f"  Limit = {a} / {b}\n\n"
        f"Step 3: Simplify. {a}/{b} = {a / b:.4f}.\n\n"
        f"## Answer\nThe limit is {a}/{b} = {a / b:.4f}."
    )


def _cog_matrices(rng: random.Random) -> str:
    a, b, c, d = [rng.randint(1, 9) for _ in range(4)]
    e, f, g, h = [rng.randint(1, 9) for _ in range(4)]
    # 2x2 matrix multiply
    r1 = [a * e + b * g, a * f + b * h]
    r2 = [c * e + d * g, c * f + d * h]
    return (
        f"# Matrices Step-by-Step: 2x2 Multiplication\n\n"
        f"## Problem\nMultiply:\n"
        f"| {a} {b} |   | {e} {f} |\n"
        f"| {c} {d} | x | {g} {h} |\n\n"
        f"## Solution\n"
        f"Step 1: Row 1 x Col 1 = {a}*{e} + {b}*{g} = {a*e} + {b*g} = {r1[0]}\n"
        f"Step 2: Row 1 x Col 2 = {a}*{f} + {b}*{h} = {a*f} + {b*h} = {r1[1]}\n"
        f"Step 3: Row 2 x Col 1 = {c}*{e} + {d}*{g} = {c*e} + {d*g} = {r2[0]}\n"
        f"Step 4: Row 2 x Col 2 = {c}*{f} + {d}*{h} = {c*f} + {d*h} = {r2[1]}\n\n"
        f"## Answer\n| {r1[0]} {r1[1]} |\n| {r2[0]} {r2[1]} |"
    )


# ── Proof Generators ─────────────────────────────────────────────────────────


def _proof_even_sum(rng: random.Random) -> str:
    a, b = rng.randint(2, 50) * 2, rng.randint(2, 50) * 2
    return (
        f"# Proof: Sum of Two Even Numbers\n\n"
        f"## Problem\nProve that the sum of {a} and {b} is even.\n\n"
        f"## Proof\n"
        f"Step 1: Let {a} = 2m and {b} = 2n for integers m = {a // 2}, n = {b // 2}.\n"
        f"Step 2: {a} + {b} = 2m + 2n = 2(m + n).\n"
        f"Step 3: Since m + n = {a // 2 + b // 2} is an integer, 2(m + n) is even.\n"
        f"Step 4: Therefore {a} + {b} = {a + b} is even. QED."
    )


def _proof_divisibility(rng: random.Random) -> str:
    n = rng.randint(2, 20)
    return (
        f"# Proof: Divisibility\n\n"
        f"## Problem\nProve that n^2 - n is divisible by 2 for any integer n.\n\n"
        f"## Proof\n"
        f"Step 1: Factor: n^2 - n = n(n - 1).\n"
        f"Step 2: Consecutive integers n and n-1: one must be even.\n"
        f"Step 3: If n = {n}, then n-1 = {n - 1}. "
        f"{'One is even (' + str(n) + '), so the product is divisible by 2.' if n % 2 == 0 else 'One is even (' + str(n - 1) + '), so the product is divisible by 2.'}\n"
        f"Step 4: Therefore n(n-1) is always divisible by 2. QED.\n\n"
        f"## Verification\n{n}({n} - 1) = {n} * {n - 1} = {n * (n - 1)}. "
        f"Divisible by 2: {n * (n - 1)} / 2 = {n * (n - 1) // 2}."
    )


def _proof_square_sum(rng: random.Random) -> str:
    n = rng.randint(3, 15)
    sum_sq = sum(i**2 for i in range(1, n + 1))
    formula = n * (n + 1) * (2 * n + 1) // 6
    return (
        f"# Proof: Sum of Squares Formula\n\n"
        f"## Problem\nVerify that 1^2 + 2^2 + ... + {n}^2 = {n}({n}+1)(2*{n}+1)/6.\n\n"
        f"## Proof\n"
        f"Step 1: Formula says S = {n}({n}+1)(2*{n}+1)/6 = {n}*{n + 1}*{2 * n + 1}/6.\n"
        f"Step 2: Compute: {n} * {n + 1} = {n * (n + 1)}.\n"
        f"  {n * (n + 1)} * {2 * n + 1} = {n * (n + 1) * (2 * n + 1)}.\n"
        f"  {n * (n + 1) * (2 * n + 1)} / 6 = {formula}.\n\n"
        f"Step 3: Direct sum: 1 + 4 + 9 + ... + {n**2} = {sum_sq}.\n\n"
        f"Step 4: Both give {formula}. The formula is verified. QED."
    )


# ── Tutorial Generators ──────────────────────────────────────────────────────


def _tutorial_fractions(rng: random.Random) -> str:
    a, b = rng.randint(1, 20), rng.randint(2, 20)
    c, d = rng.randint(1, 20), rng.randint(2, 20)
    from math import gcd
    g1 = gcd(a, b)
    _sa, _sb = a // g1, b // g1
    num = a * d + c * b
    den = b * d
    g2 = gcd(num, den)
    rn, rd = num // g2, den // g2
    return (
        f"# Tutorial: Adding Fractions\n\n"
        f"## Concept\nTo add fractions a/b + c/d, find a common denominator: ad + bc over bd.\n\n"
        f"## Example\nCompute {a}/{b} + {c}/{d}.\n\n"
        f"Step 1: Common denominator = {b} x {d} = {b * d}.\n"
        f"Step 2: {a}/{b} = {a * d}/{b * d}. {c}/{d} = {c * b}/{b * d}.\n"
        f"Step 3: {a * d} + {c * b} = {num}.\n"
        f"Step 4: Result = {num}/{den}.\n"
        f"Step 5: Simplify. GCD({num}, {den}) = {g2}. Reduced: {rn}/{rd}.\n\n"
        f"## Answer\n{a}/{b} + {c}/{d} = {rn}/{rd}"
    )


def _tutorial_exponents(rng: random.Random) -> str:
    a = rng.randint(2, 5)
    b = rng.randint(2, 4)
    c = rng.randint(2, 4)
    return (
        f"# Tutorial: Laws of Exponents\n\n"
        f"## Rules\n"
        f"1. a^m x a^n = a^(m+n)\n"
        f"2. (a^m)^n = a^(mn)\n"
        f"3. (ab)^n = a^n x b^n\n\n"
        f"## Example 1\nSimplify {a}^{b} x {a}^{c}.\n"
        f"  {a}^{b} x {a}^{c} = {a}^({b}+{c}) = {a}^{b + c} = {a ** (b + c)}.\n\n"
        f"## Example 2\nSimplify ({a}^{b})^{c}.\n"
        f"  ({a}^{b})^{c} = {a}^({b}*{c}) = {a}^{b * c} = {a ** (b * c)}.\n\n"
        f"## Example 3\nSimplify ({a} x {c})^{b}.\n"
        f"  ({a} x {c})^{b} = {a}^{b} x {c}^{b} = {a ** b} x {c ** b} = {(a * c) ** b}."
    )


def _tutorial_percentage(rng: random.Random) -> str:
    val = rng.randint(50, 500)
    pct = rng.randint(5, 75)
    result = val * pct / 100
    return (
        f"# Tutorial: Percentage Calculations\n\n"
        f"## Concept\nPercents are fractions out of 100. To find p% of x, compute (p/100) * x.\n\n"
        f"## Example\nWhat is {pct}% of {val}?\n\n"
        f"Step 1: Convert {pct}% to decimal: {pct}/100 = {pct / 100}.\n"
        f"Step 2: Multiply: {pct / 100} x {val} = {result:.0f}.\n\n"
        f"## Answer\n{pct}% of {val} = {result:.0f}.\n\n"
        f"## Reverse Example\n{val} is {pct}% of what number?\n"
        f"  x = {val} / ({pct}/100) = {val} / {pct / 100} = {val / (pct / 100):.0f}."
    )


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_GENERATORS = [
    ("step-by-step-arithmetic", _cog_arithmetic, 200),
    ("step-by-step-algebra", _cog_algebra, 200),
    ("step-by-step-quadratic", _cog_quadratic, 200),
    ("word-problems", _cog_word_problem, 300),
    ("step-by-step-geometry", _cog_geometry, 150),
    ("step-by-step-probability", _cog_probability, 100),
    ("sequences", _cog_sequences, 150),
    ("limits", _cog_limits, 100),
    ("matrices", _cog_matrices, 100),
    ("proofs-even-sum", _proof_even_sum, 80),
    ("proofs-divisibility", _proof_divisibility, 80),
    ("proofs-square-sum", _proof_square_sum, 40),
    ("tutorial-fractions", _tutorial_fractions, 100),
    ("tutorial-exponents", _tutorial_exponents, 100),
    ("tutorial-percentage", _tutorial_percentage, 100),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand training corpus with chain-of-thought and diverse formats.")
    parser.add_argument(
        "--output",
        default="experiments/corpus/expanded_v2.jsonl",
        help="Output JSONL path for new documents",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cap", type=int, default=None, help="Cap total docs (0 = skip generation)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs: list[dict] = []
    idx = 0
    for source_name, gen_fn, count in ALL_GENERATORS:
        for _i in range(count):
            text = gen_fn(rng)
            docs.append(_make_doc(text, idx, source_name))
            idx += 1
        print(f"  {source_name}: {count} docs")

    if args.cap is not None:
        docs = docs[: args.cap]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_chars = sum(len(d["text"]) for d in docs)
    print(f"\nGenerated {len(docs)} expanded docs ({total_chars:,} chars) -> {out_path}")
    print(f"Approx words: {total_chars // 5:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

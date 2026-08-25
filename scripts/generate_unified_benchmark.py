"""Unified benchmark generator (M20) - graded evaluation across every
lesson-plan domain (math L01-L13 + computing L14-L16).

Design rules:
* question PHRASINGS are deliberately different from foundation/lesson_plan.py
  training templates (contamination guard);
* every reference answer is computed in code;
* every answer is a plain number (int/float string) so the existing numeric
  metrics apply unchanged.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _arithmetic(rng):
    a, b, c = rng.randint(101, 999), rng.randint(11, 99), rng.randint(2, 12)
    return f"Compute {a} x {b} - {c}. Reply with the number only.", a * b - c


def _geometry(rng):
    w, h = rng.randint(7, 60), rng.randint(5, 40)
    return f"A rectangle measures {w} cm by {h} cm. How many square centimeters is its area?", w * h


def _probability(rng):
    faces, target = rng.choice([6, 8]), rng.randint(2, 5)
    return (
        f"One fair {faces}-sided die is rolled once. What is the probability of rolling "
        f"a number of {target} or higher? Answer as a decimal fraction.",
        round((faces - target + 1) / faces, 4),
    )


def _algebra(rng):
    k, s, m = rng.randint(3, 14), rng.randint(-25, 25), rng.randint(-60, 60)
    return f"Find n if {k} times n plus ({m}) equals {k * s + m}. Numeric answer only.", s


def _linear_systems(rng):
    x0, y0 = rng.randint(-15, 15), rng.randint(-15, 15)
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    return (
        f"Two numbers add up to {x0 + y0}; twice the first plus the second equals "
        f"{2 * x0 + y0}. Find the first number.",
        x0,
    ) if b == 0 else (
        f"Solve for x: {a}x + {b}y = {a * x0 + b * y0}, given that y = {y0}.",
        x0,
    )


def _functions(rng):
    a, b, x = rng.randint(2, 9), rng.randint(-9, 9), rng.randint(3, 12)
    return f"If g(t) = {a}t + ({b}), what is g({x})?", a * x + b


def _number_theory(rng):
    a, b = rng.randint(12, 200), rng.randint(12, 200)
    return f"Calculate the greatest common divisor of {a} and {b}.", math.gcd(a, b)


def _statistics(rng):
    mu, spread = rng.randint(10, 40), [(-2, 4), (-1, 1), (0, 0), (1, 1), (2, 4)]
    data = [mu + d for d, _ in [(d, s) for d, s in spread]]
    var = sum(s for _, s in spread) / len(spread)
    return (
        f"For the population {data}, compute the variance (mean mu = {mu}).",
        var,
    )


def _trigonometry(rng):
    # sin/cos of standard angles expressed numerically, rounded to 4 dp
    angle, val = rng.choice([
        ("30", 0.5), ("45", round(math.sqrt(2) / 2, 4)), ("60", round(math.sqrt(3) / 2, 4)),
    ])
    fn = rng.choice(["sine", "cosine"]) if angle != "30" else "sine"
    v = val if fn == "sine" else {
        "30": round(math.sqrt(3) / 2, 4), "45": round(math.sqrt(2) / 2, 4), "60": 0.5,
    }[angle]
    return f"To four decimals, what is the {fn} of {angle} degrees?", v


def _differentiation(rng):
    a, n, x = rng.randint(2, 9), rng.choice([2, 3, 4]), rng.randint(1, 6)
    deriv = a * n * x ** (n - 1)
    return (
        f"f(x) = {a}x^{n}. Evaluate the derivative f'({x}). Integer answer.",
        deriv,
    )


def _integration(rng):
    a, lo, hi = rng.randint(2, 8), 0, rng.randint(1, 5)
    val = a * hi ** 2 // 2
    while a * hi ** 2 % 2:
        a += 1
    val = a * hi ** 2 // 2
    return f"Compute the definite integral of {a}x from {lo} to {hi}.", val


def _linear_algebra(rng):
    which = rng.randrange(3)
    if which == 0:
        u, v = [rng.randint(-7, 7) for _ in range(3)], [rng.randint(-7, 7) for _ in range(3)]
        dot = sum(p * q for p, q in zip(u, v))
        return f"Tuples u = {tuple(u)} and v = {tuple(v)}. Their scalar product equals?", dot
    if which == 1:
        m = [[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        return (
            f"For the 2x2 grid [{m[0][0]}, {m[0][1]}; {m[1][0]}, {m[1][1]}], give its determinant.",
            det,
        )
    tr = rng.randint(-9, 9)
    off = rng.randint(-6, 6)
    return (
        f"A square matrix has diagonal entries {tr} and 0, and one off-diagonal entry {off}. "
        f"What is its trace?",
        tr,
    )


def _advanced_probability(rng):
    which = rng.randrange(2)
    if which == 0:
        n, k = rng.randint(6, 10), rng.randint(2, 4)
        return (
            f"From {n} candidates, how many distinct unordered teams of exactly {k} can be formed?",
            math.comb(n, k),
        )
    faces = rng.choice([6, 8])
    ev = round((faces + 1) / 2, 2)
    return (
        f"A uniform spinner labeled 1..{faces} is spun once. Expected value (2 dp)?",
        ev,
    )


def _programming(rng):
    n, step = rng.randint(3, 9), rng.choice([1, 2])
    total = sum(range(step, n * step + 1, step))
    seq = ", ".join(str(i) for i in range(step, n * step + 1, step))
    return (
        f"A loop accumulates these values into a running sum: {seq}. What is the final sum?",
        total,
    )


def _machine_language(rng):
    which = rng.randrange(2)
    x = rng.randint(9, 250)
    if which == 0:
        return f"The byte 0b{format(x, '08b')} represents which base-ten value?", x
    a, b = rng.randint(10, 200), rng.randint(10, 200)
    return f"Bitwise OR of {a} and {b} equals what integer?", a | b


def _networking(rng):
    which = rng.randrange(2)
    if which == 0:
        prefix = rng.choice([24, 25, 26, 27, 28])
        return (
            f"How many usable host addresses exist in an IPv4 /{prefix} subnet?",
            2 ** (32 - prefix) - 2,
        )
    port, svc = rng.choice([(22, "SSH"), (53, "DNS"), (80, "HTTP"), (443, "HTTPS"), (25, "SMTP")])
    return (
        f"On what TCP port does the {svc} protocol conventionally listen? Number only.",
        port,
    )


FAMILIES = [
    ("arithmetic", _arithmetic),
    ("geometry", _geometry),
    ("probability", _probability),
    ("algebra", _algebra),
    ("linear_equations", _linear_systems),
    ("functions", _functions),
    ("number_theory", _number_theory),
    ("statistics", _statistics),
    ("trigonometry", _trigonometry),
    ("differentiation", _differentiation),
    ("integration", _integration),
    ("linear_algebra", _linear_algebra),
    ("advanced_probability", _advanced_probability),
    ("programming", _programming),
    ("machine_language", _machine_language),
    ("networking", _networking),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = ROOT / "benchmarks" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for category, family in FAMILIES:
        for i in range(args.per_topic):
            prompt, reference = family(rng)
            ref_str = f"{reference:.4f}".rstrip("0").rstrip(".") if isinstance(reference, float) else str(reference)
            records.append({
                "id": f"uni-{category}-{i:04d}",
                "prompt": prompt,
                "reference": ref_str,
                "category": category,
                "task_type": "numeric",
            })

    out_path = out_dir / "unified-v2.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark": "unified-benchmark-v2",
        "seed": args.seed,
        "per_topic": args.per_topic,
        "topics": [c for c, _ in FAMILIES],
        "total": len(records),
        "path": str(out_path),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

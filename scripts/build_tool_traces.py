"""Build tool-trace SFT data (M21): teaches the model to USE tools itself.

Each record's response follows the exact protocol the serving loop speaks:

    <formula/plan line>
    <tool>{"name": "calculator", "arguments": {...}}</tool>
    <result>...</result>
    <final answer sentence>

Deterministic (M01): problems come from seeded templates; results are
computed in code. Output: experiments/sft/tool-traces-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.general_corpus import write_jsonl  # noqa: E402


def _calc(expression: str, result: str) -> str:
    return (
        f'<tool>{{"name": "calculator", "arguments": {{"expression": "{expression}"}}}}</tool>\n'
        f"<result>{result}</result>"
    )


def _circle(rng) -> dict:
    r = rng.randint(3, 25)
    area = round(math.pi * r * r, 2)
    return {
        "instruction": f"A circle has a radius of {r} cm. Find its area (use pi = 3.14159).",
        "response": (
            f"Area of a circle: A = pi * r^2.\n"
            f"{_calc(f'3.14159*{r}**2', str(area))}\n"
            f"The area is {area} square centimeters."
        ),
    }


def _percent(rng) -> dict:
    p = rng.choice([5, 10, 12, 15, 20, 25, 40, 60])
    n = rng.randint(20, 900)
    val = round(p * n / 100, 2)
    return {
        "instruction": f"What is {p}% of {n}?",
        "response": (
            f"Percentage: value = P/100 * N.\n"
            f"{_calc(f'{p}/100*{n}', str(val))}\n"
            f"So {p}% of {n} is {val}."
        ),
    }


def _rectangle(rng) -> dict:
    w, h = rng.randint(4, 40), rng.randint(3, 30)
    return {
        "instruction": (
            f"A technical drawing shows a rectangle {w} mm by {h} mm. "
            f"Give its perimeter and area."
        ),
        "response": (
            f"Perimeter: P = 2(w + h); Area: A = w * h.\n"
            f"{_calc(f'2*({w}+{h})', str(2 * (w + h)))}\n"
            f"{_calc(f'{w}*{h}', str(w * h))}\n"
            f"Perimeter = {2 * (w + h)} mm and area = {w * h} square mm."
        ),
    }


def _speed(rng) -> dict:
    km = rng.randint(60, 600)
    hrs = rng.randint(2, 9)
    v = round(km / hrs, 2)
    return {
        "instruction": f"A train covers {km} km in {hrs} hours. Find its average speed.",
        "response": (
            f"Average speed: v = distance / time.\n"
            f"{_calc(f'{km}/{hrs}', str(v))}\n"
            f"The average speed is {v} km/h."
        ),
    }


def _discount(rng) -> dict:
    price = rng.randint(80, 2000)
    off = rng.choice([10, 15, 20, 25, 30])
    final = round(price * (100 - off) / 100, 2)
    return {
        "instruction": f"An item costs ${price} with a {off}% discount. What is the final price?",
        "response": (
            f"Final price = price * (1 - discount/100).\n"
            f"{_calc(f'{price}*(1-{off}/100)', str(final))}\n"
            f"The final price is ${final}."
        ),
    }


def _pythagoras(rng) -> dict:
    triple = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17)])
    k = rng.randint(1, 4)
    a, b, c = (t * k for t in triple)
    return {
        "instruction": f"In a right triangle the legs are {a} and {b}. Find the hypotenuse.",
        "response": (
            "Pythagorean theorem: c^2 = a^2 + b^2.\n"
            f"{_calc(f'{a}**2+{b}**2', str(a * a + b * b))}\n"
            f"c = sqrt({a * a + b * b}) = {c}. The hypotenuse is {c}."
        ),
    }


def _average(rng) -> dict:
    vals = [rng.randint(2, 50) for _ in range(rng.randint(4, 6))]
    mean = round(sum(vals) / len(vals), 2)
    seq = ", ".join(map(str, vals))
    expr = "(" + "+".join(map(str, vals)) + f")/{len(vals)}"
    return {
        "instruction": f"Find the mean of: {seq}.",
        "response": (
            "Mean = sum of values / count.\n"
            f"{_calc(expr.strip('(').rstrip('/') + '/' + str(len(vals)) if False else expr[1:-1] + '/' + str(len(vals)), str(mean))}\n"
            f"The mean is {mean}."
        ),
    }


TEMPLATES = [
    _circle, _percent, _rectangle, _speed,
    _discount, _pythagoras, _average,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="experiments/sft/tool-traces-v1.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = []
    for idx, template in enumerate(TEMPLATES):
        for i in range(args.per_template):
            rec = template(rng)
            records.append({
                "id": f"tool-v1-{template.__name__.strip('_')}-{i:04d}",
                "instruction": rec["instruction"],
                "response": rec["response"],
            })

    out = ROOT / args.output
    write_jsonl(out, records)
    print(json.dumps({
        "dataset": "tool-traces-v1",
        "templates": [t.__name__ for t in TEMPLATES],
        "total": len(records),
        "path": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

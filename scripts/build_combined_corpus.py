"""Combine original + expanded corpus into a single training file.
Also builds an enhanced SFT dataset with chain-of-thought solutions.

Usage:
    python scripts/build_combined_corpus.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _combine_corpus() -> None:
    original = ROOT / "experiments/corpus/train.0.jsonl"
    expanded = ROOT / "experiments/corpus/expanded_v2.jsonl"
    output = ROOT / "experiments/corpus/combined_train.jsonl"

    docs = []
    for src in [original, expanded]:
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    docs.append(line.strip())
            print(f"  Loaded {src.name}: {len(docs)} total")

    with output.open("w", encoding="utf-8") as f:
        for line in docs:
            f.write(line + "\n")

    total_chars = sum(len(json.loads(l)["text"]) for l in docs)
    print(f"  Combined: {len(docs)} docs, {total_chars:,} chars, ~{total_chars // 5:,} words -> {output}")


def _build_enhanced_sft() -> None:
    """Build enhanced SFT with step-by-step chain-of-thought responses."""
    rng = random.Random(42)
    output = ROOT / "experiments/sft/sft-v2-cot.jsonl"

    records = []
    idx = 0

    # ── Arithmetic with steps ──
    for _ in range(500):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            steps = (
                f"{a} + {b}\n"
                f"Step 1: Add ones: {a % 10} + {b % 10} = {a % 10 + b % 10}.\n"
                f"Step 2: Add remaining: {a // 10} + {b // 10} + {(a % 10 + b % 10) // 10} = {a // 10 + b // 10 + (a % 10 + b % 10) // 10}.\n"
                f"Answer: {ans}"
            )
        elif op == "-":
            x, y = max(a, b), min(a, b)
            ans = x - y
            steps = f"{x} - {y}\nStep 1: Subtract directly: {x} - {y} = {ans}.\nAnswer: {ans}"
        else:
            ans = a * b
            steps = f"{a} x {b}\nStep 1: Multiply: {a} x {b} = {ans}.\nAnswer: {ans}"

        records.append({
            "id": f"sft-v2-arithmetic-{idx:05d}",
            "instruction": f"Compute {a} {op} {b}. Show your work step by step.",
            "response": steps,
        })
        idx += 1

    # ── Algebra with steps ──
    for _ in range(400):
        a = rng.randint(2, 15)
        b = rng.randint(1, 40)
        x = rng.randint(1, 20)
        c = a * x + b
        records.append({
            "id": f"sft-v2-algebra-{idx:05d}",
            "instruction": f"Solve for x: {a}x + {b} = {c}. Show steps.",
            "response": (
                f"{a}x + {b} = {c}\n"
                f"Step 1: Subtract {b} from both sides: {a}x = {c - b}.\n"
                f"Step 2: Divide by {a}: x = {c - b} / {a} = {x}.\n"
                f"Step 3: Check: {a}({x}) + {b} = {c}. Correct.\n"
                f"Answer: x = {x}"
            ),
        })
        idx += 1

    # ── Quadratic with steps ──
    for _ in range(200):
        r1 = rng.randint(-8, 8)
        r2 = rng.randint(-8, 8)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2
        records.append({
            "id": f"sft-v2-quadratic-{idx:05d}",
            "instruction": f"Solve x^2 + ({b_coeff})x + ({c_coeff}) = 0. Show steps.",
            "response": (
                f"x^2 + ({b_coeff})x + ({c_coeff}) = 0\n"
                f"Step 1: Find two numbers that multiply to {c_coeff} and add to {-b_coeff}.\n"
                f"  {r1} x {r2} = {r1 * r2}, {r1} + {r2} = {r1 + r2}\n"
                f"Step 2: Factor: (x - ({r1}))(x - ({r2})) = 0\n"
                f"Step 3: x = {r1} or x = {r2}\n"
                f"Answer: x = {r1} or x = {r2}"
            ),
        })
        idx += 1

    # ── Geometry with steps ──
    for _ in range(200):
        shape = rng.choice(["rectangle", "triangle", "circle"])
        if shape == "rectangle":
            w, h = rng.randint(3, 25), rng.randint(3, 25)
            area = w * h
            perim = 2 * (w + h)
            records.append({
                "id": f"sft-v2-geometry-{idx:05d}",
                "instruction": f"Find the area and perimeter of a rectangle with width {w} and height {h}. Show steps.",
                "response": (
                    f"Rectangle: width={w}, height={h}\n"
                    f"Step 1: Area = width x height = {w} x {h} = {area}.\n"
                    f"Step 2: Perimeter = 2(width + height) = 2({w} + {h}) = 2({w + h}) = {perim}.\n"
                    f"Answer: Area = {area}, Perimeter = {perim}"
                ),
            })
        elif shape == "triangle":
            b, ht = rng.randint(3, 20), rng.randint(3, 20)
            area = b * ht / 2
            area_s = f"{area:.0f}" if area == int(area) else f"{area}"
            records.append({
                "id": f"sft-v2-geometry-{idx:05d}",
                "instruction": f"Find the area of a triangle with base {b} and height {ht}. Show steps.",
                "response": (
                    f"Triangle: base={b}, height={ht}\n"
                    f"Step 1: Area = (1/2) x base x height.\n"
                    f"Step 2: Area = (1/2) x {b} x {ht} = {b * ht} / 2 = {area_s}.\n"
                    f"Answer: Area = {area_s}"
                ),
            })
        else:
            r = rng.randint(2, 15)
            from math import pi
            area = pi * r * r
            circ = 2 * pi * r
            records.append({
                "id": f"sft-v2-geometry-{idx:05d}",
                "instruction": f"Find the area and circumference of a circle with radius {r}. Show steps.",
                "response": (
                    f"Circle: radius={r}\n"
                    f"Step 1: Area = pi x r^2 = pi x {r}^2 = {area:.2f}.\n"
                    f"Step 2: Circumference = 2 x pi x r = 2 x pi x {r} = {circ:.2f}.\n"
                    f"Answer: Area = {area:.2f}, Circumference = {circ:.2f}"
                ),
            })
        idx += 1

    # ── Word problems with steps ──
    for _ in range(400):
        speed = rng.randint(20, 100)
        time = rng.randint(1, 10)
        dist = speed * time
        records.append({
            "id": f"sft-v2-wordprob-{idx:05d}",
            "instruction": f"A vehicle travels at {speed} km/h for {time} hours. How far does it travel? Show steps.",
            "response": (
                f"Given: speed = {speed} km/h, time = {time} hours\n"
                f"Step 1: Distance = Speed x Time.\n"
                f"Step 2: Distance = {speed} x {time} = {dist} km.\n"
                f"Answer: {dist} km"
            ),
        })
        idx += 1

    # ── Probability with steps ──
    for _ in range(150):
        n = rng.randint(3, 10)
        k = rng.randint(1, n - 1)
        from math import comb
        c = comb(n, k)
        total = 2**n
        records.append({
            "id": f"sft-v2-probability-{idx:05d}",
            "instruction": f"What is the probability of exactly {k} heads in {n} fair coin flips? Show steps.",
            "response": (
                f"Flipping {n} fair coins, P(exactly {k} heads)\n"
                f"Step 1: Total outcomes = 2^{n} = {total}.\n"
                f"Step 2: Ways to choose {k} positions = C({n},{k}) = {c}.\n"
                f"Step 3: Probability = {c}/{total} = {c / total:.4f}.\n"
                f"Answer: {c}/{total} = {c / total:.4f}"
            ),
        })
        idx += 1

    # ── Sequences ──
    for _ in range(150):
        a1 = rng.randint(1, 10)
        d = rng.randint(1, 10)
        n = rng.randint(5, 15)
        s = n * (2 * a1 + (n - 1) * d) // 2
        records.append({
            "id": f"sft-v2-sequences-{idx:05d}",
            "instruction": f"Find the sum of the first {n} terms of an arithmetic sequence with a1={a1} and d={d}. Show steps.",
            "response": (
                f"Arithmetic sequence: a1={a1}, d={d}, n={n}\n"
                f"Step 1: S_n = n/2 * (2*a1 + (n-1)*d).\n"
                f"Step 2: S_{n} = {n}/2 * (2*{a1} + {n-1}*{d}) = {n}/2 * ({2*a1} + {(n-1)*d}) = {n}/2 * {2*a1 + (n-1)*d}.\n"
                f"Step 3: S_{n} = {s}.\n"
                f"Answer: {s}"
            ),
        })
        idx += 1

    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Enhanced SFT: {len(records)} records -> {output}")


def main() -> int:
    print("Building combined corpus:")
    _combine_corpus()
    print("\nBuilding enhanced SFT (chain-of-thought):")
    _build_enhanced_sft()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

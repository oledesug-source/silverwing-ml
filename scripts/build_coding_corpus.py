"""Coding-agent corpus (M22) - teaches Silverwing to behave like a coding
agent: plan, write clean code, EXECUTE it via the tool protocol, verify,
then answer. Every trace's <result> block is produced by actually running
the snippet at generation time, so supervision is execution-verified.

Layers:
  A style_knowledge   conventions / idioms / complexity (deterministic Q/A)
  B function_writing  task -> typed implementation -> run-verified output
  C bug_fix           buggy snippet -> diagnosis -> fix -> verified output
  D agent_loop        multi-step tool traces (write file? no - pure compute
                      loops using calculator/python_exec protocol shapes)

Output: experiments/sft/coding-v1.jsonl  (M01 reproducible from seed)
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation.general_corpus import write_jsonl  # noqa: E402


def _run(code: str) -> str:
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        raise RuntimeError(f"snippet failed rc={r.returncode}: {out[:200]}")
    return out


def _tool(name: str, args: dict) -> str:
    return f'<tool>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool>'


def _result_block(output: str) -> str:
    return f"<result>{output}</result>"


# ------------------------------------------------------------------ A: style
STYLE_QA = [
    ("Why should Python functions have docstrings?",
     "Docstrings are the contract of a function: they document intent, "
     "parameters and return value for callers and tooling. Tools like help() "
     "and IDEs read them directly, so documentation stays attached to code."),
    ("When is a list comprehension better than a for-loop?",
     "Use a comprehension when you are transforming/filtering data into a new "
     "list in one expression; keep an explicit loop when there are side "
     "effects or multiple statements per item. Comprehensions state intent: "
     "'build this list'."),
    ("What is the time complexity of dict lookup in Python?",
     "Average case O(1): dicts are hash tables. Worst case degrades toward "
     "O(n) with pathological hash collisions, which is rare in practice."),
    ("Why prefer 'with open(...)' over manual close()?",
     "The context manager guarantees the file is closed even when an "
     "exception occurs, preventing resource leaks. Manual close depends on "
     "remembering every exit path."),
    ("What does typing.List[int] give you that a plain list does not?",
     "Static documentation of intent plus editor/mypy checking. Runtime "
     "behavior is identical; the win is catching wrong element types before "
     "shipping."),
    ("Why raise specific exceptions instead of returning error strings?",
     "Specific exceptions let callers handle failure precisely with except "
     "clauses, keep the success path typed, and carry tracebacks instead of "
     "silent stringly-typed errors."),
    ("When should a class be used instead of a function?",
     "Use a class when state and behavior belong together across calls "
     "(a connection, a parser with position). Use functions for stateless "
     "transforms - classes add structure cost that must pay for itself."),
    ("What is the single-responsibility principle in module design?",
     "Each module owns exactly one reason to change: parsing lives apart from "
     "IO, config apart from logic. Small surfaces make testing and reuse "
     "trivial."),
]

# ------------------------------------------------------- B/C: verified tasks
def _task_greet(rng):
    n = rng.choice(["Silverwing", "Ada", "Linus", "Grace"])
    code = (
        "def greet(name):\n"
        '    return "Hello, " + name + "!"\n'
        f"print(greet('{n}'))"
    )
    expected = _run(code)
    return {
        "instruction": (
            f"Write a Python function greet(name) that returns 'Hello, <name>!' "
            f"and show it working for {n}."
        ),
        "response": (
            "Plan: one typed-signature function, simple concatenation, then execute to verify.\n"
            "```python\ndef greet(name):\n    return \"Hello, \" + name + \"!\"\n```\n"
            + _tool("python_exec", {"code": code})
            + "\n" + _result_block(expected)
            + f"\nVerified: greet('{n}') returns 'Hello, {n}!' as required."
        ),
    }


def _task_fib(rng):
    k = rng.randint(7, 12)
    seq_code = (
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    out = []\n"
        "    for _ in range(n):\n"
        "        out.append(a)\n"
        "        a, b = b, a + b\n"
        "    return out\n"
        f"print(fib({k}))"
    )
    expected = _run(seq_code)
    return {
        "instruction": f"Write fib(n) returning the first n Fibonacci numbers and verify for n={k}.",
        "response": (
            "Iterative build avoids exponential recursion.\n"
            "```python\ndef fib(n):\n    a, b = 0, 1\n    out = []\n"
            "    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n"
            "    return out\n```\n"
            + _tool("python_exec", {"code": seq_code})
            + "\n" + _result_block(expected)
            + f"\nThe first {k} Fibonacci numbers verified by execution."
        ),
    }


def _task_sum_squares(rng):
    n = rng.randint(4, 15)
    code = f"print(sum(i*i for i in range(1, {n + 1})))"
    expected = _run(code)
    total = sum(i * i for i in range(1, n + 1))
    return {
        "instruction": f"Compute the sum of squares from 1 to {n} in Python and verify.",
        "response": (
            "Single generator expression inside sum() states intent cleanly.\n"
            f"```python\nsum(i*i for i in range(1, {n + 1}))\n```\n"
            + _tool("python_exec", {"code": code})
            + "\n" + _result_block(expected)
            + f"\nExecution confirms the sum of squares is {total}."
        ),
    }


def _bug_off_by_one(rng):
    n = rng.randint(3, 8)
    bad = f"total = 0\nfor i in range(1, {n}):\n    total += i\nprint(total)"
    fixed = f"total = 0\nfor i in range(1, {n + 1}):\n    total += i\nprint(total)"
    wrong = _run(bad)
    right = _run(fixed)
    return {
        "instruction": (
            f"This code should sum 1..{n} but returns {wrong}. Find and fix the bug:\n{bad}"
        ),
        "response": (
            f"Diagnosis: range(1, {n}) stops BEFORE {n} - classic off-by-one; "
            f"range excludes its stop value.\nFix: use range(1, {n + 1}).\n"
            + _tool("python_exec", {"code": fixed})
            + "\n" + _result_block(right)
            + f"\nFixed output {right} matches the expected sum of 1..{n}."
        ),
    }


def _bug_mutable_default(rng):
    code_fixed = (
        "def add_item(item, items=None):\n"
        "    if items is None:\n"
        "        items = []\n"
        "    items.append(item)\n"
        "    return items\n"
        "print(add_item('a'))\nprint(add_item('b'))"
    )
    expected = _run(code_fixed)
    return {
        "instruction": (
            "Explain and fix the mutable default argument trap:\n"
            "def add_item(item, items=[]):\n    items.append(item)\n    return items"
        ),
        "response": (
            "A default list is created ONCE at definition time, so calls share it "
            "and items accumulate across calls.\nFix: default None, create inside:\n"
            "```python\ndef add_item(item, items=None):\n    if items is None:\n"
            "        items = []\n    items.append(item)\n    return items\n```\n"
            + _tool("python_exec", {"code": code_fixed})
            + "\n" + _result_block(expected)
            + "\nNow each call starts fresh unless a list is passed in."
        ),
    }


VERIFIED_TASKS = [_task_greet, _task_fib, _task_sum_squares]
BUG_TASKS = [_bug_off_by_one, _bug_mutable_default]

AGENT_LOOP_TEMPLATE = (
    "Approach: restate the goal, choose data structures, implement, then "
    "verify by execution before answering.\n{body}"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-per-q", type=int, default=6)
    ap.add_argument("--tasks-per-t", type=int, default=25)
    ap.add_argument("--bugs-per-t", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="experiments/sft/coding-v1.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = []

    for i, (q, a) in enumerate(STYLE_QA):
        for v in range(args.style_per_q):
            records.append({
                "id": f"code-style-{i:02d}-{v:03d}",
                "instruction": q,
                "response": a,
            })

    counters = {"t": 0, "b": 0}
    for gen in VERIFIED_TASKS:
        for _ in range(args.tasks_per_t):
            rec = gen(rng)
            body = rec["response"]
            records.append({
                "id": f"code-task-{counters['t']:04d}",
                "instruction": rec["instruction"],
                "response": AGENT_LOOP_TEMPLATE.format(body=body),
            })
            counters["t"] += 1
    for gen in BUG_TASKS:
        for _ in range(args.bugs_per_t):
            rec = gen(rng)
            records.append({
                "id": f"code-bug-{counters['b']:04d}",
                "instruction": rec["instruction"],
                "response": rec["response"],
            })
            counters["b"] += 1

    rng.shuffle(records)
    out = ROOT / args.output
    write_jsonl(out, records)
    print(json.dumps({
        "dataset": "coding-v1",
        "style": len(STYLE_QA) * args.style_per_q,
        "verified_tasks": counters["t"],
        "bug_fixes": counters["b"],
        "total": len(records),
        "path": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

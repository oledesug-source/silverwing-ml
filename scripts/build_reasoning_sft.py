"""Convert reasoning-v1.jsonl to SFT format with chain-of-thought reasoning.

Output: reasoning-v1-sft.jsonl with {"id", "instruction", "response"} where
response includes the reasoning steps followed by the final answer.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "reasoning" / "reasoning-v1.jsonl"
OUTPUT = ROOT / "experiments" / "reasoning" / "reasoning-v1-sft.jsonl"

count = 0
with open(INPUT, encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        d = json.loads(line)
        steps = " ".join(d["reasoning_steps"])
        response = steps + " " + d["final_answer"]
        record = {
            "id": d["id"],
            "instruction": d["problem"],
            "response": response,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Wrote {count} records to {OUTPUT}")

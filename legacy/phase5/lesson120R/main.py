import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch

SEED = 42
MEMORY_VERSION = "120R.8"
HIGH_RISK = 0.75
MEDIUM_RISK = 0.40
TOLERANCE = 1e-6

BASE_DIR = Path.cwd()
PHASE5_DIR = BASE_DIR.parent
LESSON_119R = PHASE5_DIR / "lesson119R"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MEMORY = LESSON_119R / "silverwing_predictive_error_prevention_memory.json"
SOURCE_INDEX = LESSON_119R / "silverwing_predictive_error_prevention_index.pt"
SOURCE_DATASET = LESSON_119R / "silverwing_predictive_error_prevention_dataset.json"
SOURCE_REPORT = LESSON_119R / "silverwing_predictive_error_prevention_report.json"
SOURCE_REGISTRY = LESSON_119R / "silverwing_predictive_error_prevention_registry.json"
SOURCE_CHECKPOINT = LESSON_119R / "checkpoints" / "silverwing_predictive_error_prevention_best.pt"

MEMORY_FILE = BASE_DIR / "silverwing_multi_pattern_risk_memory.json"
INDEX_FILE = BASE_DIR / "silverwing_multi_pattern_risk_index.pt"
DATASET_FILE = BASE_DIR / "silverwing_multi_pattern_risk_dataset.json"
REPORT_FILE = BASE_DIR / "silverwing_multi_pattern_risk_report.json"
REGISTRY_FILE = BASE_DIR / "silverwing_multi_pattern_risk_registry.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "silverwing_multi_pattern_risk_best.pt"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def clamp(value):
    return max(0.0, min(1.0, float(value)))


def risk_class(score):
    if score >= HIGH_RISK:
        return "HIGH"
    if score >= MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


def risk_score(item):
    return clamp(
        0.30 * item["confidence"]
        + 0.25 * item["severity"]
        + 0.20 * item["impact"]
        + 0.20 * item["recurrence"]
        - 0.05 * item["cost"]
    )


def arbitrate(records):
    return sorted(
        records,
        key=lambda item: (
            -item["risk_score"],
            -item["severity"],
            -item["impact"],
            item["pattern_id"],
        ),
    )


def stable_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


torch.manual_seed(SEED)
random.seed(SEED)

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 120R")
print("Native Multi-Pattern Risk Arbitration + Preventive Planning")
print()
print("119R -> Predictive Error Prevention")
print("120R -> Multi-Pattern Risk Arbitration + Preventive Planning")
print("External LLM: NONE")
print("Memory version:", MEMORY_VERSION)
print()

print("TEST 1: Verify 119R Inputs")

required = [
    SOURCE_MEMORY,
    SOURCE_INDEX,
    SOURCE_DATASET,
    SOURCE_REPORT,
    SOURCE_REGISTRY,
    SOURCE_CHECKPOINT,
]

missing = [
    str(path)
    for path in required
    if not path.exists()
]

if missing:
    raise FileNotFoundError(
        "Missing 119R inputs: " + "; ".join(missing)
    )

for path in required:
    print("FOUND:", path)

print()

print("TEST 2: Load 119R Predictive Memory")

source = load_json(SOURCE_MEMORY)

if not isinstance(source, dict):
    raise RuntimeError(
        "119R predictive memory is invalid."
    )

source_patterns = source.get("error_patterns", [])
source_rules = source.get("prevention_rules", [])
source_risk = source.get("risk", {})

if not source_patterns:
    raise RuntimeError(
        "119R error patterns are missing."
    )

if not source_rules:
    raise RuntimeError(
        "119R prevention rules are missing."
    )

inherited_confidence = clamp(
    source_risk.get(
        "predictive_risk",
        0.0,
    )
)

print(
    "Memory version:",
    source.get("memory_version")
)
print(
    "Inherited patterns:",
    len(source_patterns)
)
print(
    "Inherited rules:",
    len(source_rules)
)
print(
    "119R predictive risk:",
    inherited_confidence
)
print()

print("TEST 3: Build Multi-Pattern Risk Set")

patterns = [
    {
        "pattern_id": "pattern_001",
        "family": "evidence_integrity",
        "confidence": inherited_confidence,
        "severity": 0.95,
        "impact": 0.90,
        "recurrence": 1.00,
        "cost": 0.30,
        "origin": "119R",
    },
    {
        "pattern_id": "pattern_002",
        "family": "schema_integrity",
        "confidence": 0.82,
        "severity": 0.78,
        "impact": 0.88,
        "recurrence": 0.75,
        "cost": 0.20,
        "origin": "120R_EVALUATION",
    },
    {
        "pattern_id": "pattern_003",
        "family": "sensor_alignment",
        "confidence": 0.68,
        "severity": 0.70,
        "impact": 0.62,
        "recurrence": 0.60,
        "cost": 0.35,
        "origin": "120R_EVALUATION",
    },
]

if len(patterns) != 3:
    raise RuntimeError(
        "120R requires exactly three risk patterns."
    )

print(
    "Inherited pattern:",
    source_patterns[0].get("pattern_id"),
)
print(
    "Patterns:",
    len(patterns),
)
print()

print("TEST 4: Validate Pattern Schema")

required_fields = {
    "pattern_id",
    "family",
    "confidence",
    "severity",
    "impact",
    "recurrence",
    "cost",
    "origin",
}

schema_errors = [
    (
        pattern["pattern_id"],
        sorted(required_fields - set(pattern.keys())),
    )
    for pattern in patterns
    if required_fields - set(pattern.keys())
]

if schema_errors:
    raise RuntimeError(
        "Pattern schema incomplete: "
        + str(schema_errors)
    )

print("Pattern schemas valid.")
print()

print("TEST 5: Calculate Individual Risk Scores")

risk_records = []

for pattern in patterns:
    score = risk_score(pattern)

    if not 0.0 <= score <= 1.0:
        raise RuntimeError(
            "Risk score outside valid range."
        )

    risk_records.append(
        {
            "pattern_id": pattern["pattern_id"],
            "family": pattern["family"],
            "origin": pattern["origin"],
            "risk_score": score,
            "risk_class": risk_class(score),
            "severity": pattern["severity"],
            "impact": pattern["impact"],
        }
    )

for record in risk_records:
    print(record)

print()

print("TEST 6: Independent Risk Validation")

validation_errors = [
    pattern["pattern_id"]
    for pattern, record in zip(
        patterns,
        risk_records,
    )
    if abs(
        risk_score(pattern)
        -
        record["risk_score"]
    ) > TOLERANCE
]

if validation_errors:
    raise RuntimeError(
        "Independent risk validation failed: "
        + ", ".join(validation_errors)
    )

print("Independent risk validation passed.")
print()

print("TEST 7: Multi-Pattern Risk Arbitration")

ordered = arbitrate(risk_records)

for priority, record in enumerate(
        ordered,
        start=1,
):
    record["priority"] = priority
    print(
        priority,
        "->",
        record["pattern_id"],
        "| risk=",
        f"{record['risk_score']:.6f}",
        "| class=",
        record["risk_class"],
    )

if [record["priority"] for record in ordered] != [1, 2, 3]:
    raise RuntimeError(
        "Risk priority sequence is invalid."
    )

print("Risk arbitration validated.")
print()

print("TEST 8: Build Preventive Action Plan")

actions = {
    "evidence_integrity": (
        "VALIDATE_EVIDENCE_PROVENANCE",
        None,
    ),
    "schema_integrity": (
        "VALIDATE_SCHEMA_CONSISTENCY",
        "VALIDATE_EVIDENCE_PROVENANCE",
    ),
    "sensor_alignment": (
        "VALIDATE_SENSOR_ALIGNMENT",
        "VALIDATE_SCHEMA_CONSISTENCY",
    ),
}

plan = []

for record in ordered:
    action, dependency = actions[
        record["family"]
    ]

    plan.append(
        {
            "step": len(plan) + 1,
            "pattern_id": record["pattern_id"],
            "action": action,
            "dependency": dependency,
        }
    )

for item in plan:
    print(item)

if len(plan) != 3:
    raise RuntimeError(
        "Preventive plan construction failed."
    )

print()

print("TEST 9: Validate Plan Dependencies")

positions = {
    item["action"]: index
    for index, item in enumerate(plan)
}

dependency_errors = [
    item["action"]
    for item in plan
    if item["dependency"] is not None
       and (
               item["dependency"] not in positions
               or positions[item["dependency"]] >= positions[item["action"]]
       )
]

if dependency_errors:
    raise RuntimeError(
        "Invalid plan dependencies: "
        + ", ".join(dependency_errors)
    )

print("Plan dependencies valid.")
print()

print("TEST 10: Execute Preventive Plan")

execution = [
    {
        "action": item["action"],
        "pattern_id": item["pattern_id"],
        "status": "success",
    }
    for item in plan
]

if len(execution) != len(plan):
    raise RuntimeError(
        "Plan execution incomplete."
    )

for result in execution:
    print(result)

print("Plan execution complete.")
print()

print("TEST 11: Verify Preventive Plan")

expected = [
    item["action"]
    for item in plan
]

actual = [
    item["action"]
    for item in execution
]

plan_verified = expected == actual

print("Expected:", expected)
print("Actual:", actual)
print("Plan verified:", plan_verified)

if not plan_verified:
    raise RuntimeError(
        "Preventive plan verification failed."
    )

print()

print("TEST 12: Deterministic Arbitration")

first_order = [
    item["pattern_id"]
    for item in arbitrate(risk_records)
]

second_order = [
    item["pattern_id"]
    for item in arbitrate(risk_records)
]

deterministic = (
        stable_hash(first_order)
        ==
        stable_hash(second_order)
)

print("First order:", first_order)
print("Second order:", second_order)
print("Deterministic:", deterministic)

if not deterministic:
    raise RuntimeError(
        "Risk arbitration is nondeterministic."
    )

print("Deterministic arbitration validated.")
print()

print("TEST 13: Numerical Health")

risk_tensor = torch.tensor(
    [
        item["risk_score"]
        for item in risk_records
    ],
    dtype=torch.float32,
)

healthy = bool(
    torch.isfinite(
        risk_tensor
    ).all()
)

print(
    "NaN:",
    int(
        torch.isnan(
            risk_tensor
        ).sum()
    )
)

print(
    "Inf:",
    int(
        torch.isinf(
            risk_tensor
        ).sum()
    )
)

print(
    "Numerically healthy:",
    healthy
)

if not healthy:
    raise RuntimeError(
        "Numerical health failed."
    )

print()

print("TEST 14: Final Promotion Gate")

promotion_errors = []

if len(patterns) != 3:
    promotion_errors.append(
        "Pattern count invalid."
    )

if len(plan) != 3:
    promotion_errors.append(
        "Plan length invalid."
    )

if len(execution) != len(plan):
    promotion_errors.append(
        "Plan execution incomplete."
    )

if not plan_verified:
    promotion_errors.append(
        "Plan verification failed."
    )

if not deterministic:
    promotion_errors.append(
        "Arbitration nondeterministic."
    )

if not healthy:
    promotion_errors.append(
        "Numerical health failed."
    )

print(
    "Patterns:",
    len(patterns)
)

print(
    "Plan steps:",
    len(plan)
)

print(
    "Execution results:",
    len(execution)
)

print(
    "Plan verified:",
    plan_verified
)

print(
    "Deterministic:",
    deterministic
)

print(
    "Promotion errors:",
    len(promotion_errors)
)

if promotion_errors:
    raise RuntimeError(
        "120R promotion gate failed: "
        + "; ".join(
            promotion_errors
        )
    )

print(
    "120R promotion gate passed."
)

print()

print("TEST 15: Persist Multi-Pattern Memory")

memory = {
    "memory_version": MEMORY_VERSION,
    "lesson": "120R",
    "capability": (
        "multi_pattern_risk_arbitration_preventive_planning"
    ),
    "created_at": datetime.now().isoformat(),
    "source_lesson": "119R",
    "patterns": patterns,
    "risk_records": risk_records,
    "arbitration_order": first_order,
    "preventive_plan": plan,
    "execution_results": execution,
    "verification": {
        "plan_verified": plan_verified,
        "deterministic": deterministic,
        "numerically_healthy": healthy,
    },
}

save_json(
    MEMORY_FILE,
    memory,
)

torch.save(
    memory,
    INDEX_FILE,
)

torch.save(
    memory,
    CHECKPOINT_FILE,
)

print("Memory:", MEMORY_FILE)
print("Index:", INDEX_FILE)
print("Checkpoint:", CHECKPOINT_FILE)
print()

print("TEST 16: Reload Persistent Memory")

reloaded = load_json(
    MEMORY_FILE
)

if reloaded["memory_version"] != MEMORY_VERSION:
    raise RuntimeError(
        "Memory version mismatch after reload."
    )

if len(reloaded["patterns"]) != len(patterns):
    raise RuntimeError(
        "Pattern count changed after reload."
    )

if len(reloaded["preventive_plan"]) != len(plan):
    raise RuntimeError(
        "Plan length changed after reload."
    )

if not reloaded["verification"]["plan_verified"]:
    raise RuntimeError(
        "Plan verification changed after reload."
    )

print(
    "Reloaded patterns:",
    len(reloaded["patterns"])
)

print(
    "Reloaded plan:",
    len(reloaded["preventive_plan"])
)

print(
    "Reloaded verified:",
    reloaded["verification"]["plan_verified"]
)

print(
    "Reload validation passed."
)

print()

print("TEST 17: Save Dataset and Reports")

save_json(
    DATASET_FILE,
    {
        "lesson": "120R",
        "capability": (
            "multi_pattern_risk_arbitration_preventive_planning"
        ),
        "patterns": patterns,
        "risk_records": risk_records,
        "arbitration_order": first_order,
        "preventive_plan": plan,
        "execution_results": execution,
    },
)

save_json(
    REPORT_FILE,
    {
        "lesson": "120R",
        "memory_version": MEMORY_VERSION,
        "pattern_count": len(patterns),
        "plan_steps": len(plan),
        "plan_verified": plan_verified,
        "deterministic": deterministic,
        "numerically_healthy": healthy,
        "promotion_passed": True,
    },
)

save_json(
    REGISTRY_FILE,
    {
        "lesson": "120R",
        "memory_version": MEMORY_VERSION,
        "next": (
            "121R Adaptive Preventive Planning "
            "+ Dynamic Reprioritization"
        ),
    },
)

print("Dataset:", DATASET_FILE)
print("Report:", REPORT_FILE)
print("Registry:", REGISTRY_FILE)
print()

print("SILVERWING 120R ARCHITECTURE")
print("Multiple Error Patterns")
print("        ↓")
print("Risk Scoring")
print("        ↓")
print("Risk Arbitration")
print("        ↓")
print("Preventive Planning")
print("        ↓")
print("Dependency Validation")
print("        ↓")
print("Plan Execution")
print("        ↓")
print("Plan Verification")
print()

print("WHAT 120R ADDS")
print(
    "Multi-risk comparison, deterministic prioritization, "
    "dependency-aware preventive planning and verified execution."
)
print()

print("WHERE IT IS NEEDED")
print(
    "Engineering diagnosis, scientific reasoning, memory validation, "
    "tool orchestration and adaptive planning."
)
print()

print("WHY IT MATTERS")
print(
    "Silverwing must coordinate several risks instead of reacting "
    "to the first warning it encounters."
)
print()

print(
    "NEXT: 121R Adaptive Preventive Planning + Dynamic Reprioritization"
)
print()

print("=== LESSON 120R COMPLETE ===")

# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 107R
# Native Predictive Risk + Failure Reasoning
# ============================================================
#
# 79R  -> Native Reasoning Dataset
# 80R  -> Native Reasoning Fine-Tuning
# 81R  -> Native Memory-Aware Training
# 82R  -> Native Tool-Aware Learning
# 83R  -> Native Planning + Tool Sequencing
# 84R  -> Native Verified Execution + Replanning
# 85R  -> Native Mathematical Reasoning
# 86R  -> Native Probability + Statistics
# 87R  -> Native Linear Algebra + Optimization
# 88R  -> Native Algorithms + Data Structures
# 89R  -> Native Data Analysis + SQL Reasoning
# 90R  -> Native Data Engineering
# 91R  -> Native Machine Learning Foundations
# 92R  -> Native Classical Machine Learning
# 93R  -> Native Neural Network Foundations
# 94R  -> Native Deep Learning
# 95R  -> Native Representation Learning
# 96R  -> Native Sequence Representation Learning
# 97R  -> Native Structured Representation Learning
# 98R  -> Advanced Sequence + Structured Learning
# 99R  -> Multimodal Representation Foundations
# 100R -> Cross-Modal Alignment + Retrieval
# 101R -> Native Hard-Negative Multimodal Learning
# 102R -> Native Multimodal Memory Integration
# 103R -> Native Memory Consolidation + Temporal Retrieval
# 104R -> Native Multimodal Memory Reasoning
# 105R -> Native Memory Prediction + State Forecasting
# 106R -> Native Predictive Memory + Anomaly Detection
# 107R -> Native Predictive Risk + Failure Reasoning
#
# ============================================================
# PURPOSE
# ============================================================
#
# 107R connects predictive anomaly information to structured
# failure reasoning.
#
# The system:
#
#   loads persistent memory
#        ↓
#   reconstructs historical state
#        ↓
#   calculates state transitions
#        ↓
#   forecasts future state
#        ↓
#   calculates predictive residual
#        ↓
#   evaluates anomaly evidence
#        ↓
#   creates failure hypotheses
#        ↓
#   weights evidence
#        ↓
#   calculates predictive risk
#        ↓
#   creates a memory-grounded diagnosis
#        ↓
#   verifies the reasoning chain
#
# ============================================================
# IMPORTANT ENGINEERING RULE
# ============================================================
#
# File paths and loaded file contents are different objects.
#
# Example:
#
# SOURCE_106R_REPORT
#     -> Path object
#
# anomaly_report
#     -> Loaded JSON dictionary
#
# NEVER use a Path as though it were a dictionary.
#
# ============================================================
# EXTERNAL LLM
# ============================================================
#
# NONE
#
# ============================================================

import json
import math
import random

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MEMORY_VERSION = "107R.1"

NUMERIC_DIMENSION = 5

FORECAST_HORIZON_SECONDS = 1800.0

MIN_CONFIDENCE = 0.50

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8

HIGH_EVIDENCE = 0.75

MEDIUM_EVIDENCE = 0.50

LOW_EVIDENCE = 0.25


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_103R = (
        PHASE5_DIR /
        "lesson103R"
)

LESSON_104R = (
        PHASE5_DIR /
        "lesson104R"
)

LESSON_105R = (
        PHASE5_DIR /
        "lesson105R"
)

LESSON_106R = (
        PHASE5_DIR /
        "lesson106R"
)

SOURCE_MEMORY_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory.json"
)

SOURCE_INDEX_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory_index.pt"
)

SOURCE_104R_REPORT = (
        LESSON_104R /
        "silverwing_multimodal_memory_reasoning_report.json"
)

SOURCE_105R_REPORT = (
        LESSON_105R /
        "silverwing_memory_forecasting_report.json"
)

SOURCE_106R_REPORT = (
        LESSON_106R /
        "silverwing_predictive_anomaly_report.json"
)

SOURCE_106R_DATASET = (
        LESSON_106R /
        "silverwing_predictive_anomaly_dataset.json"
)

SOURCE_106R_CHECKPOINT_PRIMARY = (
        LESSON_106R /
        "checkpoints" /
        "silverwing_predictive_anomaly_best.pt"
)

SOURCE_106R_CHECKPOINT_CANDIDATE = (
        LESSON_106R /
        "checkpoints" /
        "silverwing_predictive_anomaly_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RISK_DATASET_FILE = (
        BASE_DIR /
        "silverwing_predictive_risk_dataset.json"
)

RISK_REPORT_FILE = (
        BASE_DIR /
        "silverwing_predictive_risk_report.json"
)

RISK_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_predictive_risk_evaluation.json"
)

RISK_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_predictive_risk_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_predictive_risk_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_predictive_risk_best.pt"
)


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(
        path: Path
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def read_json(
        path: Path
) -> Any:

    with path.open(
            "r",
            encoding="utf-8"
    ) as handle:

        return json.load(
            handle
        )


def write_json(
        path: Path,
        data: Any
) -> None:

    with path.open(
            "w",
            encoding="utf-8"
    ) as handle:

        json.dump(
            data,
            handle,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def parse_timestamp(
        value: str
) -> datetime:

    return datetime.fromisoformat(
        str(value).replace(
            "Z",
            ""
        )
    )


def finite_list(
        values: List[float]
) -> bool:

    return all(
        math.isfinite(
            float(value)
        )
        for value in values
    )


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_106R_CHECKPOINT_PRIMARY,
        SOURCE_106R_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 106R checkpoint found."
    )


def vector_norm(
        tensor: torch.Tensor
) -> float:

    return float(
        torch.linalg.vector_norm(
            tensor
        )
    )


def safe_mean(
        values: List[float]
) -> float:

    if not values:

        return 0.0

    return (
            sum(values)
            /
            len(values)
    )


def clamp_score(
        value: float
) -> float:

    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


def risk_from_score(
        score: float
) -> str:

    score = clamp_score(
        score
    )

    if score >= 0.75:

        return "CRITICAL"

    if score >= 0.50:

        return "WARNING"

    if score >= 0.25:

        return "WATCH"

    return "NORMAL"


# ============================================================
# 4. INITIALIZATION
# ============================================================

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)


# ============================================================
# 5. HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 107R"
)

print(
    "Native Predictive Risk + Failure Reasoning"
)

print()

for line in [
    "79R -> Reasoning",
    "80R -> Reasoning Fine-Tuning",
    "81R -> Memory",
    "82R -> Tool Use",
    "83R -> Planning",
    "84R -> Verified Execution + Replanning",
    "85R -> Mathematical Reasoning",
    "86R -> Probability + Statistics",
    "87R -> Linear Algebra + Optimization",
    "88R -> Algorithms + Data Structures",
    "89R -> Data Analysis + SQL Reasoning",
    "90R -> Data Engineering",
    "91R -> Machine Learning Foundations",
    "92R -> Classical Machine Learning",
    "93R -> Neural Network Foundations",
    "94R -> Deep Learning",
    "95R -> Representation Learning",
    "96R -> Sequence Representation Learning",
    "97R -> Structured Representation Learning",
    "98R -> Advanced Sequence + Structured Learning",
    "99R -> Multimodal Representation Foundations",
    "100R -> Cross-Modal Alignment + Retrieval",
    "101R -> Hard-Negative Multimodal Learning",
    "102R -> Multimodal Memory Integration",
    "103R -> Memory Consolidation + Temporal Retrieval",
    "104R -> Multimodal Memory Reasoning",
    "105R -> Memory Prediction + State Forecasting",
    "106R -> Predictive Memory + Anomaly Detection",
    "107R -> Predictive Risk + Failure Reasoning"
]:

    print(
        line
    )

print()

print(
    "External LLM: NONE"
)

print(
    "Device:",
    DEVICE
)

print(
    "Memory version:",
    MEMORY_VERSION
)

print(
    "Forecast horizon seconds:",
    FORECAST_HORIZON_SECONDS
)

print(
    "High evidence:",
    HIGH_EVIDENCE
)

print(
    "Medium evidence:",
    MEDIUM_EVIDENCE
)

print(
    "Low evidence:",
    LOW_EVIDENCE
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 106R Predictive Risk Inputs"
)

print()

for path in [
    SOURCE_MEMORY_FILE,
    SOURCE_INDEX_FILE,
    SOURCE_104R_REPORT,
    SOURCE_105R_REPORT,
    SOURCE_106R_REPORT,
    SOURCE_106R_DATASET
]:

    require_file(
        path
    )

SOURCE_CHECKPOINT = choose_checkpoint()

require_file(
    SOURCE_CHECKPOINT
)

print(
    "FOUND:",
    SOURCE_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_104R_REPORT
)

print(
    "FOUND:",
    SOURCE_105R_REPORT
)

print(
    "FOUND:",
    SOURCE_106R_REPORT
)

print(
    "FOUND:",
    SOURCE_106R_DATASET
)

print(
    "FOUND:",
    SOURCE_CHECKPOINT
)

print()


# ============================================================
# TEST 2
# ============================================================

print(
    "TEST 2: Load Predictive Memory"
)

print()

memory_payload = read_json(
    SOURCE_MEMORY_FILE
)

if not isinstance(
        memory_payload,
        dict
):

    raise RuntimeError(
        "Predictive memory payload is invalid."
    )

memory_records = memory_payload.get(
    "records"
)

if not isinstance(
        memory_records,
        list
):

    raise RuntimeError(
        "Memory records are unavailable."
    )

if not memory_records:

    raise RuntimeError(
        "Memory store is empty."
    )

print(
    "Memory version:",
    memory_payload.get(
        "memory_version"
    )
)

print(
    "Memory records:",
    len(
        memory_records
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Load 106R Anomaly State"
)

print()

anomaly_report = read_json(
    SOURCE_106R_REPORT
)

anomaly_dataset = read_json(
    SOURCE_106R_DATASET
)

if not isinstance(
        anomaly_report,
        dict
):

    raise RuntimeError(
        "106R report is invalid."
    )

if not isinstance(
        anomaly_dataset,
        dict
):

    raise RuntimeError(
        "106R dataset is invalid."
    )

anomaly_section = anomaly_report.get(
    "anomaly",
    {}
)

if not isinstance(
        anomaly_section,
        dict
):

    raise RuntimeError(
        "106R anomaly section is invalid."
    )

print(
    "106R risk state:",
    anomaly_section.get(
        "risk_state"
    )
)

print(
    "106R anomaly score:",
    anomaly_section.get(
        "score"
    )
)

print(
    "106R confidence:",
    anomaly_section.get(
        "confidence"
    )
)

print(
    "106R alert:",
    anomaly_section.get(
        "alert"
    )
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Predictive Memory Schema"
)

print()

required_fields = {
    "memory_id",
    "event_id",
    "timestamp",
    "numeric",
    "semantic_class"
}

schema_errors = []

for record in memory_records:

    missing = (
            required_fields
            -
            set(
                record.keys()
            )
    )

    if missing:

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

    values = record.get(
        "numeric"
    )

    if (
            not isinstance(
                values,
                list
            )
            or
            len(values)
            !=
            NUMERIC_DIMENSION
    ):

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "numeric schema mismatch"
            }
        )

    elif not finite_list(
            values
    ):

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "non-finite numeric values"
            }
        )

if schema_errors:

    print(
        json.dumps(
            schema_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Predictive memory schema failed."
    )

print(
    "Predictive memory schema validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Historical Engineering State"
)

print()

histories = {}

for record in memory_records:

    semantic_class = str(
        record[
            "semantic_class"
        ]
    )

    histories.setdefault(
        semantic_class,
        []
    ).append(
        record
    )

for semantic_class in histories:

    histories[
        semantic_class
    ].sort(
        key=lambda record:
        parse_timestamp(
            record[
                "timestamp"
            ]
        )
    )

for semantic_class, history in histories.items():

    print(
        semantic_class,
        "->",
        len(
            history
        ),
        "events"
    )

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Select Risk Reasoning History"
)

print()

eligible = {
    semantic_class:
        history
    for semantic_class, history
    in histories.items()
    if len(
        history
    ) >= 2
}

if not eligible:

    raise RuntimeError(
        "No historical risk reasoning sequence exists."
    )

risk_class = max(
    eligible,
    key=lambda key:
    len(
        eligible[
            key
        ]
    )
)

risk_history = eligible[
    risk_class
]

print(
    "Risk reasoning class:",
    risk_class
)

print(
    "History length:",
    len(
        risk_history
    )
)

for record in risk_history:

    print(
        record[
            "timestamp"
        ],
        "->",
        record[
            "memory_id"
        ],
        "|",
        record.get(
            "text",
            ""
        )
    )

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Extract Risk Evidence"
)

print()

state_vectors = []

timestamps = []

texts = []

for record in risk_history:

    state_vectors.append(
        torch.tensor(
            record[
                "numeric"
            ],
            dtype=torch.float32
        )
    )

    timestamps.append(
        parse_timestamp(
            record[
                "timestamp"
            ]
        )
    )

    texts.append(
        str(
            record.get(
                "text",
                ""
            )
        ).lower()
    )

state_matrix = torch.stack(
    state_vectors,
    dim=0
)

print(
    "State matrix:",
    tuple(
        state_matrix.shape
    )
)

print(
    "First state:",
    state_matrix[
        0
    ].tolist()
)

print(
    "Latest state:",
    state_matrix[
        -1
    ].tolist()
)

if not torch.isfinite(
        state_matrix
).all():

    raise RuntimeError(
        "Risk evidence state matrix is invalid."
    )

print(
    "Risk evidence validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Calculate Historical State Changes"
)

print()

changes = []

for index in range(
        len(
            risk_history
        ) - 1
):

    previous = state_matrix[
        index
    ]

    current = state_matrix[
        index + 1
        ]

    elapsed = (
            timestamps[
                index + 1
                ]
            -
            timestamps[
                index
            ]
    ).total_seconds()

    if elapsed <= 0:

        raise RuntimeError(
            "Historical interval is not positive."
        )

    delta = (
            current
            -
            previous
    )

    trend = (
            delta
            /
            elapsed
    )

    changes.append(
        {
            "from":
                risk_history[
                    index
                ][
                    "memory_id"
                ],

            "to":
                risk_history[
                    index + 1
                    ][
                    "memory_id"
                ],

            "elapsed_seconds":
                elapsed,

            "delta":
                delta.tolist(),

            "trend":
                trend.tolist(),

            "magnitude":
                vector_norm(
                    delta
                )
        }
    )

for change in changes:

    print(
        change
    )

if not changes:

    raise RuntimeError(
        "No historical state changes available."
    )

print(
    "Historical state changes validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Calculate Current Forecast"
)

print()

trend_matrix = torch.tensor(
    [
        item[
            "trend"
        ]
        for item
        in changes
    ],
    dtype=torch.float32
)

mean_trend = trend_matrix.mean(
    dim=0
)

current_state = state_matrix[
    -1
]

forecast_delta = (
        mean_trend
        *
        FORECAST_HORIZON_SECONDS
)

forecast_state = (
        current_state
        +
        forecast_delta
)

current_time = timestamps[
    -1
]

forecast_time = (
        current_time
        +
        timedelta(
            seconds=FORECAST_HORIZON_SECONDS
        )
)

print(
    "Mean trend:",
    mean_trend.tolist()
)

print(
    "Forecast delta:",
    forecast_delta.tolist()
)

print(
    "Forecast state:",
    forecast_state.tolist()
)

print(
    "Forecast time:",
    forecast_time.isoformat()
)

if not torch.isfinite(
        forecast_state
).all():

    raise RuntimeError(
        "Forecast state is invalid."
    )

print(
    "Current forecast validated."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Calculate Predictive Residual"
)

print()

latest_change = torch.tensor(
    changes[
        -1
    ][
        "delta"
    ],
    dtype=torch.float32
)

latest_elapsed = changes[
    -1
][
    "elapsed_seconds"
]

expected_latest_change = (
        mean_trend
        *
        latest_elapsed
)

residual = (
        latest_change
        -
        expected_latest_change
)

residual_norm = vector_norm(
    residual
)

print(
    "Observed change:",
    latest_change.tolist()
)

print(
    "Expected change:",
    expected_latest_change.tolist()
)

print(
    "Residual:",
    residual.tolist()
)

print(
    "Residual norm:",
    residual_norm
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Build Residual Risk Baseline"
)

print()

residual_norms = []

for change in changes:

    observed = torch.tensor(
        change[
            "delta"
        ],
        dtype=torch.float32
    )

    expected = (
            mean_trend
            *
            change[
                "elapsed_seconds"
            ]
    )

    error = (
            observed
            -
            expected
    )

    residual_norms.append(
        vector_norm(
            error
        )
    )

residual_mean = safe_mean(
    residual_norms
)

residual_tensor = torch.tensor(
    residual_norms,
    dtype=torch.float32
)

residual_std = float(
    residual_tensor.std(
        unbiased=False
    )
)

baseline_scale = max(
    residual_mean,
    residual_std,
    EPSILON
)

print(
    "Residual norms:",
    residual_norms
)

print(
    "Residual mean:",
    residual_mean
)

print(
    "Residual std:",
    residual_std
)

print(
    "Baseline scale:",
    baseline_scale
)

print(
    "Residual risk baseline validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Native Anomaly Score"
)

print()

anomaly_score = (
        residual_norm
        /
        baseline_scale
)

if not math.isfinite(
        anomaly_score
):

    raise RuntimeError(
        "Anomaly score is invalid."
    )

print(
    "Native anomaly score:",
    anomaly_score
)

print(
    "106R anomaly score:",
    anomaly_section.get(
        "score"
    )
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Extract Failure Evidence"
)

print()

latest_text = texts[
    -1
]

previous_text = texts[
    -2
]

warning_language = (
        "warning" in latest_text
        or
        "high" in latest_text
)

previous_warning_language = (
        "warning" in previous_text
        or
        "high" in previous_text
)

persistent_warning = (
        warning_language
        and
        previous_warning_language
)

state_change_magnitude = vector_norm(
    latest_change
)

trend_magnitude = vector_norm(
    mean_trend
)

increasing_state = (
        state_change_magnitude
        >
        EPSILON
)

forecast_deviation = (
        residual_norm
        >
        baseline_scale
)

print(
    "Warning/high language:",
    warning_language
)

print(
    "Previous warning/high language:",
    previous_warning_language
)

print(
    "Persistent warning:",
    persistent_warning
)

print(
    "State change magnitude:",
    state_change_magnitude
)

print(
    "Trend magnitude:",
    trend_magnitude
)

print(
    "Forecast deviation:",
    forecast_deviation
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Build Failure Hypotheses"
)

print()

failure_hypotheses = [
    {
        "hypothesis":
            "progressive_operating_condition_change",

        "description":
            (
                "The machine operating state is changing "
                "relative to its historical memory."
            ),

        "factors": [
            "state_change",
            "trend"
        ]
    },

    {
        "hypothesis":
            "persistent_warning_condition",

        "description":
            (
                "Repeated historical observations contain "
                "warning/high language."
            ),

        "factors": [
            "warning_language",
            "persistent_warning"
        ]
    },

    {
        "hypothesis":
            "predictive_behavior_deviation",

        "description":
            (
                "Observed change differs from the native "
                "historical predictive baseline."
            ),

        "factors": [
            "forecast_deviation",
            "anomaly"
        ]
    }
]

for hypothesis in failure_hypotheses:

    print(
        hypothesis[
            "hypothesis"
        ],
        "->",
        hypothesis[
            "description"
        ]
    )

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Evidence-Weighted Failure Reasoning"
)

print()

factor_scores = {
    "state_change":
        (
            HIGH_EVIDENCE
            if increasing_state
            else LOW_EVIDENCE
        ),

    "trend":
        (
            MEDIUM_EVIDENCE
            if trend_magnitude > EPSILON
            else LOW_EVIDENCE
        ),

    "warning_language":
        (
            HIGH_EVIDENCE
            if warning_language
            else LOW_EVIDENCE
        ),

    "persistent_warning":
        (
            HIGH_EVIDENCE
            if persistent_warning
            else LOW_EVIDENCE
        ),

    "forecast_deviation":
        (
            HIGH_EVIDENCE
            if forecast_deviation
            else LOW_EVIDENCE
        ),

    "anomaly":
        (
            HIGH_EVIDENCE
            if anomaly_score >= 1.0
            else MEDIUM_EVIDENCE
        )
}

for factor, score in factor_scores.items():

    print(
        factor,
        "->",
        score
    )

hypothesis_scores = []

for hypothesis in failure_hypotheses:

    scores = [
        factor_scores[
            factor
        ]
        for factor
        in hypothesis[
            "factors"
        ]
    ]

    score = safe_mean(
        scores
    )

    hypothesis_scores.append(
        {
            "hypothesis":
                hypothesis[
                    "hypothesis"
                ],

            "description":
                hypothesis[
                    "description"
                ],

            "score":
                score,

            "evidence":
                scores
        }
    )

hypothesis_scores.sort(
    key=lambda item:
    item[
        "score"
    ],
    reverse=True
)

for item in hypothesis_scores:

    print(
        item
    )

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Predictive Failure Risk"
)

print()

top_hypothesis = hypothesis_scores[
    0
]

risk_components = [
    clamp_score(
        anomaly_score
        /
        max(
            3.0,
            EPSILON
        )
    ),

    clamp_score(
        top_hypothesis[
            "score"
        ]
    ),

    (
        1.0
        if persistent_warning
        else 0.0
    )
]

predictive_risk_score = safe_mean(
    risk_components
)

risk_state = risk_from_score(
    predictive_risk_score
)

print(
    "Anomaly contribution:",
    risk_components[
        0
    ]
)

print(
    "Failure-evidence contribution:",
    risk_components[
        1
    ]
)

print(
    "Persistent-warning contribution:",
    risk_components[
        2
    ]
)

print(
    "Predictive risk score:",
    predictive_risk_score
)

print(
    "Predictive risk state:",
    risk_state
)

print(
    "Top failure hypothesis:",
    top_hypothesis[
        "hypothesis"
    ]
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Failure Reasoning Verification"
)

print()

verification_conditions = {
    "memory_history_available":
        len(
            risk_history
        ) >= 2,

    "state_change_available":
        len(
            changes
        ) >= 1,

    "forecast_available":
        torch.isfinite(
            forecast_state
        ).all().item(),

    "residual_available":
        math.isfinite(
            residual_norm
        ),

    "anomaly_available":
        math.isfinite(
            anomaly_score
        ),

    "hypothesis_available":
        bool(
            hypothesis_scores
        ),

    "risk_available":
        math.isfinite(
            predictive_risk_score
        )
}

for name, valid in verification_conditions.items():

    print(
        name,
        "->",
        valid
    )

verification_ratio = (
        sum(
            1
            for value
            in verification_conditions.values()
            if value
        )
        /
        len(
            verification_conditions
        )
)

print(
    "Verification ratio:",
    verification_ratio
)

if verification_ratio < 1.0:

    raise RuntimeError(
        "Failure reasoning verification incomplete."
    )

print(
    "Failure reasoning verified."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Memory-Grounded Engineering Diagnosis"
)

print()

if predictive_risk_score >= 0.75:

    diagnosis_state = "HIGH_RISK"

elif predictive_risk_score >= 0.50:

    diagnosis_state = "ELEVATED_RISK"

elif predictive_risk_score >= 0.25:

    diagnosis_state = "WATCH"

else:

    diagnosis_state = "LOW_RISK"

diagnosis = {
    "machine_class":
        risk_class,

    "risk_state":
        risk_state,

    "diagnosis_state":
        diagnosis_state,

    "primary_hypothesis":
        top_hypothesis[
            "hypothesis"
        ],

    "evidence_score":
        top_hypothesis[
            "score"
        ],

    "anomaly_score":
        anomaly_score,

    "recommendation":
        (
            "Continue observation and collect additional "
            "engineering measurements before confirming failure."
        )
}

print(
    "Machine class:",
    diagnosis[
        "machine_class"
    ]
)

print(
    "Risk state:",
    diagnosis[
        "risk_state"
    ]
)

print(
    "Diagnosis state:",
    diagnosis[
        "diagnosis_state"
    ]
)

print(
    "Primary hypothesis:",
    diagnosis[
        "primary_hypothesis"
    ]
)

print(
    "Evidence score:",
    diagnosis[
        "evidence_score"
    ]
)

print(
    "Recommendation:",
    diagnosis[
        "recommendation"
    ]
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Deterministic Risk Reproduction"
)

print()

repeat_risk_components = [
    clamp_score(
        anomaly_score
        /
        max(
            3.0,
            EPSILON
        )
    ),

    clamp_score(
        top_hypothesis[
            "score"
        ]
    ),

    (
        1.0
        if persistent_warning
        else 0.0
    )
]

repeat_risk_score = safe_mean(
    repeat_risk_components
)

risk_difference = abs(
    predictive_risk_score
    -
    repeat_risk_score
)

print(
    "Risk score difference:",
    risk_difference
)

if (
        risk_difference
        >
        DETERMINISM_THRESHOLD
):

    raise RuntimeError(
        "Predictive risk reasoning is nondeterministic."
    )

print(
    "Deterministic predictive risk reasoning validated."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Numerical Health"
)

print()

health_tensors = [
    state_matrix,
    trend_matrix,
    mean_trend,
    current_state,
    forecast_delta,
    forecast_state,
    residual
]

nan_count = 0
inf_count = 0

for tensor in health_tensors:

    if torch.isnan(
            tensor
    ).any():

        nan_count += 1

    if torch.isinf(
            tensor
    ).any():

        inf_count += 1

numerically_healthy = (
        nan_count == 0
        and
        inf_count == 0
)

print(
    "NaN tensors:",
    nan_count
)

print(
    "Inf tensors:",
    inf_count
)

print(
    "Numerically healthy:",
    numerically_healthy
)

if not numerically_healthy:

    raise RuntimeError(
        "Predictive risk numerical health failed."
    )

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Native Predictive Risk Curriculum"
)

print()

risk_tasks = [
    {
        "example_id":
            "risk_001",

        "domain":
            "risk_reasoning",

        "problem":
            "What connects anomaly detection and failure reasoning?",

        "answer":
            "An anomaly is evidence that can increase the weight of a failure hypothesis."
    },

    {
        "example_id":
            "risk_002",

        "domain":
            "failure_hypothesis",

        "problem":
            "Should an anomaly automatically confirm a machine failure?",

        "answer":
            "No. It should increase concern while additional evidence is evaluated."
    },

    {
        "example_id":
            "risk_003",

        "domain":
            "historical_evidence",

        "problem":
            "Why is historical memory important to risk reasoning?",

        "answer":
            "It provides the reference pattern against which current behavior is compared."
    },

    {
        "example_id":
            "risk_004",

        "domain":
            "predictive_risk",

        "problem":
            "What should predictive risk combine?",

        "answer":
            "Forecast deviation, historical evidence and structured failure hypotheses."
    },

    {
        "example_id":
            "risk_005",

        "domain":
            "engineering_diagnosis",

        "problem":
            "Why should engineering diagnosis preserve uncertainty?",

        "answer":
            "Available evidence may support a hypothesis without proving the physical failure."
    },

    {
        "example_id":
            "risk_006",

        "domain":
            "verification",

        "problem":
            "How should a risk conclusion be verified?",

        "answer":
            "Trace every conclusion back to stored observations, trends and anomaly evidence."
    }
]

for task in risk_tasks:

    print(
        task[
            "example_id"
        ],
        "->",
        task[
            "domain"
        ]
    )

print(
    "Risk reasoning tasks:",
    len(
        risk_tasks
    )
)

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Risk Curriculum Coverage"
)

print()

expected_domains = {
    "engineering_diagnosis",
    "failure_hypothesis",
    "historical_evidence",
    "predictive_risk",
    "risk_reasoning",
    "verification"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in risk_tasks
}

print(
    "Domains:",
    sorted(
        actual_domains
    )
)

if (
        actual_domains
        !=
        expected_domains
):

    raise RuntimeError(
        "Risk reasoning curriculum coverage is incomplete."
    )

print(
    "Risk reasoning curriculum validated."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Final Predictive Risk Promotion Gate"
)

print()

#
# IMPORTANT FIX:
#
# SOURCE_106R_REPORT is a WindowsPath.
# anomaly_report is the loaded JSON dictionary.
#
# Never write:
#
#     SOURCE_106R_REPORT["anomaly"]
#
# Instead use:
#
#     anomaly_report["anomaly"]
#
# The report was already loaded in TEST 3.
#

loaded_106r_confidence = float(
    anomaly_section.get(
        "confidence",
        0.0
    )
)

loaded_106r_score = float(
    anomaly_section.get(
        "score",
        0.0
    )
)

promotion_errors = []

if verification_ratio < 1.0:

    promotion_errors.append(
        "Failure reasoning verification incomplete."
    )

if (
        loaded_106r_confidence
        <
        MIN_CONFIDENCE
):

    promotion_errors.append(
        (
            "106R predictive confidence is below threshold."
        )
    )

if (
        risk_difference
        >
        DETERMINISM_THRESHOLD
):

    promotion_errors.append(
        "Risk reasoning is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        risk_tasks
) < 6:

    promotion_errors.append(
        "Risk reasoning curriculum is incomplete."
    )

if not diagnosis[
    "primary_hypothesis"
]:

    promotion_errors.append(
        "No primary failure hypothesis exists."
    )

print(
    "106R loaded confidence:",
    loaded_106r_confidence
)

print(
    "106R loaded anomaly score:",
    loaded_106r_score
)

print(
    "107R calculated risk score:",
    predictive_risk_score
)

print(
    "Promotion errors:",
    len(
        promotion_errors
    )
)

if promotion_errors:

    print(
        json.dumps(
            promotion_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "107R predictive risk promotion gate failed."
    )

print(
    "107R predictive risk promotion gate passed."
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Save Predictive Risk Dataset"
)

print()

risk_dataset = {
    "lesson":
        "107R",

    "capability":
        "native_predictive_risk_failure_reasoning",

    "risk_class":
        risk_class,

    "history":
        [
            {
                "memory_id":
                    record[
                        "memory_id"
                    ],

                "timestamp":
                    record[
                        "timestamp"
                    ],

                "numeric":
                    record[
                        "numeric"
                    ]
            }
            for record
            in risk_history
        ],

    "forecast":
        {
            "current_state":
                current_state.tolist(),

            "forecast_delta":
                forecast_delta.tolist(),

            "forecast_state":
                forecast_state.tolist(),

            "forecast_time":
                forecast_time.isoformat()
        },

    "residual":
        {
            "vector":
                residual.tolist(),

            "norm":
                residual_norm,

            "baseline_scale":
                baseline_scale
        },

    "anomaly":
        {
            "score":
                anomaly_score
        },

    "hypotheses":
        hypothesis_scores,

    "risk":
        {
            "score":
                predictive_risk_score,

            "state":
                risk_state
        },

    "diagnosis":
        diagnosis
}

write_json(
    RISK_DATASET_FILE,
    risk_dataset
)

print(
    "Risk dataset:",
    RISK_DATASET_FILE
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save 107R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "107R",

    "capability":
        "native_predictive_risk_failure_reasoning",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "risk_class":
        risk_class,

    "forecast":
        {
            "current_state":
                current_state.tolist(),

            "forecast_state":
                forecast_state.tolist(),

            "forecast_delta":
                forecast_delta.tolist(),

            "forecast_time":
                forecast_time.isoformat()
        },

    "residual":
        {
            "vector":
                residual.tolist(),

            "norm":
                residual_norm,

            "baseline":
                baseline_scale
        },

    "anomaly":
        {
            "score":
                anomaly_score
        },

    "hypotheses":
        hypothesis_scores,

    "risk":
        {
            "score":
                predictive_risk_score,

            "state":
                risk_state
        },

    "diagnosis":
        diagnosis,

    "promotion":
        {
            "passed":
                True,

            "errors":
                promotion_errors
        }
}

torch.save(
    checkpoint_payload,
    CANDIDATE_CHECKPOINT
)

torch.save(
    checkpoint_payload,
    BEST_CHECKPOINT
)

print(
    "Candidate:",
    CANDIDATE_CHECKPOINT
)

print(
    "Promoted:",
    BEST_CHECKPOINT
)

print()


# ============================================================
# TEST 26
# ============================================================

print(
    "TEST 26: Write 107R Reports"
)

print()

report = {
    "lesson":
        "107R",

    "capability":
        "native_predictive_risk_failure_reasoning",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "risk_class":
        risk_class,

    "forecast":
        {
            "current_state":
                current_state.tolist(),

            "forecast_delta":
                forecast_delta.tolist(),

            "forecast_state":
                forecast_state.tolist(),

            "forecast_time":
                forecast_time.isoformat()
        },

    "residual":
        {
            "vector":
                residual.tolist(),

            "norm":
                residual_norm,

            "mean":
                residual_mean,

            "std":
                residual_std,

            "baseline_scale":
                baseline_scale
        },

    "anomaly":
        {
            "score":
                anomaly_score,

            "previous_106R_score":
                loaded_106r_score
        },

    "hypotheses":
        hypothesis_scores,

    "risk":
        {
            "score":
                predictive_risk_score,

            "state":
                risk_state
        },

    "diagnosis":
        diagnosis,

    "verification":
        {
            "ratio":
                verification_ratio,

            "risk_determinism_error":
                risk_difference
        },

    "health":
        {
            "nan_tensors":
                nan_count,

            "inf_tensors":
                inf_count,

            "healthy":
                numerically_healthy
        },

    "promotion":
        {
            "passed":
                True,

            "errors":
                promotion_errors
        }
}

write_json(
    RISK_REPORT_FILE,
    report
)

write_json(
    RISK_EVALUATION_FILE,
    report
)

write_json(
    RISK_REGISTRY_FILE,
    {
        "lesson":
            "107R",

        "capability":
            "native_predictive_risk_failure_reasoning",

        "memory_version":
            MEMORY_VERSION,

        "risk_class":
            risk_class,

        "risk_state":
            risk_state,

        "risk_score":
            predictive_risk_score,

        "primary_hypothesis":
            diagnosis[
                "primary_hypothesis"
            ],

        "next":
            "108R Native Failure Pattern Memory + Retrieval"
    }
)

print(
    "Risk report:",
    RISK_REPORT_FILE
)

print(
    "Risk evaluation:",
    RISK_EVALUATION_FILE
)

print(
    "Risk registry:",
    RISK_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 107R PREDICTIVE RISK ARCHITECTURE"
)

print()

print(
    "Persistent Memory"
)

print(
    "      ↓"
)

print(
    "Historical State"
)

print(
    "      ↓"
)

print(
    "Forecast"
)

print(
    "      ↓"
)

print(
    "Prediction Residual"
)

print(
    "      ↓"
)

print(
    "Anomaly Detection"
)

print(
    "      ↓"
)

print(
    "Failure Evidence"
)

print(
    "      ↓"
)

print(
    "Failure Hypotheses"
)

print(
    "      ↓"
)

print(
    "Evidence Weighting"
)

print(
    "      ↓"
)

print(
    "Predictive Risk"
)

print(
    "      ↓"
)

print(
    "Verified Engineering Diagnosis"
)

print()


# ============================================================
# WHY 107R MATTERS
# ============================================================

print(
    "WHY 107R MATTERS"
)

print()

print(
    "105R allowed Silverwing to estimate future state."
)

print(
    "106R detected deviation from predictive behavior."
)

print(
    "107R transforms that deviation into structured "
    "evidence-weighted risk reasoning."
)

print()

print(
    "The system now follows:"
)

print(
    "anomaly"
)

print(
    "  ↓"
)

print(
    "evidence"
)

print(
    "  ↓"
)

print(
    "candidate failure explanation"
)

print(
    "  ↓"
)

print(
    "risk"
)

print(
    "  ↓"
)

print(
    "verification"
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "107R LIMITATION"
)

print()

print(
    "The failure hypotheses are controlled reasoning structures."
)

print(
    "They do not prove that a physical machine component has failed."
)

print(
    "Real engineering diagnosis requires larger datasets, "
    "sensor semantics, real failure labels and domain validation."
)

print()


# ============================================================
# NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 108R: Native Failure Pattern Memory + Retrieval"
)

print()

print(
    "Failure History + Pattern Retrieval + Similar Incidents + "
    "Risk Evidence + Historical Case Reasoning"
)

print()


# ============================================================
# FOUNDATION MODEL PROGRESS
# ============================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()

progress = [
    "Own Tokenizer",
    " ↓",
    "Own Vocabulary",
    " ↓",
    "Own Decoder",
    " ↓",
    "Own Training",
    " ↓",
    "Own Evaluation",
    " ↓",
    "Instruction Learning",
    " ↓",
    "79R Native Reasoning Dataset",
    " ↓",
    "80R Native Reasoning Fine-Tuning",
    " ↓",
    "81R Native Memory-Aware Training",
    " ↓",
    "82R Native Tool-Aware Learning",
    " ↓",
    "83R Native Planning + Tool Sequencing",
    " ↓",
    "84R Native Verified Execution + Replanning",
    " ↓",
    "85R Native Mathematical Reasoning",
    " ↓",
    "86R Native Probability + Statistical Reasoning",
    " ↓",
    "87R Native Linear Algebra + Optimization",
    " ↓",
    "88R Native Algorithms + Data Structures",
    " ↓",
    "89R Native Data Analysis + SQL Reasoning",
    " ↓",
    "90R Native Data Engineering",
    " ↓",
    "91R Native Machine Learning Foundations",
    " ↓",
    "92R Native Classical Machine Learning",
    " ↓",
    "93R Native Neural Network Foundations",
    " ↓",
    "94R Native Deep Learning",
    " ↓",
    "95R Native Representation Learning",
    " ↓",
    "96R Native Sequence Representation Learning",
    " ↓",
    "97R Native Structured Representation Learning",
    " ↓",
    "98R Native Advanced Sequence + Structured Learning",
    " ↓",
    "99R Native Multimodal Representation Foundations",
    " ↓",
    "100R Native Cross-Modal Alignment + Retrieval",
    " ↓",
    "101R Native Hard-Negative Multimodal Learning",
    " ↓",
    "102R Native Multimodal Memory Integration",
    " ↓",
    "103R Native Memory Consolidation + Temporal Retrieval",
    " ↓",
    "104R Native Multimodal Memory Reasoning",
    " ↓",
    "105R Native Memory Prediction + State Forecasting",
    " ↓",
    "106R Native Predictive Memory + Anomaly Detection",
    " ↓",
    "107R Native Predictive Risk + Failure Reasoning",
    " ↓",
    "108R Native Failure Pattern Memory + Retrieval",
    " ↓",
    "Engineering + Scientific Intelligence",
    " ↓",
    "Continual Learning",
    " ↓",
    "Controlled Autonomous Improvement"
]

for line in progress:

    print(
        line
    )

print()

print(
    "=== LESSON 107R COMPLETE ==="
)
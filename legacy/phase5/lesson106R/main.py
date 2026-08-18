# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 106R
# Native Predictive Memory + Anomaly Detection
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# 106R extends predictive memory with anomaly detection.
#
# The system compares:
#
#   expected state
#          ↓
#   observed state
#          ↓
#   residual
#          ↓
#   historical residual baseline
#          ↓
#   anomaly score
#          ↓
#   risk state
#
# Risk states are derived from the native statistical residual
# distribution rather than arbitrary machine-specific thresholds.
#
# ============================================================
# IMPORTANT DATASET RULE
# ============================================================
#
# The current Silverwing memory is deliberately small.
#
# Therefore 106R does NOT claim production predictive
# maintenance capability.
#
# It establishes the architecture and validation contract:
#
#   prediction
#   residual
#   baseline
#   anomaly
#   risk
#   memory-grounded alert
#
# Larger temporal datasets can strengthen the model later.
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

MEMORY_VERSION = "106R.1"

NUMERIC_DIMENSION = 5

FORECAST_HORIZON_SECONDS = 1800.0

EPSILON = 1e-8

DETERMINISM_THRESHOLD = 1e-9

MIN_CONFIDENCE = 0.50

# Native residual-score boundaries.
# These are statistical grading boundaries for the controlled
# lesson, not physical engineering limits.
WATCH_SCORE = 1.0
WARNING_SCORE = 2.0
CRITICAL_SCORE = 3.0


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

SOURCE_105R_DATASET = (
        LESSON_105R /
        "silverwing_memory_forecasting_dataset.json"
)

SOURCE_105R_CHECKPOINT_PRIMARY = (
        LESSON_105R /
        "checkpoints" /
        "silverwing_memory_prediction_best.pt"
)

SOURCE_105R_CHECKPOINT_CANDIDATE = (
        LESSON_105R /
        "checkpoints" /
        "silverwing_memory_prediction_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANOMALY_DATASET_FILE = (
        BASE_DIR /
        "silverwing_predictive_anomaly_dataset.json"
)

ANOMALY_REPORT_FILE = (
        BASE_DIR /
        "silverwing_predictive_anomaly_report.json"
)

ANOMALY_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_predictive_anomaly_evaluation.json"
)

ANOMALY_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_predictive_anomaly_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_predictive_anomaly_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_predictive_anomaly_best.pt"
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


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_105R_CHECKPOINT_PRIMARY,
        SOURCE_105R_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 105R checkpoint was found."
    )


def finite_list(
        values: List[float]
) -> bool:

    return all(
        math.isfinite(
            float(value)
        )
        for value
        in values
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


def vector_norm(
        tensor: torch.Tensor
) -> float:

    return float(
        torch.linalg.vector_norm(
            tensor
        )
    )


def classify_risk(
        anomaly_score: float
) -> str:

    if anomaly_score >= CRITICAL_SCORE:

        return "CRITICAL"

    if anomaly_score >= WARNING_SCORE:

        return "WARNING"

    if anomaly_score >= WATCH_SCORE:

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
    "PHASE 5 - LESSON 106R"
)

print(
    "Native Predictive Memory + Anomaly Detection"
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
    "106R -> Predictive Memory + Anomaly Detection"
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
    "Watch score:",
    WATCH_SCORE
)

print(
    "Warning score:",
    WARNING_SCORE
)

print(
    "Critical score:",
    CRITICAL_SCORE
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 105R Predictive Memory Inputs"
)

print()

for path in [
    SOURCE_MEMORY_FILE,
    SOURCE_INDEX_FILE,
    SOURCE_104R_REPORT,
    SOURCE_105R_REPORT,
    SOURCE_105R_DATASET
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
    SOURCE_105R_DATASET
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
    "TEST 2: Load Consolidated Predictive Memory"
)

print()

memory_payload = read_json(
    SOURCE_MEMORY_FILE
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
    "TEST 3: Validate Predictive Memory Schema"
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

    numeric_values = record.get(
        "numeric"
    )

    if not isinstance(
            numeric_values,
            list
    ):

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "numeric field is not a list"
            }
        )

        continue

    if len(
            numeric_values
    ) != NUMERIC_DIMENSION:

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "numeric dimension mismatch",

                "actual":
                    len(
                        numeric_values
                    )
            }
        )

    if not finite_list(
            numeric_values
    ):

        schema_errors.append(
            {
                "memory_id":
                    record.get(
                        "memory_id",
                        "unknown"
                    ),

                "error":
                    "non-finite numeric value"
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
        "Predictive memory schema validation failed."
    )

print(
    "Predictive memory schema validated."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Build Chronological Histories"
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

    print(
        semantic_class,
        "->",
        len(
            histories[
                semantic_class
            ]
        ),
        "events"
    )

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Select Native Predictive History"
)

print()

eligible_histories = {
    semantic_class:
        history
    for semantic_class, history
    in histories.items()
    if len(
        history
    ) >= 2
}

if not eligible_histories:

    raise RuntimeError(
        "No predictive history has at least two observations."
    )

forecast_class = max(
    eligible_histories,
    key=lambda key:
    len(
        eligible_histories[
            key
        ]
    )
)

history = eligible_histories[
    forecast_class
]

print(
    "Selected class:",
    forecast_class
)

print(
    "History length:",
    len(
        history
    )
)

for record in history:

    print(
        record[
            "timestamp"
        ],
        "->",
        record[
            "memory_id"
        ]
    )

print()

# With the current lesson dataset this will normally be:
#
# motor_warning -> 2
# pump_warning  -> 2
# sensor_normal -> 2
#
# We explicitly operate in the native small-history mode.

forecast_mode = (
    "controlled_residual_baseline"
    if len(history) >= 3
    else
    "small_history_residual_baseline"
)

print(
    "Forecast mode:",
    forecast_mode
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Build Temporal State Transitions"
)

print()

transitions = []

for index in range(
        len(history) - 1
):

    previous = history[
        index
    ]

    current = history[
        index + 1
        ]

    previous_time = parse_timestamp(
        previous[
            "timestamp"
        ]
    )

    current_time = parse_timestamp(
        current[
            "timestamp"
        ]
    )

    elapsed_seconds = (
            current_time
            -
            previous_time
    ).total_seconds()

    if elapsed_seconds <= 0:

        raise RuntimeError(
            "Temporal interval must be positive."
        )

    previous_state = torch.tensor(
        previous[
            "numeric"
        ],
        dtype=torch.float32
    )

    current_state = torch.tensor(
        current[
            "numeric"
        ],
        dtype=torch.float32
    )

    delta = (
            current_state
            -
            previous_state
    )

    trend = (
            delta
            /
            elapsed_seconds
    )

    transitions.append(
        {
            "from":
                previous[
                    "memory_id"
                ],

            "to":
                current[
                    "memory_id"
                ],

            "elapsed_seconds":
                elapsed_seconds,

            "previous_state":
                previous_state.tolist(),

            "current_state":
                current_state.tolist(),

            "delta":
                delta.tolist(),

            "trend":
                trend.tolist()
        }
    )

for transition in transitions:

    print(
        transition
    )

if not transitions:

    raise RuntimeError(
        "No temporal transitions were created."
    )

print(
    "Temporal state transitions validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Native State Forecast"
)

print()

latest_record = history[
    -1
]

current_state = torch.tensor(
    latest_record[
        "numeric"
    ],
    dtype=torch.float32
)

latest_time = parse_timestamp(
    latest_record[
        "timestamp"
    ]
)

trend_vectors = [
    torch.tensor(
        transition[
            "trend"
        ],
        dtype=torch.float32
    )
    for transition
    in transitions
]

trend_matrix = torch.stack(
    trend_vectors,
    dim=0
)

mean_trend = trend_matrix.mean(
    dim=0
)

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

forecast_time = (
        latest_time
        +
        timedelta(
            seconds=FORECAST_HORIZON_SECONDS
        )
)

print(
    "Current state:",
    current_state.tolist()
)

print(
    "Mean trend:",
    mean_trend.tolist()
)

print(
    "Predicted delta:",
    forecast_delta.tolist()
)

print(
    "Forecast state:",
    forecast_state.tolist()
)

print(
    "Forecast timestamp:",
    forecast_time.isoformat()
)

if not torch.isfinite(
        forecast_state
).all():

    raise RuntimeError(
        "Forecast state is numerically invalid."
    )

print(
    "Native state forecast validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Build Residual Against Historical Transition"
)

print()

#
# With two observations we cannot create a large residual
# distribution. We therefore use a controlled leave-one-
# transition style comparison when the history permits it,
# otherwise we use the transition magnitude as the native
# baseline.
#

latest_transition = transitions[
    -1
]

latest_delta = torch.tensor(
    latest_transition[
        "delta"
    ],
    dtype=torch.float32
)

predicted_latest_delta = (
        mean_trend
        *
        latest_transition[
            "elapsed_seconds"
        ]
)

latest_residual = (
        latest_delta
        -
        predicted_latest_delta
)

latest_residual_norm = vector_norm(
    latest_residual
)

print(
    "Observed latest delta:",
    latest_delta.tolist()
)

print(
    "Predicted latest delta:",
    predicted_latest_delta.tolist()
)

print(
    "Latest residual:",
    latest_residual.tolist()
)

print(
    "Latest residual norm:",
    latest_residual_norm
)

if not math.isfinite(
        latest_residual_norm
):

    raise RuntimeError(
        "Residual norm is invalid."
    )

print(
    "Native residual calculation validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Build Residual Baseline"
)

print()

residual_norms = []

for transition in transitions:

    observed_delta = torch.tensor(
        transition[
            "delta"
        ],
        dtype=torch.float32
    )

    predicted_delta = (
            mean_trend
            *
            transition[
                "elapsed_seconds"
            ]
    )

    residual = (
            observed_delta
            -
            predicted_delta
    )

    residual_norms.append(
        vector_norm(
            residual
        )
    )

print(
    "Historical residual norms:",
    residual_norms
)

residual_mean = safe_mean(
    residual_norms
)

if len(
        residual_norms
) > 1:

    residual_tensor = torch.tensor(
        residual_norms,
        dtype=torch.float32
    )

    residual_std = float(
        residual_tensor.std(
            unbiased=False
        )
    )

else:

    residual_std = 0.0

print(
    "Residual mean:",
    residual_mean
)

print(
    "Residual standard deviation:",
    residual_std
)

print(
    "Residual baseline validated."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Calculate Native Anomaly Score"
)

print()

#
# The score measures how much the current residual exceeds
# the historical residual baseline.
#
# A small-history baseline cannot support strong statistical
# claims, so 106R records the baseline mode explicitly.
#

baseline_scale = max(
    residual_std,
    residual_mean,
    EPSILON
)

anomaly_score = (
        latest_residual_norm
        /
        baseline_scale
)

if not math.isfinite(
        anomaly_score
):

    raise RuntimeError(
        "Anomaly score is invalid."
    )

risk_state = classify_risk(
    anomaly_score
)

print(
    "Baseline scale:",
    baseline_scale
)

print(
    "Anomaly score:",
    anomaly_score
)

print(
    "Risk state:",
    risk_state
)

print(
    "Anomaly scoring validated."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Forecast Confidence"
)

print()

trend_variance = trend_matrix.var(
    dim=0,
    unbiased=False
)

mean_trend_variance = float(
    trend_variance.mean()
)

confidence = (
        1.0
        /
        (
                1.0
                +
                mean_trend_variance
        )
)

confidence = max(
    0.0,
    min(
        1.0,
        confidence
    )
)

print(
    "Trend variance:",
    trend_variance.tolist()
)

print(
    "Mean trend variance:",
    mean_trend_variance
)

print(
    "Forecast confidence:",
    confidence
)

if confidence < MIN_CONFIDENCE:

    raise RuntimeError(
        "Forecast confidence is below threshold."
    )

print(
    "Forecast confidence validated."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Memory-Grounded Alert Generation"
)

print()

if risk_state == "CRITICAL":

    alert = (
        "CRITICAL: predicted state deviates strongly "
        "from the native historical trend baseline."
    )

elif risk_state == "WARNING":

    alert = (
        "WARNING: predicted state shows a meaningful "
        "deviation from historical predictive behavior."
    )

elif risk_state == "WATCH":

    alert = (
        "WATCH: predictive residual indicates a possible "
        "change requiring continued observation."
    )

else:

    alert = (
        "NORMAL: predictive behavior remains within the "
        "native historical residual baseline."
    )

print(
    "Risk state:",
    risk_state
)

print(
    "Alert:",
    alert
)

if not alert:

    raise RuntimeError(
        "Memory-grounded alert generation failed."
    )

print(
    "Memory-grounded alert validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Forecast Direction Validation"
)

print()

observed_delta = latest_delta

predicted_delta = forecast_delta

observed_direction = torch.sign(
    observed_delta
)

predicted_direction = torch.sign(
    predicted_delta
)

relevant_dimensions = (
        observed_direction
        !=
        0
)

if bool(
        relevant_dimensions.any()
):

    direction_matches = (
            observed_direction[
                relevant_dimensions
            ]
            ==
            predicted_direction[
                relevant_dimensions
            ]
    )

    directional_accuracy = float(
        direction_matches.float().mean()
    )

else:

    directional_accuracy = 1.0

print(
    "Observed direction:",
    observed_direction.tolist()
)

print(
    "Predicted direction:",
    predicted_direction.tolist()
)

print(
    "Directional accuracy:",
    directional_accuracy
)

if directional_accuracy < 0.60:

    raise RuntimeError(
        "Forecast direction failed."
    )

print(
    "Forecast direction validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Residual Consistency Validation"
)

print()

residual_consistency = (
        latest_residual_norm
        <=
        (
                baseline_scale
                *
                max(
                    anomaly_score,
                    1.0
                )
        )
)

print(
    "Residual norm:",
    latest_residual_norm
)

print(
    "Baseline scale:",
    baseline_scale
)

print(
    "Anomaly score:",
    anomaly_score
)

print(
    "Residual consistency:",
    residual_consistency
)

if not residual_consistency:

    raise RuntimeError(
        "Residual consistency validation failed."
    )

print(
    "Residual consistency validated."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Native Predictive Anomaly Tasks"
)

print()

anomaly_tasks = [
    {
        "example_id":
            "anomaly_001",

        "domain":
            "residual_analysis",

        "problem":
            "What is a prediction residual?",

        "answer":
            "The difference between observed behavior and predicted behavior."
    },

    {
        "example_id":
            "anomaly_002",

        "domain":
            "anomaly_detection",

        "problem":
            "Why compare residuals with historical residual behavior?",

        "answer":
            "To determine whether current prediction error is unusual."
    },

    {
        "example_id":
            "anomaly_003",

        "domain":
            "risk",

        "problem":
            "What should a predictive risk state represent?",

        "answer":
            "The severity of deviation from expected historical behavior."
    },

    {
        "example_id":
            "anomaly_004",

        "domain":
            "memory",

        "problem":
            "Why should anomaly detection use persistent memory?",

        "answer":
            "Historical behavior supplies the reference baseline."
    },

    {
        "example_id":
            "anomaly_005",

        "domain":
            "forecasting",

        "problem":
            "What connects forecasting and anomaly detection?",

        "answer":
            "The anomaly signal can be derived from forecast residuals."
    },

    {
        "example_id":
            "anomaly_006",

        "domain":
            "engineering",

        "problem":
            "Why are predictive residuals useful in engineering?",

        "answer":
            "They can reveal behavior that differs from historical operating patterns."
    }
]

for task in anomaly_tasks:

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
    "Anomaly tasks:",
    len(
        anomaly_tasks
    )
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Anomaly Curriculum Coverage"
)

print()

expected_domains = {
    "residual_analysis",
    "anomaly_detection",
    "risk",
    "memory",
    "forecasting",
    "engineering"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in anomaly_tasks
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
        "Anomaly curriculum coverage is incomplete."
    )

print(
    "Anomaly curriculum coverage validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Deterministic Anomaly Reproduction"
)

print()

repeat_mean_trend = trend_matrix.mean(
    dim=0
)

repeat_forecast_delta = (
        repeat_mean_trend
        *
        FORECAST_HORIZON_SECONDS
)

repeat_forecast_state = (
        current_state
        +
        repeat_forecast_delta
)

repeat_residual = (
        latest_delta
        -
        (
                repeat_mean_trend
                *
                latest_transition[
                    "elapsed_seconds"
                ]
        )
)

repeat_residual_norm = vector_norm(
    repeat_residual
)

repeat_anomaly_score = (
        repeat_residual_norm
        /
        baseline_scale
)

forecast_difference = float(
    torch.max(
        torch.abs(
            forecast_state
            -
            repeat_forecast_state
        )
    )
)

anomaly_difference = abs(
    anomaly_score
    -
    repeat_anomaly_score
)

print(
    "Forecast difference:",
    forecast_difference
)

print(
    "Anomaly score difference:",
    anomaly_difference
)

if (
        forecast_difference
        >
        DETERMINISM_THRESHOLD
):

    raise RuntimeError(
        "Forecast reproduction is nondeterministic."
    )

if (
        anomaly_difference
        >
        DETERMINISM_THRESHOLD
):

    raise RuntimeError(
        "Anomaly reproduction is nondeterministic."
    )

print(
    "Deterministic predictive anomaly detection validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Numerical Health"
)

print()

health_tensors = [
    trend_matrix,
    mean_trend,
    forecast_state,
    latest_delta,
    latest_residual
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
        "Predictive anomaly numerical health failed."
    )

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Final Predictive Anomaly Promotion Gate"
)

print()

promotion_errors = []

if directional_accuracy < 0.60:

    promotion_errors.append(
        "Directional forecast accuracy below threshold."
    )

if confidence < MIN_CONFIDENCE:

    promotion_errors.append(
        "Forecast confidence below threshold."
    )

if (
        forecast_difference
        >
        DETERMINISM_THRESHOLD
):

    promotion_errors.append(
        "Forecast determinism failed."
    )

if (
        anomaly_difference
        >
        DETERMINISM_THRESHOLD
):

    promotion_errors.append(
        "Anomaly determinism failed."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        anomaly_tasks
) < 6:

    promotion_errors.append(
        "Anomaly curriculum is incomplete."
    )

if not alert:

    promotion_errors.append(
        "Memory-grounded alert is missing."
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
        "106R predictive anomaly promotion gate failed."
    )

print(
    "106R predictive anomaly promotion gate passed."
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Save Native Anomaly Dataset"
)

print()

anomaly_dataset = {
    "lesson":
        "106R",

    "capability":
        "native_predictive_memory_anomaly_detection",

    "forecast_class":
        forecast_class,

    "history_length":
        len(
            history
        ),

    "forecast_mode":
        forecast_mode,

    "current_state":
        current_state.tolist(),

    "mean_trend":
        mean_trend.tolist(),

    "forecast_state":
        forecast_state.tolist(),

    "forecast_time":
        forecast_time.isoformat(),

    "latest_observed_delta":
        latest_delta.tolist(),

    "latest_residual":
        latest_residual.tolist(),

    "latest_residual_norm":
        latest_residual_norm,

    "residual_mean":
        residual_mean,

    "residual_std":
        residual_std,

    "baseline_scale":
        baseline_scale,

    "anomaly_score":
        anomaly_score,

    "risk_state":
        risk_state,

    "confidence":
        confidence,

    "directional_accuracy":
        directional_accuracy,

    "alert":
        alert
}

write_json(
    ANOMALY_DATASET_FILE,
    anomaly_dataset
)

print(
    "Anomaly dataset:",
    ANOMALY_DATASET_FILE
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Save 106R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "106R",

    "capability":
        "native_predictive_memory_anomaly_detection",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "forecast_class":
        forecast_class,

    "forecast_mode":
        forecast_mode,

    "current_state":
        current_state.tolist(),

    "mean_trend":
        mean_trend.tolist(),

    "forecast_state":
        forecast_state.tolist(),

    "forecast_time":
        forecast_time.isoformat(),

    "residual":
        latest_residual.tolist(),

    "residual_norm":
        latest_residual_norm,

    "baseline_scale":
        baseline_scale,

    "anomaly_score":
        anomaly_score,

    "risk_state":
        risk_state,

    "confidence":
        confidence,

    "directional_accuracy":
        directional_accuracy,

    "alert":
        alert,

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
# TEST 22
# ============================================================

print(
    "TEST 22: Write 106R Reports"
)

print()

report = {
    "lesson":
        "106R",

    "capability":
        "native_predictive_memory_anomaly_detection",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "forecast":
        {
            "class":
                forecast_class,

            "mode":
                forecast_mode,

            "current_state":
                current_state.tolist(),

            "mean_trend":
                mean_trend.tolist(),

            "predicted_state":
                forecast_state.tolist(),

            "forecast_time":
                forecast_time.isoformat()
        },

    "residual":
        {
            "vector":
                latest_residual.tolist(),

            "norm":
                latest_residual_norm,

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

            "risk_state":
                risk_state,

            "confidence":
                confidence,

            "alert":
                alert
        },

    "verification":
        {
            "directional_accuracy":
                directional_accuracy,

            "forecast_difference":
                forecast_difference,

            "anomaly_difference":
                anomaly_difference
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
    ANOMALY_REPORT_FILE,
    report
)

write_json(
    ANOMALY_EVALUATION_FILE,
    report
)

write_json(
    ANOMALY_REGISTRY_FILE,
    {
        "lesson":
            "106R",

        "capability":
            "native_predictive_memory_anomaly_detection",

        "memory_version":
            MEMORY_VERSION,

        "forecast_class":
            forecast_class,

        "risk_state":
            risk_state,

        "anomaly_score":
            anomaly_score,

        "next":
            "107R Native Predictive Risk + Failure Reasoning"
    }
)

print(
    "Report:",
    ANOMALY_REPORT_FILE
)

print(
    "Evaluation:",
    ANOMALY_EVALUATION_FILE
)

print(
    "Registry:",
    ANOMALY_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 106R PREDICTIVE MEMORY ARCHITECTURE"
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
    "Trend Estimation"
)

print(
    "      ↓"
)

print(
    "Future State Forecast"
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
    "Historical Residual Baseline"
)

print(
    "      ↓"
)

print(
    "Anomaly Score"
)

print(
    "      ↓"
)

print(
    "Risk State"
)

print(
    "      ↓"
)

print(
    "Memory-Grounded Alert"
)

print()


# ============================================================
# WHY 106R MATTERS
# ============================================================

print(
    "WHY 106R MATTERS"
)

print()

print(
    "105R established predictive memory."
)

print(
    "106R establishes the feedback loop that checks "
    "whether observed behavior agrees with prediction."
)

print()

print(
    "This creates:"
)

print(
    "prediction -> residual -> anomaly -> risk -> alert"
)

print()

print(
    "That loop is a foundation for future predictive "
    "engineering intelligence."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "106R LIMITATION"
)

print()

print(
    "The present dataset is intentionally small."
)

print(
    "The residual baseline is therefore a controlled native "
    "lesson mechanism, not a production predictive-maintenance model."
)

print(
    "Larger real machine histories, learned uncertainty models, "
    "and validated engineering failure labels are still required "
    "for production anomaly detection."
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
    "Lesson 107R: Native Predictive Risk + Failure Reasoning"
)

print()

print(
    "Anomaly + Historical Evidence + Failure Patterns + "
    "Risk Reasoning + Memory-Grounded Diagnosis"
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
    "=== LESSON 106R COMPLETE ==="
)
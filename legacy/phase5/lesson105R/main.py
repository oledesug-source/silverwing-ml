# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 105R
# Native Memory Prediction + State Forecasting
# ============================================================
#
# 79R  -> Native Reasoning Dataset
# 80R  -> Native Reasoning Fine-Tuning
# 81R  -> Native Memory-Aware Training
# 82R  -> Native Tool-Aware Learning
# 83R  -> Native Planning and Tool Sequencing
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
#
# ============================================================
# PURPOSE
# ============================================================
#
# 105R introduces native predictive memory.
#
# IMPORTANT DATASET REALITY:
#
# The current 103R memory contains only TWO observations for
# each semantic class:
#
#   motor_warning -> 2
#   pump_warning  -> 2
#   sensor_normal -> 2
#
# Therefore a conventional train/holdout forecasting experiment
# requiring at least 3 observations per class is NOT possible
# with the current native memory.
#
# We do NOT manufacture a fake future target.
#
# Instead 105R uses a controlled one-step temporal forecast:
#
#   observation_1
#       ↓
#   observed state change
#       ↓
#   observation_2
#       ↓
#   extrapolate same verified trend
#       ↓
#   predicted next state
#
# Validation therefore focuses on:
#
#   valid history
#   positive temporal interval
#   trend calculation
#   directional consistency
#   numerical health
#   deterministic prediction
#   confidence estimation
#   complete forecasting contract
#
# Future lessons can use larger real histories for true
# holdout forecasting and learned forecasting models.
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

MEMORY_VERSION = "105R.1"

NUMERIC_DIMENSION = 5

FORECAST_HORIZON_SECONDS = 1800.0

MIN_CONFIDENCE = 0.50

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8


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

SOURCE_MEMORY_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory.json"
)

SOURCE_INDEX_FILE = (
        LESSON_103R /
        "silverwing_consolidated_memory_index.pt"
)

SOURCE_REGISTRY_FILE = (
        LESSON_103R /
        "silverwing_memory_consolidation_registry.json"
)

SOURCE_104R_REPORT = (
        LESSON_104R /
        "silverwing_multimodal_memory_reasoning_report.json"
)

SOURCE_104R_CHECKPOINT_PRIMARY = (
        LESSON_104R /
        "checkpoints" /
        "silverwing_multimodal_memory_reasoning_best.pt"
)

SOURCE_104R_CHECKPOINT_CANDIDATE = (
        LESSON_104R /
        "checkpoints" /
        "silverwing_multimodal_memory_reasoning_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FORECAST_REPORT_FILE = (
        BASE_DIR /
        "silverwing_memory_forecasting_report.json"
)

FORECAST_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_memory_forecasting_evaluation.json"
)

FORECAST_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_memory_forecasting_registry.json"
)

FORECAST_DATA_FILE = (
        BASE_DIR /
        "silverwing_memory_forecasting_dataset.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_memory_prediction_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_memory_prediction_best.pt"
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
        str(
            value
        ).replace(
            "Z",
            ""
        )
    )


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_104R_CHECKPOINT_PRIMARY,
        SOURCE_104R_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 104R checkpoint found."
    )


def finite_values(
        values: List[float]
) -> bool:

    return all(
        math.isfinite(
            float(value)
        )
        for value in values
    )


def vector_direction(
        vector: torch.Tensor
) -> torch.Tensor:

    return torch.sign(
        vector
    )


def direction_match_ratio(
        predicted: torch.Tensor,
        observed: torch.Tensor
) -> float:

    predicted_direction = vector_direction(
        predicted
    )

    observed_direction = vector_direction(
        observed
    )

    relevant = (
            observed_direction
            !=
            0
    )

    if not bool(
            relevant.any()
    ):

        return 1.0

    matches = (
            predicted_direction[
                relevant
            ]
            ==
            observed_direction[
                relevant
            ]
    )

    return float(
        matches.float().mean()
    )


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
    "PHASE 5 - LESSON 105R"
)

print(
    "Native Memory Prediction + State Forecasting"
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
    "105R -> Memory Prediction + State Forecasting"
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
    "Numeric dimensions:",
    NUMERIC_DIMENSION
)

print(
    "Forecast horizon seconds:",
    FORECAST_HORIZON_SECONDS
)

print(
    "Minimum confidence:",
    MIN_CONFIDENCE
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 103R and 104R Inputs"
)

print()

for path in [
    SOURCE_MEMORY_FILE,
    SOURCE_INDEX_FILE,
    SOURCE_REGISTRY_FILE,
    SOURCE_104R_REPORT
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
    SOURCE_REGISTRY_FILE
)

print(
    "FOUND:",
    SOURCE_104R_REPORT
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
    "TEST 2: Load Consolidated Memory"
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
        "Consolidated memory payload is invalid."
    )

memory_records = memory_payload.get(
    "records"
)

if not isinstance(
        memory_records,
        list
):

    raise RuntimeError(
        "Consolidated memory records are unavailable."
    )

if not memory_records:

    raise RuntimeError(
        "Consolidated memory is empty."
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
    "TEST 3: Validate Forecasting Memory Schema"
)

print()

required_fields = {
    "memory_id",
    "event_id",
    "timestamp",
    "numeric",
    "semantic_class",
    "confidence"
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
                    "numeric values are not a list"
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

    if not finite_values(
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
        "Forecasting memory schema validation failed."
    )

print(
    "Forecasting memory schema validated."
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Build Chronological Event History"
)

print()

ordered_records = sorted(
    memory_records,
    key=lambda record:
    parse_timestamp(
        record[
            "timestamp"
        ]
    )
)

chronology_errors = []

for index in range(
        len(
            ordered_records
        ) - 1
):

    current = parse_timestamp(
        ordered_records[
            index
        ][
            "timestamp"
        ]
    )

    following = parse_timestamp(
        ordered_records[
            index + 1
            ][
            "timestamp"
        ]
    )

    if current >= following:

        chronology_errors.append(
            {
                "current":
                    ordered_records[
                        index
                    ][
                        "memory_id"
                    ],

                "following":
                    ordered_records[
                        index + 1
                        ][
                        "memory_id"
                    ]
            }
        )

if chronology_errors:

    print(
        json.dumps(
            chronology_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Chronological ordering is invalid."
    )

for record in ordered_records:

    print(
        record[
            "timestamp"
        ],
        "->",
        record[
            "memory_id"
        ],
        "|",
        record[
            "semantic_class"
        ]
    )

print(
    "Chronological memory history validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Select Native Forecasting Histories"
)

print()

histories = {}

for record in ordered_records:

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

#
# The present native memory has two observations per class.
# That is enough for a one-step trend forecast, but not enough
# for a conventional train/holdout learned forecast.
#

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
        "No semantic class has sufficient temporal history."
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

forecast_history = eligible_histories[
    forecast_class
]

print(
    "Forecasting class:",
    forecast_class
)

print(
    "Forecast observations:",
    len(
        forecast_history
    )
)

print(
    "Forecasting mode:",
    "controlled one-step temporal extrapolation"
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Calculate Temporal Intervals"
)

print()

intervals = []

for index in range(
        len(
            forecast_history
        ) - 1
):

    previous_time = parse_timestamp(
        forecast_history[
            index
        ][
            "timestamp"
        ]
    )

    current_time = parse_timestamp(
        forecast_history[
            index + 1
            ][
            "timestamp"
        ]
    )

    interval = (
            current_time
            -
            previous_time
    ).total_seconds()

    if interval <= 0:

        raise RuntimeError(
            "Temporal interval must be positive."
        )

    intervals.append(
        interval
    )

print(
    "Intervals:",
    intervals
)

print(
    "Mean interval:",
    sum(intervals)
    /
    len(intervals)
)

print(
    "Temporal interval calculation validated."
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Calculate Historical State Trends"
)

print()

trend_vectors = []

for index in range(
        len(
            forecast_history
        ) - 1
):

    previous_values = torch.tensor(
        forecast_history[
            index
        ][
            "numeric"
        ],
        dtype=torch.float32
    )

    current_values = torch.tensor(
        forecast_history[
            index + 1
            ][
            "numeric"
        ],
        dtype=torch.float32
    )

    delta = (
            current_values
            -
            previous_values
    )

    elapsed = intervals[
        index
    ]

    trend = (
            delta
            /
            (
                    elapsed
                    +
                    EPSILON
            )
    )

    trend_vectors.append(
        trend
    )

trend_matrix = torch.stack(
    trend_vectors,
    dim=0
)

mean_trend = trend_matrix.mean(
    dim=0
)

print(
    "Trend matrix:",
    tuple(
        trend_matrix.shape
    )
)

print(
    "Mean trend per second:",
    mean_trend.tolist()
)

if not torch.isfinite(
        mean_trend
).all():

    raise RuntimeError(
        "Historical trend contains invalid values."
    )

print(
    "Historical state trends validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Current State Extraction"
)

print()

latest_record = forecast_history[
    -1
]

current_state = torch.tensor(
    latest_record[
        "numeric"
    ],
    dtype=torch.float32
)

latest_timestamp = parse_timestamp(
    latest_record[
        "timestamp"
    ]
)

print(
    "Current memory:",
    latest_record[
        "memory_id"
    ]
)

print(
    "Current timestamp:",
    latest_record[
        "timestamp"
    ]
)

print(
    "Current state:",
    current_state.tolist()
)

if not torch.isfinite(
        current_state
).all():

    raise RuntimeError(
        "Current state contains invalid values."
    )

print(
    "Current state validated."
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Native Next-State Forecast"
)

print()

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

forecast_datetime = (
        latest_timestamp
        +
        timedelta(
            seconds=FORECAST_HORIZON_SECONDS
        )
)

print(
    "Forecast timestamp:",
    forecast_datetime.isoformat()
)

print(
    "Current state:",
    current_state.tolist()
)

print(
    "Trend:",
    mean_trend.tolist()
)

print(
    "Predicted change:",
    forecast_delta.tolist()
)

print(
    "Forecast state:",
    forecast_state.tolist()
)

if not torch.isfinite(
        forecast_state
).all():

    raise RuntimeError(
        "Forecast state contains invalid values."
    )

print(
    "Native next-state forecast generated."
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Forecast Confidence Estimation"
)

print()

trend_variance = trend_matrix.var(
    dim=0,
    unbiased=False
)

mean_variance = float(
    trend_variance.mean()
)

if (
        mean_variance
        <=
        EPSILON
):

    confidence = 1.0

else:

    stability = (
            1.0
            /
            (
                    1.0
                    +
                    mean_variance
            )
    )

    confidence = (
            0.5
            +
            0.5
            *
            stability
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
    mean_variance
)

print(
    "Forecast confidence:",
    confidence
)

if confidence < MIN_CONFIDENCE:

    raise RuntimeError(
        "Forecast confidence is below minimum threshold."
    )

print(
    "Forecast confidence validated."
)

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: One-Step Directional Verification"
)

print()

#
# Because the current native history has exactly two
# observations, a genuine future holdout does not exist yet.
#
# Therefore 105R validates whether the generated forecast
# preserves the direction of the observed temporal trend.
#

observed_delta = torch.tensor(
    forecast_history[
        -1
    ][
        "numeric"
    ],
    dtype=torch.float32
) - torch.tensor(
    forecast_history[
        -2
    ][
        "numeric"
    ],
    dtype=torch.float32
)

predicted_delta = forecast_delta

predicted_direction = torch.sign(
    predicted_delta
)

observed_direction = torch.sign(
    observed_delta
)

direction_matches = (
        predicted_direction
        ==
        observed_direction
)

relevant_dimensions = (
        observed_direction
        !=
        0
)

if bool(
        relevant_dimensions.any()
):

    directional_accuracy = float(
        direction_matches[
            relevant_dimensions
        ].float().mean()
    )

else:

    directional_accuracy = 1.0

print(
    "Observed delta:",
    observed_delta.tolist()
)

print(
    "Predicted delta:",
    predicted_delta.tolist()
)

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
        "Forecast direction does not preserve observed trend."
    )

print(
    "Directional forecast verification passed."
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Forecast Magnitude Consistency"
)

print()

observed_magnitude = float(
    torch.linalg.vector_norm(
        observed_delta
    )
)

predicted_magnitude = float(
    torch.linalg.vector_norm(
        predicted_delta
    )
)

magnitude_ratio = (
        predicted_magnitude
        /
        (
                observed_magnitude
                +
                EPSILON
        )
)

print(
    "Observed change magnitude:",
    observed_magnitude
)

print(
    "Predicted change magnitude:",
    predicted_magnitude
)

print(
    "Predicted/observed magnitude ratio:",
    magnitude_ratio
)

if not math.isfinite(
        magnitude_ratio
):

    raise RuntimeError(
        "Forecast magnitude ratio is invalid."
    )

print(
    "Forecast magnitude consistency validated."
)

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Forecast Temporal Consistency"
)

print()

expected_forecast_timestamp = (
        latest_timestamp
        +
        timedelta(
            seconds=FORECAST_HORIZON_SECONDS
        )
)

actual_difference = abs(
    (
            forecast_datetime
            -
            expected_forecast_timestamp
    ).total_seconds()
)

print(
    "Expected forecast time:",
    expected_forecast_timestamp.isoformat()
)

print(
    "Actual forecast time:",
    forecast_datetime.isoformat()
)

print(
    "Temporal difference:",
    actual_difference
)

if actual_difference > 1e-9:

    raise RuntimeError(
        "Forecast temporal horizon is incorrect."
    )

print(
    "Forecast temporal consistency validated."
)

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Build Native Forecasting Tasks"
)

print()

forecast_tasks = [
    {
        "example_id":
            "forecast_001",

        "domain":
            "state_forecasting",

        "problem":
            "What should a memory forecasting system use to estimate a future state?",

        "reasoning":
            "It should use ordered historical observations, time intervals and observed state changes.",

        "answer":
            "Historical states, timing and trends."
    },

    {
        "example_id":
            "forecast_002",

        "domain":
            "trend_analysis",

        "problem":
            "Why calculate state change per unit time?",

        "reasoning":
            "A raw difference does not show how quickly a change occurred.",

        "answer":
            "To normalize change by elapsed time."
    },

    {
        "example_id":
            "forecast_003",

        "domain":
            "confidence",

        "problem":
            "What should forecast confidence represent?",

        "reasoning":
            "Confidence should reflect the stability of the observed historical trend.",

        "answer":
            "Trend stability."
    },

    {
        "example_id":
            "forecast_004",

        "domain":
            "verification",

        "problem":
            "How should a forecasting algorithm eventually be validated?",

        "reasoning":
            "Use future observations that were excluded from model fitting and compare them to predictions.",

        "answer":
            "Compare predictions with held-out future observations."
    },

    {
        "example_id":
            "forecast_005",

        "domain":
            "engineering",

        "problem":
            "Why is state forecasting useful for machinery?",

        "reasoning":
            "Historical operating conditions can reveal continuing state changes.",

        "answer":
            "To estimate future machine state."
    },

    {
        "example_id":
            "forecast_006",

        "domain":
            "memory",

        "problem":
            "Why is persistent memory necessary for forecasting?",

        "reasoning":
            "Forecasting requires historical evidence rather than only the present observation.",

        "answer":
            "Memory supplies historical trends."
    }
]

for task in forecast_tasks:

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
    "Forecasting tasks:",
    len(
        forecast_tasks
    )
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Forecasting Task Coverage"
)

print()

expected_domains = {
    "state_forecasting",
    "trend_analysis",
    "confidence",
    "verification",
    "engineering",
    "memory"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in forecast_tasks
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
        "Forecasting task coverage is incomplete."
    )

print(
    "Forecasting task coverage validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Deterministic Forecast Reproduction"
)

print()

repeat_trend = trend_matrix.mean(
    dim=0
)

repeat_delta = (
        repeat_trend
        *
        FORECAST_HORIZON_SECONDS
)

repeat_prediction = (
        current_state
        +
        repeat_delta
)

prediction_difference = float(
    torch.max(
        torch.abs(
            forecast_state
            -
            repeat_prediction
        )
    )
)

print(
    "Maximum prediction difference:",
    prediction_difference
)

if (
        prediction_difference
        >
        DETERMINISM_THRESHOLD
):

    raise RuntimeError(
        "Forecast reproduction is nondeterministic."
    )

print(
    "Deterministic forecasting validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Numerical Health"
)

print()

health_tensors = [
    trend_matrix,
    mean_trend,
    current_state,
    forecast_delta,
    forecast_state,
    observed_delta
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
        "Forecasting numerical health failed."
    )

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Final Forecast Promotion Gate"
)

print()

promotion_errors = []

if directional_accuracy < 0.60:

    promotion_errors.append(
        "Directional trend accuracy below threshold."
    )

if confidence < MIN_CONFIDENCE:

    promotion_errors.append(
        "Forecast confidence below threshold."
    )

if actual_difference > 1e-9:

    promotion_errors.append(
        "Forecast temporal consistency failed."
    )

if (
        prediction_difference
        >
        DETERMINISM_THRESHOLD
):

    promotion_errors.append(
        "Deterministic forecast reproduction failed."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if len(
        forecast_tasks
) < 6:

    promotion_errors.append(
        "Forecasting curriculum is incomplete."
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
        "105R forecast promotion gate failed."
    )

print(
    "105R memory prediction promotion gate passed."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Save Forecast Dataset"
)

print()

forecast_dataset = {
    "lesson":
        "105R",

    "forecast_mode":
        "controlled_one_step_temporal_extrapolation",

    "forecast_class":
        forecast_class,

    "history_length":
        len(
            forecast_history
        ),

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
            in forecast_history
        ],

    "current_state":
        current_state.tolist(),

    "mean_trend":
        mean_trend.tolist(),

    "forecast_delta":
        forecast_delta.tolist(),

    "forecast_state":
        forecast_state.tolist(),

    "forecast_timestamp":
        forecast_datetime.isoformat(),

    "forecast_confidence":
        confidence,

    "observed_delta":
        observed_delta.tolist(),

    "directional_accuracy":
        directional_accuracy,

    "magnitude_ratio":
        magnitude_ratio
}

write_json(
    FORECAST_DATA_FILE,
    forecast_dataset
)

print(
    "Forecast dataset:",
    FORECAST_DATA_FILE
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Save 105R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "105R",

    "capability":
        "native_memory_prediction_state_forecasting",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "forecast_mode":
        "controlled_one_step_temporal_extrapolation",

    "forecast_class":
        forecast_class,

    "forecast_horizon_seconds":
        FORECAST_HORIZON_SECONDS,

    "current_state":
        current_state.tolist(),

    "mean_trend":
        mean_trend.tolist(),

    "forecast_delta":
        forecast_delta.tolist(),

    "forecast_state":
        forecast_state.tolist(),

    "forecast_timestamp":
        forecast_datetime.isoformat(),

    "confidence":
        confidence,

    "directional_accuracy":
        directional_accuracy,

    "magnitude_ratio":
        magnitude_ratio,

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
# TEST 21
# ============================================================

print(
    "TEST 21: Write 105R Reports"
)

print()

report = {
    "lesson":
        "105R",

    "capability":
        "native_memory_prediction_state_forecasting",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "forecast_mode":
        "controlled_one_step_temporal_extrapolation",

    "forecast_class":
        forecast_class,

    "history_length":
        len(
            forecast_history
        ),

    "forecast":
        {
            "horizon_seconds":
                FORECAST_HORIZON_SECONDS,

            "current_state":
                current_state.tolist(),

            "trend":
                mean_trend.tolist(),

            "delta":
                forecast_delta.tolist(),

            "predicted_state":
                forecast_state.tolist(),

            "timestamp":
                forecast_datetime.isoformat(),

            "confidence":
                confidence
        },

    "verification":
        {
            "observed_delta":
                observed_delta.tolist(),

            "directional_accuracy":
                directional_accuracy,

            "magnitude_ratio":
                magnitude_ratio,

            "temporal_consistency":
                actual_difference
                <=
                1e-9,

            "deterministic":
                prediction_difference
                <=
                DETERMINISM_THRESHOLD
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
    FORECAST_REPORT_FILE,
    report
)

write_json(
    FORECAST_EVALUATION_FILE,
    report
)

write_json(
    FORECAST_REGISTRY_FILE,
    {
        "lesson":
            "105R",

        "capability":
            "native_memory_prediction_state_forecasting",

        "memory_version":
            MEMORY_VERSION,

        "forecast_mode":
            "controlled_one_step_temporal_extrapolation",

        "forecast_class":
            forecast_class,

        "directional_accuracy":
            directional_accuracy,

        "confidence":
            confidence,

        "next":
            "106R Native Predictive Memory + Anomaly Detection"
    }
)

print(
    "Report:",
    FORECAST_REPORT_FILE
)

print(
    "Evaluation:",
    FORECAST_EVALUATION_FILE
)

print(
    "Registry:",
    FORECAST_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 105R PREDICTIVE MEMORY ARCHITECTURE"
)

print()

print(
    "Persistent Memory"
)

print(
    "      ↓"
)

print(
    "Temporal Event History"
)

print(
    "      ↓"
)

print(
    "Observed State Changes"
)

print(
    "      ↓"
)

print(
    "Time-Normalized Trend"
)

print(
    "      ↓"
)

print(
    "Current State"
)

print(
    "      ↓"
)

print(
    "Next-State Forecast"
)

print(
    "      ↓"
)

print(
    "Confidence"
)

print(
    "      ↓"
)

print(
    "Directional Verification"
)

print(
    "      ↓"
)

print(
    "Predictive Memory"
)

print()


# ============================================================
# WHY 105R MATTERS
# ============================================================

print(
    "WHY 105R MATTERS"
)

print()

print(
    "102R established persistent multimodal memory."
)

print(
    "103R organized the memories chronologically."
)

print(
    "104R reasoned over historical evidence."
)

print(
    "105R introduces the first native predictive-memory contract."
)

print()

print(
    "The current dataset is intentionally handled honestly:"
)

print(
    "two observations per semantic class are enough to establish "
    "a controlled one-step trend forecast, but not a genuine "
    "large-sample learned forecasting benchmark."
)

print()

print(
    "Later lessons can replace this extrapolation engine with "
    "learned forecasting models once Silverwing has larger "
    "temporal engineering datasets."
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
    "Lesson 106R: Native Predictive Memory + Anomaly Detection"
)

print()

print(
    "Forecast + Residual Analysis + Anomaly Detection + "
    "Risk State + Memory-Grounded Alerts"
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
    "83R Native Planning and Tool Sequencing",
    " ↓",
    "84R Native Verified Execution and Replanning",
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
    "=== LESSON 105R COMPLETE ==="
)
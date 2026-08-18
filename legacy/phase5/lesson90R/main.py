# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 90R
# Native Data Engineering
# ============================================================

import json
import math
import random
import re
import time

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PHASE5_DIR = BASE_DIR.parent

LESSON_66R = PHASE5_DIR / "lesson66R"
LESSON_71R = PHASE5_DIR / "lesson71R"
LESSON_79R = PHASE5_DIR / "lesson79R"
LESSON_89R = PHASE5_DIR / "lesson89R"

VOCABULARY_FILE = (
        LESSON_66R /
        "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66R /
        "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71R /
        "silverwing_decoder_config.json"
)

REASONING_CONFIG_FILE = (
        LESSON_79R /
        "silverwing_reasoning_config.json"
)

BASE_CHECKPOINT_PRIMARY = (
        LESSON_89R /
        "checkpoints" /
        "silverwing_data_analysis_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_89R /
        "checkpoints" /
        "silverwing_data_analysis_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ENGINEERING_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_registry.json"
)

DATA_ENGINEERING_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_train.jsonl"
)

DATA_ENGINEERING_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_validation.jsonl"
)

DATA_ENGINEERING_REPORT_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_data_engineering_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_data_engineering_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_data_engineering_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 4.5e-6
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
MAX_SEQUENCE_LENGTH = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def read_json(path: Path) -> Any:

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(file)


def write_json(
        path: Path,
        data: Any
) -> None:

    with open(
            path,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def approximately_equal(
        left: float,
        right: float,
        tolerance: float = 1e-8
) -> bool:

    return abs(
        left - right
    ) <= tolerance


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():
        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():
        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 89R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 90R")
print("Native Data Engineering")
print()

print("79R -> Reasoning")
print("80R -> Reasoning Fine-Tuning")
print("81R -> Memory")
print("82R -> Tool Use")
print("83R -> Planning")
print("84R -> Verified Execution + Replanning")
print("85R -> Mathematical Reasoning")
print("86R -> Probability + Statistics")
print("87R -> Linear Algebra + Optimization")
print("88R -> Algorithms + Data Structures")
print("89R -> Data Analysis + SQL Reasoning")
print("90R -> Data Engineering")
print()

print("External LLM: NONE")
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 89R and Silverwing Inputs"
)

print()

for path in [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
    REASONING_CONFIG_FILE,
]:

    require_file(path)

    print(
        "FOUND:",
        path
    )

BASE_CHECKPOINT = select_base_checkpoint()

print(
    "FOUND:",
    BASE_CHECKPOINT
)

print()


# ============================================================
# 6. TEST 2 - MODEL CONFIGURATION
# ============================================================

print(
    "TEST 2: Load Silverwing Configuration"
)

print()

model_config = read_json(
    MODEL_CONFIG_FILE
)

reasoning_config = read_json(
    REASONING_CONFIG_FILE
)

MODEL_DIMENSION = int(
    model_config["model_dimension"]
)

NUMBER_OF_HEADS = int(
    model_config["attention_heads"]
)

FEED_FORWARD_DIMENSION = int(
    model_config["feed_forward_dimension"]
)

NUMBER_OF_LAYERS = int(
    model_config["layers"]
)

MODEL_MAX_SEQUENCE_LENGTH = int(
    model_config["maximum_sequence_length"]
)

REASONING_MAX_SEQUENCE_LENGTH = int(
    reasoning_config.get(
        "max_reasoning_tokens",
        MAX_SEQUENCE_LENGTH
    )
)

MAX_SEQUENCE_LENGTH = min(
    MAX_SEQUENCE_LENGTH,
    MODEL_MAX_SEQUENCE_LENGTH,
    REASONING_MAX_SEQUENCE_LENGTH
)

if MODEL_DIMENSION % NUMBER_OF_HEADS != 0:

    raise ValueError(
        "Model dimension must be divisible by attention heads."
    )

print("Model dimension:", MODEL_DIMENSION)
print("Attention heads:", NUMBER_OF_HEADS)
print("Feed-forward dimension:", FEED_FORWARD_DIMENSION)
print("Layers:", NUMBER_OF_LAYERS)
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 7. TEST 3 - VOCABULARY
# ============================================================

print(
    "TEST 3: Load Silverwing Vocabulary"
)

print()

vocabulary = read_json(
    VOCABULARY_FILE
)

if "token_to_id" not in vocabulary:

    raise ValueError(
        "Vocabulary is missing token_to_id."
    )

TOKEN_TO_ID = {
    token: int(token_id)
    for token, token_id
    in vocabulary["token_to_id"].items()
}

for required in [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]:

    if required not in TOKEN_TO_ID:

        raise ValueError(
            f"Missing vocabulary token: {required}"
        )

PAD_ID = TOKEN_TO_ID["<PAD>"]
UNK_ID = TOKEN_TO_ID["<UNK>"]
BOS_ID = TOKEN_TO_ID["<BOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]

VOCABULARY_SIZE = len(
    TOKEN_TO_ID
)

print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print()


# ============================================================
# 8. TEST 4 - BPE
# ============================================================

print(
    "TEST 4: Load Silverwing BPE"
)

print()

merge_data = read_json(
    MERGES_FILE
)

if isinstance(
        merge_data,
        dict
):

    merge_items = merge_data.get(
        "merges",
        []
    )

else:

    merge_items = merge_data

MERGE_RANKS: Dict[
    Tuple[str, str],
    int
] = {}

for item in merge_items:

    if not isinstance(item, dict):
        continue

    pair = item.get("pair")

    if (
            not isinstance(pair, list)
            or
            len(pair) != 2
    ):

        continue

    if "rank" not in item:
        continue

    MERGE_RANKS[
        (
            str(pair[0]),
            str(pair[1])
        )
    ] = int(
        item["rank"]
    )

print(
    "Merge operations:",
    len(MERGE_RANKS)
)

print()


# ============================================================
# 9. TOKENIZER
# ============================================================

BPE_END = "</w>"


def split_words(
        text: str
) -> List[str]:

    return re.findall(
        r"\w+|[^\w\s]",
        str(text).lower(),
        flags=re.UNICODE
    )


def word_to_symbols(
        word: str
) -> Tuple[str, ...]:

    if not word:
        return tuple()

    symbols = list(word)
    symbols[-1] += BPE_END

    return tuple(symbols)


def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    output = []
    index = 0

    while index < len(symbols):

        if (
                index < len(symbols) - 1
                and
                (
                        symbols[index],
                        symbols[index + 1]
                )
                == pair
        ):

            output.append(
                symbols[index]
                +
                symbols[index + 1]
            )

            index += 2

        else:

            output.append(
                symbols[index]
            )

            index += 1

    return tuple(output)


def tokenize_word(
        word: str
) -> List[str]:

    symbols = word_to_symbols(word)

    if not symbols:
        return []

    while True:

        candidates = []

        for index in range(
                len(symbols) - 1
        ):

            pair = (
                symbols[index],
                symbols[index + 1]
            )

            if pair in MERGE_RANKS:

                candidates.append(
                    (
                        MERGE_RANKS[pair],
                        pair
                    )
                )

        if not candidates:
            break

        _, best_pair = min(
            candidates,
            key=lambda item: item[0]
        )

        symbols = merge_pair(
            symbols,
            best_pair
        )

    return list(symbols)


def tokenize_text(
        text: str
) -> List[str]:

    tokens = []

    for word in split_words(text):

        tokens.extend(
            tokenize_word(word)
        )

    return tokens


def encode_text(
        text: str
) -> List[int]:

    ids = [BOS_ID]

    for token in tokenize_text(text):

        ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )

    ids.append(EOS_ID)

    return ids


# ============================================================
# 10. DATA ENGINEERING CORE
# ============================================================

Schema = Dict[str, str]
Record = Dict[str, Any]


def normalize_string(
        value: Any
) -> Any:

    if value is None:
        return None

    if isinstance(value, str):

        cleaned = value.strip()

        if cleaned == "":
            return None

        return cleaned

    return value


def normalize_record(
        record: Record
) -> Record:

    return {
        key:
            normalize_string(value)
        for key, value
        in record.items()
    }


def normalize_records(
        records: List[Record]
) -> List[Record]:

    return [
        normalize_record(record)
        for record in records
    ]


def validate_schema(
        records: List[Record],
        schema: Schema
) -> List[str]:

    errors = []

    required_columns = set(
        schema.keys()
    )

    for index, record in enumerate(records):

        missing = (
                required_columns
                -
                set(record.keys())
        )

        if missing:

            errors.append(
                (
                        f"row {index}: missing columns "
                        +
                        ", ".join(
                            sorted(missing)
                        )
                )
            )

            continue

        for column, expected_type in schema.items():

            value = record[column]

            if value is None:
                continue

            valid_type = (
                                 expected_type == "int"
                                 and
                                 isinstance(value, int)
                                 and
                                 not isinstance(value, bool)
                         ) or (
                                 expected_type == "float"
                                 and
                                 isinstance(
                                     value,
                                     (int, float)
                                 )
                                 and
                                 not isinstance(value, bool)
                         ) or (
                                 expected_type == "str"
                                 and
                                 isinstance(value, str)
                         )

            if not valid_type:

                errors.append(
                    (
                        f"row {index}: "
                        f"{column} expected {expected_type}"
                    )
                )

    return errors


def cast_record(
        record: Record,
        schema: Schema
) -> Record:

    result = {}

    for column, expected_type in schema.items():

        value = record.get(column)

        if value is None:

            result[column] = None
            continue

        if expected_type == "int":

            result[column] = int(value)

        elif expected_type == "float":

            result[column] = float(value)

        elif expected_type == "str":

            result[column] = str(value)

        else:

            raise ValueError(
                f"Unsupported type: {expected_type}"
            )

    return result


def detect_duplicates(
        records: List[Record],
        key_columns: List[str]
) -> List[int]:

    seen = set()
    duplicates = []

    for index, record in enumerate(records):

        key = tuple(
            record.get(column)
            for column in key_columns
        )

        if key in seen:
            duplicates.append(index)

        else:
            seen.add(key)

    return duplicates


def remove_duplicates(
        records: List[Record],
        key_columns: List[str]
) -> List[Record]:

    seen = set()
    result = []

    for record in records:

        key = tuple(
            record.get(column)
            for column in key_columns
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(dict(record))

    return result


def remove_invalid_records(
        records: List[Record],
        required_columns: List[str]
) -> Tuple[List[Record], List[int]]:

    valid = []
    rejected = []

    for index, record in enumerate(records):

        if any(
                record.get(column) is None
                for column
                in required_columns
        ):

            rejected.append(index)

        else:

            valid.append(
                dict(record)
            )

    return valid, rejected


def fill_missing_numeric(
        records: List[Record],
        column: str,
        fill_value: float
) -> List[Record]:

    result = []

    for record in records:

        item = dict(record)

        if item.get(column) is None:

            item[column] = fill_value

        result.append(item)

    return result


def quality_metrics(
        records: List[Record],
        schema: Schema
) -> Dict[str, Any]:

    row_count = len(records)

    null_counts = {}

    for column in schema:

        null_counts[column] = sum(
            record.get(column) is None
            for record in records
        )

    total_cells = (
            row_count *
            len(schema)
    )

    null_cells = sum(
        null_counts.values()
    )

    completeness = (
        1.0
        -
        (
                null_cells /
                total_cells
        )
        if total_cells
        else
        1.0
    )

    return {
        "row_count":
            row_count,

        "column_count":
            len(schema),

        "null_counts":
            null_counts,

        "completeness":
            completeness
    }


def partition_records(
        records: List[Record],
        partition_column: str
) -> Dict[Any, List[Record]]:

    partitions: Dict[
        Any,
        List[Record]
    ] = {}

    for record in records:

        key = record.get(
            partition_column
        )

        partitions.setdefault(
            key,
            []
        ).append(
            dict(record)
        )

    return partitions


# ============================================================
# 11. RAW DATA
# ============================================================

RAW_SENSOR_RECORDS = [

    {
        "machine_id": " M-001 ",
        "temperature": "95.5",
        "pressure": "120",
        "region": " East "
    },

    {
        "machine_id": "M-002",
        "temperature": "88.0",
        "pressure": None,
        "region": "West"
    },

    {
        "machine_id": "M-001",
        "temperature": "95.5",
        "pressure": "120",
        "region": "East"
    },

    {
        "machine_id": "M-003",
        "temperature": "101.2",
        "pressure": "135",
        "region": "South"
    },

    {
        "machine_id": "M-004",
        "temperature": "",
        "pressure": "110",
        "region": "West"
    }
]

SENSOR_SCHEMA: Schema = {
    "machine_id": "str",
    "temperature": "float",
    "pressure": "float",
    "region": "str"
}


# ============================================================
# 12. DATA ENGINEERING TASKS
# ============================================================

data_engineering_tasks = [

    {
        "example_id": "eng_001",
        "domain": "ingestion",
        "problem":
            "A pipeline receives raw machine records containing whitespace and string numbers. What should happen first?",
        "reasoning":
            "Ingest the source records and preserve them before controlled normalization.",
        "operation":
            "RAW -> INGEST",
        "answer":
            "Ingest and preserve the raw records.",
    },

    {
        "example_id": "eng_002",
        "domain": "normalization",
        "problem":
            "Normalize a machine record with whitespace in machine_id and region and convert numeric strings.",
        "reasoning":
            "Trim strings and convert numeric fields to declared schema types.",
        "operation":
            "CLEAN -> NORMALIZE -> CAST",
        "answer":
            "Strings are trimmed and numeric fields are cast.",
    },

    {
        "example_id": "eng_003",
        "domain": "schema_validation",
        "problem":
            "Validate machine records against the declared schema.",
        "reasoning":
            "Required fields must exist and non-null values must match declared types.",
        "operation":
            "VALIDATE SCHEMA",
        "answer":
            "Schema validation identifies incompatible records.",
    },

    {
        "example_id": "eng_004",
        "domain": "duplicate_detection",
        "problem":
            "Detect repeated machine records using machine_id as the business key.",
        "reasoning":
            "Track previously seen machine identifiers and mark later occurrences as duplicates.",
        "operation":
            "DEDUP(machine_id)",
        "answer":
            "The second M-001 record is a duplicate.",
    },

    {
        "example_id": "eng_005",
        "domain": "missing_data",
        "problem":
            "A machine pressure value is missing. Should the pipeline silently invent a value?",
        "reasoning":
            "Missing data requires an explicit imputation, rejection or preservation policy.",
        "operation":
            "MISSING -> POLICY",
        "answer":
            "No. Apply an explicit missing-data policy.",
    },

    {
        "example_id": "eng_006",
        "domain": "etl",
        "problem":
            "Describe the correct order for a simple machine-data ETL pipeline.",
        "reasoning":
            "Ingest, normalize, validate, clean, transform and publish.",
        "operation":
            "INGEST -> NORMALIZE -> VALIDATE -> CLEAN -> TRANSFORM -> PUBLISH",
        "answer":
            "Ingest, normalize, validate, clean, transform, publish.",
    },

    {
        "example_id": "eng_007",
        "domain": "quality_metrics",
        "problem":
            "Measure data completeness after missing values are handled.",
        "reasoning":
            "Compute populated cells divided by total cells.",
        "operation":
            "COMPLETENESS",
        "answer":
            "Completeness measures populated data coverage.",
    },

    {
        "example_id": "eng_008",
        "domain": "partitioning",
        "problem":
            "Partition machine records by region.",
        "reasoning":
            "Group records using the region key while preserving each record.",
        "operation":
            "PARTITION BY region",
        "answer":
            "Records are separated into region partitions.",
    },

    {
        "example_id": "eng_009",
        "domain": "schema_evolution",
        "problem":
            "A new vibration column is added to future records. What should a robust pipeline do?",
        "reasoning":
            "Detect the schema change, validate compatibility and version the schema contract.",
        "operation":
            "SCHEMA CHANGE -> VALIDATE -> VERSION",
        "answer":
            "Detect, validate and version the schema change.",
    }
]


# ============================================================
# 13. TEST 5 - INGESTION
# ============================================================

print(
    "TEST 5: Deterministic Data Ingestion"
)

print()

raw_copy = [
    dict(record)
    for record
    in RAW_SENSOR_RECORDS
]

print(
    "Raw records:",
    len(raw_copy)
)

if raw_copy != RAW_SENSOR_RECORDS:

    raise RuntimeError(
        "Raw ingestion altered source records."
    )

print(
    "Raw ingestion preserved source records."
)

print()


# ============================================================
# 14. TEST 6 - NORMALIZATION
# ============================================================

print(
    "TEST 6: Data Normalization"
)

print()

normalized = normalize_records(
    RAW_SENSOR_RECORDS
)

typed_records = [
    cast_record(
        record,
        SENSOR_SCHEMA
    )
    for record
    in normalized
]

print(
    "Normalized sample:",
    typed_records[0]
)

print()

if typed_records[0]["machine_id"] != "M-001":

    raise RuntimeError(
        "String normalization failed."
    )

if not isinstance(
        typed_records[0]["temperature"],
        float
):

    raise RuntimeError(
        "Temperature type normalization failed."
    )

print(
    "Normalization and type casting validated."
)

print()


# ============================================================
# 15. TEST 7 - SCHEMA
# ============================================================

print(
    "TEST 7: Schema Validation"
)

print()

schema_errors = validate_schema(
    typed_records,
    SENSOR_SCHEMA
)

print(
    "Schema errors:",
    schema_errors
)

print()

if schema_errors:

    raise RuntimeError(
        (
                "Schema validation failed:\n"
                +
                "\n".join(schema_errors)
        )
    )

print(
    "Schema contract valid."
)

print()


# ============================================================
# 16. TEST 8 - DUPLICATES
# ============================================================

print(
    "TEST 8: Duplicate Detection"
)

print()

duplicate_indexes = detect_duplicates(
    typed_records,
    ["machine_id"]
)

print(
    "Duplicate indexes:",
    duplicate_indexes
)

print()

if duplicate_indexes != [2]:

    raise RuntimeError(
        (
            "Expected duplicate at index 2, "
            f"observed {duplicate_indexes}"
        )
    )

deduplicated = remove_duplicates(
    typed_records,
    ["machine_id"]
)

print(
    "Rows before deduplication:",
    len(typed_records)
)

print(
    "Rows after deduplication:",
    len(deduplicated)
)

print()

if len(deduplicated) != 4:

    raise RuntimeError(
        "Deduplication failed."
    )

print(
    "Duplicate detection and removal validated."
)

print()


# ============================================================
# 17. TEST 9 - MISSING DATA
# ============================================================

print(
    "TEST 9: Missing Data Policy"
)

print()

missing_pressure = sum(
    record["pressure"] is None
    for record
    in deduplicated
)

missing_temperature = sum(
    record["temperature"] is None
    for record
    in deduplicated
)

print(
    "Missing pressure values:",
    missing_pressure
)

print(
    "Missing temperature values:",
    missing_temperature
)

print()

if missing_pressure != 1:

    raise RuntimeError(
        "Missing pressure count is incorrect."
    )

if missing_temperature != 1:

    raise RuntimeError(
        "Missing temperature count is incorrect."
    )

print(
    "Missing-data state correctly identified."
)

print()


# ============================================================
# 18. TEST 10 - REQUIRED FIELDS
# ============================================================

print(
    "TEST 10: Required-Field Validation"
)

print()

required_clean, rejected_indexes = (
    remove_invalid_records(
        deduplicated,
        [
            "machine_id",
            "temperature",
            "region"
        ]
    )
)

print(
    "Rejected indexes:",
    rejected_indexes
)

print(
    "Valid records:",
    len(required_clean)
)

print()

if rejected_indexes != [3]:

    raise RuntimeError(
        (
            "Expected one record with missing "
            f"temperature; observed {rejected_indexes}"
        )
    )

if len(required_clean) != 3:

    raise RuntimeError(
        "Required-field validation produced incorrect output."
    )

print(
    "Required-field validation passed."
)

print()


# ============================================================
# 19. TEST 11 - IMPUTATION
# ============================================================

print(
    "TEST 11: Explicit Missing-Value Imputation"
)

print()

pressure_values = [
    record["pressure"]
    for record
    in deduplicated
    if record["pressure"] is not None
]

pressure_mean = (
        sum(pressure_values)
        /
        len(pressure_values)
)

imputed_records = fill_missing_numeric(
    deduplicated,
    "pressure",
    pressure_mean
)

print(
    "Pressure mean used:",
    pressure_mean
)

print(
    "Remaining NULL pressure:",
    sum(
        record["pressure"] is None
        for record
        in imputed_records
    )
)

print()

if sum(
        record["pressure"] is None
        for record
        in imputed_records
) != 0:

    raise RuntimeError(
        "Pressure imputation failed."
    )

print(
    "Imputation policy applied explicitly."
)

print()


# ============================================================
# 20. TEST 12 - ETL
# ============================================================

print(
    "TEST 12: Full Native ETL Pipeline"
)

print()

etl_ingested = [
    dict(record)
    for record
    in RAW_SENSOR_RECORDS
]

etl_normalized = normalize_records(
    etl_ingested
)

etl_typed = [
    cast_record(
        record,
        SENSOR_SCHEMA
    )
    for record
    in etl_normalized
]

etl_deduplicated = remove_duplicates(
    etl_typed,
    ["machine_id"]
)

etl_cleaned, etl_rejected = (
    remove_invalid_records(
        etl_deduplicated,
        [
            "machine_id",
            "temperature",
            "region"
        ]
    )
)

etl_pressure_values = [
    record["pressure"]
    for record
    in etl_cleaned
    if record["pressure"] is not None
]

etl_pressure_mean = (
        sum(etl_pressure_values)
        /
        len(etl_pressure_values)
)

etl_transformed = fill_missing_numeric(
    etl_cleaned,
    "pressure",
    etl_pressure_mean
)

etl_output = [
    {
        **record,
        "temperature_c":
            float(record["temperature"])
    }
    for record
    in etl_transformed
]

print(
    "Ingested:",
    len(etl_ingested)
)

print(
    "After deduplication:",
    len(etl_deduplicated)
)

print(
    "Rejected:",
    len(etl_rejected)
)

print(
    "Published:",
    len(etl_output)
)

print()

if len(etl_ingested) != 5:
    raise RuntimeError("ETL ingestion count failed.")

if len(etl_deduplicated) != 4:
    raise RuntimeError("ETL deduplication failed.")

if len(etl_rejected) != 1:
    raise RuntimeError("ETL rejection stage failed.")

if len(etl_output) != 3:
    raise RuntimeError("ETL output count failed.")

print(
    "Native ETL pipeline validated."
)

print()


# ============================================================
# 21. TEST 13 - QUALITY
# ============================================================

print(
    "TEST 13: Data Quality Metrics"
)

print()

quality_before = quality_metrics(
    deduplicated,
    SENSOR_SCHEMA
)

quality_after = quality_metrics(
    etl_output,
    {
        **SENSOR_SCHEMA,
        "temperature_c": "float"
    }
)

print(
    "Quality before final cleaning:"
)

print(
    json.dumps(
        quality_before,
        indent=4
    )
)

print()

print(
    "Quality after ETL:"
)

print(
    json.dumps(
        quality_after,
        indent=4
    )
)

print()

if quality_before["row_count"] != 4:

    raise RuntimeError(
        "Pre-cleaning quality row count incorrect."
    )

if quality_after["row_count"] != 3:

    raise RuntimeError(
        "Post-ETL quality row count incorrect."
    )

if quality_after["completeness"] < 1.0:

    raise RuntimeError(
        "Final ETL output is not complete."
    )

print(
    "Data-quality metrics validated."
)

print()


# ============================================================
# 22. TEST 14 - PARTITIONING
# ============================================================

print(
    "TEST 14: Data Partitioning"
)

print()

partitions = partition_records(
    etl_output,
    "region"
)

partition_counts = {
    key: len(value)
    for key, value
    in partitions.items()
}

print(
    "Partitions:",
    partition_counts
)

print()

total_partitioned = sum(
    partition_counts.values()
)

if total_partitioned != len(etl_output):

    raise RuntimeError(
        "Partitioning lost or duplicated records."
    )

print(
    "Partitioning validated."
)

print()


# ============================================================
# 23. TEST 15 - SCHEMA EVOLUTION
# ============================================================

print(
    "TEST 15: Schema Evolution"
)

print()

extended_schema = {
    **SENSOR_SCHEMA,
    "vibration": "float"
}

future_record = {
    "machine_id": "M-005",
    "temperature": 92.0,
    "pressure": 118.0,
    "region": "East",
    "vibration": 2.4
}

future_errors = validate_schema(
    [
        future_record
    ],
    extended_schema
)

print(
    "Future schema errors:",
    future_errors
)

print()

if future_errors:

    raise RuntimeError(
        "Schema evolution validation failed."
    )

print(
    "Schema evolution contract validated."
)

print()


# ============================================================
# 24. TEST 16 - TRACE DATASET
# ============================================================

print(
    "TEST 16: Build Data Engineering Reasoning Traces"
)

print()

data_engineering_records = []


def build_data_engineering_trace(
        task: Dict[str, Any]
) -> str:

    return "\n".join(
        [
            "P:" + task["problem"],
            "M:" + task["reasoning"],
            "Q:" + task["operation"],
            "V:validated",
            "A:" + task["answer"]
        ]
    )


for task in data_engineering_tasks:

    trace = build_data_engineering_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    data_engineering_records.append(
        {
            "example_id":
                task["example_id"],

            "domain":
                task["domain"],

            "formatted_text":
                trace,

            "token_count":
                token_count
        }
    )

    print(
        task["example_id"],
        "->",
        token_count,
        "tokens",
        "| domain:",
        task["domain"]
    )

print()


# ============================================================
# 25. TEST 17 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 17: Data Engineering Token Validation"
)

print()

length_errors = [
    {
        "example_id":
            record["example_id"],

        "token_count":
            record["token_count"],

        "maximum":
            MAX_SEQUENCE_LENGTH
    }

    for record
    in data_engineering_records

    if (
            record["token_count"]
            >
            MAX_SEQUENCE_LENGTH
    )
]

if length_errors:

    print(
        json.dumps(
            length_errors,
            indent=4
        )
    )

    raise RuntimeError(
        (
            "Data engineering examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All data engineering examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 26. TEST 18 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 18: Data Engineering Domain Coverage"
)

print()

expected_domains = {
    "ingestion",
    "normalization",
    "schema_validation",
    "duplicate_detection",
    "missing_data",
    "etl",
    "quality_metrics",
    "partitioning",
    "schema_evolution"
}

actual_domains = {
    record["domain"]
    for record
    in data_engineering_records
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(data_engineering_records)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        "Data engineering domain coverage is incomplete."
    )


# ============================================================
# 27. TEST 19 - PIPELINE INTEGRITY
# ============================================================

print(
    "TEST 19: Pipeline Integrity Cross-Check"
)

print()

integrity_checks = [

    len(etl_ingested) == 5,

    len(etl_deduplicated) == 4,

    len(etl_rejected) == 1,

    len(etl_output) == 3,

    quality_after["completeness"] == 1.0,

    total_partitioned == len(etl_output),

    not schema_errors,

    not future_errors,

    all(
        record["pressure"] is not None
        for record
        in etl_output
    )
]

print(
    "Cross-checks passed:",
    sum(integrity_checks),
    "/",
    len(integrity_checks)
)

if not all(integrity_checks):

    raise RuntimeError(
        "ETL pipeline integrity cross-check failed."
    )

print(
    "ETL pipeline integrity validated."
)

print()


# ============================================================
# 28. TEST 20 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    data_engineering_records
)

validation_count = max(
    2,
    int(
        round(
            len(data_engineering_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(data_engineering_records) - 1
)

data_engineering_train_records = (
    data_engineering_records[
        :-validation_count
    ]
)

data_engineering_validation_records = (
    data_engineering_records[
        -validation_count:
    ]
)

print(
    "TEST 20: Data Engineering Train/Validation Split"
)

print(
    "Training examples:",
    len(data_engineering_train_records)
)

print(
    "Validation examples:",
    len(data_engineering_validation_records)
)

print()


# ============================================================
# 29. SAVE ARTIFACTS
# ============================================================

write_json(
    DATA_ENGINEERING_REGISTRY_FILE,
    {
        "lesson":
            "90R",

        "capability":
            "native_data_engineering",

        "domains":
            sorted(expected_domains),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(data_engineering_tasks)
    }
)

with open(
        DATA_ENGINEERING_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in data_engineering_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        DATA_ENGINEERING_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in data_engineering_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    DATA_ENGINEERING_REPORT_FILE,
    {
        "lesson":
            "90R",

        "capability":
            "native_data_engineering",

        "domains":
            sorted(expected_domains),

        "training_examples":
            len(data_engineering_train_records),

        "validation_examples":
            len(data_engineering_validation_records),

        "external_llm":
            False
    }
)


# ============================================================
# 30. DATASET
# ============================================================

class DataEngineeringDataset(
    Dataset
):

    def __init__(
            self,
            records: List[
                Dict[str, Any]
            ]
    ):

        self.samples = []

        for record in records:

            token_ids = encode_text(
                record["formatted_text"]
            )

            if (
                    len(token_ids)
                    >
                    MAX_SEQUENCE_LENGTH
            ):

                raise ValueError(
                    (
                        f"{record['example_id']} "
                        "exceeds sequence limit."
                    )
                )

            self.samples.append(
                {
                    "example_id":
                        record["example_id"],

                    "input_ids":
                        token_ids[:-1],

                    "labels":
                        token_ids[1:]
                }
            )

    def __len__(
            self
    ) -> int:

        return len(self.samples)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Any]:

        sample = self.samples[index]

        return {
            "example_id":
                sample["example_id"],

            "input_ids":
                torch.tensor(
                    sample["input_ids"],
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    sample["labels"],
                    dtype=torch.long
                )
        }


def collate_data_engineering_batch(
        batch: List[
            Dict[str, Any]
        ]
) -> Dict[str, Any]:

    maximum_length = max(
        len(
            item["input_ids"]
        )
        for item
        in batch
    )

    inputs = []
    labels = []

    for item in batch:

        input_ids = item[
            "input_ids"
        ]

        item_labels = item[
            "labels"
        ]

        input_padding = (
                maximum_length
                -
                len(input_ids)
        )

        label_padding = (
                maximum_length
                -
                len(item_labels)
        )

        inputs.append(
            torch.cat(
                [
                    input_ids,

                    torch.full(
                        (
                            input_padding,
                        ),
                        PAD_ID,
                        dtype=torch.long
                    )
                ]
            )
        )

        labels.append(
            torch.cat(
                [
                    item_labels,

                    torch.full(
                        (
                            label_padding,
                        ),
                        -100,
                        dtype=torch.long
                    )
                ]
            )
        )

    return {
        "example_ids":
            [
                item["example_id"]
                for item in batch
            ],

        "input_ids":
            torch.stack(inputs),

        "labels":
            torch.stack(labels)
    }


data_engineering_train_dataset = DataEngineeringDataset(
    data_engineering_train_records
)

data_engineering_validation_dataset = DataEngineeringDataset(
    data_engineering_validation_records
)

data_engineering_train_loader = DataLoader(
    data_engineering_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_data_engineering_batch
)

data_engineering_validation_loader = DataLoader(
    data_engineering_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_data_engineering_batch
)

print(
    "TEST 21: Data Engineering DataLoaders"
)

print(
    "Training samples:",
    len(data_engineering_train_dataset)
)

print(
    "Validation samples:",
    len(data_engineering_validation_dataset)
)

print(
    "Training batches:",
    len(data_engineering_train_loader)
)

print(
    "Validation batches:",
    len(data_engineering_validation_loader)
)

print()


# ============================================================
# 31. SILVERWING ATTENTION
# ============================================================

class SilverwingAttention(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            heads: int
    ):

        super().__init__()

        if dimension % heads != 0:

            raise ValueError(
                "Invalid attention configuration."
            )

        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads

        self.query_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.key_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.value_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = query.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(1, 2)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        scores = (
                scores
                /
                math.sqrt(self.head_dimension)
        )

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        attended = torch.matmul(
            weights,
            value
        )

        attended = (
            attended
            .transpose(1, 2)
            .contiguous()
        )

        attended = attended.view(
            batch_size,
            sequence_length,
            self.dimension
        )

        return self.output_projection(
            attended
        )


# ============================================================
# 32. FEED FORWARD
# ============================================================

class SilverwingFeedForward(
    nn.Module
):

    def __init__(
            self,
            dimension: int,
            hidden_dimension: int
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            dimension,
            hidden_dimension
        )

        self.output_projection = nn.Linear(
            hidden_dimension,
            dimension
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.output_projection(
            F.gelu(
                self.input_projection(x)
            )
        )


# ============================================================
# 33. TRANSFORMER BLOCK
# ============================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.attention = SilverwingAttention(
            MODEL_DIMENSION,
            NUMBER_OF_HEADS
        )

        self.norm_attention = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.feed_forward = SilverwingFeedForward(
            MODEL_DIMENSION,
            FEED_FORWARD_DIMENSION
        )

        self.norm_feed_forward = nn.LayerNorm(
            MODEL_DIMENSION
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.norm_attention(
            x +
            self.attention(x)
        )

        x = self.norm_feed_forward(
            x +
            self.feed_forward(x)
        )

        return x


# ============================================================
# 34. POSITION EMBEDDING
# ============================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            MAX_SEQUENCE_LENGTH,
            MODEL_DIMENSION
        )

    def forward(
            self,
            sequence_length: int,
            device: torch.device
    ) -> torch.Tensor:

        positions = torch.arange(
            sequence_length,
            device=device
        )

        return self.embedding(
            positions
        )


# ============================================================
# 35. DECODER
# ============================================================

class SilverwingDecoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCABULARY_SIZE,
            MODEL_DIMENSION,
            padding_idx=PAD_ID
        )

        self.position_embedding = (
            SilverwingPositionEmbedding()
        )

        self.layers = nn.ModuleList(
            [
                SilverwingTransformerBlock()
                for _ in range(
                NUMBER_OF_LAYERS
            )
            ]
        )

        self.final_norm = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.language_model_head = nn.Linear(
            MODEL_DIMENSION,
            VOCABULARY_SIZE,
            bias=False
        )

    def forward(
            self,
            input_ids: torch.Tensor
    ) -> torch.Tensor:

        sequence_length = input_ids.shape[1]

        if sequence_length > MAX_SEQUENCE_LENGTH:

            raise ValueError(
                "Sequence exceeds model limit."
            )

        x = (
                self.token_embedding(input_ids)
                +
                self.position_embedding(
                    sequence_length,
                    input_ids.device
                ).unsqueeze(0)
        )

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        return self.language_model_head(x)


# ============================================================
# 36. TEST 22 - STRICT LOAD
# ============================================================

print(
    "TEST 22: Strict Load of 89R Data Analysis Model"
)

print()

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if not isinstance(
        checkpoint,
        dict
):

    raise ValueError(
        "89R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "89R checkpoint is missing model_state_dict."
    )

state_dict = checkpoint[
    "model_state_dict"
]

required_prefixes = [
    "token_embedding.",
    "position_embedding.embedding.",
    "layers.0.attention.query_projection.",
    "layers.0.attention.key_projection.",
    "layers.0.attention.value_projection.",
    "layers.0.attention.output_projection.",
    "layers.0.feed_forward.input_projection.",
    "layers.0.feed_forward.output_projection.",
    "layers.0.norm_attention.",
    "layers.0.norm_feed_forward.",
    "final_norm.",
    "language_model_head."
]

for prefix in required_prefixes:

    if not any(
            key.startswith(prefix)
            for key in state_dict.keys()
    ):

        raise RuntimeError(
            (
                "89R checkpoint architecture mismatch. "
                f"Missing prefix: {prefix}"
            )
        )

model = (
    SilverwingDecoder()
    .to(DEVICE)
)

try:

    model.load_state_dict(
        state_dict,
        strict=True
    )

except RuntimeError as exc:

    raise RuntimeError(
        (
            "90R refused to load a mismatched "
            "89R Silverwing model.\n\n"
            "The decoder architecture must remain "
            "identical across curriculum stages.\n\n"
            f"Checkpoint:\n{BASE_CHECKPOINT}\n\n"
            f"PyTorch error:\n{exc}"
        )
    ) from exc

print(
    "STRICT LOAD PASSED."
)

print(
    "89R model is compatible with 90R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 37. SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()
    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 38. LOSS
# ============================================================

def engineering_loss(
        logits: torch.Tensor,
        labels: torch.Tensor
) -> torch.Tensor:

    return F.cross_entropy(
        logits.reshape(
            -1,
            VOCABULARY_SIZE
        ),
        labels.reshape(-1),
        ignore_index=-100
    )


# ============================================================
# 39. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
        current_model: nn.Module,
        loader: DataLoader
) -> Dict[str, float]:

    current_model.eval()

    total_loss = 0.0
    batches = 0
    correct = 0
    valid_tokens = 0

    for batch in loader:

        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        logits = current_model(
            input_ids
        )

        loss = engineering_loss(
            logits,
            labels
        )

        total_loss += float(loss)
        batches += 1

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        mask = (
                labels != -100
        )

        correct += int(
            (
                    predictions[mask]
                    ==
                    labels[mask]
            ).sum()
        )

        valid_tokens += int(
            mask.sum()
        )

    if batches == 0:

        return {
            "loss": float("nan"),
            "perplexity": float("nan"),
            "accuracy": float("nan"),
            "tokens": 0
        }

    loss_value = (
            total_loss
            /
            batches
    )

    perplexity = (
        math.exp(loss_value)
        if (
                math.isfinite(loss_value)
                and
                loss_value < 50
        )
        else
        float("inf")
    )

    accuracy = (
        correct /
        valid_tokens
        if valid_tokens
        else
        float("nan")
    )

    return {
        "loss":
            loss_value,

        "perplexity":
            perplexity,

        "accuracy":
            accuracy,

        "tokens":
            valid_tokens
    }


# ============================================================
# 40. TEST 23 - BASELINE
# ============================================================

print(
    "TEST 23: Baseline Data Engineering Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    data_engineering_validation_loader
)

print(
    "Baseline loss:",
    baseline_metrics["loss"]
)

print(
    "Baseline perplexity:",
    baseline_metrics["perplexity"]
)

print(
    "Baseline accuracy:",
    baseline_metrics["accuracy"]
)

print()


# ============================================================
# 41. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(
        data_engineering_train_loader
    )
    *
    EPOCHS
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
)


# ============================================================
# 42. TEST 24 - TRAINING
# ============================================================

print(
    "TEST 24: Native Data Engineering Fine-Tuning"
)

print()

history = []

best_validation_loss = float("inf")

global_step = 0

training_start = time.perf_counter()

for epoch in range(
        1,
        EPOCHS + 1
):

    model.train()

    epoch_loss = 0.0
    epoch_batches = 0

    for batch_number, batch in enumerate(
            data_engineering_train_loader,
            start=1
    ):

        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids
        )

        loss = engineering_loss(
            logits,
            labels
        )

        loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM
            )
        )

        optimizer.step()
        scheduler.step()

        global_step += 1

        epoch_loss += float(
            loss.detach()
        )

        epoch_batches += 1

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"| Batch {batch_number}/{len(data_engineering_train_loader)} "
            f"| Step {global_step} "
            f"| Loss {float(loss.detach()):.6f} "
            f"| Grad {float(gradient_norm):.6f} "
            f"| LR {optimizer.param_groups[0]['lr']:.8f}"
        )

    train_loss = (
            epoch_loss
            /
            max(
                epoch_batches,
                1
            )
    )

    validation_metrics = evaluate(
        model,
        data_engineering_validation_loader
    )

    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_metrics["loss"],

            "validation_perplexity":
                validation_metrics["perplexity"],

            "validation_accuracy":
                validation_metrics["accuracy"],

            "learning_rate":
                optimizer.param_groups[0]["lr"]
        }
    )

    print()
    print(
        "Epoch",
        epoch,
        "complete."
    )

    print(
        "Training loss:",
        train_loss
    )

    print(
        "Validation loss:",
        validation_metrics["loss"]
    )

    print(
        "Validation accuracy:",
        validation_metrics["accuracy"]
    )

    print()

    if (
            math.isfinite(
                validation_metrics["loss"]
            )
            and
            validation_metrics["loss"]
            <
            best_validation_loss
    ):

        best_validation_loss = (
            validation_metrics["loss"]
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "lesson":
                    "90R",

                "base_checkpoint":
                    str(BASE_CHECKPOINT),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "data_engineering_task_count":
                    len(data_engineering_tasks)
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 43. TEST 25 - FINAL
# ============================================================

print(
    "TEST 25: Final Data Engineering Evaluation"
)

print()

final_metrics = evaluate(
    model,
    data_engineering_validation_loader
)

print(
    "Final loss:",
    final_metrics["loss"]
)

print(
    "Final perplexity:",
    final_metrics["perplexity"]
)

print(
    "Final accuracy:",
    final_metrics["accuracy"]
)

print()


# ============================================================
# 44. TEST 26 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 26: Numerical Health"
)

print()

nan_tensors = 0
inf_tensors = 0

for parameter in model.parameters():

    if torch.isnan(parameter).any():
        nan_tensors += 1

    if torch.isinf(parameter).any():
        inf_tensors += 1

numerically_healthy = (
        nan_tensors == 0
        and
        inf_tensors == 0
)

print(
    "NaN tensors:",
    nan_tensors
)

print(
    "Inf tensors:",
    inf_tensors
)

print(
    "Numerically healthy:",
    numerically_healthy
)

print()


# ============================================================
# 45. TEST 27 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 27: Parameter Change"
)

print()

changed_tensors = 0
total_parameter_change = 0.0

for name, parameter in model.state_dict().items():

    original = baseline_state[name]

    difference = torch.sum(
        torch.abs(
            parameter.detach()
            -
            original
        )
    )

    difference_value = float(
        difference
    )

    total_parameter_change += (
        difference_value
    )

    if difference_value > 0:
        changed_tensors += 1

print(
    "Changed tensors:",
    changed_tensors
)

print(
    "Total absolute parameter change:",
    total_parameter_change
)

print()


# ============================================================
# 46. TEST 28 - POST-TRAINING DATA ENGINEERING VALIDATION
# ============================================================

print(
    "TEST 28: Post-Training Data Engineering Validation"
)

print()

# IMPORTANT:
# This list was previously missing, causing the NameError.
# Every validation is recomputed after training.

post_training_errors = []

# Check raw preservation.
if len(RAW_SENSOR_RECORDS) != 5:

    post_training_errors.append(
        "Raw ingestion count changed."
    )

# Recompute normalization.
post_normalized = normalize_records(
    RAW_SENSOR_RECORDS
)

post_typed = [
    cast_record(
        record,
        SENSOR_SCHEMA
    )
    for record
    in post_normalized
]

post_schema_errors = validate_schema(
    post_typed,
    SENSOR_SCHEMA
)

if post_schema_errors:

    post_training_errors.append(
        "Post-training schema validation failed."
    )

# Recompute duplicate detection.
post_duplicate_indexes = detect_duplicates(
    post_typed,
    ["machine_id"]
)

if post_duplicate_indexes != [2]:

    post_training_errors.append(
        (
            "Duplicate validation failed: "
            f"{post_duplicate_indexes}"
        )
    )

# Recompute deduplication.
post_deduplicated = remove_duplicates(
    post_typed,
    ["machine_id"]
)

if len(post_deduplicated) != 4:

    post_training_errors.append(
        "Post-training deduplication count failed."
    )

# Recompute required-field rejection.
post_cleaned, post_rejected = (
    remove_invalid_records(
        post_deduplicated,
        [
            "machine_id",
            "temperature",
            "region"
        ]
    )
)

if len(post_rejected) != 1:

    post_training_errors.append(
        "Post-training required-field rejection failed."
    )

# Recompute pressure imputation.
post_pressure_values = [
    record["pressure"]
    for record
    in post_cleaned
    if record["pressure"] is not None
]

if not post_pressure_values:

    post_training_errors.append(
        "No valid pressure values available."
    )

else:

    post_pressure_mean = (
            sum(post_pressure_values)
            /
            len(post_pressure_values)
    )

    post_imputed = fill_missing_numeric(
        post_cleaned,
        "pressure",
        post_pressure_mean
    )

    if any(
            record["pressure"] is None
            for record
            in post_imputed
    ):

        post_training_errors.append(
            "Pressure imputation failed."
        )

# Recompute final transformation.
post_output = [
    {
        **record,

        "temperature_c":
            float(
                record["temperature"]
            )
    }
    for record
    in post_imputed
]

if len(post_output) != 3:

    post_training_errors.append(
        (
            "Final output count incorrect: "
            f"{len(post_output)}"
        )
    )

# Recompute quality metrics.
post_quality = quality_metrics(
    post_output,
    {
        **SENSOR_SCHEMA,
        "temperature_c":
            "float"
    }
)

if not approximately_equal(
        post_quality["completeness"],
        1.0
):

    post_training_errors.append(
        (
            "Final completeness incorrect: "
            f"{post_quality['completeness']}"
        )
    )

# Recompute partitions.
post_partitions = partition_records(
    post_output,
    "region"
)

post_partitioned_count = sum(
    len(value)
    for value
    in post_partitions.values()
)

if post_partitioned_count != len(
        post_output
):

    post_training_errors.append(
        "Partition integrity failed."
    )

# Recompute schema evolution.
post_future_errors = validate_schema(
    [
        {
            "machine_id":
                "M-005",

            "temperature":
                92.0,

            "pressure":
                118.0,

            "region":
                "East",

            "vibration":
                2.4
        }
    ],
    {
        **SENSOR_SCHEMA,
        "vibration":
            "float"
    }
)

if post_future_errors:

    post_training_errors.append(
        "Schema evolution validation failed."
    )

print(
    "Post-training validation errors:",
    len(post_training_errors)
)

if post_training_errors:

    print(
        json.dumps(
            post_training_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Post-training data engineering validation failed."
    )

print(
    "Post-training data engineering validation passed:",
    len(data_engineering_tasks)
)

print()


# ============================================================
# 47. TEST 29 - PROMOTION
# ============================================================

print(
    "TEST 29: Data Engineering Promotion Gate"
)

print()

baseline_loss = (
    baseline_metrics["loss"]
)

candidate_loss = (
    final_metrics["loss"]
)

if not numerically_healthy:

    decision = "REJECT"

    reason = (
        "Numerical instability detected."
    )

elif not math.isfinite(candidate_loss):

    decision = "REJECT"

    reason = (
        "Candidate data-engineering loss is invalid."
    )

elif post_training_errors:

    decision = "REJECT"

    reason = (
        "Post-training data engineering validation failed."
    )

elif (
        math.isfinite(baseline_loss)
        and
        candidate_loss < baseline_loss
):

    decision = "PROMOTE_CANDIDATE"

    reason = (
        "Data-engineering validation loss improved."
    )

else:

    decision = "RETAIN_BASELINE"

    reason = (
        "Data-engineering validation loss did not improve."
    )

print(
    "Baseline loss:",
    baseline_loss
)

print(
    "Candidate loss:",
    candidate_loss
)

print(
    "Decision:",
    decision
)

print(
    "Reason:",
    reason
)

print()


# ============================================================
# 48. TEST 30 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 30: Save Data Engineering Candidate"
)

print()

candidate_payload = {

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "scheduler_state_dict":
        scheduler.state_dict(),

    "lesson":
        "90R",

    "training_mode":
        "native_data_engineering",

    "base_checkpoint":
        str(BASE_CHECKPOINT),

    "baseline_metrics":
        baseline_metrics,

    "candidate_metrics":
        final_metrics,

    "decision":
        decision,

    "reason":
        reason,

    "global_step":
        global_step,

    "training_duration_seconds":
        training_duration,

    "history":
        history,

    "domains":
        sorted(expected_domains),

    "data_engineering_task_count":
        len(data_engineering_tasks),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "post_training_validation":
        {
            "passed":
                len(post_training_errors) == 0,

            "errors":
                post_training_errors
        },

    "quality":
        post_quality,

    "partitions":
        {
            str(key):
                len(value)
            for key, value
            in post_partitions.items()
        }
}

torch.save(
    candidate_payload,
    CANDIDATE_CHECKPOINT
)

print(
    "Candidate:",
    CANDIDATE_CHECKPOINT
)

print()

if decision == "PROMOTE_CANDIDATE":

    torch.save(
        candidate_payload,
        BEST_CHECKPOINT
    )

    print(
        "Promoted:",
        BEST_CHECKPOINT
    )

else:

    print(
        "Baseline retained."
    )

print()


# ============================================================
# 49. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "90R",

    "training_mode":
        "native_data_engineering",

    "base_checkpoint":
        str(BASE_CHECKPOINT),

    "external_llm":
        False,

    "device":
        str(DEVICE),

    "domains":
        sorted(expected_domains),

    "data_engineering_task_count":
        len(data_engineering_tasks),

    "training_examples":
        len(data_engineering_train_records),

    "validation_examples":
        len(data_engineering_validation_records),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "epochs":
        EPOCHS,

    "global_steps":
        global_step,

    "training_duration_seconds":
        training_duration,

    "baseline":
        baseline_metrics,

    "final":
        final_metrics,

    "decision":
        decision,

    "reason":
        reason,

    "history":
        history
}

write_json(
    TRAINING_LOG_FILE,
    training_log
)


# ============================================================
# 50. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "90R",

    "capability":
        "native_data_engineering",

    "domains":
        sorted(expected_domains),

    "data_engineering_task_count":
        len(data_engineering_tasks),

    "training_examples":
        len(data_engineering_train_records),

    "validation_examples":
        len(data_engineering_validation_records),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH,

    "baseline":
        baseline_metrics,

    "candidate":
        final_metrics,

    "numerical_health":
        {
            "nan_tensors":
                nan_tensors,

            "inf_tensors":
                inf_tensors,

            "healthy":
                numerically_healthy
        },

    "parameter_change":
        {
            "changed_tensors":
                changed_tensors,

            "total_absolute_parameter_change":
                total_parameter_change
        },

    "pipeline":
        {
            "raw_records":
                len(RAW_SENSOR_RECORDS),

            "deduplicated_records":
                len(post_deduplicated),

            "rejected_records":
                len(post_rejected),

            "published_records":
                len(post_output)
        },

    "quality":
        post_quality,

    "partitions":
        {
            str(key):
                len(value)
            for key, value
            in post_partitions.items()
        },

    "schema_evolution":
        {
            "passed":
                len(post_future_errors) == 0
        },

    "independent_validation":
        {
            "passed":
                len(post_training_errors) == 0,

            "errors":
                post_training_errors
        },

    "promotion":
        {
            "decision":
                decision,

            "reason":
                reason
        }
}

write_json(
    EVALUATION_FILE,
    evaluation_report
)


# ============================================================
# 51. DATA ENGINEERING INTELLIGENCE STACK
# ============================================================

print(
    "SILVERWING DATA ENGINEERING STACK"
)

print()

print("Raw Data Ingestion")
print(" ↓")
print("Schema Validation")
print(" ↓")
print("Normalization")
print(" ↓")
print("Type Casting")
print(" ↓")
print("Duplicate Detection")
print(" ↓")
print("Missing-Data Handling")
print(" ↓")
print("Transformation")
print(" ↓")
print("Quality Metrics")
print(" ↓")
print("Partitioning")
print(" ↓")
print("Schema Evolution")
print(" ↓")
print("Future: Streaming Pipelines")
print(" ↓")
print("Future: Distributed Processing")
print(" ↓")
print("Future: Feature Engineering")

print()


# ============================================================
# 52. WHY 90R MATTERS
# ============================================================

print(
    "WHY 90R MATTERS"
)

print()

print(
    "Data engineering converts raw information into "
    "reliable structured information."
)

print()

print(
    "Reliable data is a prerequisite for reliable machine learning."
)

print()

print(
    "90R is the bridge between Silverwing's data foundation "
    "and its machine-learning foundation."
)

print()


# ============================================================
# 53. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "90R implements a deterministic local ETL foundation."
)

print(
    "90R does not yet implement distributed data systems."
)

print(
    "90R does not yet implement streaming ingestion."
)

print(
    "90R does not yet implement advanced feature stores."
)

print(
    "90R does not yet implement production data orchestration."
)

print(
    "90R does not yet establish full data-engineering competence."
)

print()


# ============================================================
# 54. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 91R: Native Machine Learning Foundations"
)

print()

print(
    "Datasets + Features + Targets + Train/Test Splits + "
    "Baseline Models + Metrics + Generalization"
)

print()


# ============================================================
# 55. FOUNDATION MODEL PROGRESS
# ============================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()

print("Own Tokenizer")
print(" ↓")
print("Own Vocabulary")
print(" ↓")
print("Own Decoder")
print(" ↓")
print("Own Training")
print(" ↓")
print("Own Evaluation")
print(" ↓")
print("Instruction Learning")
print(" ↓")
print("79R Native Reasoning Dataset")
print(" ↓")
print("80R Native Reasoning Fine-Tuning")
print(" ↓")
print("81R Native Memory-Aware Training")
print(" ↓")
print("82R Native Tool-Aware Learning")
print(" ↓")
print("83R Native Planning and Tool Sequencing")
print(" ↓")
print("84R Native Verified Execution and Replanning")
print(" ↓")
print("85R Native Mathematical Reasoning Foundation")
print(" ↓")
print("86R Native Probability and Statistical Reasoning")
print(" ↓")
print("87R Native Linear Algebra and Optimization")
print(" ↓")
print("88R Native Algorithms and Data Structures")
print(" ↓")
print("89R Native Data Analysis and SQL Reasoning")
print(" ↓")
print("90R Native Data Engineering")
print(" ↓")
print("91R Native Machine Learning Foundations")
print(" ↓")
print("Deep Learning")
print(" ↓")
print("LLM Architecture and Advanced Learning")
print(" ↓")
print("Engineering + Scientific Intelligence")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 56. COMPLETE
# ============================================================

print(
    "=== LESSON 90R COMPLETE ==="
)
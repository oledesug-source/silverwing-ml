# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 89R
# Native Data Analysis and SQL Reasoning
# ============================================================

import json
import math
import random
import re
import statistics
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
LESSON_88R = PHASE5_DIR / "lesson88R"

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
        LESSON_88R /
        "checkpoints" /
        "silverwing_algorithm_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_88R /
        "checkpoints" /
        "silverwing_algorithm_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_registry.json"
)

DATA_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_train.jsonl"
)

DATA_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_validation.jsonl"
)

DATA_REPORT_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_data_analysis_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_data_analysis_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_data_analysis_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5.0e-6
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
            "No Lesson 88R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 89R")
print("Native Data Analysis and SQL Reasoning")
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
print()

print("External LLM: NONE")
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 88R and Silverwing Inputs"
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

print(
    "Model dimension:",
    MODEL_DIMENSION
)

print(
    "Attention heads:",
    NUMBER_OF_HEADS
)

print(
    "Feed-forward dimension:",
    FEED_FORWARD_DIMENSION
)

print(
    "Layers:",
    NUMBER_OF_LAYERS
)

print(
    "Sequence limit:",
    MAX_SEQUENCE_LENGTH
)

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

    if not isinstance(
            item,
            dict
    ):

        continue

    pair = item.get(
        "pair"
    )

    if (
            not isinstance(pair, list)
            or len(pair) != 2
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

    symbols = list(
        word
    )

    symbols[-1] += BPE_END

    return tuple(
        symbols
    )


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

    return tuple(
        output
    )


def tokenize_word(
        word: str
) -> List[str]:

    symbols = word_to_symbols(
        word
    )

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

    return list(
        symbols
    )


def tokenize_text(
        text: str
) -> List[str]:

    tokens = []

    for word in split_words(
            text
    ):

        tokens.extend(
            tokenize_word(
                word
            )
        )

    return tokens


def encode_text(
        text: str
) -> List[int]:

    ids = [
        BOS_ID
    ]

    for token in tokenize_text(
            text
    ):

        ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )

    ids.append(
        EOS_ID
    )

    return ids


# ============================================================
# 10. TABLE OPERATIONS
# ============================================================

Table = List[
    Dict[str, Any]
]


def table_columns(
        table: Table
) -> List[str]:

    if not table:

        return []

    return list(
        table[0].keys()
    )


def select_columns(
        table: Table,
        columns: List[str]
) -> Table:

    available = set(
        table_columns(table)
    )

    missing = [
        column
        for column
        in columns
        if column not in available
    ]

    if missing:

        raise KeyError(
            (
                    "Missing columns: "
                    +
                    ", ".join(missing)
            )
        )

    return [
        {
            column:
                row[column]
            for column
            in columns
        }
        for row
        in table
    ]


def filter_rows(
        table: Table,
        predicate
) -> Table:

    return [
        dict(row)
        for row
        in table
        if predicate(row)
    ]


def sort_rows(
        table: Table,
        column: str,
        descending: bool = False
) -> Table:

    if column not in table_columns(
            table
    ):

        raise KeyError(
            f"Column '{column}' does not exist."
        )

    return sorted(
        [dict(row) for row in table],
        key=lambda row: (
            row[column] is None,
            row[column]
        ),
        reverse=descending
    )


def count_rows(
        table: Table
) -> int:

    return len(
        table
    )


def sum_column(
        table: Table,
        column: str
) -> float:

    values = [
        float(row[column])
        for row
        in table
        if row[column] is not None
    ]

    return sum(
        values
    )


def mean_column(
        table: Table,
        column: str
) -> float:

    values = [
        float(row[column])
        for row
        in table
        if row[column] is not None
    ]

    if not values:

        raise ValueError(
            f"No numeric values in '{column}'."
        )

    return (
            sum(values)
            /
            len(values)
    )


def group_sum(
        table: Table,
        group_column: str,
        value_column: str
) -> Dict[Any, float]:

    result: Dict[
        Any,
        float
    ] = {}

    for row in table:

        key = row[
            group_column
        ]

        value = row[
            value_column
        ]

        if value is None:

            continue

        result[key] = (
                result.get(
                    key,
                    0.0
                )
                +
                float(value)
        )

    return result


def inner_join(
        left: Table,
        right: Table,
        left_key: str,
        right_key: str
) -> Table:

    right_index: Dict[
        Any,
        List[Dict[str, Any]]
    ] = {}

    for row in right:

        key = row[
            right_key
        ]

        right_index.setdefault(
            key,
            []
        ).append(
            row
        )

    result = []

    for left_row in left:

        key = left_row[
            left_key
        ]

        matches = right_index.get(
            key,
            []
        )

        for right_row in matches:

            merged = dict(
                left_row
            )

            for column, value in (
                    right_row.items()
            ):

                if column == right_key:

                    continue

                if column in merged:

                    merged[
                        "right_" + column
                        ] = value

                else:

                    merged[column] = value

            result.append(
                merged
            )

    return result


# ============================================================
# 11. SQL OPERATIONS
# ============================================================

def sql_where(
        table: Table,
        column: str,
        operator: str,
        value: Any
) -> Table:

    def predicate(
            row: Dict[str, Any]
    ) -> bool:

        current = row[
            column
        ]

        if current is None:

            return False

        if operator == "=":
            return current == value

        if operator == ">":
            return current > value

        if operator == "<":
            return current < value

        if operator == ">=":
            return current >= value

        if operator == "<=":
            return current <= value

        if operator == "!=":
            return current != value

        raise ValueError(
            f"Unsupported operator: {operator}"
        )

    return filter_rows(
        table,
        predicate
    )


def sql_count(
        table: Table
) -> int:

    return count_rows(
        table
    )


def sql_sum(
        table: Table,
        column: str
) -> float:

    return sum_column(
        table,
        column
    )


def sql_avg(
        table: Table,
        column: str
) -> float:

    return mean_column(
        table,
        column
    )


# ============================================================
# 12. SAMPLE DATABASE
# ============================================================

SALES_TABLE: Table = [

    {
        "id": 1,
        "region": "east",
        "product": "pump",
        "quantity": 10,
        "revenue": 1000.0
    },

    {
        "id": 2,
        "region": "west",
        "product": "pump",
        "quantity": 7,
        "revenue": 840.0
    },

    {
        "id": 3,
        "region": "east",
        "product": "compressor",
        "quantity": 4,
        "revenue": 720.0
    },

    {
        "id": 4,
        "region": "south",
        "product": "pump",
        "quantity": 8,
        "revenue": 800.0
    },

    {
        "id": 5,
        "region": "west",
        "product": "compressor",
        "quantity": 5,
        "revenue": 900.0
    }
]


CUSTOMER_TABLE: Table = [

    {
        "customer_id": 1,
        "region": "east",
        "customer": "A"
    },

    {
        "customer_id": 2,
        "region": "west",
        "customer": "B"
    },

    {
        "customer_id": 3,
        "region": "south",
        "customer": "C"
    }
]


# ============================================================
# 13. DATA TASKS
# ============================================================

data_tasks = [

    {
        "example_id":
            "data_001",

        "domain":
            "filtering",

        "problem":
            "From the sales table, select rows where revenue is greater than 850.",

        "reasoning":
            "Apply a WHERE condition to the revenue column.",

        "operation":
            "WHERE revenue > 850",

        "answer":
            "rows with ids 1, 5",

        "validation":
            {
                "type":
                    "filter",

                "column":
                    "revenue",

                "operator":
                    ">",

                "value":
                    850,

                "expected_ids":
                    [
                        1,
                        5
                    ]
            }
    },

    {
        "example_id":
            "data_002",

        "domain":
            "projection",

        "problem":
            "Select product and revenue from the sales table.",

        "reasoning":
            "Projection keeps only the requested columns.",

        "operation":
            "SELECT product, revenue",

        "answer":
            "five rows with product and revenue.",

        "validation":
            {
                "type":
                    "projection",

                "columns":
                    [
                        "product",
                        "revenue"
                    ]
            }
    },

    {
        "example_id":
            "data_003",

        "domain":
            "aggregation",

        "problem":
            "Find the total sales revenue.",

        "reasoning":
            "Sum all non-NULL values in the revenue column.",

        "operation":
            "SUM(revenue)",

        "answer":
            "4260.0",

        "validation":
            {
                "type":
                    "sum",

                "column":
                    "revenue",

                "expected":
                    4260.0
            }
    },

    {
        "example_id":
            "data_004",

        "domain":
            "grouping",

        "problem":
            "Find total revenue by region.",

        "reasoning":
            "Group rows by region and sum revenue within each group.",

        "operation":
            "GROUP BY region; SUM(revenue)",

        "answer":
            "east=1720; west=1740; south=800",

        "validation":
            {
                "type":
                    "group_sum",

                "group_column":
                    "region",

                "value_column":
                    "revenue",

                "expected":
                    {
                        "east":
                            1720.0,

                        "west":
                            1740.0,

                        "south":
                            800.0
                    }
            }
    },

    {
        "example_id":
            "data_005",

        "domain":
            "average",

        "problem":
            "Find the average revenue per sale.",

        "reasoning":
            "Divide total revenue by the number of sales.",

        "operation":
            "AVG(revenue)",

        "answer":
            "852.0",

        "validation":
            {
                "type":
                    "average",

                "column":
                    "revenue",

                "expected":
                    852.0
            }
    },

    {
        "example_id":
            "data_006",

        "domain":
            "sorting",

        "problem":
            "Sort sales from highest revenue to lowest.",

        "reasoning":
            "Order rows by revenue descending.",

        "operation":
            "ORDER BY revenue DESC",

        "answer":
            "ids 1, 5, 2, 4, 3",

        "validation":
            {
                "type":
                    "sort",

                "column":
                    "revenue",

                "descending":
                    True,

                "expected_ids":
                    [
                        1,
                        5,
                        2,
                        4,
                        3
                    ]
            }
    },

    {
        "example_id":
            "data_007",

        "domain":
            "join",

        "problem":
            "Join sales with customers using region.",

        "reasoning":
            "Match rows where sales.region equals customer.region.",

        "operation":
            "INNER JOIN customers ON region",

        "answer":
            "all five sales rows match a customer region.",

        "validation":
            {
                "type":
                    "join",

                "expected_count":
                    5
            }
    },

    {
        "example_id":
            "data_008",

        "domain":
            "null_reasoning",

        "problem":
            "A revenue field is NULL. How should SUM(revenue) handle it?",

        "reasoning":
            "NULL represents a missing or unknown value and is ignored by SUM.",

        "operation":
            "SUM(revenue) ignores NULL inputs.",

        "answer":
            "NULL is ignored by SUM.",

        "validation":
            {
                "type":
                    "null_sum"
            }
    }
]


# ============================================================
# 14. TEST 5 - SCHEMA
# ============================================================

print(
    "TEST 5: Table Schema Validation"
)

print()

sales_columns = table_columns(
    SALES_TABLE
)

customer_columns = table_columns(
    CUSTOMER_TABLE
)

print(
    "Sales columns:",
    sales_columns
)

print(
    "Customer columns:",
    customer_columns
)

print()

if set(sales_columns) != {
    "id",
    "region",
    "product",
    "quantity",
    "revenue"
}:

    raise RuntimeError(
        "Sales table schema validation failed."
    )

if set(customer_columns) != {
    "customer_id",
    "region",
    "customer"
}:

    raise RuntimeError(
        "Customer table schema validation failed."
    )

print(
    "Table schemas valid."
)

print()


# ============================================================
# 15. TEST 6 - DETERMINISTIC DATA ANALYSIS
# ============================================================

print(
    "TEST 6: Deterministic Data Analysis"
)

print()

filtered = sql_where(
    SALES_TABLE,
    "revenue",
    ">",
    850
)

projection = select_columns(
    SALES_TABLE,
    [
        "product",
        "revenue"
    ]
)

total_revenue = sql_sum(
    SALES_TABLE,
    "revenue"
)

average_revenue = sql_avg(
    SALES_TABLE,
    "revenue"
)

grouped_revenue = group_sum(
    SALES_TABLE,
    "region",
    "revenue"
)

sorted_sales = sort_rows(
    SALES_TABLE,
    "revenue",
    descending=True
)

joined_data = inner_join(
    SALES_TABLE,
    CUSTOMER_TABLE,
    "region",
    "region"
)

print(
    "Filtered ids:",
    [
        row["id"]
        for row
        in filtered
    ]
)

print(
    "Projection rows:",
    len(projection)
)

print(
    "Total revenue:",
    total_revenue
)

print(
    "Average revenue:",
    average_revenue
)

print(
    "Grouped revenue:",
    grouped_revenue
)

print(
    "Sorted ids:",
    [
        row["id"]
        for row
        in sorted_sales
    ]
)

print(
    "Joined rows:",
    len(joined_data)
)

print()


# ============================================================
# 16. TEST 7 - DATA ANALYSIS VALIDATION
# ============================================================

print(
    "TEST 7: Independent Data Analysis Validation"
)

print()

analysis_errors = []

expected_filtered_ids = [
    1,
    5
]

observed_filtered_ids = [
    row["id"]
    for row
    in filtered
]

if observed_filtered_ids != expected_filtered_ids:

    analysis_errors.append(
        (
            "Filtering result incorrect: "
            f"observed={observed_filtered_ids}, "
            f"expected={expected_filtered_ids}"
        )
    )

if total_revenue != 4260.0:

    analysis_errors.append(
        "Total revenue incorrect."
    )

if not approximately_equal(
        average_revenue,
        852.0
):

    analysis_errors.append(
        "Average revenue incorrect."
    )

if grouped_revenue != {
    "east":
        1720.0,

    "west":
        1740.0,

    "south":
        800.0
}:

    analysis_errors.append(
        "Grouped revenue incorrect."
    )

expected_sorted_ids = [
    1,
    5,
    2,
    4,
    3
]

observed_sorted_ids = [
    row["id"]
    for row
    in sorted_sales
]

if observed_sorted_ids != expected_sorted_ids:

    analysis_errors.append(
        (
            "Sort result incorrect: "
            f"observed={observed_sorted_ids}, "
            f"expected={expected_sorted_ids}"
        )
    )

if len(joined_data) != 5:

    analysis_errors.append(
        "Join row count incorrect."
    )

if analysis_errors:

    print(
        json.dumps(
            analysis_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Data analysis validation failed."
    )

print(
    "Data analysis operations validated."
)

print()


# ============================================================
# 17. TEST 8 - SQL SEMANTIC VALIDATION
# ============================================================

print(
    "TEST 8: SQL Semantic Validation"
)

print()

sql_filter = sql_where(
    SALES_TABLE,
    "quantity",
    ">",
    7
)

sql_checks = {

    "filter_quantity_gt_7":
        [
            row["id"]
            for row
            in sql_filter
        ],

    "count":
        sql_count(
            SALES_TABLE
        ),

    "sum_revenue":
        sql_sum(
            SALES_TABLE,
            "revenue"
        ),

    "avg_quantity":
        sql_avg(
            SALES_TABLE,
            "quantity"
        )
}

print(
    json.dumps(
        sql_checks,
        indent=4,
        default=str
    )
)

if sql_checks[
    "filter_quantity_gt_7"
] != [
    1,
    4
]:

    raise RuntimeError(
        "SQL WHERE semantics failed."
    )

if sql_checks[
    "count"
] != 5:

    raise RuntimeError(
        "SQL COUNT semantics failed."
    )

if sql_checks[
    "sum_revenue"
] != 4260.0:

    raise RuntimeError(
        "SQL SUM semantics failed."
    )

if not approximately_equal(
        sql_checks[
            "avg_quantity"
        ],
        6.8
):

    raise RuntimeError(
        "SQL AVG semantics failed."
    )

print(
    "SQL semantic contracts valid."
)

print()


# ============================================================
# 18. TEST 9 - NULL HANDLING
# ============================================================

print(
    "TEST 9: NULL Handling Validation"
)

print()

null_table = [

    {
        "id":
            1,

        "revenue":
            100.0
    },

    {
        "id":
            2,

        "revenue":
            None
    },

    {
        "id":
            3,

        "revenue":
            50.0
    }
]

null_sum = sum_column(
    null_table,
    "revenue"
)

null_avg = mean_column(
    null_table,
    "revenue"
)

print(
    "SUM with NULL:",
    null_sum
)

print(
    "AVG with NULL:",
    null_avg
)

print()

if null_sum != 150.0:

    raise RuntimeError(
        "NULL SUM handling failed."
    )

if not approximately_equal(
        null_avg,
        75.0
):

    raise RuntimeError(
        "NULL AVG handling failed."
    )

print(
    "NULL handling validated."
)

print()


# ============================================================
# 19. TEST 10 - REASONING TRACES
# ============================================================

print(
    "TEST 10: Build Data Analysis Reasoning Traces"
)

print()

data_records = []


def build_data_trace(
        task: Dict[str, Any]
) -> str:

    return "\n".join(
        [
            "P:" +
            task["problem"],

            "M:" +
            task["reasoning"],

            "Q:" +
            task["operation"],

            "V:validated",

            "A:" +
            task["answer"]
        ]
    )


for task in data_tasks:

    trace = build_data_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    data_records.append(
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
# 20. TEST 11 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 11: Data Analysis Token Validation"
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
    in data_records

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

    print()

    print(
        "OVERSIZED DATA ANALYSIS TRACES:"
    )

    for record in data_records:

        if (
                record["token_count"]
                >
                MAX_SEQUENCE_LENGTH
        ):

            print()

            print(
                "-----",
                record["example_id"],
                "-----"
            )

            print(
                record["formatted_text"]
            )

    raise RuntimeError(
        (
            "Data analysis examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All data analysis examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 21. TEST 12 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 12: Data Analysis Domain Coverage"
)

print()

expected_domains = {
    "filtering",
    "projection",
    "aggregation",
    "grouping",
    "average",
    "sorting",
    "join",
    "null_reasoning"
}

actual_domains = {
    record["domain"]
    for record
    in data_records
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(data_records)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        "Data analysis domain coverage is incomplete."
    )


# ============================================================
# 22. TEST 13 - SQL CROSS-CHECK
# ============================================================

print(
    "TEST 13: SQL Result Cross-Check"
)

print()

cross_checks = []

cross_checks.append(
    sql_sum(
        SALES_TABLE,
        "revenue"
    )
    ==
    sum(
        row["revenue"]
        for row
        in SALES_TABLE
    )
)

cross_checks.append(
    sql_count(
        SALES_TABLE
    )
    ==
    len(SALES_TABLE)
)

cross_checks.append(
    approximately_equal(
        sql_avg(
            SALES_TABLE,
            "quantity"
        ),
        statistics.mean(
            [
                row["quantity"]
                for row
                in SALES_TABLE
            ]
        )
    )
)

cross_checks.append(
    len(
        inner_join(
            SALES_TABLE,
            CUSTOMER_TABLE,
            "region",
            "region"
        )
    )
    ==
    5
)

print(
    "Cross-checks passed:",
    sum(cross_checks),
    "/",
    len(cross_checks)
)

if not all(
        cross_checks
):

    raise RuntimeError(
        "SQL/data-analysis cross-check failed."
    )

print(
    "SQL and analysis cross-check passed."
)

print()


# ============================================================
# 23. TEST 14 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    data_records
)

validation_count = max(
    2,
    int(
        round(
            len(data_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(data_records) - 1
)

data_train_records = (
    data_records[
        :-validation_count
    ]
)

data_validation_records = (
    data_records[
        -validation_count:
    ]
)

print(
    "TEST 14: Data Analysis Train/Validation Split"
)

print(
    "Training examples:",
    len(data_train_records)
)

print(
    "Validation examples:",
    len(data_validation_records)
)

print()


# ============================================================
# 24. SAVE ARTIFACTS
# ============================================================

write_json(
    DATA_REGISTRY_FILE,
    {
        "lesson":
            "89R",

        "capability":
            "native_data_analysis_and_sql_reasoning",

        "domains":
            sorted(expected_domains),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(data_tasks)
    }
)

with open(
        DATA_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in data_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        DATA_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in data_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    DATA_REPORT_FILE,
    {
        "lesson":
            "89R",

        "capability":
            "native_data_analysis_and_sql_reasoning",

        "domains":
            sorted(expected_domains),

        "training_examples":
            len(data_train_records),

        "validation_examples":
            len(data_validation_records),

        "external_llm":
            False
    }
)


# ============================================================
# 25. DATASET
# ============================================================

class DataAnalysisDataset(
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
                record[
                    "formatted_text"
                ]
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

        return len(
            self.samples
        )

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Any]:

        sample = self.samples[
            index
        ]

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


def collate_data_batch(
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
                for item
                in batch
            ],

        "input_ids":
            torch.stack(inputs),

        "labels":
            torch.stack(labels)
    }


data_train_dataset = DataAnalysisDataset(
    data_train_records
)

data_validation_dataset = DataAnalysisDataset(
    data_validation_records
)

data_train_loader = DataLoader(
    data_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_data_batch
)

data_validation_loader = DataLoader(
    data_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_data_batch
)

print(
    "TEST 15: Data Analysis DataLoaders"
)

print(
    "Training samples:",
    len(data_train_dataset)
)

print(
    "Validation samples:",
    len(data_validation_dataset)
)

print(
    "Training batches:",
    len(data_train_loader)
)

print(
    "Validation batches:",
    len(data_validation_loader)
)

print()


# ============================================================
# 26. SILVERWING ATTENTION
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

        self.head_dimension = (
                dimension
                //
                heads
        )

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

        query = self.query_projection(
            x
        )

        key = self.key_projection(
            x
        )

        value = self.value_projection(
            x
        )

        query = query.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        key = key.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        value = value.view(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dimension
        ).transpose(
            1,
            2
        )

        scores = torch.matmul(
            query,
            key.transpose(
                -2,
                -1
            )
        )

        scores = (
                scores
                /
                math.sqrt(
                    self.head_dimension
                )
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
            .transpose(
                1,
                2
            )
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
# 27. FEED FORWARD
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
# 28. TRANSFORMER BLOCK
# ============================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.attention = (
            SilverwingAttention(
                MODEL_DIMENSION,
                NUMBER_OF_HEADS
            )
        )

        self.norm_attention = nn.LayerNorm(
            MODEL_DIMENSION
        )

        self.feed_forward = (
            SilverwingFeedForward(
                MODEL_DIMENSION,
                FEED_FORWARD_DIMENSION
            )
        )

        self.norm_feed_forward = nn.LayerNorm(
            MODEL_DIMENSION
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        x = self.norm_attention(
            x
            +
            self.attention(
                x
            )
        )

        x = self.norm_feed_forward(
            x
            +
            self.feed_forward(
                x
            )
        )

        return x


# ============================================================
# 29. POSITION EMBEDDING
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
# 30. SILVERWING DECODER
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

        if (
                sequence_length
                >
                MAX_SEQUENCE_LENGTH
        ):

            raise ValueError(
                "Sequence exceeds model limit."
            )

        x = (
                self.token_embedding(
                    input_ids
                )
                +
                self.position_embedding(
                    sequence_length,
                    input_ids.device
                ).unsqueeze(0)
        )

        for layer in self.layers:

            x = layer(x)

        x = self.final_norm(
            x
        )

        return self.language_model_head(
            x
        )


# ============================================================
# 31. TEST 16 - STRICT LOAD
# ============================================================

print(
    "TEST 16: Strict Load of 88R Algorithmic Model"
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
        "88R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "88R checkpoint is missing model_state_dict."
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
                "88R checkpoint architecture mismatch. "
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
            "89R refused to load a mismatched "
            "88R Silverwing model.\n\n"
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
    "88R model is compatible with 89R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 32. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()
    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 33. LOSS
# ============================================================

def data_loss(
        logits: torch.Tensor,
        labels: torch.Tensor
) -> torch.Tensor:

    return F.cross_entropy(
        logits.reshape(
            -1,
            VOCABULARY_SIZE
        ),
        labels.reshape(
            -1
        ),
        ignore_index=-100
    )


# ============================================================
# 34. EVALUATION
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

        loss = data_loss(
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
            "loss":
                float("nan"),

            "perplexity":
                float("nan"),

            "accuracy":
                float("nan"),

            "tokens":
                0
        }

    loss_value = (
            total_loss
            /
            batches
    )

    if (
            math.isfinite(loss_value)
            and
            loss_value < 50
    ):

        perplexity = math.exp(
            loss_value
        )

    else:

        perplexity = float(
            "inf"
        )

    accuracy = (
        correct
        /
        valid_tokens
        if valid_tokens
        else float("nan")
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
# 35. TEST 17 - BASELINE
# ============================================================

print(
    "TEST 17: Baseline Data Analysis Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    data_validation_loader
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
# 36. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(data_train_loader)
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
# 37. TEST 18 - TRAINING
# ============================================================

print(
    "TEST 18: Native Data Analysis Fine-Tuning"
)

print()

history = []

best_validation_loss = float(
    "inf"
)

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
            data_train_loader,
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

        loss = data_loss(
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
            f"| Batch {batch_number}/{len(data_train_loader)} "
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
        data_validation_loader
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
                    "89R",

                "base_checkpoint":
                    str(BASE_CHECKPOINT),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "data_task_count":
                    len(data_tasks)
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 38. TEST 19 - FINAL EVALUATION
# ============================================================

print(
    "TEST 19: Final Data Analysis Evaluation"
)

print()

final_metrics = evaluate(
    model,
    data_validation_loader
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
# 39. TEST 20 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 20: Numerical Health"
)

print()

nan_tensors = 0
inf_tensors = 0

for parameter in model.parameters():

    if torch.isnan(
            parameter
    ).any():

        nan_tensors += 1

    if torch.isinf(
            parameter
    ).any():

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
# 40. TEST 21 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 21: Parameter Change"
)

print()

changed_tensors = 0
total_parameter_change = 0.0

for name, parameter in (
        model.state_dict().items()
):

    original = baseline_state[
        name
    ]

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
# 41. TEST 22 - POST-TRAINING VALIDATION
# ============================================================

print(
    "TEST 22: Post-Training Data Analysis Validation"
)

print()

post_training_errors = []

for task in data_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "filter":

        observed = sql_where(
            SALES_TABLE,
            data["column"],
            data["operator"],
            data["value"]
        )

        observed_ids = [
            row["id"]
            for row
            in observed
        ]

        valid = (
                observed_ids
                ==
                data["expected_ids"]
        )

    elif validation_type == "projection":

        observed = select_columns(
            SALES_TABLE,
            data["columns"]
        )

        valid = all(
            set(row.keys())
            ==
            set(data["columns"])
            for row
            in observed
        )

    elif validation_type == "sum":

        observed = sql_sum(
            SALES_TABLE,
            data["column"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "group_sum":

        observed = group_sum(
            SALES_TABLE,
            data["group_column"],
            data["value_column"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "average":

        observed = sql_avg(
            SALES_TABLE,
            data["column"]
        )

        valid = approximately_equal(
            observed,
            data["expected"]
        )

    elif validation_type == "sort":

        observed = sort_rows(
            SALES_TABLE,
            data["column"],
            data["descending"]
        )

        observed_ids = [
            row["id"]
            for row
            in observed
        ]

        valid = (
                observed_ids
                ==
                data["expected_ids"]
        )

    elif validation_type == "join":

        observed = inner_join(
            SALES_TABLE,
            CUSTOMER_TABLE,
            "region",
            "region"
        )

        valid = (
                len(observed)
                ==
                data["expected_count"]
        )

    elif validation_type == "null_sum":

        null_test = [
            {
                "value":
                    100.0
            },

            {
                "value":
                    None
            },

            {
                "value":
                    50.0
            }
        ]

        observed = sum(
            row["value"]
            for row
            in null_test
            if row["value"] is not None
        )

        valid = (
                observed
                ==
                150.0
        )

    if not valid:

        post_training_errors.append(
            {
                "example_id":
                    task["example_id"],

                "domain":
                    task["domain"],

                "error":
                    "Data analysis validation failed."
            }
        )


if post_training_errors:

    print(
        json.dumps(
            post_training_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Post-training data analysis validation failed."
    )

print(
    "Post-training data analysis validation passed:",
    len(data_tasks)
)

print()


# ============================================================
# 42. TEST 23 - PROMOTION
# ============================================================

print(
    "TEST 23: Data Analysis Promotion Gate"
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

elif not math.isfinite(
        candidate_loss
):

    decision = "REJECT"

    reason = (
        "Candidate data-analysis loss is invalid."
    )

elif (
        math.isfinite(
            baseline_loss
        )
        and
        candidate_loss
        <
        baseline_loss
):

    decision = "PROMOTE_CANDIDATE"

    reason = (
        "Data-analysis validation loss improved."
    )

else:

    decision = "RETAIN_BASELINE"

    reason = (
        "Data-analysis validation loss did not improve."
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
# 43. TEST 24 - SAVE
# ============================================================

print(
    "TEST 24: Save Data Analysis Candidate"
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
        "89R",

    "training_mode":
        "native_data_analysis_and_sql_reasoning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

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
        sorted(
            expected_domains
        ),

    "data_task_count":
        len(data_tasks),

    "sequence_limit":
        MAX_SEQUENCE_LENGTH
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
# 44. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "89R",

    "training_mode":
        "native_data_analysis_and_sql_reasoning",

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "domains":
        sorted(
            expected_domains
        ),

    "data_task_count":
        len(data_tasks),

    "training_examples":
        len(data_train_records),

    "validation_examples":
        len(data_validation_records),

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
# 45. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "89R",

    "capability":
        "native_data_analysis_and_sql_reasoning",

    "domains":
        sorted(
            expected_domains
        ),

    "data_task_count":
        len(data_tasks),

    "training_examples":
        len(data_train_records),

    "validation_examples":
        len(data_validation_records),

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

    "independent_validation":
        {
            "passed":
                len(post_training_errors)
                ==
                0
        },

    "sql_validation":
        {
            "filter":
                True,

            "count":
                True,

            "sum":
                True,

            "average":
                True,

            "join":
                True,

            "null_handling":
                True
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
# 46. DATA INTELLIGENCE STACK
# ============================================================

print(
    "SILVERWING DATA INTELLIGENCE STACK"
)

print()

print("Tables")
print(" ↓")
print("Schemas")
print(" ↓")
print("Filtering")
print(" ↓")
print("Projection")
print(" ↓")
print("Sorting")
print(" ↓")
print("Aggregation")
print(" ↓")
print("Grouping")
print(" ↓")
print("Joins")
print(" ↓")
print("NULL Reasoning")
print(" ↓")
print("SQL Semantics")
print(" ↓")
print("Future: Advanced SQL")
print(" ↓")
print("Future: Query Optimization")
print(" ↓")
print("Future: Data Pipelines")

print()


# ============================================================
# 47. WHY 89R MATTERS
# ============================================================

print(
    "WHY 89R MATTERS"
)

print()

print(
    "Data connects Silverwing's mathematics, algorithms, "
    "machine learning and real-world knowledge."
)

print()

print(
    "SQL reasoning provides a structured mechanism for "
    "retrieving and transforming information."
)

print()

print(
    "This layer prepares Silverwing for the data-engineering stage."
)

print()


# ============================================================
# 48. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "89R uses a small deterministic analytical database."
)

print(
    "89R does not yet implement a complete SQL parser."
)

print(
    "89R does not yet implement query optimization."
)

print(
    "89R does not yet implement indexing."
)

print(
    "89R does not yet implement transactions."
)

print(
    "89R does not yet implement distributed data processing."
)

print(
    "89R does not yet establish production-grade database intelligence."
)

print()


# ============================================================
# 49. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 90R: Native Data Engineering"
)

print()

print(
    "Data Cleaning + Validation + ETL + Pipelines + "
    "Schema Evolution + Quality Monitoring"
)

print()


# ============================================================
# 50. FOUNDATION MODEL PROGRESS
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
print("Engineering + Scientific Intelligence")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# 51. COMPLETE
# ============================================================

print(
    "=== LESSON 89R COMPLETE ==="
)
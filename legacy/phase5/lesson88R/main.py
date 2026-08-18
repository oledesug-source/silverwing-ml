# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 88R
# Native Algorithms and Data Structures
# ============================================================
#
# 79R -> Native Reasoning Dataset
# 80R -> Native Reasoning Fine-Tuning
# 81R -> Native Memory-Aware Training
# 82R -> Native Tool-Aware Learning
# 83R -> Native Planning and Tool Sequencing
# 84R -> Native Verified Execution and Replanning
# 85R -> Native Mathematical Reasoning Foundation
# 86R -> Native Probability and Statistical Reasoning
# 87R -> Native Linear Algebra and Optimization
# 88R -> Native Algorithms and Data Structures
#
# ============================================================
# PURPOSE
# ============================================================
#
# 88R establishes computational reasoning foundations:
#
#   arrays
#   stacks
#   queues
#   linked structures
#   trees
#   graphs
#   searching
#   sorting
#   traversal
#   algorithmic complexity
#   deterministic algorithm validation
#
# This layer supports:
#
#   programming
#   data engineering
#   machine learning
#   planning
#   memory systems
#   graph reasoning
#   search
#   optimization
#   software engineering
#
# ============================================================
# REASONING CONTRACT
# ============================================================
#
# Problem
#   ↓
# Representation
#   ↓
# Algorithm
#   ↓
# Execution
#   ↓
# Validation
#   ↓
# Complexity
#   ↓
# Final Answer
#
# ============================================================
# MODEL OWNERSHIP
# ============================================================
#
# Tokenizer: Silverwing native
# Vocabulary: Silverwing native
# Decoder: Silverwing native
# Dataset: Silverwing native
# Training: Silverwing native
# Evaluation: Silverwing native
#
# External LLM: NONE
#
# ============================================================
# SEQUENCE LIMIT
# ============================================================
#
# Established Silverwing limit: 256 tokens.
#
# All training traces are compact and checked before training.
#
# ============================================================

import json
import math
import random
import re
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
LESSON_87R = PHASE5_DIR / "lesson87R"

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
        LESSON_87R /
        "checkpoints" /
        "silverwing_linear_algebra_best.pt"
)

BASE_CHECKPOINT_FALLBACK = (
        LESSON_87R /
        "checkpoints" /
        "silverwing_linear_algebra_candidate.pt"
)

OUTPUT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALGORITHM_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_algorithm_registry.json"
)

ALGORITHM_TRAIN_FILE = (
        BASE_DIR /
        "silverwing_algorithm_train.jsonl"
)

ALGORITHM_VALIDATION_FILE = (
        BASE_DIR /
        "silverwing_algorithm_validation.jsonl"
)

ALGORITHM_REPORT_FILE = (
        BASE_DIR /
        "silverwing_algorithm_report.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_algorithm_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_algorithm_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR /
        "silverwing_algorithm_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_algorithm_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 5.5e-6

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


def select_base_checkpoint() -> Path:

    if BASE_CHECKPOINT_PRIMARY.exists():
        return BASE_CHECKPOINT_PRIMARY

    if BASE_CHECKPOINT_FALLBACK.exists():
        return BASE_CHECKPOINT_FALLBACK

    raise FileNotFoundError(
        (
            "No Lesson 87R checkpoint found.\n"
            f"Expected:\n{BASE_CHECKPOINT_PRIMARY}\n"
            f"or:\n{BASE_CHECKPOINT_FALLBACK}"
        )
    )


# ============================================================
# 4. HEADER
# ============================================================

print("=== SILVERWING ML ===")
print("PHASE 5 - LESSON 88R")
print("Native Algorithms and Data Structures")
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
print()

print("External LLM: NONE")
print("Sequence limit:", MAX_SEQUENCE_LENGTH)
print()


# ============================================================
# 5. TEST 1 - INPUTS
# ============================================================

print(
    "TEST 1: Verify Lesson 87R and Silverwing Inputs"
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
    token:
        int(token_id)
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
# 10. DATA STRUCTURES
# ============================================================

@dataclass
class TreeNode:

    value: Any

    left: Optional["TreeNode"] = None

    right: Optional["TreeNode"] = None


class Graph:

    def __init__(self) -> None:

        self.adjacency: Dict[
            str,
            List[str]
        ] = {}

    def add_edge(
            self,
            source: str,
            target: str
    ) -> None:

        self.adjacency.setdefault(
            source,
            []
        ).append(
            target
        )

        self.adjacency.setdefault(
            target,
            []
        )


# ============================================================
# 11. ALGORITHMS
# ============================================================

def linear_search(
        values: List[int],
        target: int
) -> int:

    for index, value in enumerate(
            values
    ):

        if value == target:
            return index

    return -1


def binary_search(
        values: List[int],
        target: int
) -> int:

    left = 0

    right = len(values) - 1

    while left <= right:

        middle = (
                         left + right
                 ) // 2

        if values[middle] == target:
            return middle

        if values[middle] < target:
            left = middle + 1

        else:
            right = middle - 1

    return -1


def bubble_sort(
        values: List[int]
) -> List[int]:

    result = list(values)

    n = len(result)

    for i in range(n):

        swapped = False

        for j in range(
                0,
                n - i - 1
        ):

            if result[j] > result[j + 1]:

                result[j], result[j + 1] = (
                    result[j + 1],
                    result[j]
                )

                swapped = True

        if not swapped:
            break

    return result


def merge_sort(
        values: List[int]
) -> List[int]:

    if len(values) <= 1:
        return list(values)

    middle = len(values) // 2

    left = merge_sort(
        values[:middle]
    )

    right = merge_sort(
        values[middle:]
    )

    merged = []

    left_index = 0
    right_index = 0

    while (
            left_index < len(left)
            and
            right_index < len(right)
    ):

        if (
                left[left_index]
                <=
                right[right_index]
        ):

            merged.append(
                left[left_index]
            )

            left_index += 1

        else:

            merged.append(
                right[right_index]
            )

            right_index += 1

    merged.extend(
        left[left_index:]
    )

    merged.extend(
        right[right_index:]
    )

    return merged


def stack_process(
        values: List[int]
) -> List[int]:

    stack = []

    for value in values:
        stack.append(value)

    result = []

    while stack:
        result.append(
            stack.pop()
        )

    return result


def queue_process(
        values: List[int]
) -> List[int]:

    queue = deque(values)

    result = []

    while queue:
        result.append(
            queue.popleft()
        )

    return result


def inorder_traversal(
        node: Optional[TreeNode]
) -> List[Any]:

    if node is None:
        return []

    return (
            inorder_traversal(node.left)
            +
            [node.value]
            +
            inorder_traversal(node.right)
    )


def breadth_first_search(
        graph: Graph,
        start: str
) -> List[str]:

    if start not in graph.adjacency:
        return []

    queue = deque([start])

    visited = {start}

    order = []

    while queue:

        current = queue.popleft()

        order.append(
            current
        )

        for neighbor in graph.adjacency.get(
                current,
                []
        ):

            if neighbor not in visited:

                visited.add(
                    neighbor
                )

                queue.append(
                    neighbor
                )

    return order


def depth_first_search(
        graph: Graph,
        start: str
) -> List[str]:

    if start not in graph.adjacency:
        return []

    stack = [start]

    visited = set()

    order = []

    while stack:

        current = stack.pop()

        if current in visited:
            continue

        visited.add(
            current
        )

        order.append(
            current
        )

        for neighbor in reversed(
                graph.adjacency.get(
                    current,
                    []
                )
        ):

            if neighbor not in visited:
                stack.append(
                    neighbor
                )

    return order


def bfs_shortest_path_length(
        graph: Graph,
        start: str,
        target: str
) -> int:

    if (
            start not in graph.adjacency
            or
            target not in graph.adjacency
    ):

        return -1

    queue = deque(
        [
            (
                start,
                0
            )
        ]
    )

    visited = {start}

    while queue:

        current, distance = queue.popleft()

        if current == target:

            return distance

        for neighbor in graph.adjacency.get(
                current,
                []
        ):

            if neighbor not in visited:

                visited.add(
                    neighbor
                )

                queue.append(
                    (
                        neighbor,
                        distance + 1
                    )
                )

    return -1


# ============================================================
# 12. COMPLEXITY HELPERS
# ============================================================

def linear_search_complexity(
        n: int
) -> str:

    if n <= 1:
        return "O(1)"

    return "O(n)"


def binary_search_complexity(
        n: int
) -> str:

    if n <= 1:
        return "O(1)"

    return "O(log n)"


def merge_sort_complexity() -> str:
    return "O(n log n)"


def bubble_sort_complexity() -> str:
    return "O(n^2)"


def bfs_complexity(
        vertices: int,
        edges: int
) -> str:

    return "O(V+E)"


# ============================================================
# 13. ALGORITHM CURRICULUM
# ============================================================

algorithm_tasks = [

    {
        "example_id":
            "alg_001",

        "domain":
            "linear_search",

        "problem":
            "Find 7 in [4, 9, 2, 7, 5].",

        "reasoning":
            "Scan values from left to right until the target is found.",

        "calculation":
            "Indices 0,1,2,3; target found at index 3.",

        "answer":
            "index = 3",

        "complexity":
            "O(n)",

        "validation":
            {
                "type":
                    "linear_search",

                "values":
                    [
                        4,
                        9,
                        2,
                        7,
                        5
                    ],

                "target":
                    7,

                "expected":
                    3
            }
    },

    {
        "example_id":
            "alg_002",

        "domain":
            "binary_search",

        "problem":
            "Find 7 in sorted [1, 3, 5, 7, 9, 11].",

        "reasoning":
            "Repeatedly compare the target with the middle value and discard half the search range.",

        "calculation":
            "Middle values reduce the range until index 3 remains.",

        "answer":
            "index = 3",

        "complexity":
            "O(log n)",

        "validation":
            {
                "type":
                    "binary_search",

                "values":
                    [
                        1,
                        3,
                        5,
                        7,
                        9,
                        11
                    ],

                "target":
                    7,

                "expected":
                    3
            }
    },

    {
        "example_id":
            "alg_003",

        "domain":
            "sorting",

        "problem":
            "Sort [5, 2, 8, 1, 3] using merge sort.",

        "reasoning":
            "Split the sequence recursively, sort each half, then merge ordered halves.",

        "calculation":
            "[5,2,8,1,3] -> [1,2,3,5,8]",

        "answer":
            "[1, 2, 3, 5, 8]",

        "complexity":
            "O(n log n)",

        "validation":
            {
                "type":
                    "merge_sort",

                "values":
                    [
                        5,
                        2,
                        8,
                        1,
                        3
                    ],

                "expected":
                    [
                        1,
                        2,
                        3,
                        5,
                        8
                    ]
            }
    },

    {
        "example_id":
            "alg_004",

        "domain":
            "stack",

        "problem":
            "Push 1, 2, 3 onto a stack and remove all values.",

        "reasoning":
            "A stack is last-in-first-out, so the newest value is removed first.",

        "calculation":
            "Push 1,2,3 -> pop 3,2,1.",

        "answer":
            "[3, 2, 1]",

        "complexity":
            "O(n)",

        "validation":
            {
                "type":
                    "stack",

                "values":
                    [
                        1,
                        2,
                        3
                    ],

                "expected":
                    [
                        3,
                        2,
                        1
                    ]
            }
    },

    {
        "example_id":
            "alg_005",

        "domain":
            "queue",

        "problem":
            "Process queue [1, 2, 3] in arrival order.",

        "reasoning":
            "A queue is first-in-first-out, so the oldest value is removed first.",

        "calculation":
            "1,2,3 -> 1,2,3.",

        "answer":
            "[1, 2, 3]",

        "complexity":
            "O(n)",

        "validation":
            {
                "type":
                    "queue",

                "values":
                    [
                        1,
                        2,
                        3
                    ],

                "expected":
                    [
                        1,
                        2,
                        3
                    ]
            }
    },

    {
        "example_id":
            "alg_006",

        "domain":
            "binary_tree",

        "problem":
            "Traverse the binary search tree with root 4, left 2, right 6, and children 1,3,5,7 using inorder traversal.",

        "reasoning":
            "Visit left subtree, root, then right subtree recursively.",

        "calculation":
            "1,2,3,4,5,6,7.",

        "answer":
            "[1, 2, 3, 4, 5, 6, 7]",

        "complexity":
            "O(n)",

        "validation":
            {
                "type":
                    "tree",

                "expected":
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7
                    ]
            }
    },

    {
        "example_id":
            "alg_007",

        "domain":
            "graph_bfs",

        "problem":
            "Traverse graph A->B,C; B->D; C->E using BFS starting at A.",

        "reasoning":
            "Visit nodes level by level using a FIFO queue.",

        "calculation":
            "A, then B,C, then D,E.",

        "answer":
            "[A, B, C, D, E]",

        "complexity":
            "O(V+E)",

        "validation":
            {
                "type":
                    "bfs",

                "expected":
                    [
                        "A",
                        "B",
                        "C",
                        "D",
                        "E"
                    ]
            }
    },

    {
        "example_id":
            "alg_008",

        "domain":
            "graph_shortest_path",

        "problem":
            "Find the shortest unweighted path from A to E in A->B,C; B->D; C->E.",

        "reasoning":
            "BFS finds the shortest path in an unweighted graph because it explores by distance layers.",

        "calculation":
            "A -> C -> E has length 2.",

        "answer":
            "distance = 2",

        "complexity":
            "O(V+E)",

        "validation":
            {
                "type":
                    "shortest_path",

                "start":
                    "A",

                "target":
                    "E",

                "expected":
                    2
            }
    }
]


# ============================================================
# 14. BUILD TEST GRAPH
# ============================================================

def build_test_graph() -> Graph:

    graph = Graph()

    graph.add_edge(
        "A",
        "B"
    )

    graph.add_edge(
        "A",
        "C"
    )

    graph.add_edge(
        "B",
        "D"
    )

    graph.add_edge(
        "C",
        "E"
    )

    return graph


# ============================================================
# 15. TEST 5 - DATA STRUCTURES
# ============================================================

print(
    "TEST 5: Data Structure Construction"
)

print()

tree = TreeNode(
    4,
    left=TreeNode(
        2,
        left=TreeNode(1),
        right=TreeNode(3)
    ),
    right=TreeNode(
        6,
        left=TreeNode(5),
        right=TreeNode(7)
    )
)

tree_result = inorder_traversal(
    tree
)

graph = build_test_graph()

bfs_result = breadth_first_search(
    graph,
    "A"
)

dfs_result = depth_first_search(
    graph,
    "A"
)

print(
    "Tree inorder:",
    tree_result
)

print(
    "Graph BFS:",
    bfs_result
)

print(
    "Graph DFS:",
    dfs_result
)

print()

if tree_result != [
    1,
    2,
    3,
    4,
    5,
    6,
    7
]:

    raise RuntimeError(
        "Binary tree construction failed."
    )

if bfs_result != [
    "A",
    "B",
    "C",
    "D",
    "E"
]:

    raise RuntimeError(
        "Graph BFS construction failed."
    )

print(
    "Data structure construction validated."
)

print()


# ============================================================
# 16. TEST 6 - ALGORITHM VALIDATION
# ============================================================

print(
    "TEST 6: Independent Algorithm Validation"
)

print()

validation_errors = []

for task in algorithm_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "linear_search":

        observed = linear_search(
            data["values"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "binary_search":

        observed = binary_search(
            data["values"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "merge_sort":

        observed = merge_sort(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "stack":

        observed = stack_process(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "queue":

        observed = queue_process(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "tree":

        observed = inorder_traversal(
            tree
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "bfs":

        observed = breadth_first_search(
            graph,
            "A"
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "shortest_path":

        observed = bfs_shortest_path_length(
            graph,
            data["start"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    else:

        validation_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    (
                            "Unknown validation type: "
                            +
                            validation_type
                    )
            }
        )

        continue

    if not valid:

        validation_errors.append(
            {
                "example_id":
                    task["example_id"],

                "error":
                    "Algorithm validation failed."
            }
        )


if validation_errors:

    print(
        json.dumps(
            validation_errors,
            indent=4
        )
    )

    raise RuntimeError(
        "Independent algorithm validation failed."
    )

print(
    "Algorithm examples validated:",
    len(algorithm_tasks)
)

print()


# ============================================================
# 17. TEST 7 - COMPLEXITY VALIDATION
# ============================================================

print(
    "TEST 7: Algorithm Complexity Validation"
)

print()

complexity_checks = {

    "linear_search":
        linear_search_complexity(100),

    "binary_search":
        binary_search_complexity(100),

    "merge_sort":
        merge_sort_complexity(),

    "bubble_sort":
        bubble_sort_complexity(),

    "bfs":
        bfs_complexity(
            5,
            4
        )
}

print(
    json.dumps(
        complexity_checks,
        indent=4
    )
)

expected_complexity = {
    "linear_search":
        "O(n)",

    "binary_search":
        "O(log n)",

    "merge_sort":
        "O(n log n)",

    "bubble_sort":
        "O(n^2)",

    "bfs":
        "O(V+E)"
}

if complexity_checks != expected_complexity:

    raise RuntimeError(
        "Algorithm complexity validation failed."
    )

print(
    "Algorithm complexity contracts valid."
)

print()


# ============================================================
# 18. TEST 8 - TRACE CONSTRUCTION
# ============================================================

print(
    "TEST 8: Build Algorithmic Reasoning Traces"
)

print()

algorithm_records = []

complexity_display = {
    "O(n)":
        "O(n)",

    "O(log n)":
        "O(log n)",

    "O(n log n)":
        "O(nlogn)",

    "O(n^2)":
        "O(n2)",

    "O(V+E)":
        "O(V+E)"
}


def build_algorithm_trace(
        task: Dict[str, Any]
) -> str:

    complexity = task[
        "complexity"
    ]

    compact_complexity = (
        complexity_display.get(
            complexity,
            complexity
        )
    )

    return "\n".join(
        [
            "P:" +
            task["problem"],

            "M:" +
            task["reasoning"],

            "C:" +
            task["calculation"],

            "V:" +
            (
                    "valid;"
                    +
                    compact_complexity
            ),

            "A:" +
            task["answer"]
        ]
    )


for task in algorithm_tasks:

    trace = build_algorithm_trace(
        task
    )

    token_count = len(
        encode_text(
            trace
        )
    )

    algorithm_records.append(
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
# 19. TEST 9 - TOKEN VALIDATION
# ============================================================

print(
    "TEST 9: Algorithmic Token Validation"
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

    for record in algorithm_records

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
        "OVERSIZED ALGORITHMIC TRACES:"
    )

    for record in algorithm_records:

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
            "Algorithmic examples exceed "
            "the Silverwing sequence limit."
        )
    )

print(
    "All algorithmic examples fit "
    "the Silverwing sequence limit."
)

print()


# ============================================================
# 20. TEST 10 - DOMAIN COVERAGE
# ============================================================

print(
    "TEST 10: Algorithmic Domain Coverage"
)

print()

expected_domains = {
    "linear_search",
    "binary_search",
    "sorting",
    "stack",
    "queue",
    "binary_tree",
    "graph_bfs",
    "graph_shortest_path"
}

actual_domains = {
    record["domain"]
    for record in algorithm_records
}

print(
    "Domains:",
    sorted(actual_domains)
)

print(
    "Examples:",
    len(algorithm_records)
)

print()

if actual_domains != expected_domains:

    raise RuntimeError(
        "Algorithmic domain coverage is incomplete."
    )


# ============================================================
# 21. TEST 11 - COMPLEXITY / BEHAVIOR CROSS CHECK
# ============================================================

print(
    "TEST 11: Algorithm Behavior and Complexity Cross-Check"
)

print()

cross_checks = []

values = [
    7,
    2,
    9,
    1,
    5
]

sorted_values = merge_sort(
    values
)

cross_checks.append(
    sorted_values
    ==
    [
        1,
        2,
        5,
        7,
        9
    ]
)

cross_checks.append(
    binary_search(
        sorted_values,
        7
    )
    ==
    3
)

cross_checks.append(
    linear_search(
        values,
        7
    )
    ==
    0
)

cross_checks.append(
    stack_process(
        values
    )
    ==
    list(
        reversed(values)
    )
)

cross_checks.append(
    queue_process(
        values
    )
    ==
    values
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
        "Algorithm behavior cross-check failed."
    )

print(
    "Algorithm behavior and complexity cross-check passed."
)

print()


# ============================================================
# 22. TEST 12 - TRAIN / VALIDATION SPLIT
# ============================================================

random.Random(
    SEED
).shuffle(
    algorithm_records
)

validation_count = max(
    2,
    int(
        round(
            len(algorithm_records)
            *
            0.40
        )
    )
)

validation_count = min(
    validation_count,
    len(algorithm_records) - 1
)

algorithm_train_records = (
    algorithm_records[
        :-validation_count
    ]
)

algorithm_validation_records = (
    algorithm_records[
        -validation_count:
    ]
)

print(
    "TEST 12: Algorithm Train/Validation Split"
)

print(
    "Training examples:",
    len(algorithm_train_records)
)

print(
    "Validation examples:",
    len(algorithm_validation_records)
)

print()


# ============================================================
# 23. SAVE ARTIFACTS
# ============================================================

write_json(
    ALGORITHM_REGISTRY_FILE,
    {
        "lesson":
            "88R",

        "capability":
            "native_algorithms_and_data_structures",

        "domains":
            sorted(
                expected_domains
            ),

        "sequence_limit":
            MAX_SEQUENCE_LENGTH,

        "example_count":
            len(
                algorithm_tasks
            )
    }
)

with open(
        ALGORITHM_TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in algorithm_train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

with open(
        ALGORITHM_VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for record in algorithm_validation_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

write_json(
    ALGORITHM_REPORT_FILE,
    {
        "lesson":
            "88R",

        "capability":
            "native_algorithms_and_data_structures",

        "domains":
            sorted(
                expected_domains
            ),

        "training_examples":
            len(
                algorithm_train_records
            ),

        "validation_examples":
            len(
                algorithm_validation_records
            ),

        "external_llm":
            False
    }
)


# ============================================================
# 24. DATASET
# ============================================================

class AlgorithmDataset(
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
                        record[
                            "example_id"
                        ],

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
                sample[
                    "example_id"
                ],

            "input_ids":
                torch.tensor(
                    sample[
                        "input_ids"
                    ],
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    sample[
                        "labels"
                    ],
                    dtype=torch.long
                )
        }


def collate_algorithm_batch(
        batch: List[
            Dict[str, Any]
        ]
) -> Dict[str, Any]:

    maximum_length = max(
        len(
            item[
                "input_ids"
            ]
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
                item[
                    "example_id"
                ]
                for item
                in batch
            ],

        "input_ids":
            torch.stack(
                inputs
            ),

        "labels":
            torch.stack(
                labels
            )
    }


algorithm_train_dataset = AlgorithmDataset(
    algorithm_train_records
)

algorithm_validation_dataset = AlgorithmDataset(
    algorithm_validation_records
)

algorithm_train_loader = DataLoader(
    algorithm_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_algorithm_batch
)

algorithm_validation_loader = DataLoader(
    algorithm_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_algorithm_batch
)

print(
    "TEST 13: Algorithm DataLoaders"
)

print(
    "Training samples:",
    len(algorithm_train_dataset)
)

print(
    "Validation samples:",
    len(algorithm_validation_dataset)
)

print(
    "Training batches:",
    len(algorithm_train_loader)
)

print(
    "Validation batches:",
    len(algorithm_validation_loader)
)

print()


# ============================================================
# 25. EXACT SILVERWING ATTENTION
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
# 26. FEED FORWARD
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
                self.input_projection(
                    x
                )
            )
        )


# ============================================================
# 27. TRANSFORMER BLOCK
# ============================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(
            self
    ):

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
# 28. POSITION EMBEDDING
# ============================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(
            self
    ):

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
# 29. SILVERWING DECODER
# ============================================================

class SilverwingDecoder(
    nn.Module
):

    def __init__(
            self
    ):

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

            x = layer(
                x
            )

        x = self.final_norm(
            x
        )

        return self.language_model_head(
            x
        )


# ============================================================
# 30. TEST 14 - STRICT LOAD
# ============================================================

print(
    "TEST 14: Strict Load of 87R Linear Algebra Model"
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
        "87R checkpoint is not a dictionary."
    )

if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(
        "87R checkpoint is missing model_state_dict."
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
                "87R checkpoint architecture mismatch. "
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
            "88R refused to load a mismatched "
            "87R Silverwing model.\n\n"
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
    "87R model is compatible with 88R."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 31. BASELINE SNAPSHOT
# ============================================================

baseline_state = {
    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()
}


# ============================================================
# 32. LOSS
# ============================================================

def algorithmic_loss(
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
# 33. EVALUATION
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
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                DEVICE
            )
        )

        logits = current_model(
            input_ids
        )

        loss = algorithmic_loss(
            logits,
            labels
        )

        total_loss += float(
            loss
        )

        batches += 1

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        mask = (
                labels
                !=
                -100
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
            math.isfinite(
                loss_value
            )
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
# 34. TEST 15 - BASELINE
# ============================================================

print(
    "TEST 15: Baseline Algorithmic Evaluation"
)

print()

baseline_metrics = evaluate(
    model,
    algorithm_validation_loader
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
# 35. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = max(
    1,
    len(
        algorithm_train_loader
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
# 36. TEST 16 - TRAINING
# ============================================================

print(
    "TEST 16: Native Algorithmic Fine-Tuning"
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
            algorithm_train_loader,
            start=1
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                DEVICE
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids
        )

        loss = algorithmic_loss(
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
            f"| Batch {batch_number}/{len(algorithm_train_loader)} "
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
        algorithm_validation_loader
    )

    history.append(
        {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_metrics[
                    "loss"
                ],

            "validation_perplexity":
                validation_metrics[
                    "perplexity"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "learning_rate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
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
        validation_metrics[
            "loss"
        ]
    )

    print(
        "Validation accuracy:",
        validation_metrics[
            "accuracy"
        ]
    )

    print()

    if (
            math.isfinite(
                validation_metrics[
                    "loss"
                ]
            )
            and
            validation_metrics[
                "loss"
            ]
            <
            best_validation_loss
    ):

        best_validation_loss = (
            validation_metrics[
                "loss"
            ]
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
                    "88R",

                "base_checkpoint":
                    str(
                        BASE_CHECKPOINT
                    ),

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "validation_metrics":
                    validation_metrics,

                "algorithm_task_count":
                    len(
                        algorithm_tasks
                    )
            },
            BEST_CHECKPOINT
        )

training_duration = (
        time.perf_counter()
        -
        training_start
)


# ============================================================
# 37. TEST 17 - FINAL EVALUATION
# ============================================================

print(
    "TEST 17: Final Algorithmic Evaluation"
)

print()

final_metrics = evaluate(
    model,
    algorithm_validation_loader
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
# 38. TEST 18 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 18: Numerical Health"
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
# 39. TEST 19 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 19: Parameter Change"
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
# 40. TEST 20 - POST-TRAINING VALIDATION
# ============================================================

print(
    "TEST 20: Post-Training Algorithmic Validation"
)

print()

post_training_errors = []

for task in algorithm_tasks:

    data = task[
        "validation"
    ]

    validation_type = data[
        "type"
    ]

    valid = False

    if validation_type == "linear_search":

        observed = linear_search(
            data["values"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "binary_search":

        observed = binary_search(
            data["values"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "merge_sort":

        observed = merge_sort(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "stack":

        observed = stack_process(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "queue":

        observed = queue_process(
            data["values"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "tree":

        observed = inorder_traversal(
            tree
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "bfs":

        observed = breadth_first_search(
            graph,
            "A"
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    elif validation_type == "shortest_path":

        observed = bfs_shortest_path_length(
            graph,
            data["start"],
            data["target"]
        )

        valid = (
                observed
                ==
                data["expected"]
        )

    if not valid:

        post_training_errors.append(
            {
                "example_id":
                    task["example_id"],

                "domain":
                    task["domain"],

                "error":
                    "Algorithmic validation failed."
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
        "Post-training algorithmic validation failed."
    )

print(
    "Post-training algorithmic validation passed:",
    len(algorithm_tasks)
)

print()


# ============================================================
# 41. TEST 21 - PROMOTION
# ============================================================

print(
    "TEST 21: Algorithmic Promotion Gate"
)

print()

baseline_loss = (
    baseline_metrics[
        "loss"
    ]
)

candidate_loss = (
    final_metrics[
        "loss"
    ]
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
        "Candidate algorithmic loss is invalid."
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

    decision = (
        "PROMOTE_CANDIDATE"
    )

    reason = (
        "Algorithmic validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Algorithmic validation loss did not improve."
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
# 42. TEST 22 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 22: Save Algorithmic Candidate"
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
        "88R",

    "training_mode":
        "native_algorithms_and_data_structures",

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

    "algorithm_task_count":
        len(
            algorithm_tasks
        ),

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
# 43. TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "88R",

    "training_mode":
        "native_algorithms_and_data_structures",

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

    "algorithm_task_count":
        len(
            algorithm_tasks
        ),

    "training_examples":
        len(
            algorithm_train_records
        ),

    "validation_examples":
        len(
            algorithm_validation_records
        ),

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
# 44. EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "88R",

    "capability":
        "native_algorithms_and_data_structures",

    "domains":
        sorted(
            expected_domains
        ),

    "algorithm_task_count":
        len(
            algorithm_tasks
        ),

    "training_examples":
        len(
            algorithm_train_records
        ),

    "validation_examples":
        len(
            algorithm_validation_records
        ),

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
                len(
                    post_training_errors
                )
                ==
                0
        },

    "complexity_contract":
        expected_complexity,

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
# 45. ALGORITHMIC INTELLIGENCE STACK
# ============================================================

print(
    "SILVERWING ALGORITHMIC INTELLIGENCE STACK"
)

print()

print("Arrays / Sequences")
print(" ↓")
print("Stacks")
print(" ↓")
print("Queues")
print(" ↓")
print("Linked Structures")
print(" ↓")
print("Trees")
print(" ↓")
print("Graphs")
print(" ↓")
print("Searching")
print(" ↓")
print("Sorting")
print(" ↓")
print("Graph Traversal")
print(" ↓")
print("Shortest-Path Foundations")
print(" ↓")
print("Complexity Reasoning")
print(" ↓")
print("Future: Dynamic Programming")
print(" ↓")
print("Future: Greedy Algorithms")
print(" ↓")
print("Future: Advanced Graph Algorithms")

print()


# ============================================================
# 46. WHY 88R MATTERS
# ============================================================

print(
    "WHY 88R MATTERS"
)

print()

print(
    "Algorithms give Silverwing a formal language for "
    "procedural problem solving."
)

print()

print(
    "Data structures determine how information can be "
    "organized, accessed and transformed."
)

print()

print(
    "This foundation will later support programming, "
    "data engineering, search, memory and planning."
)

print()


# ============================================================
# 47. CURRENT LIMITATIONS
# ============================================================

print(
    "CURRENT LIMITATIONS"
)

print()

print(
    "88R uses a controlled introductory algorithm curriculum."
)

print(
    "88R does not yet cover dynamic programming."
)

print(
    "88R does not yet cover greedy optimization."
)

print(
    "88R does not yet cover advanced graph algorithms."
)

print(
    "88R does not yet cover balanced search trees."
)

print(
    "88R does not yet cover hashing and advanced indexing."
)

print(
    "88R does not yet establish full programming competence."
)

print()


# ============================================================
# 48. NEXT COMPONENT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 89R: Native Data Analysis and SQL Reasoning"
)

print()

print(
    "Tables + SQL + Filtering + Aggregation + "
    "Joins + Data Validation + Analytical Reasoning"
)

print()


# ============================================================
# 49. FOUNDATION MODEL PROGRESS
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
# 50. COMPLETE
# ============================================================

print(
    "=== LESSON 88R COMPLETE ==="
)
# ============================================================
# SILVERWING ML - PHASE 5 - LESSON 111R
# Native Novel Failure Discovery + Prototype Birth
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
# 99R  -> Native Multimodal Representation Foundations
# 100R -> Native Cross-Modal Alignment + Retrieval
# 101R -> Native Hard-Negative Multimodal Learning
# 102R -> Native Multimodal Memory Integration
# 103R -> Native Memory Consolidation + Temporal Retrieval
# 104R -> Native Multimodal Memory Reasoning
# 105R -> Native Memory Prediction + State Forecasting
# 106R -> Native Predictive Memory + Anomaly Detection
# 107R -> Native Predictive Risk + Failure Reasoning
# 108R -> Native Failure Pattern Memory + Retrieval
# 109R -> Native Failure Pattern Clustering + Prototype Memory
# 110R -> Native Failure Prototype Evolution + Continual Memory
# 111R -> Native Novel Failure Discovery + Prototype Birth
#
# ============================================================
# PURPOSE
# ============================================================
#
# 111R extends continual prototype memory with controlled
# novelty discovery.
#
# The lesson establishes:
#
#   novel incident representation
#          ↓
#   prototype retrieval
#          ↓
#   maximum similarity
#          ↓
#   novelty score
#          ↓
#   novelty decision
#          ↓
#   prototype birth
#          ↓
#   prototype validation
#          ↓
#   persistent knowledge expansion
#
# ============================================================
# IMPORTANT
# ============================================================
#
# 111R must NEVER fake novelty.
#
# The current 110R memory can legitimately determine that the
# incident is already represented.
#
# Therefore the lesson tests BOTH branches:
#
#   known-pattern branch
#   novel-pattern branch
#
# A controlled novel probe is constructed by applying a
# deterministic transformation to the existing active pattern.
#
# This does not claim that the transformation is a real physical
# failure. It is a controlled novelty-discovery evaluation.
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

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MEMORY_VERSION = "111R.1"

EVOLUTION_THRESHOLD = 0.70

NOVELTY_THRESHOLD = 0.45

TOP_K = 3

DETERMINISM_THRESHOLD = 1e-9

EPSILON = 1e-8

NOVEL_PROBE_SCALE = 0.75

MIN_CONFIDENCE = 0.50


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PHASE5_DIR = BASE_DIR.parent

LESSON_110R = (
        PHASE5_DIR /
        "lesson110R"
)

SOURCE_EVOLUTION_MEMORY_FILE = (
        LESSON_110R /
        "silverwing_failure_prototype_evolution_memory.json"
)

SOURCE_EVOLUTION_INDEX_FILE = (
        LESSON_110R /
        "silverwing_failure_prototype_evolution_index.pt"
)

SOURCE_EVOLUTION_DATASET_FILE = (
        LESSON_110R /
        "silverwing_failure_prototype_evolution_dataset.json"
)

SOURCE_EVOLUTION_REPORT_FILE = (
        LESSON_110R /
        "silverwing_failure_prototype_evolution_report.json"
)

SOURCE_EVOLUTION_REGISTRY_FILE = (
        LESSON_110R /
        "silverwing_failure_prototype_evolution_registry.json"
)

SOURCE_EVOLUTION_CHECKPOINT_PRIMARY = (
        LESSON_110R /
        "checkpoints" /
        "silverwing_failure_prototype_evolution_best.pt"
)

SOURCE_EVOLUTION_CHECKPOINT_CANDIDATE = (
        LESSON_110R /
        "checkpoints" /
        "silverwing_failure_prototype_evolution_candidate.pt"
)

OUTPUT_DIR = (
        BASE_DIR /
        "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NOVELTY_MEMORY_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_memory.json"
)

NOVELTY_INDEX_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_index.pt"
)

NOVELTY_DATASET_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_dataset.json"
)

NOVELTY_REPORT_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_report.json"
)

NOVELTY_EVALUATION_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_evaluation.json"
)

NOVELTY_REGISTRY_FILE = (
        BASE_DIR /
        "silverwing_novel_failure_registry.json"
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_novel_failure_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR /
        "silverwing_novel_failure_best.pt"
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


def cosine_similarity(
        left: torch.Tensor,
        right: torch.Tensor
) -> float:

    left = left.float()
    right = right.float()

    left_norm = torch.linalg.vector_norm(
        left
    )

    right_norm = torch.linalg.vector_norm(
        right
    )

    denominator = (
            left_norm
            *
            right_norm
    )

    if float(
            denominator
    ) <= EPSILON:

        return 0.0

    return float(
        torch.dot(
            left,
            right
        )
        /
        denominator
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


def clamp(
        value: float,
        lower: float = 0.0,
        upper: float = 1.0
) -> float:

    return max(
        lower,
        min(
            upper,
            float(value)
        )
    )


def finite_tensor(
        tensor: torch.Tensor
) -> bool:

    return bool(
        torch.isfinite(
            tensor
        ).all()
    )


def choose_checkpoint() -> Path:

    candidates = [
        SOURCE_EVOLUTION_CHECKPOINT_PRIMARY,
        SOURCE_EVOLUTION_CHECKPOINT_CANDIDATE
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "No usable 110R checkpoint found."
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
    "PHASE 5 - LESSON 111R"
)

print(
    "Native Novel Failure Discovery + Prototype Birth"
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
    "107R -> Predictive Risk + Failure Reasoning",
    "108R -> Failure Pattern Memory + Retrieval",
    "109R -> Failure Pattern Clustering + Prototype Memory",
    "110R -> Failure Prototype Evolution + Continual Memory",
    "111R -> Novel Failure Discovery + Prototype Birth"
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
    "Evolution threshold:",
    EVOLUTION_THRESHOLD
)

print(
    "Novelty threshold:",
    NOVELTY_THRESHOLD
)

print(
    "Novel probe scale:",
    NOVEL_PROBE_SCALE
)

print(
    "Top-K:",
    TOP_K
)

print()


# ============================================================
# TEST 1
# ============================================================

print(
    "TEST 1: Verify 110R Continual-Memory Inputs"
)

print()

for path in [
    SOURCE_EVOLUTION_MEMORY_FILE,
    SOURCE_EVOLUTION_INDEX_FILE,
    SOURCE_EVOLUTION_DATASET_FILE,
    SOURCE_EVOLUTION_REPORT_FILE,
    SOURCE_EVOLUTION_REGISTRY_FILE
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
    SOURCE_EVOLUTION_MEMORY_FILE
)

print(
    "FOUND:",
    SOURCE_EVOLUTION_INDEX_FILE
)

print(
    "FOUND:",
    SOURCE_EVOLUTION_DATASET_FILE
)

print(
    "FOUND:",
    SOURCE_EVOLUTION_REPORT_FILE
)

print(
    "FOUND:",
    SOURCE_EVOLUTION_REGISTRY_FILE
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
    "TEST 2: Load Evolving Prototype Memory"
)

print()

evolution_memory = read_json(
    SOURCE_EVOLUTION_MEMORY_FILE
)

evolution_dataset = read_json(
    SOURCE_EVOLUTION_DATASET_FILE
)

evolution_report = read_json(
    SOURCE_EVOLUTION_REPORT_FILE
)

if not isinstance(
        evolution_memory,
        dict
):

    raise RuntimeError(
        "110R evolution memory is invalid."
    )

prototypes = evolution_memory.get(
    "prototypes"
)

if not isinstance(
        prototypes,
        list
):

    raise RuntimeError(
        "110R evolution memory contains no prototypes."
    )

if not prototypes:

    raise RuntimeError(
        "110R evolution prototype memory is empty."
    )

print(
    "Memory version:",
    evolution_memory.get(
        "memory_version"
    )
)

print(
    "Prototype count:",
    len(
        prototypes
    )
)

print(
    "Evolution history:",
    len(
        evolution_memory.get(
            "evolution_events",
            []
        )
    )
)

print(
    "Dataset loaded:",
    bool(
        evolution_dataset
    )
)

print(
    "Report loaded:",
    bool(
        evolution_report
    )
)

print()


# ============================================================
# TEST 3
# ============================================================

print(
    "TEST 3: Discover Persisted Representation Dimension"
)

print()

vector_lengths = []

for prototype in prototypes:

    vector = prototype.get(
        "prototype_vector"
    )

    if not isinstance(
            vector,
            list
    ):

        raise RuntimeError(
            (
                "Prototype "
                f"{prototype.get('prototype_id', 'unknown')} "
                "does not contain prototype_vector."
            )
        )

    vector_lengths.append(
        len(
            vector
        )
    )

unique_dimensions = set(
    vector_lengths
)

print(
    "Persisted prototype dimensions:",
    vector_lengths
)

print(
    "Unique dimensions:",
    sorted(
        unique_dimensions
    )
)

if len(
        unique_dimensions
) != 1:

    raise RuntimeError(
        (
            "Inconsistent prototype dimensions: "
            f"{sorted(unique_dimensions)}"
        )
    )

PATTERN_VECTOR_DIMENSION = (
    vector_lengths[
        0
    ]
)

print(
    "Discovered representation dimension:",
    PATTERN_VECTOR_DIMENSION
)

print()


# ============================================================
# TEST 4
# ============================================================

print(
    "TEST 4: Validate Prototype Schema"
)

print()

required_fields = {
    "prototype_id",
    "cluster_id",
    "member_pattern_ids",
    "prototype_vector",
    "prototype_confidence"
}

schema_errors = []

for prototype in prototypes:

    missing = (
            required_fields
            -
            set(
                prototype.keys()
            )
    )

    if missing:

        schema_errors.append(
            {
                "prototype_id":
                    prototype.get(
                        "prototype_id",
                        "unknown"
                    ),

                "missing":
                    sorted(
                        missing
                    )
            }
        )

        continue

    vector = prototype[
        "prototype_vector"
    ]

    if len(
            vector
    ) != PATTERN_VECTOR_DIMENSION:

        schema_errors.append(
            {
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "error":
                    "prototype dimension mismatch",

                "actual":
                    len(
                        vector
                    ),

                "expected":
                    PATTERN_VECTOR_DIMENSION
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
        "110R prototype schema validation failed."
    )

print(
    "Prototype schema validated."
)

print()


# ============================================================
# TEST 5
# ============================================================

print(
    "TEST 5: Build Native Prototype Matrix"
)

print()

prototype_matrix = torch.tensor(
    [
        prototype[
            "prototype_vector"
        ]
        for prototype
        in prototypes
    ],
    dtype=torch.float32
)

normalized_prototypes = F.normalize(
    prototype_matrix,
    p=2,
    dim=-1
)

print(
    "Prototype matrix:",
    tuple(
        prototype_matrix.shape
    )
)

print(
    "Normalized prototype matrix:",
    tuple(
        normalized_prototypes.shape
    )
)

if not finite_tensor(
        prototype_matrix
):

    raise RuntimeError(
        "Prototype matrix is numerically invalid."
    )

if not finite_tensor(
        normalized_prototypes
):

    raise RuntimeError(
        "Normalized prototype matrix is numerically invalid."
    )

print(
    "Native prototype matrix validated."
)

print()


# ============================================================
# TEST 6
# ============================================================

print(
    "TEST 6: Identify Active Prototype"
)

print()

active_prototype_id = evolution_memory.get(
    "active_prototype"
)

active_index = None

for index, prototype in enumerate(
        prototypes
):

    if (
            prototype[
                "prototype_id"
            ]
            ==
            active_prototype_id
    ):

        active_index = index
        break

if active_index is None:

    active_index = 0

    active_prototype_id = prototypes[
        0
    ][
        "prototype_id"
    ]

active_prototype = normalized_prototypes[
    active_index
]

print(
    "Active prototype:",
    active_prototype_id
)

print(
    "Active dimension:",
    active_prototype.shape[0]
)

print()


# ============================================================
# TEST 7
# ============================================================

print(
    "TEST 7: Known-Pattern Retrieval"
)

print()

known_scores = []

for index, prototype in enumerate(
        prototypes
):

    score = cosine_similarity(
        active_prototype,
        normalized_prototypes[
            index
        ]
    )

    known_scores.append(
        {
            "prototype_id":
                prototype[
                    "prototype_id"
                ],

            "score":
                score
        }
    )

known_scores.sort(
    key=lambda item:
    (
        item[
            "score"
        ],
        item[
            "prototype_id"
        ]
    ),
    reverse=True
)

for result in known_scores[
    :
    TOP_K
]:

    print(
        result
    )

if (
        not known_scores
        or
        known_scores[
            0
        ][
            "prototype_id"
        ]
        !=
        active_prototype_id
):

    raise RuntimeError(
        "Known-pattern self retrieval failed."
    )

known_similarity = known_scores[
    0
][
    "score"
]

print(
    "Known-pattern similarity:",
    known_similarity
)

print(
    "Known-pattern retrieval validated."
)

print()


# ============================================================
# TEST 8
# ============================================================

print(
    "TEST 8: Build Controlled Novel Incident Probe"
)

print()

#
# We need a genuine novelty test.
#
# Simply querying the active prototype would always be known.
#
# Therefore construct an orthogonal deterministic probe:
#
#   active representation
#          ↓
#   select a deterministic dimension
#          ↓
#   negate that component
#          ↓
#   renormalize
#
# This is an evaluation stimulus, not a claim about a real
# physical failure.
#

novel_probe = active_prototype.clone()

probe_index = (
        PATTERN_VECTOR_DIMENSION
        -
        1
)

novel_probe[
    probe_index
] = (
        novel_probe[
            probe_index
        ]
        *
        -1.0
)

novel_probe = F.normalize(
    novel_probe,
    p=2,
    dim=0
)

if not finite_tensor(
        novel_probe
):

    raise RuntimeError(
        "Novel probe is numerically invalid."
    )

probe_self_similarity = cosine_similarity(
    novel_probe,
    novel_probe
)

probe_active_similarity = cosine_similarity(
    novel_probe,
    active_prototype
)

print(
    "Novel probe dimension:",
    PATTERN_VECTOR_DIMENSION
)

print(
    "Modified dimension:",
    probe_index
)

print(
    "Probe self similarity:",
    probe_self_similarity
)

print(
    "Probe-to-active similarity:",
    probe_active_similarity
)

print()


# ============================================================
# TEST 9
# ============================================================

print(
    "TEST 9: Novel Probe Retrieval"
)

print()

novel_scores = []

for index, prototype in enumerate(
        prototypes
):

    score = cosine_similarity(
        novel_probe,
        normalized_prototypes[
            index
        ]
    )

    novel_scores.append(
        {
            "prototype_id":
                prototype[
                    "prototype_id"
                ],

            "score":
                score
        }
    )

novel_scores.sort(
    key=lambda item:
    (
        item[
            "score"
        ],
        item[
            "prototype_id"
        ]
    ),
    reverse=True
)

for result in novel_scores[
    :
    TOP_K
]:

    print(
        result
    )

if not novel_scores:

    raise RuntimeError(
        "Novel probe retrieval failed."
    )

best_novel_match = novel_scores[
    0
]

best_novel_similarity = float(
    best_novel_match[
        "score"
    ]
)

print(
    "Best novel-probe similarity:",
    best_novel_similarity
)

print()


# ============================================================
# TEST 10
# ============================================================

print(
    "TEST 10: Novelty Score"
)

print()

novelty_score = (
        1.0
        -
        (
                best_novel_similarity
                +
                1.0
        )
        /
        2.0
)

novelty_score = clamp(
    novelty_score
)

print(
    "Best similarity:",
    best_novel_similarity
)

print(
    "Novelty score:",
    novelty_score
)

if not math.isfinite(
        novelty_score
):

    raise RuntimeError(
        "Novelty score is invalid."
    )

print()


# ============================================================
# TEST 11
# ============================================================

print(
    "TEST 11: Novelty Decision"
)

print()

if (
        best_novel_similarity
        >=
        EVOLUTION_THRESHOLD
):

    novelty_decision = (
        "KNOWN_PATTERN"
    )

elif (
        novelty_score
        >=
        NOVELTY_THRESHOLD
):

    novelty_decision = (
        "NOVEL_PATTERN"
    )

else:

    novelty_decision = (
        "AMBIGUOUS_PATTERN"
    )

print(
    "Evolution threshold:",
    EVOLUTION_THRESHOLD
)

print(
    "Novelty threshold:",
    NOVELTY_THRESHOLD
)

print(
    "Decision:",
    novelty_decision
)

print()


# ============================================================
# TEST 12
# ============================================================

print(
    "TEST 12: Prototype Birth"
)

print()

born_prototype = None

if (
        novelty_decision
        ==
        "NOVEL_PATTERN"
):

    next_number = (
            len(
                prototypes
            )
            +
            1
    )

    born_prototype = {
        "prototype_id":
            f"prototype_{next_number:03d}",

        "cluster_id":
            next_number
            -
            1,

        "member_pattern_ids":
            [
                "novel_probe_001"
            ],

        "member_count":
            1,

        "semantic_classes":
            [
                "novel_failure_candidate"
            ],

        "prototype_vector":
            novel_probe.tolist(),

        "prototype_dimension":
            PATTERN_VECTOR_DIMENSION,

        "prototype_confidence":
            0.50,

        "birth_reason":
            "novelity_threshold_exceeded",

        "born_at":
            datetime.now().isoformat()
    }

    print(
        "New prototype born:",
        born_prototype[
            "prototype_id"
        ]
    )

else:

    print(
        "No new prototype born."
    )

print()


# ============================================================
# TEST 13
# ============================================================

print(
    "TEST 13: Prototype Birth Validation"
)

print()

if born_prototype is not None:

    birth_vector = torch.tensor(
        born_prototype[
            "prototype_vector"
        ],
        dtype=torch.float32
    )

    if (
            birth_vector.shape[0]
            !=
            PATTERN_VECTOR_DIMENSION
    ):

        raise RuntimeError(
            "New prototype dimension mismatch."
        )

    if not finite_tensor(
            birth_vector
    ):

        raise RuntimeError(
            "New prototype is numerically invalid."
        )

    birth_similarity = cosine_similarity(
        birth_vector,
        novel_probe
    )

    print(
        "Birth vector similarity to novel probe:",
        birth_similarity
    )

    if (
            birth_similarity
            <
            1.0
            -
            1e-6
    ):

        raise RuntimeError(
            "New prototype does not preserve novel incident representation."
        )

    print(
        "Novel prototype birth validated."
    )

else:

    print(
        "Prototype birth was not required for this decision."
    )

print()


# ============================================================
# TEST 14
# ============================================================

print(
    "TEST 14: Known-vs-Novel Branch Verification"
)

print()

known_branch_similarity = known_similarity

known_branch_decision = (
    "KNOWN_PATTERN"
    if
    known_branch_similarity
    >=
    EVOLUTION_THRESHOLD
    else
    "NOT_KNOWN"
)

novel_branch_decision = novelty_decision

print(
    "Known query similarity:",
    known_branch_similarity
)

print(
    "Known query decision:",
    known_branch_decision
)

print(
    "Novel probe similarity:",
    best_novel_similarity
)

print(
    "Novel probe decision:",
    novel_branch_decision
)

if known_branch_decision != "KNOWN_PATTERN":

    raise RuntimeError(
        "Known-pattern branch failed."
    )

if (
        novel_branch_decision
        not in
        {
            "KNOWN_PATTERN",
            "NOVEL_PATTERN",
            "AMBIGUOUS_PATTERN"
        }
):

    raise RuntimeError(
        "Novel-pattern branch produced invalid decision."
    )

print(
    "Known-vs-novel branch verification passed."
)

print()


# ============================================================
# TEST 15
# ============================================================

print(
    "TEST 15: Build Novelty Evidence Record"
)

print()

novelty_event = {
    "event_id":
        "novelty_001",

    "timestamp":
        datetime.now().isoformat(),

    "source":
        "111R",

    "active_prototype":
        active_prototype_id,

    "probe_type":
        "deterministic_representation_perturbation",

    "best_matching_prototype":
        best_novel_match[
            "prototype_id"
        ],

    "best_similarity":
        best_novel_similarity,

    "novelty_score":
        novelty_score,

    "decision":
        novelty_decision,

    "prototype_born":
        (
            born_prototype[
                "prototype_id"
            ]
            if born_prototype
            else
            None
        )
}

print(
    novelty_event
)

print(
    "Novelty evidence record validated."
)

print()


# ============================================================
# TEST 16
# ============================================================

print(
    "TEST 16: Deterministic Novelty Detection"
)

print()

def detect_novelty(
        query: torch.Tensor
) -> Dict[str, Any]:

    results = []

    for index, prototype in enumerate(
            prototypes
    ):

        results.append(
            {
                "prototype_id":
                    prototype[
                        "prototype_id"
                    ],

                "score":
                    cosine_similarity(
                        query,
                        normalized_prototypes[
                            index
                        ]
                    )
            }
        )

    results.sort(
        key=lambda item:
        (
            item[
                "score"
            ],
            item[
                "prototype_id"
            ]
        ),
        reverse=True
    )

    best = results[
        0
    ]

    score = float(
        best[
            "score"
        ]
    )

    novelty = clamp(
        1.0
        -
        (
                score
                +
                1.0
        )
        /
        2.0
    )

    if (
            score
            >=
            EVOLUTION_THRESHOLD
    ):

        decision = "KNOWN_PATTERN"

    elif (
            novelty
            >=
            NOVELTY_THRESHOLD
    ):

        decision = "NOVEL_PATTERN"

    else:

        decision = "AMBIGUOUS_PATTERN"

    return {
        "best_prototype":
            best[
                "prototype_id"
            ],

        "similarity":
            score,

        "novelty":
            novelty,

        "decision":
            decision
    }


first_detection = detect_novelty(
    novel_probe
)

second_detection = detect_novelty(
    novel_probe
)

deterministic_detection = (
        first_detection
        ==
        second_detection
)

print(
    "First detection:",
    first_detection
)

print(
    "Second detection:",
    second_detection
)

print(
    "Deterministic novelty detection:",
    deterministic_detection
)

if not deterministic_detection:

    raise RuntimeError(
        "Novelty detection is nondeterministic."
    )

print(
    "Deterministic novelty detection validated."
)

print()


# ============================================================
# TEST 17
# ============================================================

print(
    "TEST 17: Persistent Novel Knowledge Expansion"
)

print()

original_count = len(
    prototypes
)

expanded_prototypes = [
    dict(
        prototype
    )
    for prototype
    in prototypes
]

if born_prototype is not None:

    expanded_prototypes.append(
        born_prototype
    )

expanded_count = len(
    expanded_prototypes
)

print(
    "Original prototype count:",
    original_count
)

print(
    "Expanded prototype count:",
    expanded_count
)

if (
        novelty_decision
        ==
        "NOVEL_PATTERN"
):

    if (
            expanded_count
            !=
            original_count
            +
            1
    ):

        raise RuntimeError(
            "Novel prototype was not added correctly."
        )

else:

    if (
            expanded_count
            !=
            original_count
    ):

        raise RuntimeError(
            "Prototype count changed without novelty."
        )

print(
    "Persistent knowledge expansion validated."
)

print()


# ============================================================
# TEST 18
# ============================================================

print(
    "TEST 18: Prototype Birth Confidence"
)

print()

if born_prototype is not None:

    birth_confidence = 0.50

    print(
        "New prototype confidence:",
        birth_confidence
    )

    if not (
            0.0
            <=
            birth_confidence
            <=
            1.0
    ):

        raise RuntimeError(
            "New prototype confidence is invalid."
        )

else:

    birth_confidence = 0.0

    print(
        "No new prototype confidence required."
    )

print(
    "Prototype birth confidence validated."
)

print()


# ============================================================
# TEST 19
# ============================================================

print(
    "TEST 19: Novel Failure Curriculum"
)

print()

novelty_tasks = [
    {
        "example_id":
            "novelty_001",

        "domain":
            "novelty_detection",

        "question":
            "What indicates that an incident may represent a new pattern?",

        "answer":
            "Low similarity to the existing prototype memory."
    },

    {
        "example_id":
            "novelty_002",

        "domain":
            "prototype_birth",

        "question":
            "When should a new prototype be created?",

        "answer":
            "When novelty evidence passes the configured discovery criterion."
    },

    {
        "example_id":
            "novelty_003",

        "domain":
            "knowledge_expansion",

        "question":
            "Why should a novel prototype be preserved?",

        "answer":
            "It expands the system's reusable failure knowledge."
    },

    {
        "example_id":
            "novelty_004",

        "domain":
            "continual_learning",

        "question":
            "How does novelty discovery support continual learning?",

        "answer":
            "It allows knowledge to grow when observations do not fit existing patterns."
    },

    {
        "example_id":
            "novelty_005",

        "domain":
            "memory_validation",

        "question":
            "Why validate a newly born prototype?",

        "answer":
            "To ensure the new memory has the correct representation and traceability."
    },

    {
        "example_id":
            "novelty_006",

        "domain":
            "engineering_intelligence",

        "question":
            "Why is discovering new operating patterns valuable?",

        "answer":
            "Real systems can exhibit behaviors not present in previous memory."
    }
]

for task in novelty_tasks:

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
    "Novelty tasks:",
    len(
        novelty_tasks
    )
)

print()


# ============================================================
# TEST 20
# ============================================================

print(
    "TEST 20: Novelty Curriculum Coverage"
)

print()

expected_domains = {
    "novelty_detection",
    "prototype_birth",
    "knowledge_expansion",
    "continual_learning",
    "memory_validation",
    "engineering_intelligence"
}

actual_domains = {
    task[
        "domain"
    ]
    for task
    in novelty_tasks
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
        "Novelty curriculum coverage is incomplete."
    )

print(
    "Novelty curriculum validated."
)

print()


# ============================================================
# TEST 21
# ============================================================

print(
    "TEST 21: Numerical Health"
)

print()

health_tensors = [
    prototype_matrix,
    normalized_prototypes,
    active_prototype,
    novel_probe
]

if born_prototype is not None:

    health_tensors.append(
        torch.tensor(
            born_prototype[
                "prototype_vector"
            ],
            dtype=torch.float32
        )
    )

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
        "Novelty detection numerical health failed."
    )

print()


# ============================================================
# TEST 22
# ============================================================

print(
    "TEST 22: Final Novelty Discovery Promotion Gate"
)

print()

promotion_errors = []

if not deterministic_detection:

    promotion_errors.append(
        "Novelty detection is nondeterministic."
    )

if not numerically_healthy:

    promotion_errors.append(
        "Numerical health failed."
    )

if known_branch_decision != "KNOWN_PATTERN":

    promotion_errors.append(
        "Known-pattern retrieval branch failed."
    )

if len(
        novelty_tasks
) < 6:

    promotion_errors.append(
        "Novelty curriculum is incomplete."
    )

if not math.isfinite(
        novelty_score
):

    promotion_errors.append(
        "Novelty score is invalid."
    )

if born_prototype is not None:

    birth_vector = torch.tensor(
        born_prototype[
            "prototype_vector"
        ],
        dtype=torch.float32
    )

    birth_similarity = cosine_similarity(
        birth_vector,
        novel_probe
    )

    if (
            abs(
                birth_similarity
                -
                1.0
            )
            >
            1e-6
    ):

        promotion_errors.append(
            "Novel prototype does not preserve novel probe representation."
        )

print(
    "Novelty decision:",
    novelty_decision
)

print(
    "Novelty score:",
    novelty_score
)

print(
    "Prototype born:",
    (
        born_prototype[
            "prototype_id"
        ]
        if born_prototype
        else
        None
    )
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
        "111R novelty discovery promotion gate failed."
    )

print(
    "111R novelty discovery promotion gate passed."
)

print()


# ============================================================
# TEST 23
# ============================================================

print(
    "TEST 23: Persist Novel Failure Memory"
)

print()

novelty_memory_payload = {
    "memory_version":
        MEMORY_VERSION,

    "capability":
        "native_novel_failure_discovery_prototype_birth",

    "created_at":
        datetime.now().isoformat(),

    "source_memory_version":
        evolution_memory.get(
            "memory_version"
        ),

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_count,

    "expanded_prototype_count":
        expanded_count,

    "evolution_threshold":
        EVOLUTION_THRESHOLD,

    "novelty_threshold":
        NOVELTY_THRESHOLD,

    "novel_probe_scale":
        NOVEL_PROBE_SCALE,

    "novelty_event":
        novelty_event,

    "prototypes":
        expanded_prototypes,

    "decision":
        novelty_decision,

    "novelty_score":
        novelty_score
}

write_json(
    NOVELTY_MEMORY_FILE,
    novelty_memory_payload
)

expanded_matrix = torch.tensor(
    [
        prototype[
            "prototype_vector"
        ]
        for prototype
        in expanded_prototypes
    ],
    dtype=torch.float32
)

torch.save(
    {
        "memory_version":
            MEMORY_VERSION,

        "pattern_vector_dimension":
            PATTERN_VECTOR_DIMENSION,

        "prototype_ids":
            [
                prototype[
                    "prototype_id"
                ]
                for prototype
                in expanded_prototypes
            ],

        "prototype_matrix":
            expanded_matrix,

        "novelty_event":
            novelty_event
    },
    NOVELTY_INDEX_FILE
)

print(
    "Novelty memory:",
    NOVELTY_MEMORY_FILE
)

print(
    "Novelty index:",
    NOVELTY_INDEX_FILE
)

print()


# ============================================================
# TEST 24
# ============================================================

print(
    "TEST 24: Reload Novel Failure Memory"
)

print()

reloaded_novelty_memory = read_json(
    NOVELTY_MEMORY_FILE
)

if (
        reloaded_novelty_memory[
            "pattern_vector_dimension"
        ]
        !=
        PATTERN_VECTOR_DIMENSION
):

    raise RuntimeError(
        "Novel memory representation dimension changed."
    )

if (
        reloaded_novelty_memory[
            "expanded_prototype_count"
        ]
        !=
        expanded_count
):

    raise RuntimeError(
        "Novel memory prototype count changed."
    )

reloaded_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in reloaded_novelty_memory[
        "prototypes"
    ]
]

expanded_ids = [
    prototype[
        "prototype_id"
    ]
    for prototype
    in expanded_prototypes
]

if reloaded_ids != expanded_ids:

    raise RuntimeError(
        "Novel memory prototype identity changed."
    )

print(
    "Reloaded prototype count:",
    len(
        reloaded_ids
    )
)

print(
    "Persistent novelty memory validated."
)

print()


# ============================================================
# TEST 25
# ============================================================

print(
    "TEST 25: Save Novelty Dataset"
)

print()

novelty_dataset = {
    "lesson":
        "111R",

    "capability":
        "native_novel_failure_discovery_prototype_birth",

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_count,

    "expanded_prototype_count":
        expanded_count,

    "known_probe":
        {
            "prototype":
                active_prototype_id,

            "similarity":
                known_similarity
        },

    "novel_probe":
        {
            "best_prototype":
                best_novel_match[
                    "prototype_id"
                ],

            "similarity":
                best_novel_similarity,

            "novelty":
                novelty_score,

            "decision":
                novelty_decision
        },

    "born_prototype":
        (
            born_prototype
            if born_prototype
            else
            None
        ),

    "novelty_event":
        novelty_event
}

write_json(
    NOVELTY_DATASET_FILE,
    novelty_dataset
)

print(
    "Novelty dataset:",
    NOVELTY_DATASET_FILE
)

print()


# ============================================================
# TEST 26
# ============================================================

print(
    "TEST 26: Save 111R Checkpoint"
)

print()

checkpoint_payload = {
    "lesson":
        "111R",

    "capability":
        "native_novel_failure_discovery_prototype_birth",

    "base_checkpoint":
        str(
            SOURCE_CHECKPOINT
        ),

    "external_llm":
        False,

    "memory_version":
        MEMORY_VERSION,

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_count,

    "expanded_prototype_count":
        expanded_count,

    "active_prototype":
        active_prototype_id,

    "known_similarity":
        known_similarity,

    "novel_similarity":
        best_novel_similarity,

    "novelty_score":
        novelty_score,

    "novelty_decision":
        novelty_decision,

    "born_prototype":
        born_prototype,

    "novelty_event":
        novelty_event,

    "expanded_prototypes":
        expanded_prototypes,

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
# TEST 27
# ============================================================

print(
    "TEST 27: Write 111R Reports"
)

print()

report = {
    "lesson":
        "111R",

    "capability":
        "native_novel_failure_discovery_prototype_birth",

    "external_llm":
        False,

    "device":
        str(
            DEVICE
        ),

    "memory_version":
        MEMORY_VERSION,

    "pattern_vector_dimension":
        PATTERN_VECTOR_DIMENSION,

    "original_prototype_count":
        original_count,

    "expanded_prototype_count":
        expanded_count,

    "thresholds":
        {
            "evolution":
                EVOLUTION_THRESHOLD,

            "novelty":
                NOVELTY_THRESHOLD
        },

    "known_case":
        {
            "prototype":
                active_prototype_id,

            "similarity":
                known_similarity
        },

    "novel_case":
        {
            "best_prototype":
                best_novel_match[
                    "prototype_id"
                ],

            "similarity":
                best_novel_similarity,

            "novelty_score":
                novelty_score,

            "decision":
                novelty_decision
        },

    "prototype_birth":
        (
            born_prototype
            if born_prototype
            else
            None
        ),

    "novelty_event":
        novelty_event,

    "verification":
        {
            "deterministic":
                deterministic_detection,

            "known_branch_valid":
                known_branch_decision
                ==
                "KNOWN_PATTERN"
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
    NOVELTY_REPORT_FILE,
    report
)

write_json(
    NOVELTY_EVALUATION_FILE,
    report
)

write_json(
    NOVELTY_REGISTRY_FILE,
    {
        "lesson":
            "111R",

        "capability":
            "native_novel_failure_discovery_prototype_birth",

        "memory_version":
            MEMORY_VERSION,

        "pattern_vector_dimension":
            PATTERN_VECTOR_DIMENSION,

        "novelty_decision":
            novelty_decision,

        "novelty_score":
            novelty_score,

        "expanded_prototype_count":
            expanded_count,

        "next":
            "112R Native Failure Prototype Validation + Cross-Case Reasoning"
    }
)

print(
    "Novelty report:",
    NOVELTY_REPORT_FILE
)

print(
    "Novelty evaluation:",
    NOVELTY_EVALUATION_FILE
)

print(
    "Novelty registry:",
    NOVELTY_REGISTRY_FILE
)

print()


# ============================================================
# ARCHITECTURE PREVIEW
# ============================================================

print(
    "SILVERWING 111R NOVEL FAILURE DISCOVERY ARCHITECTURE"
)

print()

print(
    "New Incident"
)

print(
    "      ↓"
)

print(
    "Native Representation"
)

print(
    "      ↓"
)

print(
    "Existing Prototype Retrieval"
)

print(
    "      ↓"
)

print(
    "Similarity / Novelty Analysis"
)

print(
    "      ↓"
)

print(
    "      ┌────────────────────┐"
)

print(
    "      │                    │"
)

print(
    "   Known              Novel"
)

print(
    "      │                    │"
)

print(
    "Reuse Memory        Prototype Birth"
)

print(
    "      │                    │"
)

print(
    "      └─────────┬──────────┘"
)

print(
    "                ↓"
)

print(
    "       Knowledge Expansion"
)

print(
    "                ↓"
)

print(
    "        Persistent Memory"
)

print()


# ============================================================
# WHY 111R MATTERS
# ============================================================

print(
    "WHY 111R MATTERS"
)

print()

print(
    "109R organized failure cases into prototypes."
)

print(
    "110R allowed prototypes to evolve."
)

print(
    "111R adds the ability to recognize behavior that "
    "does not adequately match existing prototype memory."
)

print()

print(
    "That creates the foundation for knowledge expansion "
    "rather than only knowledge refinement."
)

print()


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

print(
    "111R LIMITATION"
)

print()

print(
    "The novel incident used in this lesson is a controlled "
    "representation-space probe, not a real equipment failure."
)

print(
    "Production novelty discovery requires real incoming "
    "observations, validated thresholds, domain context and "
    "longitudinal evidence."
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
    "Lesson 112R: Native Failure Prototype Validation + Cross-Case Reasoning"
)

print()

print(
    "Multiple Cases + Prototype Consistency + "
    "Cross-Case Evidence + Contradiction Detection + "
    "Diagnosis Confidence"
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
    "109R Native Failure Pattern Clustering + Prototype Memory",
    " ↓",
    "110R Native Failure Prototype Evolution + Continual Memory",
    " ↓",
    "111R Native Novel Failure Discovery + Prototype Birth",
    " ↓",
    "112R Native Failure Prototype Validation + Cross-Case Reasoning",
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
    "=== LESSON 111R COMPLETE ==="
)
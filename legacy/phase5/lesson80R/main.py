# ============================================================
# SILVERWING ML
# PHASE 5 - LESSON 80R
# Native Reasoning Fine-Tuning Engine
#
# 79R -> Native Reasoning Dataset + Evaluation
# 80R -> Native Reasoning Fine-Tuning
#
# IMPORTANT:
#   - No GPT-2
#   - No Qwen
#   - No external reasoning model
#   - Uses Silverwing's established decoder architecture
#
# The checkpoint architecture is intentionally kept EXACTLY
# compatible with the established Silverwing checkpoint:
#
#   token_embedding
#   position_embedding.embedding
#   layers.N.attention.query_projection
#   layers.N.attention.key_projection
#   layers.N.attention.value_projection
#   layers.N.attention.output_projection
#   layers.N.feed_forward.input_projection
#   layers.N.feed_forward.output_projection
#   layers.N.norm_attention
#   layers.N.norm_feed_forward
#   final_norm
#   language_model_head
#
# ============================================================

import json
import math
import random
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
LESSON_78R = PHASE5_DIR / "lesson78R"

REASONING_CONFIG_FILE = (
        LESSON_79R / "silverwing_reasoning_config.json"
)

REASONING_DATASET_FILE = (
        LESSON_79R / "silverwing_reasoning_dataset.jsonl"
)

REASONING_TRAIN_FILE = (
        LESSON_79R / "silverwing_reasoning_train.jsonl"
)

REASONING_VALIDATION_FILE = (
        LESSON_79R / "silverwing_reasoning_validation.jsonl"
)

REASONING_REPORT_FILE = (
        LESSON_79R / "silverwing_reasoning_report.json"
)

REASONING_EVAL_FILE = (
        LESSON_79R / "silverwing_reasoning_evaluation.json"
)

VOCABULARY_FILE = (
        LESSON_66R / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66R / "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71R / "silverwing_decoder_config.json"
)

BASE_CHECKPOINT = (
        LESSON_78R
        / "checkpoints"
        / "silverwing_instruction_best.pt"
)

OUTPUT_DIR = (
        BASE_DIR / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_reasoning_candidate.pt"
)

BEST_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_reasoning_best.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_reasoning_training_log.json"
)

EVALUATION_FILE = (
        BASE_DIR
        / "silverwing_reasoning_evaluation.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

MAX_SEQUENCE_LENGTH = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        SEED
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

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


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


def load_jsonl(
        path: Path
) -> List[Dict[str, Any]]:

    records = []

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
                file,
                start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    (
                        f"Invalid JSONL at "
                        f"{path}:{line_number}: "
                        f"{exc}"
                    )
                ) from exc

    return records


# ============================================================
# 4. HEADER
# ============================================================

print(
    "=== SILVERWING ML ==="
)

print(
    "PHASE 5 - LESSON 80R"
)

print(
    "Native Reasoning Fine-Tuning Engine"
)

print()

print(
    "Dataset source: Lesson 79R"
)

print(
    "External LLM: NONE"
)

print()


# ============================================================
# 5. TEST 1 - VERIFY 79R
# ============================================================

print(
    "TEST 1: Verify Lesson 79R Artifacts"
)

print()


for path in [

    REASONING_CONFIG_FILE,

    REASONING_DATASET_FILE,

    REASONING_TRAIN_FILE,

    REASONING_VALIDATION_FILE,

    REASONING_REPORT_FILE,

    REASONING_EVAL_FILE,

    VOCABULARY_FILE,

    MERGES_FILE,

    MODEL_CONFIG_FILE,

    BASE_CHECKPOINT,

]:

    require_file(
        path
    )

    print(
        "FOUND:",
        path
    )


print()


# ============================================================
# 6. TEST 2 - LOAD 79R CONFIG
# ============================================================

print(
    "TEST 2: Load Lesson 79R Configuration"
)

print()


reasoning_config = read_json(
    REASONING_CONFIG_FILE
)


dataset_name = reasoning_config.get(
    "dataset",
    "Silverwing-Reasoning-v1"
)


configured_limit = int(
    reasoning_config.get(
        "max_reasoning_tokens",
        MAX_SEQUENCE_LENGTH
    )
)


MAX_SEQUENCE_LENGTH = min(
    MAX_SEQUENCE_LENGTH,
    configured_limit
)


print(
    "Dataset:",
    dataset_name
)

print(
    "Configured maximum tokens:",
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
    in vocabulary[
        "token_to_id"
    ].items()

}


required_tokens = [

    "<PAD>",

    "<UNK>",

    "<BOS>",

    "<EOS>"

]


missing_tokens = [

    token

    for token
    in required_tokens

    if token not in TOKEN_TO_ID

]


if missing_tokens:

    raise ValueError(
        (
            "Vocabulary is missing required "
            f"tokens: {missing_tokens}"
        )
    )


PAD_ID = TOKEN_TO_ID[
    "<PAD>"
]

UNK_ID = TOKEN_TO_ID[
    "<UNK>"
]

BOS_ID = TOKEN_TO_ID[
    "<BOS>"
]

EOS_ID = TOKEN_TO_ID[
    "<EOS>"
]

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


MERGE_RANKS = {}


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
            not isinstance(
                pair,
                list
            )
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
        item[
            "rank"
        ]
    )


print(
    "Merge operations:",
    len(MERGE_RANKS)
)

print()


# ============================================================
# 9. TEST 5 - MODEL CONFIG
# ============================================================

print(
    "TEST 5: Load Silverwing Decoder Configuration"
)

print()


model_config = read_json(
    MODEL_CONFIG_FILE
)


MODEL_DIMENSION = int(
    model_config[
        "model_dimension"
    ]
)

NUMBER_OF_HEADS = int(
    model_config[
        "attention_heads"
    ]
)

FEED_FORWARD_DIMENSION = int(
    model_config[
        "feed_forward_dimension"
    ]
)

NUMBER_OF_LAYERS = int(
    model_config[
        "layers"
    ]
)

MODEL_MAX_SEQUENCE_LENGTH = int(
    model_config[
        "maximum_sequence_length"
    ]
)


MAX_SEQUENCE_LENGTH = min(
    MAX_SEQUENCE_LENGTH,
    MODEL_MAX_SEQUENCE_LENGTH
)


if (
        MODEL_DIMENSION
        %
        NUMBER_OF_HEADS
        !=
        0
):

    raise ValueError(
        (
            "Model dimension must be divisible "
            "by attention heads."
        )
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
# 10. TOKENIZER
# ============================================================

BPE_END = "</w>"


def split_words(
        text: str
) -> List[str]:

    import re

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

    result = []

    index = 0


    while index < len(symbols):

        if (

                index
                <
                len(symbols) - 1

                and

                (
                        symbols[index],
                        symbols[index + 1]
                )
                ==
                pair

        ):

            result.append(

                symbols[index]
                +
                symbols[index + 1]

            )

            index += 2

        else:

            result.append(
                symbols[index]
            )

            index += 1


    return tuple(
        result
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

    token_ids = [

        BOS_ID

    ]


    for token in tokenize_text(
            text
    ):

        token_ids.append(

            TOKEN_TO_ID.get(

                token,

                UNK_ID

            )

        )


    token_ids.append(
        EOS_ID
    )


    return token_ids


# ============================================================
# 11. REASONING FORMAT
# ============================================================

def format_reasoning_text(
        record: Dict[str, Any]
) -> str:

    problem = str(
        record.get(
            "problem",
            ""
        )
    ).strip()


    context = str(
        record.get(
            "context",
            ""
        )
    ).strip()


    reasoning_steps = record.get(

        "reasoning_steps",

        []

    )


    final_answer = str(
        record.get(
            "final_answer",
            ""
        )
    ).strip()


    lines = [

        "Problem:",

        problem

    ]


    if context:

        lines.extend(

            [

                "",

                "Context:",

                context

            ]

        )


    lines.extend(

        [

            "",

            "Reasoning:"

        ]

    )


    for number, step in enumerate(

            reasoning_steps,

            start=1

    ):

        lines.append(

            f"{number}. "
            f"{str(step).strip()}"

        )


    lines.extend(

        [

            "",

            "Final Answer:",

            final_answer

        ]

    )


    return "\n".join(
        lines
    )


# ============================================================
# 12. TEST 6 - LOAD 79R REASONING DATA
# ============================================================

print(
    "TEST 6: Load Lesson 79R Reasoning Data"
)

print()


train_records = load_jsonl(
    REASONING_TRAIN_FILE
)

validation_records = load_jsonl(
    REASONING_VALIDATION_FILE
)


if not train_records:

    raise RuntimeError(
        "Lesson 79R training dataset is empty."
    )


print(
    "Training examples:",
    len(train_records)
)

print(
    "Validation examples:",
    len(validation_records)
)

print()


# ============================================================
# 13. TEST 7 - RECHECK TOKEN LIMITS
# ============================================================

print(
    "TEST 7: Recheck 79R Token Limits"
)

print()


length_errors = []


for record in (

        train_records
        +
        validation_records

):

    token_count = len(

        encode_text(

            format_reasoning_text(
                record
            )

        )

    )


    record[
        "_80R_token_count"
    ] = token_count


    print(

        record.get(
            "example_id",
            "unknown"
        ),

        "->",

        token_count,

        "tokens"

    )


    if (

            token_count
            >
            MAX_SEQUENCE_LENGTH

    ):

        length_errors.append(

            {

                "example_id":
                    record.get(
                        "example_id",
                        "unknown"
                    ),

                "token_count":
                    token_count,

                "maximum":
                    MAX_SEQUENCE_LENGTH

            }

        )


if length_errors:

    print(

        json.dumps(

            length_errors,

            indent=4

        )

    )


    raise RuntimeError(

        "Lesson 79R contains reasoning examples "
        "above the Lesson 80R sequence limit."

    )


print()


# ============================================================
# 14. DATASET
# ============================================================

class ReasoningDataset(
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

                format_reasoning_text(
                    record
                )

            )


            if (

                    len(token_ids)
                    >
                    MAX_SEQUENCE_LENGTH

            ):

                raise ValueError(

                    (

                        f"{record['example_id']} "
                        f"exceeds "
                        f"{MAX_SEQUENCE_LENGTH} "
                        "tokens."

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


def collate_batch(
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


    input_batch = []

    label_batch = []


    for item in batch:

        input_ids = item[
            "input_ids"
        ]

        labels = item[
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
                len(labels)

        )


        input_batch.append(

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


        label_batch.append(

            torch.cat(

                [

                    labels,

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
                input_batch
            ),

        "labels":

            torch.stack(
                label_batch
            )

    }


train_dataset = ReasoningDataset(
    train_records
)

validation_dataset = ReasoningDataset(
    validation_records
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_batch

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_batch

)


print(
    "TEST 8: Build Reasoning DataLoaders"
)

print()

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(validation_loader)
)

print()


# ============================================================
# 15. EXACT SILVERWING ATTENTION
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


        if (

                dimension
                %
                heads
                !=
                0

        ):

            raise ValueError(

                "Model dimension must be "
                "divisible by attention heads."

            )


        self.dimension = dimension

        self.heads = heads

        self.head_dimension = (

                dimension
                //
                heads

        )


        # EXACT CHECKPOINT NAMES

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

        batch_size = x.shape[
            0
        ]

        sequence_length = x.shape[
            1
        ]


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


        attention_weights = F.softmax(

            scores,

            dim=-1

        )


        attended = torch.matmul(

            attention_weights,

            value

        )


        attended = attended.transpose(

            1,

            2

        ).contiguous()


        attended = attended.view(

            batch_size,

            sequence_length,

            self.dimension

        )


        return self.output_projection(

            attended

        )


# ============================================================
# 16. EXACT SILVERWING FEED FORWARD
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


        # EXACT CHECKPOINT NAMES

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

        x = self.input_projection(
            x
        )


        x = F.gelu(
            x
        )


        return self.output_projection(
            x
        )


# ============================================================
# 17. EXACT SILVERWING TRANSFORMER BLOCK
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


        # EXACT CHECKPOINT NAMES

        self.norm_attention = nn.LayerNorm(

            MODEL_DIMENSION

        )


        self.feed_forward = (

            SilverwingFeedForward(

                MODEL_DIMENSION,

                FEED_FORWARD_DIMENSION

            )

        )


        # EXACT CHECKPOINT NAMES

        self.norm_feed_forward = nn.LayerNorm(

            MODEL_DIMENSION

        )


    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        attention_output = (

            self.attention(
                x
            )

        )


        x = self.norm_attention(

            x
            +
            attention_output

        )


        feed_forward_output = (

            self.feed_forward(
                x
            )

        )


        x = self.norm_feed_forward(

            x
            +
            feed_forward_output

        )


        return x


# ============================================================
# 18. EXACT SILVERWING POSITION EMBEDDING
# ============================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(
            self
    ):

        super().__init__()


        # EXACT CHECKPOINT HIERARCHY:
        #
        # position_embedding.embedding.weight

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
# 19. EXACT SILVERWING DECODER
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

        sequence_length = input_ids.shape[
            1
        ]


        if (

                sequence_length
                >
                MAX_SEQUENCE_LENGTH

        ):

            raise ValueError(

                "Sequence exceeds model limit."

            )


        token_embeddings = (

            self.token_embedding(

                input_ids

            )

        )


        position_embeddings = (

            self.position_embedding(

                sequence_length,

                input_ids.device

            )

        )


        x = (

                token_embeddings

                +

                position_embeddings.unsqueeze(
                    0
                )

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
# 20. TEST 9 - INSPECT ESTABLISHED CHECKPOINT
# ============================================================

print(
    "TEST 9: Inspect Established Silverwing Checkpoint"
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

        "Silverwing checkpoint must be a dictionary."

    )


if (
        "model_state_dict"
        not in checkpoint
):

    raise ValueError(

        "Checkpoint missing model_state_dict."

    )


state_dict = checkpoint[
    "model_state_dict"
]


print(
    "Checkpoint:",
    BASE_CHECKPOINT
)

print(
    "State tensors:",
    len(state_dict)
)

print()


# ============================================================
# 21. TEST 10 - VERIFY EXACT ARCHITECTURE MARKERS
# ============================================================

print(
    "TEST 10: Verify Established Architecture"
)

print()


required_parameter_prefixes = [

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

    "language_model_head.",

]


missing_markers = []


for prefix in required_parameter_prefixes:

    if not any(

            key.startswith(
                prefix
            )

            for key
            in state_dict.keys()

    ):

        missing_markers.append(
            prefix
        )


if missing_markers:

    raise RuntimeError(

        (

                "The selected checkpoint does not match "
                "the established Silverwing architecture.\n\n"

                "Missing architecture markers:\n"

                +

                "\n".join(
                    missing_markers
                )

        )

    )


print(
    "All established architecture markers found."
)

print()


# ============================================================
# 22. TEST 11 - STRICT MODEL LOAD
# ============================================================

print(
    "TEST 11: Strict Silverwing Checkpoint Load"
)

print()


model = (

    SilverwingDecoder()

    .to(
        DEVICE
    )

)


try:

    model.load_state_dict(

        state_dict,

        strict=True

    )


except RuntimeError as exc:

    raise RuntimeError(

        (

            "\nEXACT SILVERWING ARCHITECTURE LOAD FAILED.\n\n"

            "Lesson 80R will not use strict=False.\n"

            "Lesson 80R will not rename weights.\n"

            "Lesson 80R will not partially initialize the model.\n\n"

            "The established Silverwing checkpoint must "
            "load exactly before training starts.\n\n"

            f"Checkpoint:\n{BASE_CHECKPOINT}\n\n"

            f"PyTorch error:\n{exc}"

        )

    ) from exc


print(
    "STRICT LOAD PASSED."
)

print(
    "Silverwing checkpoint architecture is compatible."
)

print(
    "Device:",
    DEVICE
)

print()


# ============================================================
# 23. BASELINE SNAPSHOT
# ============================================================

baseline_state = {

    name:
        parameter.detach().clone()

    for name, parameter
    in model.state_dict().items()

}


# ============================================================
# 24. LOSS
# ============================================================

def reasoning_loss(
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
# 25. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(

        current_model: nn.Module,

        loader: DataLoader

) -> Dict[str, float]:

    current_model.eval()


    total_loss = 0.0

    batch_count = 0

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


        loss = reasoning_loss(

            logits,

            labels

        )


        total_loss += float(
            loss
        )


        batch_count += 1


        predictions = torch.argmax(

            logits,

            dim=-1

        )


        valid_mask = (

                labels
                !=
                -100

        )


        correct += int(

            (

                    predictions[
                        valid_mask
                    ]

                    ==

                    labels[
                        valid_mask
                    ]

            ).sum()

        )


        valid_tokens += int(

            valid_mask.sum()

        )


    if batch_count == 0:

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


    average_loss = (

            total_loss
            /
            batch_count

    )


    if (

            math.isfinite(
                average_loss
            )

            and

            average_loss < 50

    ):

        perplexity = math.exp(

            average_loss

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
            average_loss,

        "perplexity":
            perplexity,

        "accuracy":
            accuracy,

        "tokens":
            valid_tokens

    }


# ============================================================
# 26. TEST 12 - BASELINE
# ============================================================

print(
    "TEST 12: Baseline Reasoning Evaluation"
)

print()


baseline_metrics = evaluate(

    model,

    validation_loader

)


print(
    "Baseline loss:",
    baseline_metrics[
        "loss"
    ]
)

print(
    "Baseline perplexity:",
    baseline_metrics[
        "perplexity"
    ]
)

print(
    "Baseline accuracy:",
    baseline_metrics[
        "accuracy"
    ]
)

print()


# ============================================================
# 27. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


total_steps = max(

    1,

    len(
        train_loader
    )
    *
    EPOCHS

)


scheduler = (

    torch.optim.lr_scheduler

    .CosineAnnealingLR(

        optimizer,

        T_max=total_steps

    )

)


# ============================================================
# 28. TEST 13 - TRAIN
# ============================================================

print(
    "TEST 13: Native Reasoning Fine-Tuning"
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

            train_loader,

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


        loss = reasoning_loss(

            logits,

            labels

        )


        loss.backward()


        gradient_norm = (

            torch.nn.utils

            .clip_grad_norm_(

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
            f"| Batch {batch_number}/{len(train_loader)} "
            f"| Step {global_step} "
            f"| Loss {float(loss.detach()):.6f} "
            f"| Grad {float(gradient_norm):.6f} "
            f"| LR {optimizer.param_groups[0]['lr']:.8f}"

        )


    average_train_loss = (

            epoch_loss
            /
            max(
                epoch_batches,
                1
            )

    )


    validation_metrics = evaluate(

        model,

        validation_loader

    )


    epoch_record = {

        "epoch":
            epoch,

        "train_loss":
            average_train_loss,

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


    history.append(
        epoch_record
    )


    print()

    print(
        "Epoch",
        epoch,
        "complete."
    )

    print(
        "Training loss:",
        average_train_loss
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
                    "80R",

                "dataset":
                    dataset_name,

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

                "architecture":
                    {

                        "model_dimension":
                            MODEL_DIMENSION,

                        "attention_heads":
                            NUMBER_OF_HEADS,

                        "feed_forward_dimension":
                            FEED_FORWARD_DIMENSION,

                        "layers":
                            NUMBER_OF_LAYERS,

                        "vocabulary_size":
                            VOCABULARY_SIZE,

                        "sequence_length":
                            MAX_SEQUENCE_LENGTH

                    }

            },

            BEST_CHECKPOINT

        )


training_duration = (

        time.perf_counter()

        -

        training_start

)


# ============================================================
# 29. TEST 14 - FINAL EVALUATION
# ============================================================

print(
    "TEST 14: Final Reasoning Evaluation"
)

print()


final_metrics = evaluate(

    model,

    validation_loader

)


print(
    "Final loss:",
    final_metrics[
        "loss"
    ]
)

print(
    "Final perplexity:",
    final_metrics[
        "perplexity"
    ]
)

print(
    "Final accuracy:",
    final_metrics[
        "accuracy"
    ]
)

print()


# ============================================================
# 30. TEST 15 - NUMERICAL HEALTH
# ============================================================

print(
    "TEST 15: Numerical Health"
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
# 31. TEST 16 - PARAMETER CHANGE
# ============================================================

print(
    "TEST 16: Parameter Change"
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
# 32. TEST 17 - PROMOTION GATE
# ============================================================

print(
    "TEST 17: Reasoning Promotion Gate"
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
        "Candidate validation loss is invalid."
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
        "Reasoning validation loss improved."
    )

else:

    decision = (
        "RETAIN_BASELINE"
    )

    reason = (
        "Reasoning validation loss did not improve."
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
# 33. TEST 18 - SAVE CANDIDATE
# ============================================================

print(
    "TEST 18: Save Reasoning Candidate"
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
        "80R",

    "dataset":
        dataset_name,

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "baseline_metrics":
        baseline_metrics,

    "candidate_metrics":
        final_metrics,

    "history":
        history,

    "global_step":
        global_step,

    "training_duration_seconds":
        training_duration,

    "promotion_decision":
        decision,

    "promotion_reason":
        reason,

    "architecture":
        {

            "model_dimension":
                MODEL_DIMENSION,

            "attention_heads":
                NUMBER_OF_HEADS,

            "feed_forward_dimension":
                FEED_FORWARD_DIMENSION,

            "layers":
                NUMBER_OF_LAYERS,

            "vocabulary_size":
                VOCABULARY_SIZE,

            "sequence_length":
                MAX_SEQUENCE_LENGTH

        }

}


torch.save(

    candidate_payload,

    CANDIDATE_CHECKPOINT

)


print(
    "Candidate saved:",
    CANDIDATE_CHECKPOINT
)

print()


if (

        decision
        ==
        "PROMOTE_CANDIDATE"

):

    torch.save(

        candidate_payload,

        BEST_CHECKPOINT

    )


    print(
        "Promoted reasoning model:",
        BEST_CHECKPOINT
    )

else:

    print(
        "Baseline retained."
    )


print()


# ============================================================
# 34. TEST 19 - TRAINING LOG
# ============================================================

training_log = {

    "lesson":
        "80R",

    "training_mode":
        "native_reasoning_fine_tuning",

    "dataset":
        dataset_name,

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

    "external_llm":
        False,

    "external_reasoning_model":
        False,

    "device":
        str(
            DEVICE
        ),

    "training_examples":
        len(
            train_dataset
        ),

    "validation_examples":
        len(
            validation_dataset
        ),

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


print(
    "TEST 19: Save Training Log"
)

print()

print(
    "Saved:",
    TRAINING_LOG_FILE
)

print()


# ============================================================
# 35. TEST 20 - EVALUATION REPORT
# ============================================================

evaluation_report = {

    "lesson":
        "80R",

    "model":
        "Silverwing native decoder",

    "dataset":
        dataset_name,

    "external_llm":
        False,

    "external_reasoning_model":
        False,

    "base_checkpoint":
        str(
            BASE_CHECKPOINT
        ),

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


print(
    "TEST 20: Save Evaluation Report"
)

print()

print(
    "Saved:",
    EVALUATION_FILE
)

print()


# ============================================================
# 36. 79R -> 80R CONTRACT
# ============================================================

print(
    "LESSON 79R -> 80R CONTRACT"
)

print()

print(
    "79R:"
)

print(
    "Native reasoning dataset"
)

print(
    "        ↓"
)

print(
    "Native reasoning validation"
)

print(
    "        ↓"
)

print(
    "80R:"
)

print(
    "Exact established Silverwing decoder"
)

print(
    "        ↓"
)

print(
    "Native reasoning fine-tuning"
)

print(
    "        ↓"
)

print(
    "Reasoning validation"
)

print(
    "        ↓"
)

print(
    "Promotion gate"
)

print()


# ============================================================
# 37. MODEL OWNERSHIP
# ============================================================

print(
    "MODEL OWNERSHIP CHECK"
)

print()

print(
    "Tokenizer: Silverwing native"
)

print(
    "Vocabulary: Silverwing native"
)

print(
    "Decoder: Silverwing native"
)

print(
    "Dataset: Silverwing native reasoning data"
)

print(
    "Training: Silverwing native fine-tuning pipeline"
)

print(
    "External LLM: NONE"
)

print()


# ============================================================
# 38. NEXT
# ============================================================

print(
    "NEXT COMPONENT"
)

print()

print(
    "Lesson 81R will consume the promoted 80R model "
    "and add the next capability without changing "
    "the established Silverwing decoder architecture."
)

print()


# ============================================================
# 39. PROGRESS
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
print("81R Next Capability")
print(" ↓")
print("Memory")
print(" ↓")
print("Tools")
print(" ↓")
print("Planning")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ============================================================
# COMPLETE
# ============================================================

print(
    "=== LESSON 80R COMPLETE ==="
)
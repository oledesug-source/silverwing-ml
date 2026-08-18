# Silverwing ML
# Phase 5 - Lesson 78R
# Silverwing Own Foundation Model
# Native Instruction Fine-Tuning Engine
#
# Purpose:
# Fine-tune Silverwing's own decoder using the native
# instruction dataset created in Lesson 77R.
#
# No GPT-2.
# No Qwen.
# No external instruction-tuned model.
#
# Training objective:
#
# Instruction + Context + Response
#           ↓
#      Silverwing Decoder
#           ↓
#        Token Logits
#           ↓
#   Response-only Cross Entropy
#           ↓
#      Backpropagation
#           ↓
#      Candidate Model
#           ↓
#        Evaluation
#           ↓
#    Promotion / Rejection


import json
import math
import time

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LESSON_66_DIR = (
        BASE_DIR.parent / "lesson66R"
)

LESSON_71_DIR = (
        BASE_DIR.parent / "lesson71R"
)

LESSON_73_DIR = (
        BASE_DIR.parent / "lesson73R"
)

LESSON_77_DIR = (
        BASE_DIR.parent / "lesson77R"
)


VOCABULARY_FILE = (
        LESSON_66_DIR
        / "silverwing_subword_vocabulary.json"
)


MERGES_FILE = (
        LESSON_66_DIR
        / "silverwing_bpe_merges.json"
)


MODEL_CONFIG_FILE = (
        LESSON_71_DIR
        / "silverwing_decoder_config.json"
)


PRETRAINED_CHECKPOINT = (
        LESSON_73_DIR
        / "checkpoints"
        / "silverwing_best.pt"
)


INSTRUCTION_CONFIG_FILE = (
        LESSON_77_DIR
        / "silverwing_instruction_config.json"
)


INSTRUCTION_TRAIN_FILE = (
        LESSON_77_DIR
        / "silverwing_instruction_train.jsonl"
)


INSTRUCTION_VALIDATION_FILE = (
        LESSON_77_DIR
        / "silverwing_instruction_validation.jsonl"
)


OUTPUT_DIR = (
        BASE_DIR
        / "checkpoints"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CANDIDATE_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_instruction_candidate.pt"
)


BEST_INSTRUCTION_CHECKPOINT = (
        OUTPUT_DIR
        / "silverwing_instruction_best.pt"
)


TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_instruction_training_log.json"
)


EVALUATION_REPORT_FILE = (
        BASE_DIR
        / "silverwing_instruction_evaluation.json"
)


# ==================================================
# 2. TRAINING CONFIGURATION
# ==================================================

SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ==================================================
# 3. HELPERS
# ==================================================

def read_json(
        path: Path
):

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        return json.load(file)


def write_json(
        path: Path,
        data
):

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


def require_file(
        path: Path
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


# ==================================================
# 4. HEADER
# ==================================================

print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 78R")
print("Silverwing Own Foundation Model")
print("Native Instruction Fine-Tuning Engine")
print()


# ==================================================
# 5. VERIFY ARTIFACTS
# ==================================================

print("TEST 1: Verify Foundation Artifacts")
print()


required_files = [

    VOCABULARY_FILE,

    MERGES_FILE,

    MODEL_CONFIG_FILE,

    PRETRAINED_CHECKPOINT,

    INSTRUCTION_CONFIG_FILE,

    INSTRUCTION_TRAIN_FILE,

    INSTRUCTION_VALIDATION_FILE

]


for path in required_files:

    require_file(
        path
    )

    print(
        "FOUND:",
        path
    )


print()


# ==================================================
# 6. LOAD VOCABULARY
# ==================================================

print("TEST 2: Load Vocabulary")
print()


vocabulary_data = read_json(
    VOCABULARY_FILE
)


TOKEN_TO_ID = {

    token:
        int(token_id)

    for token, token_id
    in vocabulary_data[
        "token_to_id"
    ].items()

}


ID_TO_TOKEN = {

    token_id:
        token

    for token, token_id
    in TOKEN_TO_ID.items()

}


VOCABULARY_SIZE = len(
    TOKEN_TO_ID
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


print(
    "Vocabulary size:",
    VOCABULARY_SIZE
)

print()


# ==================================================
# 7. LOAD MODEL CONFIGURATION
# ==================================================

print("TEST 3: Load Model Configuration")
print()


model_config = read_json(
    MODEL_CONFIG_FILE
)


MODEL_DIMENSION = (
    model_config[
        "model_dimension"
    ]
)


NUMBER_OF_HEADS = (
    model_config[
        "attention_heads"
    ]
)


FEED_FORWARD_DIMENSION = (
    model_config[
        "feed_forward_dimension"
    ]
)


NUMBER_OF_LAYERS = (
    model_config[
        "layers"
    ]
)


MAX_SEQUENCE_LENGTH = (
    model_config[
        "maximum_sequence_length"
    ]
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
    "Maximum sequence length:",
    MAX_SEQUENCE_LENGTH
)

print()


# ==================================================
# 8. LOAD BPE MERGES
# ==================================================

print("TEST 4: Load BPE Merges")
print()


merge_data = read_json(
    MERGES_FILE
)


MERGE_RANKS = {}


for item in merge_data:

    pair = item[
        "pair"
    ]


    if (
            not isinstance(
                pair,
                list
            )
            or
            len(pair) != 2
    ):

        raise ValueError(
            f"Invalid merge pair: {pair}"
        )


    MERGE_RANKS[
        (
            pair[0],
            pair[1]
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


# ==================================================
# 9. SILVERWING TOKENIZER
# ==================================================

BPE_END = "</w>"


def normalize_text(
        text: str
) -> str:

    return (
        str(text)
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
        .strip()
    )


def split_words(
        text: str
) -> List[str]:

    import re

    return re.findall(
        r"\w+|[^\w\s]",
        normalize_text(
            text
        ).lower(),
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
                ==
                pair
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

    tokens = tokenize_text(
        text
    )


    ids = [
        BOS_ID
    ]


    for token in tokens:

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


# ==================================================
# 10. INSTRUCTION FORMATTING
# ==================================================

def format_instruction_prompt(
        instruction: str,
        context: str
) -> str:

    parts = [
        "Instruction:",
        instruction.strip()
    ]


    if context.strip():

        parts.extend(
            [
                "",
                "Context:",
                context.strip()
            ]
        )


    parts.extend(
        [
            "",
            "Response:"
        ]
    )


    return "\n".join(
        parts
    )


# ==================================================
# 11. LOAD INSTRUCTION DATA
# ==================================================

print("TEST 5: Load Instruction Dataset")
print()


def load_jsonl(
        path: Path
) -> List[Dict]:

    records = []


    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()


            if not line:

                continue


            records.append(
                json.loads(
                    line
                )
            )


    return records


train_records = load_jsonl(
    INSTRUCTION_TRAIN_FILE
)


validation_records = load_jsonl(
    INSTRUCTION_VALIDATION_FILE
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


if not train_records:

    raise RuntimeError(
        "Instruction training dataset is empty."
    )


# ==================================================
# 12. INSTRUCTION DATASET
# ==================================================

class InstructionFineTuneDataset(
    Dataset
):

    def __init__(
            self,
            records: List[Dict]
    ):

        self.samples = []


        for record in records:

            instruction = record[
                "instruction"
            ]

            context = record.get(
                "context",
                ""
            )

            response = record[
                "response"
            ]


            prompt_text = (
                format_instruction_prompt(
                    instruction,
                    context
                )
            )


            full_text = (
                    prompt_text
                    +
                    "\n"
                    +
                    response.strip()
            )


            prompt_ids = encode_text(
                prompt_text
            )


            full_ids = encode_text(
                full_text
            )


            if len(full_ids) > (
                    MAX_SEQUENCE_LENGTH
            ):

                raise ValueError(
                    "Instruction example exceeds "
                    "model maximum sequence length."
                )


            response_start = max(
                1,
                len(prompt_ids) - 1
            )


            labels = list(
                full_ids
            )


            for index in range(
                    min(
                        response_start,
                        len(labels)
                    )
            ):

                labels[index] = -100


            response_token_count = sum(
                1
                for value in labels
                if value != -100
            )


            if response_token_count == 0:

                raise ValueError(
                    "Instruction example has no "
                    "supervised response tokens."
                )


            self.samples.append(
                {
                    "input_ids":
                        full_ids,

                    "labels":
                        labels,

                    "response_start":
                        response_start,

                    "response_token_count":
                        response_token_count
                }
            )


    def __len__(
            self
    ):

        return len(
            self.samples
        )


    def __getitem__(
            self,
            index
    ):

        sample = self.samples[
            index
        ]


        return {

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


train_dataset = (
    InstructionFineTuneDataset(
        train_records
    )
)


validation_dataset = (
    InstructionFineTuneDataset(
        validation_records
    )
)


print("TEST 6: Build Fine-Tuning Dataset")
print()


print(
    "Train samples:",
    len(train_dataset)
)


print(
    "Validation samples:",
    len(validation_dataset)
)

print()


# ==================================================
# 13. BATCH COLLATION
# ==================================================

def collate_instruction_batch(
        batch: List[Dict]
) -> Dict[str, torch.Tensor]:

    maximum_length = max(
        len(
            item[
                "input_ids"
            ]
        )
        for item in batch
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


        padded_inputs = torch.cat(
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


        padded_labels = torch.cat(
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


        input_batch.append(
            padded_inputs
        )


        label_batch.append(
            padded_labels
        )


    return {

        "input_ids":
            torch.stack(
                input_batch
            ),

        "labels":
            torch.stack(
                label_batch
            )

    }


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_instruction_batch
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_instruction_batch
)


print("TEST 7: DataLoaders")
print()


print(
    "Training batches:",
    len(train_loader)
)


print(
    "Validation batches:",
    len(validation_loader)
)

print()


# ==================================================
# 14. RESPONSE MASK TEST
# ==================================================

print("TEST 8: Response-Only Masking")
print()


if len(train_dataset) > 0:

    sample = train_dataset[0]


    labels = sample[
        "labels"
    ]


    supervised_tokens = int(
        (
                labels
                !=
                -100
        ).sum()
    )


    masked_tokens = int(
        (
                labels
                ==
                -100
        ).sum()
    )


    print(
        "Total tokens:",
        len(labels)
    )


    print(
        "Masked prompt tokens:",
        masked_tokens
    )


    print(
        "Supervised response tokens:",
        supervised_tokens
    )


print()


# ==================================================
# 15. ATTENTION
# ==================================================

class SilverwingAttentionMath:

    @staticmethod
    def scaled_dot_product(
            queries,
            keys,
            values,
            mask=None
    ):

        dimension = (
            queries.shape[-1]
        )


        scores = torch.matmul(
            queries,
            keys.transpose(
                -2,
                -1
            )
        )


        scores = (
                scores
                /
                math.sqrt(
                    dimension
                )
        )


        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )


        weights = F.softmax(
            scores,
            dim=-1
        )


        return torch.matmul(
            weights,
            values
        )


class SilverwingMultiHeadAttention(
    nn.Module
):

    def __init__(
            self,
            model_dimension,
            number_of_heads
    ):

        super().__init__()


        if (
                model_dimension
                %
                number_of_heads
                !=
                0
        ):

            raise ValueError(
                "Model dimension must be "
                "divisible by number of heads."
            )


        self.model_dimension = (
            model_dimension
        )


        self.number_of_heads = (
            number_of_heads
        )


        self.head_dimension = (
                model_dimension
                //
                number_of_heads
        )


        self.query_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )


        self.key_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )


        self.value_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )


        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False
        )


    def split_heads(
            self,
            x
    ):

        batch_size = x.shape[0]

        sequence_length = x.shape[1]


        x = x.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension
        )


        return x.transpose(
            1,
            2
        )


    def merge_heads(
            self,
            x
    ):

        batch_size = x.shape[0]

        sequence_length = x.shape[2]


        x = x.transpose(
            1,
            2
        ).contiguous()


        return x.view(
            batch_size,
            sequence_length,
            self.model_dimension
        )


    def forward(
            self,
            x,
            mask
    ):

        queries = self.split_heads(
            self.query_projection(
                x
            )
        )


        keys = self.split_heads(
            self.key_projection(
                x
            )
        )


        values = self.split_heads(
            self.value_projection(
                x
            )
        )


        output = (
            SilverwingAttentionMath
            .scaled_dot_product(
                queries,
                keys,
                values,
                mask
            )
        )


        output = self.merge_heads(
            output
        )


        return self.output_projection(
            output
        )


# ==================================================
# 16. FEED-FORWARD
# ==================================================

class SilverwingFeedForward(
    nn.Module
):

    def __init__(
            self,
            model_dimension,
            hidden_dimension
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            model_dimension,
            hidden_dimension
        )


        self.output_projection = nn.Linear(
            hidden_dimension,
            model_dimension
        )


        self.activation = nn.GELU()


    def forward(
            self,
            x
    ):

        x = self.input_projection(
            x
        )


        x = self.activation(
            x
        )


        return self.output_projection(
            x
        )


# ==================================================
# 17. TRANSFORMER BLOCK
# ==================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(
            self
    ):

        super().__init__()


        self.attention = (
            SilverwingMultiHeadAttention(
                MODEL_DIMENSION,
                NUMBER_OF_HEADS
            )
        )


        self.feed_forward = (
            SilverwingFeedForward(
                MODEL_DIMENSION,
                FEED_FORWARD_DIMENSION
            )
        )


        self.norm_attention = (
            nn.LayerNorm(
                MODEL_DIMENSION
            )
        )


        self.norm_feed_forward = (
            nn.LayerNorm(
                MODEL_DIMENSION
            )
        )


    def forward(
            self,
            x,
            mask
    ):

        attention_output = (
            self.attention(
                x,
                mask
            )
        )


        x = (
                x
                +
                attention_output
        )


        x = (
            self.norm_attention(
                x
            )
        )


        feed_forward_output = (
            self.feed_forward(
                x
            )
        )


        x = (
                x
                +
                feed_forward_output
        )


        return (
            self.norm_feed_forward(
                x
            )
        )


# ==================================================
# 18. POSITION EMBEDDING
# ==================================================

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
            sequence_length,
            device
    ):

        positions = torch.arange(
            sequence_length,
            device=device
        )


        return self.embedding(
            positions
        )


# ==================================================
# 19. SILVERWING DECODER
# ==================================================

class SilverwingDecoderLM(
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


    def create_causal_mask(
            self,
            sequence_length,
            device
    ):

        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device
            )
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )


    def forward(
            self,
            input_ids
    ):

        sequence_length = (
            input_ids.shape[1]
        )


        if sequence_length > (
                MAX_SEQUENCE_LENGTH
        ):

            raise ValueError(
                "Sequence exceeds model maximum."
            )


        x = self.token_embedding(
            input_ids
        )


        positions = (
            self.position_embedding(
                sequence_length,
                input_ids.device
            )
        )


        x = (
                x
                +
                positions.unsqueeze(0)
        )


        mask = self.create_causal_mask(
            sequence_length,
            input_ids.device
        )


        for layer in self.layers:

            x = layer(
                x,
                mask
            )


        x = self.final_norm(
            x
        )


        return self.language_model_head(
            x
        )


# ==================================================
# 20. LOAD FOUNDATION CHECKPOINT
# ==================================================

print("TEST 9: Load Silverwing Foundation Checkpoint")
print()


model = (
    SilverwingDecoderLM()
    .to(DEVICE)
)


checkpoint = torch.load(
    PRETRAINED_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Foundation checkpoint loaded."
)


print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch"
    )
)


print(
    "Checkpoint step:",
    checkpoint.get(
        "global_step"
    )
)

print()


# ==================================================
# 21. BASELINE STATE
# ==================================================

baseline_state = {

    key:
        value.detach().clone()

    for key, value
    in model.state_dict().items()

}


# ==================================================
# 22. OPTIMIZER
# ==================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


total_steps = max(

    1,

    len(train_loader)
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


# ==================================================
# 23. LOSS
# ==================================================

def instruction_loss(
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


# ==================================================
# 24. EVALUATION
# ==================================================

@torch.no_grad()
def evaluate(
        current_model,
        loader
):

    current_model.eval()


    total_loss = 0.0

    total_batches = 0

    correct = 0

    supervised_tokens = 0


    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )


        labels = batch[
            "labels"
        ].to(
            DEVICE
        )


        logits = current_model(
            input_ids
        )


        loss = instruction_loss(
            logits,
            labels
        )


        total_loss += float(
            loss
        )


        total_batches += 1


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


        supervised_tokens += int(
            valid_mask.sum()
        )


    if total_batches == 0:

        return {

            "loss":
                float("nan"),

            "perplexity":
                float("nan"),

            "accuracy":
                float("nan"),

            "supervised_tokens":
                0

        }


    average_loss = (
            total_loss
            /
            total_batches
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
        supervised_tokens

        if supervised_tokens

        else float("nan")

    )


    return {

        "loss":
            average_loss,

        "perplexity":
            perplexity,

        "accuracy":
            accuracy,

        "supervised_tokens":
            supervised_tokens

    }


# ==================================================
# 25. BASELINE EVALUATION
# ==================================================

print("TEST 10: Baseline Instruction Evaluation")
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
    "Baseline response-token accuracy:",
    baseline_metrics[
        "accuracy"
    ]
)

print()


# ==================================================
# 26. TRAINING
# ==================================================

print("TEST 11: Native Instruction Fine-Tuning")
print()


training_history = []

best_validation_loss = float(
    "inf"
)

global_step = 0


training_start = (
    time.perf_counter()
)


for epoch in range(
        1,
        EPOCHS + 1
):

    model.train()


    epoch_loss = 0.0

    epoch_batches = 0


    for batch_index, batch in enumerate(
            train_loader,
            start=1
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )


        labels = batch[
            "labels"
        ].to(
            DEVICE
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            input_ids
        )


        loss = instruction_loss(
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
            f"| Batch {batch_index}/{len(train_loader)} "
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
        validation_loader
    )


    epoch_record = {

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


    training_history.append(
        epoch_record
    )


    print()

    print(
        "Epoch",
        epoch,
        "complete."
    )


    print(
        "Train loss:",
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

                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "baseline_checkpoint":
                    str(
                        PRETRAINED_CHECKPOINT
                    ),

                "baseline_metrics":
                    baseline_metrics,

                "instruction_metrics":
                    validation_metrics,

                "configuration": {

                    "batch_size":
                        BATCH_SIZE,

                    "epochs":
                        EPOCHS,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "gradient_clip_norm":
                        GRADIENT_CLIP_NORM

                }

            },

            BEST_INSTRUCTION_CHECKPOINT

        )


training_duration = (
        time.perf_counter()
        -
        training_start
)


# ==================================================
# 27. FINAL EVALUATION
# ==================================================

print("TEST 12: Final Candidate Evaluation")
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


# ==================================================
# 28. NUMERICAL HEALTH
# ==================================================

print("TEST 13: Numerical Health")
print()


nan_parameters = 0

inf_parameters = 0


for parameter in model.parameters():

    if torch.isnan(
            parameter
    ).any():

        nan_parameters += 1


    if torch.isinf(
            parameter
    ).any():

        inf_parameters += 1


numerically_healthy = (

        nan_parameters == 0

        and

        inf_parameters == 0

)


print(
    "NaN parameter tensors:",
    nan_parameters
)


print(
    "Inf parameter tensors:",
    inf_parameters
)


print(
    "Numerically healthy:",
    numerically_healthy
)

print()


# ==================================================
# 29. PARAMETER CHANGE
# ==================================================

print("TEST 14: Parameter Change")
print()


total_parameter_change = 0.0

changed_tensors = 0


for name, parameter in (
        model.state_dict().items()
):

    baseline_parameter = (
        baseline_state[
            name
        ]
    )


    difference = torch.sum(
        torch.abs(
            parameter.detach()
            -
            baseline_parameter
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


# ==================================================
# 30. PROMOTION DECISION
# ==================================================

print("TEST 15: Instruction Model Promotion Gate")
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

    promotion_decision = "REJECT"

    promotion_reason = (
        "Numerical instability detected."
    )


elif not math.isfinite(
        candidate_loss
):

    promotion_decision = "REJECT"

    promotion_reason = (
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

    promotion_decision = (
        "PROMOTE_CANDIDATE"
    )

    promotion_reason = (
        "Instruction validation loss improved."
    )


else:

    promotion_decision = (
        "RETAIN_BASELINE"
    )

    promotion_reason = (
        "Instruction validation loss did not improve."
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
    promotion_decision
)


print(
    "Reason:",
    promotion_reason
)

print()


# ==================================================
# 31. SAVE CANDIDATE
# ==================================================

print("TEST 16: Save Candidate Checkpoint")
print()


candidate_payload = {

    "model_state_dict":
        model.state_dict(),

    "baseline_checkpoint":
        str(
            PRETRAINED_CHECKPOINT
        ),

    "baseline_metrics":
        baseline_metrics,

    "candidate_metrics":
        final_metrics,

    "training_history":
        training_history,

    "promotion_decision":
        promotion_decision,

    "promotion_reason":
        promotion_reason,

    "configuration": {

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM

    }

}


torch.save(
    candidate_payload,
    CANDIDATE_CHECKPOINT
)


print(
    "Saved:",
    CANDIDATE_CHECKPOINT
)

print()


if (
        promotion_decision
        ==
        "PROMOTE_CANDIDATE"
):

    torch.save(
        candidate_payload,
        BEST_INSTRUCTION_CHECKPOINT
    )


    print(
        "Promoted candidate checkpoint saved:"
    )


    print(
        BEST_INSTRUCTION_CHECKPOINT
    )


else:

    print(
        "Baseline retained."
    )


print()


# ==================================================
# 32. TRAINING LOG
# ==================================================

training_log = {

    "model":
        "Silverwing-Decoder-v1",

    "base_checkpoint":
        str(
            PRETRAINED_CHECKPOINT
        ),

    "dataset":
        "Silverwing-Instruction-v1",

    "epochs":
        EPOCHS,

    "global_steps":
        global_step,

    "duration_seconds":
        training_duration,

    "baseline_metrics":
        baseline_metrics,

    "final_metrics":
        final_metrics,

    "promotion_decision":
        promotion_decision,

    "promotion_reason":
        promotion_reason,

    "history":
        training_history

}


write_json(
    TRAINING_LOG_FILE,
    training_log
)


print("TEST 17: Save Training Log")
print()


print(
    "Saved:",
    TRAINING_LOG_FILE
)

print()


# ==================================================
# 33. EVALUATION REPORT
# ==================================================

evaluation_report = {

    "lesson":
        "78R",

    "model":
        "Silverwing-Decoder-v1",

    "training_mode":
        "supervised_instruction_fine_tuning",

    "response_only_loss":
        True,

    "device":
        str(
            DEVICE
        ),

    "baseline":
        baseline_metrics,

    "candidate":
        final_metrics,

    "numerical_health": {

        "nan_parameters":
            nan_parameters,

        "inf_parameters":
            inf_parameters,

        "healthy":
            numerically_healthy

    },

    "parameter_change": {

        "changed_tensors":
            changed_tensors,

        "total_absolute_change":
            total_parameter_change

    },

    "promotion": {

        "decision":
            promotion_decision,

        "reason":
            promotion_reason

    }

}


write_json(
    EVALUATION_REPORT_FILE,
    evaluation_report
)


print(
    "TEST 18: Save Evaluation Report"
)

print()


print(
    "Saved:",
    EVALUATION_REPORT_FILE
)

print()


# ==================================================
# 34. INSTRUCTION LEARNING PIPELINE
# ==================================================

print(
    "SILVERWING INSTRUCTION LEARNING PIPELINE"
)

print()


print(
    "Instruction"
)

print(
    "    ↓"
)

print(
    "Context"
)

print(
    "    ↓"
)

print(
    "Response"
)

print(
    "    ↓"
)

print(
    "Silverwing Tokenizer"
)

print(
    "    ↓"
)

print(
    "Prompt Tokens Masked"
)

print(
    "    ↓"
)

print(
    "Response Tokens Supervised"
)

print(
    "    ↓"
)

print(
    "Silverwing Decoder"
)

print(
    "    ↓"
)

print(
    "Cross-Entropy"
)

print(
    "    ↓"
)

print(
    "Gradient Update"
)

print(
    "    ↓"
)

print(
    "Candidate Model"
)

print(
    "    ↓"
)

print(
    "Evaluation"
)

print(
    "    ↓"
)

print(
    "Promotion Gate"
)

print()


# ==================================================
# 35. WHY RESPONSE-ONLY SUPERVISION
# ==================================================

print(
    "WHY RESPONSE-ONLY SUPERVISION"
)

print()


print(
    "The instruction and context tell Silverwing "
    "what task is being requested."
)

print()


print(
    "The response is the desired behavior."
)

print()


print(
    "Masking the prompt concentrates the supervised "
    "learning signal on producing the desired response."
)

print()


# ==================================================
# 36. PRETRAINING VS INSTRUCTION TRAINING
# ==================================================

print(
    "PRETRAINING VS INSTRUCTION FINE-TUNING"
)

print()


print(
    "PRETRAINING:"
)

print(
    "Learn broad language and token relationships."
)

print()


print(
    "INSTRUCTION FINE-TUNING:"
)

print(
    "Learn how to respond to explicit goals, "
    "constraints, context, and tasks."
)

print()


# ==================================================
# 37. CONTROLLED ADAPTATION
# ==================================================

print(
    "CONTROLLED ADAPTATION"
)

print()


print(
    "Base Model"
)

print(
    "    ↓"
)

print(
    "Instruction Dataset"
)

print(
    "    ↓"
)

print(
    "Candidate Training"
)

print(
    "    ↓"
)

print(
    "Independent Validation"
)

print(
    "    ↓"
)

print(
    "Compare Against Base"
)

print(
    "    ↓"
)

print(
    "Promote or Reject"
)

print()


# ==================================================
# 38. IMPORTANT ENGINEERING PRINCIPLE
# ==================================================

print(
    "IMPORTANT ENGINEERING PRINCIPLE"
)

print()


print(
    "Fine-tuning does not automatically replace "
    "the foundation model."
)

print()


print(
    "The candidate remains isolated until evaluation "
    "shows that the intended behavior improved."
)

print()


# ==================================================
# 39. CURRENT LIMITATION
# ==================================================

print(
    "CURRENT LIMITATION"
)

print()


print(
    "The instruction dataset is still very small."
)

print()


print(
    "This validates the native fine-tuning mechanism; "
    "it does not make Silverwing a modern general-purpose "
    "assistant yet."
)

print()


# ==================================================
# 40. NEXT COMPONENT
# ==================================================

print(
    "NEXT COMPONENT"
)

print()


print(
    "Lesson 79R: Silverwing Native Reasoning Dataset "
    "and Reasoning-Specific Evaluation."
)

print()


# ==================================================
# 41. FOUNDATION PROGRESS
# ==================================================

print(
    "SILVERWING FOUNDATION MODEL PROGRESS"
)

print()


print(
    "Own BPE Tokenizer"
)

print(
    " ↓"
)

print(
    "Own Subword Vocabulary"
)

print(
    " ↓"
)

print(
    "Own Token IDs"
)

print(
    " ↓"
)

print(
    "Own Embedding System"
)

print(
    " ↓"
)

print(
    "Own Position Encoding"
)

print(
    " ↓"
)

print(
    "Own Self-Attention"
)

print(
    " ↓"
)

print(
    "Own Transformer Block"
)

print(
    " ↓"
)

print(
    "Own Decoder Language Model"
)

print(
    " ↓"
)

print(
    "Own Training Data Pipeline"
)

print(
    " ↓"
)

print(
    "Own Pretraining Engine"
)

print(
    " ↓"
)

print(
    "Own Evaluation Framework"
)

print(
    " ↓"
)

print(
    "Own Experiment / Promotion System"
)

print(
    " ↓"
)

print(
    "Own Curriculum Engine"
)

print(
    " ↓"
)

print(
    "Own Instruction Dataset"
)

print(
    " ↓"
)

print(
    "OWN INSTRUCTION FINE-TUNING ENGINE"
)

print(
    " ↓"
)

print(
    "Reasoning Training"
)

print(
    " ↓"
)

print(
    "Memory-Aware Training"
)

print(
    " ↓"
)

print(
    "Multitask Training"
)

print(
    " ↓"
)

print(
    "Continual Learning"
)

print(
    " ↓"
)

print(
    "Controlled Autonomous Improvement"
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print(
    "=== LESSON 78R COMPLETE ==="
)
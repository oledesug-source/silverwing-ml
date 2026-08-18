# Silverwing ML
# Phase 5 - Lesson 73R
# Silverwing Own Foundation Model
# Native Pretraining Engine
#
# This lesson trains Silverwing's own decoder model.
#
# Pipeline:
# Dataset
#   ↓
# Forward Pass
#   ↓
# Causal Language-Model Loss
#   ↓
# Backpropagation
#   ↓
# Gradient Clipping
#   ↓
# Optimizer
#   ↓
# Validation
#   ↓
# Checkpoint
#
# No GPT-2, Qwen, or other pretrained language model
# is loaded in this lesson.


import json
import math
import time

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 73R")
print("Silverwing Own Foundation Model")
print("Native Pretraining Engine")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LESSON_66_DIR = (
        BASE_DIR.parent / "lesson66R"
)

LESSON_71_DIR = (
        BASE_DIR.parent / "lesson71R"
)

LESSON_72_DIR = (
        BASE_DIR.parent / "lesson72R"
)

VOCABULARY_FILE = (
        LESSON_66_DIR
        / "silverwing_subword_vocabulary.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71_DIR
        / "silverwing_decoder_config.json"
)

TRAIN_FILE = (
        LESSON_72_DIR
        / "silverwing_train.jsonl"
)

VALIDATION_FILE = (
        LESSON_72_DIR
        / "silverwing_validation.jsonl"
)

DATASET_CONFIG_FILE = (
        LESSON_72_DIR
        / "silverwing_dataset_config.json"
)

CHECKPOINT_DIR = (
        BASE_DIR / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
        CHECKPOINT_DIR
        / "silverwing_best.pt"
)

LATEST_CHECKPOINT = (
        CHECKPOINT_DIR
        / "silverwing_latest.pt"
)

TRAINING_LOG_FILE = (
        BASE_DIR
        / "silverwing_training_log.json"
)


SEED = 42

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0

LOG_EVERY = 1

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


print("TEST 1: Configuration")
print()

print(
    "Training data:",
    TRAIN_FILE
)

print(
    "Validation data:",
    VALIDATION_FILE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "Gradient clip norm:",
    GRADIENT_CLIP_NORM
)

print(
    "Device:",
    DEVICE
)

print()


# ==================================================
# 2. VERIFY ARTIFACTS
# ==================================================

print("TEST 2: Verify Foundation Artifacts")
print()


required_files = [
    VOCABULARY_FILE,
    MODEL_CONFIG_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
    DATASET_CONFIG_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n{file_path}"
        )

    print(
        "FOUND:",
        file_path
    )


print()


# ==================================================
# 3. LOAD VOCABULARY
# ==================================================

print("TEST 3: Load Vocabulary")
print()


with open(
        VOCABULARY_FILE,
        "r",
        encoding="utf-8"
) as file:

    vocabulary_data = json.load(
        file
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

print(
    "PAD:",
    PAD_ID
)

print(
    "UNK:",
    UNK_ID
)

print(
    "BOS:",
    BOS_ID
)

print(
    "EOS:",
    EOS_ID
)

print()


# ==================================================
# 4. LOAD MODEL CONFIGURATION
# ==================================================

print("TEST 4: Model Configuration")
print()


with open(
        MODEL_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    model_config = json.load(
        file
    )


MODEL_DIMENSION = model_config[
    "model_dimension"
]

NUMBER_OF_HEADS = model_config[
    "attention_heads"
]

FEED_FORWARD_DIMENSION = (
    model_config[
        "feed_forward_dimension"
    ]
)

NUMBER_OF_LAYERS = model_config[
    "layers"
]

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
# 5. LOAD DATASET CONFIG
# ==================================================

print("TEST 5: Dataset Configuration")
print()


with open(
        DATASET_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:

    dataset_config = json.load(
        file
    )


SEQUENCE_LENGTH = (
    dataset_config[
        "sequence_length"
    ]
)


print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Training sequences:",
    dataset_config[
        "training_sequences"
    ]
)

print(
    "Validation sequences:",
    dataset_config[
        "validation_sequences"
    ]
)

print()


if SEQUENCE_LENGTH > (
        MAX_SEQUENCE_LENGTH
):

    raise ValueError(
        "Dataset sequence length exceeds "
        "model maximum sequence length."
    )


# ==================================================
# 6. DATASET
# ==================================================

class SilverwingLanguageDataset(
    Dataset
):

    def __init__(
            self,
            file_path: Path
    ):

        self.samples = []

        with open(
                file_path,
                "r",
                encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()


                if not line:
                    continue


                sample = json.loads(
                    line
                )


                input_ids = sample[
                    "input_ids"
                ]

                labels = sample[
                    "labels"
                ]


                if len(input_ids) != (
                        len(labels)
                ):

                    raise ValueError(
                        "Input and label lengths "
                        "must match."
                    )


                self.samples.append(
                    {
                        "input_ids":
                            input_ids,

                        "labels":
                            labels
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
    SilverwingLanguageDataset(
        TRAIN_FILE
    )
)


validation_dataset = (
    SilverwingLanguageDataset(
        VALIDATION_FILE
    )
)


print("TEST 6: Dataset")
print()

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print()


if len(train_dataset) == 0:

    raise RuntimeError(
        "Training dataset is empty."
    )


# ==================================================
# 7. DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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
# 8. ATTENTION
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


        output = torch.matmul(
            weights,
            values
        )


        return output


class SilverwingMultiHeadAttention(
    nn.Module
):

    def __init__(
            self,
            model_dimension,
            number_of_heads,
            dropout=0.0
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
                "Model dimension must be divisible "
                "by number of heads."
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


        self.dropout = nn.Dropout(
            dropout
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
            mask=None
    ):

        queries = self.query_projection(
            x
        )

        keys = self.key_projection(
            x
        )

        values = self.value_projection(
            x
        )


        queries = self.split_heads(
            queries
        )

        keys = self.split_heads(
            keys
        )

        values = self.split_heads(
            values
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


        output = self.output_projection(
            output
        )


        return self.dropout(
            output
        )


# ==================================================
# 9. FEED-FORWARD
# ==================================================

class SilverwingFeedForward(
    nn.Module
):

    def __init__(
            self,
            model_dimension,
            hidden_dimension,
            dropout=0.0
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

        self.dropout = nn.Dropout(
            dropout
        )


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

        x = self.dropout(
            x
        )

        x = self.output_projection(
            x
        )

        return x


# ==================================================
# 10. TRANSFORMER BLOCK
# ==================================================

class SilverwingTransformerBlock(
    nn.Module
):

    def __init__(
            self,
            model_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout=0.0
    ):

        super().__init__()


        self.attention = (
            SilverwingMultiHeadAttention(
                model_dimension,
                number_of_heads,
                dropout
            )
        )


        self.feed_forward = (
            SilverwingFeedForward(
                model_dimension,
                feed_forward_dimension,
                dropout
            )
        )


        self.norm_attention = (
            nn.LayerNorm(
                model_dimension
            )
        )

        self.norm_feed_forward = (
            nn.LayerNorm(
                model_dimension
            )
        )


        self.dropout_attention = (
            nn.Dropout(
                dropout
            )
        )

        self.dropout_feed_forward = (
            nn.Dropout(
                dropout
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
                self.dropout_attention(
                    attention_output
                )
        )


        x = self.norm_attention(
            x
        )


        feed_forward_output = (
            self.feed_forward(
                x
            )
        )


        x = (
                x
                +
                self.dropout_feed_forward(
                    feed_forward_output
                )
        )


        x = self.norm_feed_forward(
            x
        )


        return x


# ==================================================
# 11. POSITION EMBEDDING
# ==================================================

class SilverwingPositionEmbedding(
    nn.Module
):

    def __init__(
            self,
            maximum_sequence_length,
            dimension
    ):

        super().__init__()


        self.embedding = nn.Embedding(
            maximum_sequence_length,
            dimension
        )


        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
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
# 12. SILVERWING DECODER
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
            SilverwingPositionEmbedding(
                MAX_SEQUENCE_LENGTH,
                MODEL_DIMENSION
            )
        )


        self.layers = nn.ModuleList(
            [
                SilverwingTransformerBlock(
                    MODEL_DIMENSION,
                    NUMBER_OF_HEADS,
                    FEED_FORWARD_DIMENSION,
                    dropout=0.0
                )
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


        self.reset_parameters()


    def reset_parameters(
            self
    ):

        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02
        )


        with torch.no_grad():

            self.token_embedding.weight[
                PAD_ID
            ].zero_()


        nn.init.normal_(
            self.language_model_head.weight,
            mean=0.0,
            std=0.02
        )


    def causal_mask(
            self,
            sequence_length,
            device
    ):

        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool
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
                "Sequence is longer than the "
                "model maximum."
            )


        token_vectors = (
            self.token_embedding(
                input_ids
            )
        )


        position_vectors = (
            self.position_embedding(
                sequence_length,
                input_ids.device
            )
        )


        position_vectors = (
            position_vectors.unsqueeze(
                0
            )
        )


        x = (
                token_vectors
                +
                position_vectors
        )


        mask = self.causal_mask(
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


        logits = (
            self.language_model_head(
                x
            )
        )


        return logits


    def parameter_count(
            self
    ):

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


# ==================================================
# 13. CREATE MODEL
# ==================================================

print("TEST 8: Create Silverwing Model")
print()


model = SilverwingDecoderLM().to(
    DEVICE
)


print(
    "Model created."
)

print(
    "Parameters:",
    model.parameter_count()
)

print()


# ==================================================
# 14. OPTIMIZER
# ==================================================

print("TEST 9: Optimizer")
print()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


print(
    "Optimizer: AdamW"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print()


# ==================================================
# 15. LEARNING-RATE SCHEDULER
# ==================================================

total_training_steps = max(
    1,
    len(train_loader) * EPOCHS
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_training_steps
)


print("TEST 10: Scheduler")
print()

print(
    "Training steps:",
    total_training_steps
)

print(
    "Scheduler: CosineAnnealingLR"
)

print()


# ==================================================
# 16. LOSS FUNCTION
# ==================================================

def compute_loss(
        logits,
        labels
):

    return F.cross_entropy(
        logits.reshape(
            -1,
            VOCABULARY_SIZE
        ),
        labels.reshape(
            -1
        ),
        ignore_index=PAD_ID
    )


# ==================================================
# 17. TRAINING STEP
# ==================================================

def training_step(
        model,
        batch,
        optimizer
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


    loss = compute_loss(
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


    return (
        float(
            loss.detach()
        ),
        float(
            gradient_norm
        )
    )


# ==================================================
# 18. VALIDATION
# ==================================================

@torch.no_grad()
def evaluate(
        model,
        data_loader
):

    model.eval()


    total_loss = 0.0
    total_batches = 0


    for batch in data_loader:

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


        logits = model(
            input_ids
        )


        loss = compute_loss(
            logits,
            labels
        )


        total_loss += float(
            loss
        )

        total_batches += 1


    if total_batches == 0:

        return float(
            "nan"
        )


    return (
            total_loss
            /
            total_batches
    )


# ==================================================
# 19. CHECKPOINT FUNCTIONS
# ==================================================

def save_checkpoint(
        path,
        epoch,
        global_step,
        train_loss,
        validation_loss,
        best_validation_loss
):

    checkpoint = {
        "epoch":
            epoch,

        "global_step":
            global_step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "configuration": {
            "vocabulary_size":
                VOCABULARY_SIZE,

            "model_dimension":
                MODEL_DIMENSION,

            "attention_heads":
                NUMBER_OF_HEADS,

            "feed_forward_dimension":
                FEED_FORWARD_DIMENSION,

            "layers":
                NUMBER_OF_LAYERS,

            "max_sequence_length":
                MAX_SEQUENCE_LENGTH,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY
        }
    }


    torch.save(
        checkpoint,
        path
    )


# ==================================================
# 20. TRAINING INITIALIZATION
# ==================================================

print("TEST 11: Training Initialization")
print()


global_step = 0

best_validation_loss = float(
    "inf"
)

training_history = []


print(
    "Initial best validation loss:",
    best_validation_loss
)

print()


# ==================================================
# 21. PRETRAINING
# ==================================================

print("TEST 12: Native Pretraining")
print()

training_start = time.perf_counter()


for epoch in range(
        1,
        EPOCHS + 1
):

    model.train()


    epoch_loss = 0.0

    epoch_steps = 0

    epoch_start = time.perf_counter()


    for batch_index, batch in enumerate(
            train_loader,
            start=1
    ):

        loss, gradient_norm = (
            training_step(
                model,
                batch,
                optimizer
            )
        )


        global_step += 1

        epoch_loss += loss

        epoch_steps += 1


        if (
                global_step
                %
                LOG_EVERY
                ==
                0
        ):

            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )


            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Batch {batch_index}/{len(train_loader)} "
                f"| Step {global_step} "
                f"| Loss {loss:.6f} "
                f"| Grad {gradient_norm:.6f} "
                f"| LR {current_lr:.8f}"
            )


    average_train_loss = (
            epoch_loss
            /
            max(
                epoch_steps,
                1
            )
    )


    validation_loss = evaluate(
        model,
        validation_loader
    )


    epoch_duration = (
            time.perf_counter()
            -
            epoch_start
    )


    perplexity = (
        math.exp(
            validation_loss
        )
        if math.isfinite(
            validation_loss
        )
           and
           validation_loss < 50
        else float(
            "inf"
        )
    )


    epoch_record = {
        "epoch":
            epoch,

        "train_loss":
            average_train_loss,

        "validation_loss":
            validation_loss,

        "validation_perplexity":
            perplexity,

        "learning_rate":
            optimizer.param_groups[
                0
            ]["lr"],

        "duration_seconds":
            epoch_duration
    }


    training_history.append(
        epoch_record
    )


    print()

    print(
        f"Epoch {epoch} complete"
    )

    print(
        "Train loss:",
        round(
            average_train_loss,
            6
        )
    )

    print(
        "Validation loss:",
        round(
            validation_loss,
            6
        )
    )

    print(
        "Validation perplexity:",
        perplexity
    )

    print(
        "Epoch duration:",
        round(
            epoch_duration,
            3
        ),
        "seconds"
    )

    print()


    save_checkpoint(
        LATEST_CHECKPOINT,
        epoch,
        global_step,
        average_train_loss,
        validation_loss,
        best_validation_loss
    )


    if (
            math.isfinite(
                validation_loss
            )
            and
            validation_loss
            <
            best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )


        save_checkpoint(
            BEST_CHECKPOINT,
            epoch,
            global_step,
            average_train_loss,
            validation_loss,
            best_validation_loss
        )


        print(
            "New best checkpoint saved."
        )

    else:

        print(
            "Best checkpoint unchanged."
        )

    print()


training_duration = (
        time.perf_counter()
        -
        training_start
)


# ==================================================
# 22. FINAL EVALUATION
# ==================================================

print("TEST 13: Final Evaluation")
print()


final_validation_loss = evaluate(
    model,
    validation_loader
)


if (
        math.isfinite(
            final_validation_loss
        )
        and
        final_validation_loss < 50
):

    final_perplexity = math.exp(
        final_validation_loss
    )

else:

    final_perplexity = float(
        "inf"
    )


print(
    "Final validation loss:",
    final_validation_loss
)

print(
    "Final perplexity:",
    final_perplexity
)

print()


# ==================================================
# 23. SAVE TRAINING LOG
# ==================================================

print("TEST 14: Save Training Log")
print()


training_log = {
    "model":
        "Silverwing-Decoder-v1",

    "dataset":
        "Silverwing-Corpus-v1",

    "epochs":
        EPOCHS,

    "global_steps":
        global_step,

    "best_validation_loss":
        best_validation_loss,

    "final_validation_loss":
        final_validation_loss,

    "final_perplexity":
        final_perplexity,

    "training_duration_seconds":
        training_duration,

    "history":
        training_history
}


with open(
        TRAINING_LOG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        training_log,
        file,
        indent=4
    )


print(
    "Saved:",
    TRAINING_LOG_FILE
)

print()


# ==================================================
# 24. TRAINING CURVE SUMMARY
# ==================================================

print("TEST 15: Training History")
print()


for record in training_history:

    print(
        "Epoch:",
        record["epoch"],
        "| train:",
        round(
            record["train_loss"],
            6
        ),
        "| validation:",
        round(
            record["validation_loss"],
            6
        )
    )


print()


# ==================================================
# 25. MODEL STATE SUMMARY
# ==================================================

print("TEST 16: Model State")
print()


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


print(
    "Model parameters:",
    parameter_count
)

print(
    "Global steps:",
    global_step
)

print(
    "Best validation loss:",
    best_validation_loss
)

print()


# ==================================================
# 26. CHECKPOINT STATUS
# ==================================================

print("TEST 17: Checkpoint Status")
print()


print(
    "Latest checkpoint:",
    LATEST_CHECKPOINT.exists()
)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT.exists()
)

print()


# ==================================================
# 27. SIMPLE NEXT-TOKEN TEST
# ==================================================

print("TEST 18: Post-Training Next-Token Test")
print()


model.eval()


if len(validation_dataset) > 0:

    sample = validation_dataset[
        0
    ]


    sample_input = (
        sample["input_ids"]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    with torch.no_grad():

        sample_logits = model(
            sample_input
        )


    last_logits = (
        sample_logits[
            0,
            -1
        ]
    )


    probabilities = F.softmax(
        last_logits,
        dim=-1
    )


    top_k = min(
        5,
        VOCABULARY_SIZE
    )


    values, indices = torch.topk(
        probabilities,
        k=top_k
    )


    print(
        "Top next-token candidates:"
    )


    for probability, token_id in zip(
            values.cpu().tolist(),
            indices.cpu().tolist()
    ):

        print(
            repr(
                ID_TO_TOKEN.get(
                    int(token_id),
                    "<UNK>"
                )
            ),
            "->",
            round(
                probability,
                6
            )
        )

else:

    print(
        "Validation set is empty."
    )


print()


# ==================================================
# 28. TRAINING ARCHITECTURE
# ==================================================

print("SILVERWING PRETRAINING ARCHITECTURE")
print()

print("Training Dataset")
print("      ↓")
print("Batch")
print("      ↓")
print("Silverwing Decoder")
print("      ↓")
print("Vocabulary Logits")
print("      ↓")
print("Causal Cross-Entropy")
print("      ↓")
print("Backward Pass")
print("      ↓")
print("Gradient Clipping")
print("      ↓")
print("AdamW")
print("      ↓")
print("Scheduler")
print("      ↓")
print("Validation")
print("      ↓")
print("Checkpoint")
print("      ↓")
print("Next Epoch")

print()


# ==================================================
# 29. WHAT PRETRAINING ACTUALLY DOES
# ==================================================

print("WHAT PRETRAINING DOES")
print()

print(
    "The model begins with randomly initialized "
    "parameters."
)

print()

print(
    "Each training example generates a prediction "
    "error."
)

print()

print(
    "Backpropagation computes how parameters "
    "contributed to that error."
)

print()

print(
    "The optimizer changes the parameters to reduce "
    "future error."
)

print()

print(
    "Repeated exposure to sufficient data gradually "
    "creates learned representations."
)

print()


# ==================================================
# 30. CURRENT LIMITATION
# ==================================================

print("CURRENT LIMITATION")
print()

print(
    "This corpus is intentionally tiny."
)

print()

print(
    "Training on it does not create a capable "
    "general-purpose AI."
)

print()

print(
    "It validates the complete native training "
    "mechanism."
)

print()

print(
    "Scaling Silverwing will require substantially "
    "larger and better-curated datasets, stronger "
    "evaluation, distributed training, and more "
    "efficient model infrastructure."
)

print()


# ==================================================
# 31. SELF-GROWTH FOUNDATION
# ==================================================

print("SELF-GROWTH FOUNDATION")
print()

print("Experience")
print("   ↓")
print("Validated Data")
print("   ↓")
print("Dataset Version")
print("   ↓")
print("Training Experiment")
print("   ↓")
print("Candidate Model")
print("   ↓")
print("Evaluation")
print("   ↓")
print("Promotion / Reject")
print("   ↓")
print("New Silverwing Version")

print()


# ==================================================
# 32. IMPORTANT ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Autonomous improvement should operate through "
    "controlled experiments and versioned artifacts."
)

print()

print(
    "A model should not silently overwrite its "
    "production weights."
)

print()

print(
    "Every candidate should be measurable, "
    "reproducible, and reversible."
)

print()


# ==================================================
# 33. FOUNDATION MODEL PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own BPE Tokenizer")
print(" ↓")
print("Own Subword Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("Own Embedding System")
print(" ↓")
print("Own Position Encoding")
print(" ↓")
print("Own Self-Attention")
print(" ↓")
print("Own Transformer Block")
print(" ↓")
print("Own Decoder Language Model")
print(" ↓")
print("Own Training Data Pipeline")
print(" ↓")
print("OWN PRETRAINING ENGINE  ← COMPLETE")
print(" ↓")
print("Evaluation Framework  ← NEXT")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Multitask Training")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 73R COMPLETE ===")
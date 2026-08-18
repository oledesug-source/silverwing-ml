# Silverwing ML
# Phase 5 - Lesson 74R
# Silverwing Own Foundation Model
# Native Evaluation Framework
#
# Goal:
# Build a reproducible evaluation framework for
# Silverwing model checkpoints.
#
# The evaluator measures:
# - cross-entropy loss
# - perplexity
# - token accuracy
# - top-k accuracy
# - confidence
# - calibration-style statistics
# - checkpoint comparison
# - regression detection
#
# No external pretrained language model is used.


import json
import math
import time

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 74R")
print("Silverwing Own Foundation Model")
print("Native Evaluation Framework")
print()


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

LESSON_72_DIR = (
        BASE_DIR.parent / "lesson72R"
)

LESSON_73_DIR = (
        BASE_DIR.parent / "lesson73R"
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

LATEST_CHECKPOINT = (
        LESSON_73_DIR
        / "checkpoints"
        / "silverwing_latest.pt"
)

BEST_CHECKPOINT = (
        LESSON_73_DIR
        / "checkpoints"
        / "silverwing_best.pt"
)

TRAINING_LOG_FILE = (
        LESSON_73_DIR
        / "silverwing_training_log.json"
)

EVALUATION_REPORT = (
        BASE_DIR
        / "silverwing_evaluation_report.json"
)

COMPARISON_REPORT = (
        BASE_DIR
        / "silverwing_model_comparison.json"
)


BATCH_SIZE = 2

TOP_K_VALUES = [
    1,
    3,
    5,
    10
]

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


torch.manual_seed(
    SEED
)


print("TEST 1: Configuration")
print()

print(
    "Latest checkpoint:",
    LATEST_CHECKPOINT
)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Validation dataset:",
    VALIDATION_FILE
)

print(
    "Device:",
    DEVICE
)

print()


# ==================================================
# 2. VERIFY FILES
# ==================================================

print("TEST 2: Verify Evaluation Artifacts")
print()


required_files = [
    VOCABULARY_FILE,
    MODEL_CONFIG_FILE,
    VALIDATION_FILE,
    DATASET_CONFIG_FILE,
    LATEST_CHECKPOINT,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n"
            f"{file_path}\n"
            f"Complete the required previous "
            f"lesson first."
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

print()


# ==================================================
# 4. MODEL CONFIGURATION
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
    "Dimension:",
    MODEL_DIMENSION
)

print(
    "Heads:",
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

print()


# ==================================================
# 5. DATASET
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
                        "Input and label "
                        "lengths differ."
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


validation_dataset = (
    SilverwingLanguageDataset(
        VALIDATION_FILE
    )
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("TEST 5: Validation Dataset")
print()

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Validation batches:",
    len(validation_loader)
)

print()


# ==================================================
# 6. ATTENTION
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
                "Model dimension must be "
                "divisible by the number "
                "of heads."
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


        output = self.output_projection(
            output
        )


        return self.dropout(
            output
        )


# ==================================================
# 7. FEED-FORWARD
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

        return self.output_projection(
            x
        )


# ==================================================
# 8. TRANSFORMER BLOCK
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


        return self.norm_feed_forward(
            x
        )


# ==================================================
# 9. POSITION EMBEDDING
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


    def forward(
            self,
            sequence_length,
            device
    ):

        position_ids = torch.arange(
            sequence_length,
            device=device
        )


        return self.embedding(
            position_ids
        )


# ==================================================
# 10. SILVERWING DECODER
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
                "Sequence exceeds model "
                "maximum sequence length."
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


        mask = (
            self.create_causal_mask(
                sequence_length,
                input_ids.device
            )
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


    def parameter_count(
            self
    ):

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


# ==================================================
# 11. LOAD CHECKPOINT
# ==================================================

def load_checkpoint(
        checkpoint_path
):

    model = (
        SilverwingDecoderLM()
        .to(DEVICE)
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    return (
        model,
        checkpoint
    )


print("TEST 6: Load Latest Checkpoint")
print()


model, latest_checkpoint = (
    load_checkpoint(
        LATEST_CHECKPOINT
    )
)


print(
    "Checkpoint loaded."
)

print(
    "Epoch:",
    latest_checkpoint.get(
        "epoch"
    )
)

print(
    "Global step:",
    latest_checkpoint.get(
        "global_step"
    )
)

print(
    "Training loss:",
    latest_checkpoint.get(
        "train_loss"
    )
)

print(
    "Validation loss recorded during training:",
    latest_checkpoint.get(
        "validation_loss"
    )
)

print()


# ==================================================
# 12. MODEL PARAMETER COUNT
# ==================================================

print("TEST 7: Parameter Count")
print()

print(
    "Parameters:",
    model.parameter_count()
)

print()


# ==================================================
# 13. LOSS EVALUATION
# ==================================================

@torch.no_grad()
def evaluate_loss(
        model,
        loader
):

    model.eval()


    total_loss = 0.0

    total_batches = 0


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


        logits = model(
            input_ids
        )


        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCABULARY_SIZE
            ),
            labels.reshape(
                -1
            ),
            ignore_index=PAD_ID
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


print("TEST 8: Validation Loss")
print()


validation_loss = evaluate_loss(
    model,
    validation_loader
)


if math.isfinite(
        validation_loss
):

    perplexity = math.exp(
        validation_loss
    )

else:

    perplexity = float(
        "inf"
    )


print(
    "Validation loss:",
    validation_loss
)

print(
    "Perplexity:",
    perplexity
)

print()


# ==================================================
# 14. TOKEN ACCURACY
# ==================================================

@torch.no_grad()
def evaluate_accuracy(
        model,
        loader
):

    model.eval()


    correct = 0
    total = 0


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


        logits = model(
            input_ids
        )


        predictions = torch.argmax(
            logits,
            dim=-1
        )


        valid_mask = (
                labels
                !=
                PAD_ID
        )


        correct += int(
            (
                    predictions[valid_mask]
                    ==
                    labels[valid_mask]
            ).sum()
        )


        total += int(
            valid_mask.sum()
        )


    if total == 0:

        return float(
            "nan"
        )


    return (
            correct
            /
            total
    )


print("TEST 9: Next-Token Accuracy")
print()


accuracy = evaluate_accuracy(
    model,
    validation_loader
)


print(
    "Token accuracy:",
    accuracy
)

print()


# ==================================================
# 15. TOP-K ACCURACY
# ==================================================

@torch.no_grad()
def evaluate_top_k(
        model,
        loader,
        k
):

    model.eval()


    correct = 0
    total = 0


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


        logits = model(
            input_ids
        )


        _, top_indices = torch.topk(
            logits,
            k=min(
                k,
                VOCABULARY_SIZE
            ),
            dim=-1
        )


        valid_mask = (
                labels
                !=
                PAD_ID
        )


        valid_labels = (
            labels[
                valid_mask
            ]
        )


        valid_predictions = (
            top_indices[
                valid_mask
            ]
        )


        matches = (
                valid_predictions
                ==
                valid_labels.unsqueeze(
                    -1
                )
        )


        correct += int(
            matches.any(
                dim=-1
            ).sum()
        )


        total += int(
            valid_mask.sum()
        )


    if total == 0:

        return float(
            "nan"
        )


    return (
            correct
            /
            total
    )


print("TEST 10: Top-K Accuracy")
print()


top_k_results = {}


for k in TOP_K_VALUES:

    score = evaluate_top_k(
        model,
        validation_loader,
        k
    )


    top_k_results[
        f"top_{k}"
    ] = score


    print(
        f"Top-{k} accuracy:",
        score
    )


print()


# ==================================================
# 16. CONFIDENCE ANALYSIS
# ==================================================

@torch.no_grad()
def evaluate_confidence(
        model,
        loader
):

    model.eval()


    confidence_sum = 0.0

    correct_confidence = 0.0

    incorrect_confidence = 0.0

    correct_count = 0

    incorrect_count = 0


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


        logits = model(
            input_ids
        )


        probabilities = F.softmax(
            logits,
            dim=-1
        )


        predicted_probability, predictions = (
            torch.max(
                probabilities,
                dim=-1
            )
        )


        valid_mask = (
                labels
                !=
                PAD_ID
        )


        correct_mask = (
                predictions
                ==
                labels
        )


        correct_mask = (
                correct_mask
                &
                valid_mask
        )


        incorrect_mask = (
                (~correct_mask)
                &
                valid_mask
        )


        confidence_sum += float(
            predicted_probability[
                valid_mask
            ].sum()
        )


        correct_confidence += float(
            predicted_probability[
                correct_mask
            ].sum()
        )


        incorrect_confidence += float(
            predicted_probability[
                incorrect_mask
            ].sum()
        )


        correct_count += int(
            correct_mask.sum()
        )


        incorrect_count += int(
            incorrect_mask.sum()
        )


    total = (
            correct_count
            +
            incorrect_count
    )


    return {
        "mean_confidence":
            (
                confidence_sum
                /
                total
                if total
                else float("nan")
            ),

        "correct_confidence":
            (
                correct_confidence
                /
                correct_count
                if correct_count
                else float("nan")
            ),

        "incorrect_confidence":
            (
                incorrect_confidence
                /
                incorrect_count
                if incorrect_count
                else float("nan")
            )
    }


print("TEST 11: Confidence")
print()


confidence_results = (
    evaluate_confidence(
        model,
        validation_loader
    )
)


print(
    "Mean confidence:",
    confidence_results[
        "mean_confidence"
    ]
)

print(
    "Correct prediction confidence:",
    confidence_results[
        "correct_confidence"
    ]
)

print(
    "Incorrect prediction confidence:",
    confidence_results[
        "incorrect_confidence"
    ]
)

print()


# ==================================================
# 17. CALIBRATION BINS
# ==================================================

@torch.no_grad()
def calibration_report(
        model,
        loader,
        number_of_bins=10
):

    model.eval()


    bin_data = [
        {
            "count": 0,
            "correct": 0,
            "confidence_sum": 0.0
        }
        for _ in range(
            number_of_bins
        )
    ]


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


        logits = model(
            input_ids
        )


        probabilities = F.softmax(
            logits,
            dim=-1
        )


        confidence, predictions = torch.max(
            probabilities,
            dim=-1
        )


        valid_mask = (
                labels
                !=
                PAD_ID
        )


        valid_confidence = (
            confidence[
                valid_mask
            ]
        )


        valid_predictions = (
            predictions[
                valid_mask
            ]
        )


        valid_labels = (
            labels[
                valid_mask
            ]
        )


        for conf, prediction, label in zip(
                valid_confidence.cpu().tolist(),
                valid_predictions.cpu().tolist(),
                valid_labels.cpu().tolist()
        ):

            bin_index = min(
                int(
                    conf
                    *
                    number_of_bins
                ),
                number_of_bins - 1
            )


            bin_data[
                bin_index
            ]["count"] += 1


            bin_data[
                bin_index
            ]["confidence_sum"] += conf


            if prediction == label:

                bin_data[
                    bin_index
                ]["correct"] += 1


    report = []


    for index, item in enumerate(
            bin_data
    ):

        if item["count"] == 0:

            continue


        average_confidence = (
                item["confidence_sum"]
                /
                item["count"]
        )


        accuracy_value = (
                item["correct"]
                /
                item["count"]
        )


        report.append(
            {
                "bin":
                    index,

                "lower":
                    index
                    /
                    number_of_bins,

                "upper":
                    (
                            index + 1
                    )
                    /
                    number_of_bins,

                "count":
                    item["count"],

                "confidence":
                    average_confidence,

                "accuracy":
                    accuracy_value,

                "gap":
                    abs(
                        average_confidence
                        -
                        accuracy_value
                    )
            }
        )


    return report


print("TEST 12: Calibration")
print()


calibration = calibration_report(
    model,
    validation_loader
)


for item in calibration:

    print(
        f"[{item['lower']:.1f}, "
        f"{item['upper']:.1f}) "
        f"count={item['count']} "
        f"confidence={item['confidence']:.4f} "
        f"accuracy={item['accuracy']:.4f} "
        f"gap={item['gap']:.4f}"
    )


print()


# ==================================================
# 18. CROSS-ENTROPY BASELINE
# ==================================================

print("TEST 13: Random Baseline")
print()


random_baseline_loss = math.log(
    VOCABULARY_SIZE
)


random_baseline_perplexity = (
    math.exp(
        random_baseline_loss
    )
)


print(
    "Random loss:",
    random_baseline_loss
)

print(
    "Random perplexity:",
    random_baseline_perplexity
)

print()


# ==================================================
# 19. MODEL HEALTH CHECK
# ==================================================

print("TEST 14: Model Numerical Health")
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


print(
    "Parameters containing NaN:",
    nan_parameters
)

print(
    "Parameters containing Inf:",
    inf_parameters
)


if (
        nan_parameters
        >
        0
        or
        inf_parameters
        >
        0
):

    raise RuntimeError(
        "Model contains invalid numerical values."
    )


print(
    "Model numerical state is valid."
)

print()


# ==================================================
# 20. CHECKPOINT METADATA
# ==================================================

print("TEST 15: Checkpoint Metadata")
print()


checkpoint_metadata = {
    "epoch":
        latest_checkpoint.get(
            "epoch"
        ),

    "global_step":
        latest_checkpoint.get(
            "global_step"
        ),

    "training_loss":
        latest_checkpoint.get(
            "train_loss"
        ),

    "recorded_validation_loss":
        latest_checkpoint.get(
            "validation_loss"
        ),

    "best_validation_loss":
        latest_checkpoint.get(
            "best_validation_loss"
        )
}


print(
    json.dumps(
        checkpoint_metadata,
        indent=4
    )
)

print()


# ==================================================
# 21. CURRENT VS CHECKPOINT VALIDATION
# ==================================================

recorded_validation_loss = (
    latest_checkpoint.get(
        "validation_loss"
    )
)


if (
        recorded_validation_loss
        is not None
        and
        math.isfinite(
            recorded_validation_loss
        )
        and
        math.isfinite(
            validation_loss
        )
):

    validation_difference = (
            validation_loss
            -
            recorded_validation_loss
    )

else:

    validation_difference = float(
        "nan"
    )


print("TEST 16: Recorded vs Recomputed")
print()


print(
    "Recorded validation loss:",
    recorded_validation_loss
)

print(
    "Recomputed validation loss:",
    validation_loss
)

print(
    "Difference:",
    validation_difference
)

print()


# ==================================================
# 22. EVALUATION SCORECARD
# ==================================================

print("TEST 17: Evaluation Scorecard")
print()


scorecard = {
    "validation_loss":
        validation_loss,

    "perplexity":
        perplexity,

    "token_accuracy":
        accuracy,

    "top_k_accuracy":
        top_k_results,

    "confidence":
        confidence_results,

    "calibration":
        calibration,

    "model_parameters":
        model.parameter_count(),

    "checkpoint_epoch":
        latest_checkpoint.get(
            "epoch"
        ),

    "checkpoint_step":
        latest_checkpoint.get(
            "global_step"
        ),

    "numerically_healthy":
        True
}


print(
    json.dumps(
        scorecard,
        indent=4,
        default=str
    )
)

print()


# ==================================================
# 23. MODEL REGRESSION LOGIC
# ==================================================

def compare_losses(
        current_loss,
        baseline_loss,
        tolerance=1e-6
):

    if not math.isfinite(
            current_loss
    ):

        return {
            "status":
                "invalid",

            "improvement":
                False
        }


    if not math.isfinite(
            baseline_loss
    ):

        return {
            "status":
                "no_valid_baseline",

            "improvement":
                True
        }


    difference = (
            current_loss
            -
            baseline_loss
    )


    if difference < -tolerance:

        status = "improved"

        improvement = True

    elif difference > tolerance:

        status = "regressed"

        improvement = False

    else:

        status = "unchanged"

        improvement = False


    return {
        "status":
            status,

        "improvement":
            improvement,

        "difference":
            difference
    }


baseline_loss = (
    latest_checkpoint.get(
        "best_validation_loss"
    )
)


comparison = compare_losses(
    validation_loss,
    baseline_loss
)


print("TEST 18: Regression Detection")
print()


print(
    "Baseline loss:",
    baseline_loss
)

print(
    "Current loss:",
    validation_loss
)

print(
    "Status:",
    comparison[
        "status"
    ]
)

print(
    "Improvement:",
    comparison[
        "improvement"
    ]
)

print()


# ==================================================
# 24. BEST CHECKPOINT COMPARISON
# ==================================================

best_comparison = None


if BEST_CHECKPOINT.exists():

    print(
        "TEST 19: Best Checkpoint Comparison"
    )

    print()


    best_model, best_checkpoint = (
        load_checkpoint(
            BEST_CHECKPOINT
        )
    )


    best_loss = evaluate_loss(
        best_model,
        validation_loader
    )


    best_accuracy = (
        evaluate_accuracy(
            best_model,
            validation_loader
        )
    )


    best_comparison = {
        "latest_validation_loss":
            validation_loss,

        "best_checkpoint_validation_loss":
            best_loss,

        "latest_accuracy":
            accuracy,

        "best_checkpoint_accuracy":
            best_accuracy,

        "loss_difference":
            (
                    validation_loss
                    -
                    best_loss
            ),

        "accuracy_difference":
            (
                    accuracy
                    -
                    best_accuracy
            )
    }


    print(
        json.dumps(
            best_comparison,
            indent=4
        )
    )

    print()


# ==================================================
# 25. SAVE EVALUATION REPORT
# ==================================================

print("TEST 20: Save Evaluation Report")
print()


evaluation_report = {
    "system":
        "Silverwing",

    "evaluation_version":
        "74R",

    "timestamp":
        time.time(),

    "model":
        {
            "architecture":
                "Silverwing-Decoder-v1",

            "parameters":
                model.parameter_count(),

            "vocabulary_size":
                VOCABULARY_SIZE,

            "layers":
                NUMBER_OF_LAYERS,

            "model_dimension":
                MODEL_DIMENSION,

            "attention_heads":
                NUMBER_OF_HEADS
        },

    "checkpoint":
        checkpoint_metadata,

    "metrics":
        {
            "validation_loss":
                validation_loss,

            "perplexity":
                perplexity,

            "token_accuracy":
                accuracy,

            "top_k":
                top_k_results,

            "confidence":
                confidence_results,

            "calibration":
                calibration
        },

    "health":
        {
            "nan_parameters":
                nan_parameters,

            "inf_parameters":
                inf_parameters,

            "numerically_healthy":
                True
        },

    "comparison":
        comparison,

    "best_checkpoint_comparison":
        best_comparison
}


with open(
        EVALUATION_REPORT,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        evaluation_report,
        file,
        indent=4,
        default=str
    )


print(
    "Saved:",
    EVALUATION_REPORT
)

print()


# ==================================================
# 26. SAVE COMPARISON REPORT
# ==================================================

print("TEST 21: Save Model Comparison")
print()


comparison_report = {
    "baseline":
        {
            "checkpoint":
                str(
                    BEST_CHECKPOINT
                ),

            "validation_loss":
                baseline_loss
        },

    "candidate":
        {
            "checkpoint":
                str(
                    LATEST_CHECKPOINT
                ),

            "validation_loss":
                validation_loss,

            "accuracy":
                accuracy,

            "perplexity":
                perplexity
        },

    "decision":
        {
            "status":
                comparison[
                    "status"
                ],

            "improved":
                comparison[
                    "improvement"
                ]
        }
}


with open(
        COMPARISON_REPORT,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        comparison_report,
        file,
        indent=4,
        default=str
    )


print(
    "Saved:",
    COMPARISON_REPORT
)

print()


# ==================================================
# 27. EVALUATION GATE
# ==================================================

print("TEST 22: Promotion Gate")
print()


def promotion_gate(
        report
):

    metrics = report[
        "metrics"
    ]

    health = report[
        "health"
    ]


    if not health[
        "numerically_healthy"
    ]:

        return {
            "decision":
                "REJECT",

            "reason":
                "Numerical instability detected."
        }


    loss = metrics[
        "validation_loss"
    ]


    if not math.isfinite(
            loss
    ):

        return {
            "decision":
                "REJECT",

            "reason":
                "Invalid validation loss."
        }


    # This is deliberately a conservative
    # educational gate. Future versions should
    # use a richer multi-metric policy.

    return {
        "decision":
            "PASS",

        "reason":
            "Candidate passed basic numerical "
            "and validation checks."
    }


promotion_decision = promotion_gate(
    evaluation_report
)


print(
    "Decision:",
    promotion_decision[
        "decision"
    ]
)

print(
    "Reason:",
    promotion_decision[
        "reason"
    ]
)

print()


# ==================================================
# 28. WHAT THIS ENABLES
# ==================================================

print("WHAT THIS ENABLES")
print()

print(
    "Silverwing can now compare model versions "
    "using measured evidence instead of intuition."
)

print()

print(
    "This is the foundation of controlled "
    "self-improvement."
)

print()


# ==================================================
# 29. FUTURE SELF-IMPROVEMENT LOOP
# ==================================================

print("FUTURE SILVERWING SELF-IMPROVEMENT LOOP")
print()

print("Current Model")
print("      ↓")
print("Identify Weakness")
print("      ↓")
print("Create Candidate Training Data")
print("      ↓")
print("Train Candidate")
print("      ↓")
print("Evaluate Candidate")
print("      ↓")
print("Compare")
print("      ↓")
print("Promotion Gate")
print("   ┌──┴──┐")
print(" PASS  REJECT")
print("   ↓      ↓")
print("Version  Keep Current")
print("   ↓")
print("New Silverwing")

print()


# ==================================================
# 30. BIO-INSPIRED EVALUATION CONNECTION
# ==================================================

print("BIO-INSPIRED EVALUATION CONNECTION")
print()

print(
    "A self-improving system needs internal feedback."
)

print()

print(
    "Evaluation acts as a computational feedback "
    "signal indicating whether an adaptation helped "
    "or harmed performance."
)

print()

print(
    "Future Silverwing loops can combine this signal "
    "with resource state, memory, task success, "
    "prediction error, and environmental feedback."
)

print()


# ==================================================
# 31. IMPORTANT ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Self-improvement without evaluation is simply "
    "uncontrolled mutation."
)

print()

print(
    "Silverwing therefore requires an evaluation gate "
    "between every candidate version and production "
    "promotion."
)

print()


# ==================================================
# 32. CURRENT LIMITATION
# ==================================================

print("CURRENT LIMITATION")
print()

print(
    "The current evaluation dataset is very small."
)

print()

print(
    "Its metrics are useful for validating the "
    "engineering pipeline, not for claiming that "
    "Silverwing is intelligent or broadly capable."
)

print()

print(
    "As the corpus grows, evaluation must become "
    "larger, more diverse, task-specific, and "
    "independent from the training data."
)

print()


# ==================================================
# 33. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "The model can now be trained and objectively "
    "measured."
)

print()

print(
    "The next stage is to build a stronger training "
    "and evaluation curriculum rather than immediately "
    "connecting the model to external LLMs."
)

print()

print(
    "Lesson 75R will establish Silverwing's "
    "controlled experiment runner and versioned "
    "model promotion system."
)

print()


# ==================================================
# 34. FOUNDATION MODEL PROGRESS
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
print("Own Pretraining Engine")
print(" ↓")
print("OWN EVALUATION FRAMEWORK")
print(" ↓")
print("Experiment Runner")
print(" ↓")
print("Version Promotion")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 74R COMPLETE ===")
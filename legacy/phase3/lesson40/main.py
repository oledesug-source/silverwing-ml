# Silverwing ML
# Phase 3 - Lesson 40
# Language Model Training, Cross-Entropy and Perplexity


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 40")
print("Language Model Training, Cross-Entropy and Perplexity")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42

torch.manual_seed(SEED)


print("TEST 1: Configuration")
print()

print("Random seed:", SEED)
print("PyTorch version:", torch.__version__)

print()


# ==================================================
# 2. EXAMPLE VOCABULARY
# ==================================================

vocabulary = [
    "<PAD>",
    "machine",
    "temperature",
    "pressure",
    "is",
    "normal",
    "high",
    "warning",
    "critical",
    "detected"
]


VOCAB_SIZE = len(vocabulary)


print("TEST 2: Vocabulary")
print()

for index, token in enumerate(vocabulary):

    print(
        index,
        "->",
        token
    )

print()

print(
    "Vocabulary size:",
    VOCAB_SIZE
)

print()


# ==================================================
# 3. CREATE EXAMPLE LOGITS
# ==================================================

print("TEST 3: Model Logits")
print()


logits = torch.tensor([
    [
        0.2,
        0.5,
        1.5,
        0.3,
        0.7,
        0.4,
        2.5,
        0.2,
        0.1,
        0.4
    ]
])


print(
    "Logits:"
)

print(
    logits
)

print()


# ==================================================
# 4. CONVERT LOGITS TO PROBABILITIES
# ==================================================

print("TEST 4: Softmax")
print()


probabilities = F.softmax(
    logits,
    dim=-1
)


print(
    "Probabilities:"
)

print(
    probabilities
)

print()

print(
    "Probability sum:",
    probabilities.sum().item()
)

print()


# ==================================================
# 5. TRUE TARGET TOKEN
# ==================================================

print("TEST 5: Target Token")
print()


target_token_id = torch.tensor([
    6
])


print(
    "Target token ID:",
    target_token_id.item()
)

print(
    "Target token:",
    vocabulary[
        target_token_id.item()
    ]
)

print()


# ==================================================
# 6. CROSS-ENTROPY LOSS
# ==================================================

print("TEST 6: Cross-Entropy Loss")
print()


loss_function = nn.CrossEntropyLoss()


loss = loss_function(
    logits,
    target_token_id
)


print(
    "Cross-entropy loss:",
    loss.item()
)

print()


# ==================================================
# 7. MANUAL NEGATIVE LOG PROBABILITY
# ==================================================

print("TEST 7: Negative Log Probability")
print()


target_probability = probabilities[
    0,
    target_token_id.item()
]


negative_log_probability = -torch.log(
    target_probability
)


print(
    "Target probability:",
    target_probability.item()
)

print(
    "Negative log probability:",
    negative_log_probability.item()
)

print()


# ==================================================
# 8. COMPARE BOTH VALUES
# ==================================================

print("TEST 8: Cross-Entropy Verification")
print()


print(
    "Cross-entropy:",
    loss.item()
)

print(
    "Negative log probability:",
    negative_log_probability.item()
)

print()


print(
    "Difference:",
    abs(
        loss.item()
        -
        negative_log_probability.item()
    )
)

print()


# ==================================================
# 9. HIGH CONFIDENCE VS LOW CONFIDENCE
# ==================================================

print("TEST 9: Confidence and Loss")
print()


high_confidence_logits = torch.tensor([
    [0.0, 0.0, 0.0, 6.0]
])


low_confidence_logits = torch.tensor([
    [1.5, 1.5, 1.5, 1.5]
])


target = torch.tensor([
    3
])


high_confidence_loss = (
    loss_function(
        high_confidence_logits,
        target
    )
)


low_confidence_loss = (
    loss_function(
        low_confidence_logits,
        target
    )
)


print(
    "High-confidence correct prediction loss:",
    high_confidence_loss.item()
)

print(
    "Low-confidence prediction loss:",
    low_confidence_loss.item()
)

print()


# ==================================================
# 10. WRONG HIGH-CONFIDENCE PREDICTION
# ==================================================

print("TEST 10: Wrong High-Confidence Prediction")
print()


wrong_confidence_logits = torch.tensor([
    [6.0, 0.0, 0.0, 0.0]
])


wrong_confidence_loss = (
    loss_function(
        wrong_confidence_logits,
        target
    )
)


print(
    "Loss:",
    wrong_confidence_loss.item()
)

print()

print(
    "A highly confident incorrect prediction "
    "can receive a large loss."
)

print()


# ==================================================
# 11. MULTIPLE TOKENS
# ==================================================

print("TEST 11: Multiple Token Predictions")
print()


sequence_logits = torch.tensor([
    [
        [2.0, 0.5, 1.0, 0.2],
        [0.5, 2.5, 0.3, 0.1],
        [0.2, 0.3, 2.8, 0.1],
        [0.1, 0.2, 0.4, 3.0]
    ]
])


sequence_targets = torch.tensor([
    [
        0,
        1,
        2,
        3
    ]
])


sequence_loss = loss_function(
    sequence_logits.reshape(
        -1,
        4
    ),
    sequence_targets.reshape(
        -1
    )
)


print(
    "Sequence cross-entropy:",
    sequence_loss.item()
)

print()


# ==================================================
# 12. PAD TOKENS
# ==================================================

print("TEST 12: Padding")
print()


PAD_ID = 0


loss_with_padding = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


padded_logits = torch.tensor([
    [
        [2.0, 0.5, 1.0, 0.2],
        [0.5, 2.5, 0.3, 0.1],
        [0.2, 0.3, 2.8, 0.1],
        [0.1, 0.2, 0.4, 3.0],
        [4.0, 0.1, 0.1, 0.1]
    ]
])


padded_targets = torch.tensor([
    [
        0,
        1,
        2,
        3,
        0
    ]
])


padded_loss = loss_with_padding(
    padded_logits.reshape(
        -1,
        4
    ),
    padded_targets.reshape(
        -1
    )
)


print(
    "Loss with padding ignored:",
    padded_loss.item()
)

print()


# ==================================================
# 13. PERPLEXITY
# ==================================================

print("TEST 13: Perplexity")
print()


perplexity = math.exp(
    sequence_loss.item()
)


print(
    "Cross-entropy:",
    sequence_loss.item()
)

print(
    "Perplexity:",
    perplexity
)

print()


# ==================================================
# 14. PERPLEXITY FUNCTION
# ==================================================

print("TEST 14: Perplexity Function")
print()


def calculate_perplexity(loss):

    return math.exp(
        loss
    )


example_losses = [
    0.1,
    0.5,
    1.0,
    2.0,
    3.0
]


for example_loss in example_losses:

    print(
        "Loss:",
        example_loss,
        "-> Perplexity:",
        round(
            calculate_perplexity(
                example_loss
            ),
            4
        )
    )


print()


# ==================================================
# 15. SIMPLE LANGUAGE MODEL
# ==================================================

print("TEST 15: Mini Language Model")
print()


class TinyLanguageModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dimension,
            hidden_dimension
    ):

        super().__init__()


        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dimension
        )


        self.network = nn.Sequential(

            nn.Linear(
                embedding_dimension,
                hidden_dimension
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dimension,
                vocab_size
            )
        )


    def forward(
            self,
            token_ids
    ):

        embeddings = self.embedding(
            token_ids
        )

        logits = self.network(
            embeddings
        )

        return logits


model = TinyLanguageModel(
    vocab_size=VOCAB_SIZE,
    embedding_dimension=16,
    hidden_dimension=32
)


print(model)

print()


# ==================================================
# 16. TRAINING EXAMPLES
# ==================================================

print("TEST 16: Training Examples")
print()


X = torch.tensor([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])


y = torch.tensor([
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
])


print(
    "Inputs:"
)

print(X.squeeze())

print()

print(
    "Targets:"
)

print(y)

print()


# ==================================================
# 17. TRAINING COMPONENTS
# ==================================================

loss_function = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)


# ==================================================
# 18. TRAIN MINI LANGUAGE MODEL
# ==================================================

print("TEST 17: Train Tiny Language Model")
print()


EPOCHS = 500


for epoch in range(
        EPOCHS
):

    logits = model(
        X
    )


    loss = loss_function(
        logits.squeeze(1),
        y
    )


    optimizer.zero_grad()


    loss.backward()


    optimizer.step()


    if (
            epoch == 0
            or
            (epoch + 1) % 100 == 0
    ):

        current_perplexity = (
            calculate_perplexity(
                loss.item()
            )
        )


        print(
            "Epoch:",
            epoch + 1,
            "| Loss:",
            round(
                loss.item(),
                6
            ),
            "| Perplexity:",
            round(
                current_perplexity,
                4
            )
        )


print()


# ==================================================
# 19. FINAL MODEL EVALUATION
# ==================================================

print("TEST 18: Final Evaluation")
print()


model.eval()


with torch.no_grad():

    final_logits = model(
        X
    )


    final_loss = loss_function(
        final_logits.squeeze(1),
        y
    )


final_perplexity = (
    calculate_perplexity(
        final_loss.item()
    )
)


print(
    "Final loss:",
    final_loss.item()
)

print(
    "Final perplexity:",
    final_perplexity
)

print()


# ==================================================
# 20. NEXT TOKEN PREDICTIONS
# ==================================================

print("TEST 19: Predictions")
print()


with torch.no_grad():

    predictions = torch.argmax(
        final_logits,
        dim=-1
    )


for input_token, target_token, predicted_token in zip(
        X.squeeze(),
        y,
        predictions.squeeze()
):

    print(
        "Input:",
        vocabulary[
            input_token.item()
        ],

        "| Target:",
        vocabulary[
            target_token.item()
        ],

        "| Predicted:",
        vocabulary[
            predicted_token.item()
        ]
    )


print()


# ==================================================
# 21. TOKEN ACCURACY
# ==================================================

print("TEST 20: Token Accuracy")
print()


correct = (
        predictions.squeeze()
        ==
        y
)


accuracy = (
    correct.float().mean()
)


print(
    "Token accuracy:",
    accuracy.item()
)

print()


# ==================================================
# 22. LOSS VS PERPLEXITY
# ==================================================

print("TEST 21: Loss vs Perplexity")
print()

print(
    "Lower cross-entropy generally means "
    "the model assigns more probability to "
    "the observed target tokens."
)

print()

print(
    "Perplexity is the exponential of average "
    "cross-entropy under this formulation."
)

print()

print(
    "Lower perplexity generally indicates "
    "better predictive performance on the "
    "same evaluation setup."
)

print()


# ==================================================
# 23. IMPORTANT PERPLEXITY CAVEAT
# ==================================================

print("TEST 22: Perplexity Caveat")
print()

print(
    "Perplexity values are meaningful when "
    "comparing models evaluated on the same "
    "or carefully controlled tokenization and dataset."
)

print()

print(
    "A perplexity value from one tokenizer or "
    "dataset should not automatically be compared "
    "with a value from a different setup."
)

print()


# ==================================================
# 24. TRAINING LOOP
# ==================================================

print("LANGUAGE MODEL TRAINING LOOP")
print()

print("Input tokens")
print("      ↓")
print("Transformer / Neural Network")
print("      ↓")
print("Vocabulary logits")
print("      ↓")
print("Cross-Entropy Loss")
print("      ↓")
print("Backpropagation")
print("      ↓")
print("Optimizer")
print("      ↓")
print("Updated parameters")
print("      ↓")
print("Repeat")

print()


# ==================================================
# 25. EVALUATION LOOP
# ==================================================

print("LANGUAGE MODEL EVALUATION")
print()

print("Held-out text")
print("      ↓")
print("Model predictions")
print("      ↓")
print("Cross-Entropy")
print("      ↓")
print("Perplexity")
print("      ↓")
print("Compare models")

print()


# ==================================================
# 26. CONNECTION TO LLM TRAINING
# ==================================================

print("LLM TRAINING CONNECTION")
print()

print(
    "Large language models are trained on "
    "very large token sequences."
)

print()

print(
    "For autoregressive training, the model "
    "learns to predict target tokens from "
    "preceding tokens."
)

print()

print(
    "Cross-entropy is a central training objective "
    "for many autoregressive language models."
)

print()


# ==================================================
# 27. WHAT WE HAVE ACHIEVED
# ==================================================

print("CURRENT LANGUAGE-MODEL FOUNDATION")
print()

print("Tokenization")
print("      ↓")
print("Embeddings")
print("      ↓")
print("Transformer")
print("      ↓")
print("Next-token prediction")
print("      ↓")
print("Cross-entropy")
print("      ↓")
print("Backpropagation")
print("      ↓")
print("Optimization")
print("      ↓")
print("Perplexity / Evaluation")

print()


# ==================================================
# 28. SILVERWING PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Classical ML")
print("      ↓")
print("Deep Learning")
print("      ↓")
print("Embeddings")
print("      ↓")
print("Attention")
print("      ↓")
print("Transformers")
print("      ↓")
print("Language Model")
print("      ↓")
print("Training Objective")
print("      ↓")
print("Evaluation")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 40 COMPLETE ===")

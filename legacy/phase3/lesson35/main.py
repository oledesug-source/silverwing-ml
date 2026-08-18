# Silverwing ML
# Phase 3 - Lesson 35
# Tokenization and Sequence Modeling
#
# Goal:
# Learn how text becomes token sequences and
# how a neural network can learn next-token prediction.

import torch
import torch.nn as nn
import torch.optim as optim


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 35")
print("Tokenization and Sequence Modeling")
print()


# ==================================================
# 1. CREATE A SMALL VOCABULARY
# ==================================================

print("TEST 1: Vocabulary")
print()

vocabulary = {
    "<PAD>": 0,
    "<UNK>": 1,
    "machine": 2,
    "temperature": 3,
    "pressure": 4,
    "is": 5,
    "normal": 6,
    "high": 7,
    "warning": 8,
    "critical": 9,
    "detected": 10,
    "pump": 11,
    "engine": 12,
    "running": 13,
    "the": 14,
    "system": 15
}

id_to_token = {
    token_id: token
    for token, token_id in vocabulary.items()
}


for token, token_id in vocabulary.items():
    print(token, "->", token_id)

print()


# ==================================================
# 2. TOKENIZER
# ==================================================

print("TEST 2: Tokenization")
print()


def tokenize(text):
    """
    Convert text into token IDs.
    Unknown words become <UNK>.
    """

    words = text.lower().split()

    token_ids = []

    for word in words:

        token_id = vocabulary.get(
            word,
            vocabulary["<UNK>"]
        )

        token_ids.append(token_id)

    return token_ids


text = "machine temperature is high"

tokens = tokenize(text)


print("Text:")
print(text)

print()

print("Token IDs:")
print(tokens)

print()


# ==================================================
# 3. DETOKENIZATION
# ==================================================

print("TEST 3: Detokenization")
print()


def detokenize(token_ids):
    """
    Convert token IDs back into tokens.
    """

    tokens = []

    for token_id in token_ids:

        token = id_to_token.get(
            int(token_id),
            "<UNK>"
        )

        tokens.append(token)

    return " ".join(tokens)


reconstructed_text = detokenize(tokens)


print(
    "Reconstructed text:"
)

print(
    reconstructed_text
)

print()


# ==================================================
# 4. CREATE SENTENCE SEQUENCES
# ==================================================

print("TEST 4: Sequence Creation")
print()


sentences = [
    "machine temperature is high",
    "machine temperature is normal",
    "machine pressure is high",
    "pump is running",
    "engine is running",
    "system warning detected",
    "system critical warning",
    "temperature warning detected"
]


for sentence in sentences:

    print(
        sentence,
        "->",
        tokenize(sentence)
    )

print()


# ==================================================
# 5. NEXT-TOKEN TRAINING PAIRS
# ==================================================

print("TEST 5: Next-Token Pairs")
print()


def create_training_pairs(token_ids):
    """
    For a sequence:
        [a, b, c, d]

    create:
        input [a]       -> target b
        input [a, b]    -> target c
        input [a,b,c]   -> target d
    """

    pairs = []

    for index in range(
            1,
            len(token_ids)
    ):

        input_sequence = token_ids[
            :index
        ]

        target_token = token_ids[
            index
        ]

        pairs.append(
            (
                input_sequence,
                target_token
            )
        )

    return pairs


for sentence in sentences:

    token_ids = tokenize(sentence)

    pairs = create_training_pairs(
        token_ids
    )

    print("Sentence:", sentence)

    for inputs, target in pairs:

        print(
            "Input:",
            inputs,
            "-> Target:",
            target,
            "(",
            id_to_token[target],
            ")"
        )

    print()


# ==================================================
# 6. PAD SEQUENCES
# ==================================================

print("TEST 6: Sequence Padding")
print()


PAD_ID = vocabulary["<PAD>"]


def pad_sequence(
        sequence,
        max_length
):
    """
    Pad a sequence with PAD tokens.
    """

    padded = list(sequence)

    while len(padded) < max_length:

        padded.append(PAD_ID)

    return padded[:max_length]


example_sequence = tokenize(
    "machine temperature is high"
)


padded_sequence = pad_sequence(
    example_sequence,
    max_length=6
)


print(
    "Original:",
    example_sequence
)

print()

print(
    "Padded:",
    padded_sequence
)

print()


# ==================================================
# 7. CREATE BATCH
# ==================================================

print("TEST 7: Batch of Sequences")
print()


max_sequence_length = 6

batch = []


for sentence in sentences:

    token_ids = tokenize(sentence)

    padded = pad_sequence(
        token_ids,
        max_sequence_length
    )

    batch.append(padded)


batch_tensor = torch.tensor(
    batch,
    dtype=torch.long
)


print(
    "Batch tensor:"
)

print(
    batch_tensor
)

print()

print(
    "Batch shape:",
    batch_tensor.shape
)

print()


# ==================================================
# 8. EMBEDDING LAYER
# ==================================================

print("TEST 8: Sequence Embeddings")
print()


embedding_dimension = 16

embedding = nn.Embedding(
    len(vocabulary),
    embedding_dimension,
    padding_idx=PAD_ID
)


embedded_sequences = embedding(
    batch_tensor
)


print(
    "Embedding shape:"
)

print(
    embedded_sequences.shape
)

print()

print(
    "Expected shape:"
)

print(
    "(batch, sequence_length, embedding_dimension)"
)

print()


# ==================================================
# 9. SIMPLE SEQUENCE MODEL
# ==================================================

print("TEST 9: Sequence Model")
print()


class NextTokenModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_dim,
            vocab_size
        )


    def forward(self, input_ids):

        embeddings = self.embedding(
            input_ids
        )

        sequence_output, hidden = (
            self.gru(embeddings)
        )

        logits = self.output(
            sequence_output
        )

        return logits


model = NextTokenModel(
    vocab_size=len(vocabulary),
    embedding_dim=16,
    hidden_dim=32
)


print(model)

print()


# ==================================================
# 10. CREATE NEXT-TOKEN DATA
# ==================================================

print("TEST 10: Next-Token Dataset")
print()


input_sequences = []
target_tokens = []


for sentence in sentences:

    token_ids = tokenize(sentence)

    for index in range(
            1,
            len(token_ids)
    ):

        input_sequence = token_ids[
            :index
        ]

        target = token_ids[
            index
        ]

        padded_input = pad_sequence(
            input_sequence,
            max_sequence_length
        )

        input_sequences.append(
            padded_input
        )

        target_tokens.append(
            target
        )


X = torch.tensor(
    input_sequences,
    dtype=torch.long
)

y = torch.tensor(
    target_tokens,
    dtype=torch.long
)


print(
    "Input shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

print()


# ==================================================
# 11. CREATE OPTIMIZER
# ==================================================

print("TEST 11: Training Components")
print()


loss_function = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


print(
    "Loss:",
    type(loss_function).__name__
)

print(
    "Optimizer:",
    type(optimizer).__name__
)

print()


# ==================================================
# 12. TRAIN NEXT-TOKEN MODEL
# ==================================================

print("TEST 12: Train Sequence Model")
print()


epochs = 500


for epoch in range(epochs):

    logits = model(X)

    # We only care about the final real token
    # of each padded input sequence.

    last_logits = logits[:, -1, :]


    loss = loss_function(
        last_logits,
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

        print(
            "Epoch:",
            epoch + 1,
            "| Loss:",
            loss.item()
        )


print()


# ==================================================
# 13. NEXT-TOKEN PREDICTION FUNCTION
# ==================================================

print("TEST 13: Next-Token Prediction")
print()


def predict_next_token(text):

    token_ids = tokenize(text)

    padded = pad_sequence(
        token_ids,
        max_sequence_length
    )

    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long
    )


    with torch.no_grad():

        logits = model(
            input_tensor
        )

        next_token_logits = (
            logits[0, -1]
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=0
        )

        predicted_id = torch.argmax(
            probabilities
        ).item()


    return (
        id_to_token[predicted_id],
        probabilities[predicted_id].item()
    )


# ==================================================
# 14. TEST PREDICTIONS
# ==================================================

test_inputs = [
    "machine temperature is",
    "machine pressure is",
    "pump is",
    "engine is",
    "system warning",
    "temperature warning"
]


for text in test_inputs:

    token, probability = (
        predict_next_token(text)
    )

    print(
        "Input:",
        text
    )

    print(
        "Predicted next token:",
        token
    )

    print(
        "Probability:",
        round(
            probability,
            4
        )
    )

    print()


# ==================================================
# 15. TOP NEXT TOKENS
# ==================================================

print("TEST 14: Top Next Tokens")
print()


def top_next_tokens(
        text,
        number_of_tokens=5
):

    token_ids = tokenize(text)

    padded = pad_sequence(
        token_ids,
        max_sequence_length
    )

    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long
    )


    with torch.no_grad():

        logits = model(
            input_tensor
        )

        next_token_logits = (
            logits[0, -1]
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=0
        )


    top_probabilities, top_ids = (
        torch.topk(
            probabilities,
            k=number_of_tokens
        )
    )


    results = []

    for token_id, probability in zip(
            top_ids,
            top_probabilities
    ):

        results.append(
            (
                id_to_token[
                    token_id.item()
                ],
                probability.item()
            )
        )


    return results


for text in [
    "machine temperature is",
    "machine pressure is",
    "system warning"
]:

    print(
        "Input:",
        text
    )

    for token, probability in (
            top_next_tokens(text)
    ):

        print(
            token,
            "->",
            round(
                probability,
                4
            )
        )

    print()


# ==================================================
# 16. SEQUENCE MODEL CONCEPT
# ==================================================

print("SEQUENCE MODELING")
print()

print(
    "A sequence model processes information "
    "according to token order."
)

print()

print(
    "The model uses earlier tokens to help "
    "predict a later token."
)

print()


# ==================================================
# 17. WHY ORDER MATTERS
# ==================================================

print("WHY TOKEN ORDER MATTERS")
print()

sentence_a = (
    "machine temperature is high"
)

sentence_b = (
    "high temperature is machine"
)


print(
    "Sentence A:"
)

print(
    tokenize(sentence_a)
)

print()

print(
    "Sentence B:"
)

print(
    tokenize(sentence_b)
)

print()

print(
    "The same vocabulary can produce "
    "different sequences."
)

print(
    "A language model must process these "
    "ordering relationships."
)

print()


# ==================================================
# 18. GRU CONNECTION
# ==================================================

print("GRU CONNECTION")
print()

print(
    "This lesson uses a GRU recurrent neural "
    "network to process token sequences."
)

print()

print(
    "RNN-family models process sequences "
    "step by step."
)

print()

print(
    "Transformers later changed sequence modeling "
    "by using attention instead of recurrence as "
    "their central mechanism."
)

print()


# ==================================================
# 19. LANGUAGE MODELING
# ==================================================

print("LANGUAGE MODELING")
print()

print(
    "The model learns to estimate:"
)

print()

print(
    "P(next token | previous tokens)"
)

print()

print(
    "That is a fundamental language-modeling idea."
)

print()


# ==================================================
# 20. CONNECTION TO LLMS
# ==================================================

print("CONNECTION TO LLMS")
print()

print(
    "Large language models also generate text "
    "one token at a time during autoregressive generation."
)

print()

print(
    "Modern LLMs use much larger datasets, "
    "tokenizers, embedding spaces, and Transformer "
    "architectures than this small teaching model."
)

print()


# ==================================================
# 21. COMMUNICATIVE AI PIPELINE
# ==================================================

print("COMMUNICATIVE AI PIPELINE")
print()

print("User text")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Token sequence")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Sequence model")
print(" ↓")
print("Next-token probabilities")
print(" ↓")
print("Generated tokens")
print(" ↓")
print("Response")

print()


# ==================================================
# 22. SILVERWING PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Neural Networks")
print("      ↓")
print("Embeddings")
print("      ↓")
print("Tokenization")
print("      ↓")
print("Sequence Modeling")
print("      ↓")
print("Attention")
print("      ↓")
print("Transformers")
print("      ↓")
print("Language Models")
print("      ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 35 COMPLETE ===")

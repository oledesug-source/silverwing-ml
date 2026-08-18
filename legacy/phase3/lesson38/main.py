# Silverwing ML
# Phase 3 - Lesson 38
# Mini Transformer Language Model
#
# Educational implementation of a tiny
# decoder-style autoregressive language model.


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 38")
print("Mini Transformer Language Model")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42

torch.manual_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)

print()


# ==================================================
# 2. SMALL VOCABULARY
# ==================================================

print("TEST 2: Vocabulary")
print()


vocabulary = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "machine": 4,
    "temperature": 5,
    "pressure": 6,
    "is": 7,
    "normal": 8,
    "high": 9,
    "warning": 10,
    "critical": 11,
    "detected": 12,
    "pump": 13,
    "engine": 14,
    "running": 15,
    "the": 16,
    "system": 17,
    "and": 18,
    "stable": 19,
    "needs": 20,
    "inspection": 21
}


id_to_token = {
    token_id: token
    for token, token_id in vocabulary.items()
}


VOCAB_SIZE = len(vocabulary)

PAD_ID = vocabulary["<PAD>"]
BOS_ID = vocabulary["<BOS>"]
EOS_ID = vocabulary["<EOS>"]


print("Vocabulary size:", VOCAB_SIZE)

print()


# ==================================================
# 3. TOKENIZER
# ==================================================

print("TEST 3: Tokenizer")
print()


def tokenize(text):
    """
    Convert text to token IDs.
    """

    words = text.lower().split()

    token_ids = [
        BOS_ID
    ]


    for word in words:

        token_ids.append(
            vocabulary.get(
                word,
                vocabulary["<UNK>"]
            )
        )


    token_ids.append(
        EOS_ID
    )


    return token_ids


def detokenize(token_ids):
    """
    Convert token IDs back into text.
    """

    words = []


    for token_id in token_ids:

        token = id_to_token.get(
            int(token_id),
            "<UNK>"
        )


        if token in {
            "<PAD>",
            "<BOS>",
            "<EOS>"
        }:

            continue


        words.append(token)


    return " ".join(words)


sample_text = (
    "machine temperature is high"
)


sample_tokens = tokenize(
    sample_text
)


print(
    "Text:",
    sample_text
)

print()

print(
    "Token IDs:",
    sample_tokens
)

print()

print(
    "Decoded:",
    detokenize(
        sample_tokens
    )
)

print()


# ==================================================
# 4. TRAINING CORPUS
# ==================================================

print("TEST 4: Training Corpus")
print()


corpus = [
    "machine temperature is normal",
    "machine temperature is high",
    "machine temperature warning detected",
    "machine pressure is normal",
    "machine pressure is high",
    "pump is running",
    "engine is running",
    "the machine is stable",
    "the machine needs inspection",
    "system warning detected",
    "system critical warning",
    "temperature warning detected",
    "pressure is high",
    "temperature is high",
    "machine is normal",
    "machine is critical"
]


for sentence in corpus:

    print(
        sentence
    )


print()


# ==================================================
# 5. CREATE TRAINING SEQUENCES
# ==================================================

print("TEST 5: Training Sequences")
print()


MAX_SEQUENCE_LENGTH = 8


def prepare_sequence(text):
    """
    Convert text into fixed-length token IDs.
    """

    token_ids = tokenize(text)


    if len(token_ids) > MAX_SEQUENCE_LENGTH:

        token_ids = token_ids[
            :MAX_SEQUENCE_LENGTH
        ]


    while len(token_ids) < MAX_SEQUENCE_LENGTH:

        token_ids.append(
            PAD_ID
        )


    return token_ids


training_sequences = [
    prepare_sequence(sentence)
    for sentence in corpus
]


sequence_tensor = torch.tensor(
    training_sequences,
    dtype=torch.long
)


print(
    "Training tensor shape:",
    sequence_tensor.shape
)

print()

print(
    sequence_tensor
)

print()


# ==================================================
# 6. INPUT / TARGET PAIRS
# ==================================================

print("TEST 6: Language-Model Targets")
print()


# For autoregressive language modeling:
#
# Input:
#     [BOS, machine, temperature, is]
#
# Target:
#     [machine, temperature, is, high]


X = sequence_tensor[:, :-1]

Y = sequence_tensor[:, 1:]


print(
    "Input shape:",
    X.shape
)

print(
    "Target shape:",
    Y.shape
)

print()


# ==================================================
# 7. POSITION EMBEDDINGS
# ==================================================

print("TEST 7: Positional Information")
print()


EMBEDDING_DIMENSION = 32

MAX_POSITION = MAX_SEQUENCE_LENGTH


# ==================================================
# 8. TRANSFORMER CONFIGURATION
# ==================================================

NUMBER_OF_HEADS = 4

FEED_FORWARD_DIMENSION = 64

NUMBER_OF_LAYERS = 2

DROPOUT = 0.1


print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION
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
    "Transformer layers:",
    NUMBER_OF_LAYERS
)

print()


# ==================================================
# 9. CAUSAL SELF-ATTENTION MASK
# ==================================================

print("TEST 8: Causal Mask")
print()


def create_causal_mask(size):

    mask = torch.triu(
        torch.ones(
            size,
            size,
            dtype=torch.bool
        ),
        diagonal=1
    )


    return mask


causal_mask = create_causal_mask(
    X.shape[1]
).to(
    DEVICE
)


print(
    causal_mask
)

print()


# ==================================================
# 10. TRANSFORMER BLOCK
# ==================================================

class DecoderBlock(nn.Module):

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout
    ):

        super().__init__()


        self.attention = (
            nn.MultiheadAttention(
                embed_dim=embedding_dimension,
                num_heads=number_of_heads,
                dropout=dropout,
                batch_first=True
            )
        )


        self.norm1 = nn.LayerNorm(
            embedding_dimension
        )


        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_dimension,
                feed_forward_dimension
            ),

            nn.GELU(),

            nn.Linear(
                feed_forward_dimension,
                embedding_dimension
            ),

            nn.Dropout(
                dropout
            )
        )


        self.norm2 = nn.LayerNorm(
            embedding_dimension
        )


    def forward(
            self,
            x,
            causal_mask
    ):

        attention_output, _ = (
            self.attention(
                x,
                x,
                x,
                attn_mask=causal_mask,
                need_weights=False
            )
        )


        x = self.norm1(
            x
            +
            attention_output
        )


        feed_forward_output = (
            self.feed_forward(x)
        )


        x = self.norm2(
            x
            +
            feed_forward_output
        )


        return x


# ==================================================
# 11. MINI LANGUAGE MODEL
# ==================================================

class MiniLanguageModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            max_sequence_length,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            number_of_layers,
            dropout
    ):

        super().__init__()


        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dimension,
            padding_idx=PAD_ID
        )


        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dimension
        )


        self.blocks = nn.ModuleList([

            DecoderBlock(
                embedding_dimension,
                number_of_heads,
                feed_forward_dimension,
                dropout
            )

            for _ in range(
                number_of_layers
            )
        ])


        self.final_norm = nn.LayerNorm(
            embedding_dimension
        )


        self.output_head = nn.Linear(
            embedding_dimension,
            vocab_size,
            bias=False
        )


    def forward(
            self,
            input_ids
    ):

        batch_size, sequence_length = (
            input_ids.shape
        )


        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )


        token_vectors = (
            self.token_embedding(
                input_ids
            )
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        x = (
                token_vectors
                +
                position_vectors
        )


        causal_mask = create_causal_mask(
            sequence_length
        ).to(
            input_ids.device
        )


        for block in self.blocks:

            x = block(
                x,
                causal_mask
            )


        x = self.final_norm(x)


        logits = self.output_head(
            x
        )


        return logits


# ==================================================
# 12. CREATE MODEL
# ==================================================

print("TEST 9: Create Mini Language Model")
print()


model = MiniLanguageModel(
    vocab_size=VOCAB_SIZE,
    max_sequence_length=MAX_SEQUENCE_LENGTH - 1,
    embedding_dimension=EMBEDDING_DIMENSION,
    number_of_heads=NUMBER_OF_HEADS,
    feed_forward_dimension=FEED_FORWARD_DIMENSION,
    number_of_layers=NUMBER_OF_LAYERS,
    dropout=DROPOUT
).to(
    DEVICE
)


print(
    model
)

print()


# ==================================================
# 13. COUNT PARAMETERS
# ==================================================

print("TEST 10: Model Parameters")
print()


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "Trainable parameters:",
    parameter_count
)

print()


# ==================================================
# 14. PREPARE DATA FOR DEVICE
# ==================================================

X = X.to(
    DEVICE
)

Y = Y.to(
    DEVICE
)


# ==================================================
# 15. LOSS FUNCTION
# ==================================================

loss_function = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


# ==================================================
# 16. OPTIMIZER
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003
)


# ==================================================
# 17. TRAINING
# ==================================================

print("TEST 11: Train Mini Language Model")
print()


model.train()


EPOCHS = 800


for epoch in range(
        EPOCHS
):

    logits = model(
        X
    )


    batch_size, sequence_length, vocabulary_size = (
        logits.shape
    )


    loss = loss_function(
        logits.reshape(
            -1,
            vocabulary_size
        ),
        Y.reshape(
            -1
        )
    )


    optimizer.zero_grad()

    loss.backward()


    # Gradient clipping helps prevent
    # excessively large updates.

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


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
            round(
                loss.item(),
                6
            )
        )


print()


# ==================================================
# 18. FINAL TRAINING LOSS
# ==================================================

print("TEST 12: Final Training Loss")
print()


model.eval()


with torch.no_grad():

    final_logits = model(
        X
    )


    final_loss = loss_function(
        final_logits.reshape(
            -1,
            VOCAB_SIZE
        ),
        Y.reshape(
            -1
        )
    )


print(
    "Final loss:",
    final_loss.item()
)

print()


# ==================================================
# 19. NEXT-TOKEN PREDICTION
# ==================================================

print("TEST 13: Next-Token Prediction")
print()


def predict_next_token(
        text,
        temperature=1.0
):

    token_ids = tokenize(
        text
    )


    token_ids = token_ids[
        -(
                MAX_SEQUENCE_LENGTH - 1
        ):
    ]


    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


    with torch.no_grad():

        logits = model(
            input_tensor
        )


        next_token_logits = (
            logits[
                0,
                -1,
                :
            ]
        )


        scaled_logits = (
                next_token_logits
                /
                temperature
        )


        probabilities = torch.softmax(
            scaled_logits,
            dim=-1
        )


        next_token_id = torch.argmax(
            probabilities
        ).item()


    return (
        next_token_id,
        probabilities[
            next_token_id
        ].item()
    )


test_prompts = [
    "machine temperature is",
    "machine pressure is",
    "pump is",
    "engine is",
    "system warning"
]


for prompt in test_prompts:

    token_id, probability = (
        predict_next_token(
            prompt
        )
    )


    print(
        "Prompt:",
        prompt
    )

    print(
        "Next token:",
        id_to_token[token_id]
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
# 20. TOP-K PREDICTIONS
# ==================================================

print("TEST 14: Top-K Next Tokens")
print()


def top_k_predictions(
        text,
        k=5
):

    token_ids = tokenize(
        text
    )


    token_ids = token_ids[
        -(
                MAX_SEQUENCE_LENGTH - 1
        ):
    ]


    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


    with torch.no_grad():

        logits = model(
            input_tensor
        )


        last_logits = (
            logits[
                0,
                -1
            ]
        )


        probabilities = (
            torch.softmax(
                last_logits,
                dim=-1
            )
        )


        values, indices = torch.topk(
            probabilities,
            k=k
        )


    results = []


    for value, index in zip(
            values,
            indices
    ):

        results.append(
            (
                id_to_token[
                    index.item()
                ],
                value.item()
            )
        )


    return results


for prompt in [
    "machine temperature is",
    "system warning",
    "the machine"
]:

    print(
        "Prompt:",
        prompt
    )


    for token, probability in (
            top_k_predictions(
                prompt
            )
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
# 21. AUTOREGRESSIVE GENERATION
# ==================================================

print("TEST 15: Text Generation")
print()


def generate_text(
        prompt,
        max_new_tokens=8
):

    token_ids = tokenize(
        prompt
    )


    for _ in range(
            max_new_tokens
    ):

        input_ids = token_ids[
            -(
                    MAX_SEQUENCE_LENGTH - 1
            ):
        ]


        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=DEVICE
        )


        with torch.no_grad():

            logits = model(
                input_tensor
            )


            next_token_logits = (
                logits[
                    0,
                    -1
                ]
            )


            next_token_id = torch.argmax(
                next_token_logits
            ).item()


        token_ids.append(
            next_token_id
        )


        if next_token_id == EOS_ID:

            break


    return detokenize(
        token_ids
    )


generation_prompts = [
    "machine temperature",
    "machine pressure",
    "the machine",
    "system warning"
]


for prompt in generation_prompts:

    generated = generate_text(
        prompt,
        max_new_tokens=6
    )


    print(
        "Prompt:",
        prompt
    )

    print(
        "Generated:",
        generated
    )

    print()


# ==================================================
# 22. WHY CAUSAL MASKING MATTERS
# ==================================================

print("TEST 16: Causal Language Modeling")
print()

print(
    "At position t, the model can attend "
    "to earlier positions and the current position."
)

print()

print(
    "It cannot use future tokens when predicting "
    "the next token."
)

print()

print(
    "This makes the model autoregressive."
)

print()


# ==================================================
# 23. TRAINING OBJECTIVE
# ==================================================

print("TEST 17: Training Objective")
print()

print(
    "The model learns to predict the next token "
    "from the previous token sequence."
)

print()

print(
    "Input:"
)

print(
    "[BOS, machine, temperature, is]"
)

print()

print(
    "Target:"
)

print(
    "[machine, temperature, is, high]"
)

print()


# ==================================================
# 24. TRANSFORMER LANGUAGE MODEL FLOW
# ==================================================

print("TRANSFORMER LANGUAGE MODEL FLOW")
print()

print("Text")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Token + Position Embeddings")
print(" ↓")
print("Causal Self-Attention")
print(" ↓")
print("Feed-Forward Network")
print(" ↓")
print("Transformer Block")
print(" ↓")
print("Transformer Block")
print(" ↓")
print("Vocabulary Logits")
print(" ↓")
print("Next-Token Probability")
print(" ↓")
print("Next Token")
print(" ↓")
print("Repeat")

print()


# ==================================================
# 25. WHAT THIS MODEL IS
# ==================================================

print("MODEL STATUS")
print()

print(
    "This is a miniature educational "
    "decoder-style Transformer."
)

print()

print(
    "It demonstrates the architecture and "
    "training objective used by autoregressive "
    "language models."
)

print()


# ==================================================
# 26. WHAT THIS MODEL IS NOT
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This is NOT a production LLM."
)

print()

print(
    "Its vocabulary, dataset, model size, "
    "training time, and capabilities are "
    "extremely small."
)

print()

print(
    "A useful LLM requires much larger training "
    "data, considerably more model capacity, "
    "careful training, evaluation, and substantial "
    "computational resources."
)

print()


# ==================================================
# 27. SILVERWING AI PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Python")
print(" ↓")
print("Mathematics")
print(" ↓")
print("Classical ML")
print(" ↓")
print("Neural Networks")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Tokenization")
print(" ↓")
print("Sequence Modeling")
print(" ↓")
print("Attention")
print(" ↓")
print("Transformer Block")
print(" ↓")
print("Mini Language Model")
print(" ↓")
print("Larger Language Model")
print(" ↓")
print("Communicative AI")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 38 COMPLETE ===")

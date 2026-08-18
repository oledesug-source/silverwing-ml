# Silverwing ML
# Phase 3 - Lesson 34
# Embeddings and Representations
# Foundation for Language Models


import torch
import torch.nn as nn


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 34")
print("Embeddings and Representations")
print()


# ==================================================
# 1. PYTORCH INFORMATION
# ==================================================

print("TEST 1: PyTorch Information")
print()

print(
    "PyTorch version:",
    torch.__version__
)

print()


# ==================================================
# 2. CREATE A SMALL VOCABULARY
# ==================================================

print("TEST 2: Vocabulary")
print()


vocabulary = {
    "machine": 0,
    "pump": 1,
    "engine": 2,
    "temperature": 3,
    "pressure": 4,
    "failure": 5,
    "normal": 6,
    "warning": 7
}


for word, token_id in vocabulary.items():

    print(
        word,
        "->",
        token_id
    )


print()


# ==================================================
# 3. TEXT TO TOKEN IDS
# ==================================================

print("TEST 3: Tokenization")
print()


text = "machine temperature warning"


words = text.split()


token_ids = [
    vocabulary[word]
    for word in words
]


print(
    "Text:",
    text
)

print()

print(
    "Words:",
    words
)

print()

print(
    "Token IDs:",
    token_ids
)

print()


# ==================================================
# 4. TOKEN IDS AS A TENSOR
# ==================================================

print("TEST 4: Token Tensor")
print()


tokens = torch.tensor(
    token_ids,
    dtype=torch.long
)


print(
    "Token tensor:",
    tokens
)

print()

print(
    "Shape:",
    tokens.shape
)

print()


# ==================================================
# 5. CREATE AN EMBEDDING LAYER
# ==================================================

print("TEST 5: Embedding Layer")
print()


vocabulary_size = len(vocabulary)

embedding_dimension = 8


embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)


print(
    "Vocabulary size:",
    vocabulary_size
)

print(
    "Embedding dimension:",
    embedding_dimension
)

print()

print(
    embedding
)

print()


# ==================================================
# 6. GENERATE EMBEDDINGS
# ==================================================

print("TEST 6: Generate Embeddings")
print()


word_embeddings = embedding(
    tokens
)


print(
    "Token embeddings:"
)

print(
    word_embeddings
)

print()

print(
    "Embedding shape:",
    word_embeddings.shape
)

print()


# ==================================================
# 7. EXAMINE ONE WORD
# ==================================================

print("TEST 7: Single Word Embedding")
print()


machine_id = torch.tensor([
    vocabulary["machine"]
])


machine_embedding = embedding(
    machine_id
)


print(
    "Machine token ID:",
    machine_id
)

print()

print(
    "Machine embedding:"
)

print(
    machine_embedding
)

print()


# ==================================================
# 8. EMBEDDING MATRIX
# ==================================================

print("TEST 8: Embedding Matrix")
print()


embedding_matrix = embedding.weight


print(
    "Embedding matrix:"
)

print(
    embedding_matrix
)

print()

print(
    "Embedding matrix shape:",
    embedding_matrix.shape
)

print()


# ==================================================
# 9. COMPARE TWO WORD VECTORS
# ==================================================

print("TEST 9: Compare Word Vectors")
print()


pump_embedding = embedding(
    torch.tensor([
        vocabulary["pump"]
    ])
)


engine_embedding = embedding(
    torch.tensor([
        vocabulary["engine"]
    ])
)


print(
    "Pump embedding:"
)

print(
    pump_embedding
)

print()

print(
    "Engine embedding:"
)

print(
    engine_embedding
)

print()


# ==================================================
# 10. COSINE SIMILARITY
# ==================================================

print("TEST 10: Cosine Similarity")
print()


cosine_similarity = nn.CosineSimilarity(
    dim=1
)


similarity = cosine_similarity(
    pump_embedding,
    engine_embedding
)


print(
    "Pump / Engine similarity:",
    similarity.item()
)

print()


# ==================================================
# 11. IDENTICAL WORD COMPARISON
# ==================================================

print("TEST 11: Identical Embeddings")
print()


pump_embedding_2 = embedding(
    torch.tensor([
        vocabulary["pump"]
    ])
)


similarity_same = cosine_similarity(
    pump_embedding,
    pump_embedding_2
)


print(
    "Pump / Pump similarity:",
    similarity_same.item()
)

print()


# ==================================================
# 12. EMBEDDING A SENTENCE
# ==================================================

print("TEST 12: Sentence Embedding")
print()


sentence = [
    "machine",
    "temperature",
    "warning"
]


sentence_ids = torch.tensor([
    vocabulary[word]
    for word in sentence
])


sentence_embeddings = embedding(
    sentence_ids
)


print(
    "Sentence:",
    sentence
)

print()

print(
    "Token IDs:",
    sentence_ids
)

print()

print(
    "Token embeddings:"
)

print(
    sentence_embeddings
)

print()


# ==================================================
# 13. AVERAGE SENTENCE REPRESENTATION
# ==================================================

print("TEST 13: Average Representation")
print()


sentence_vector = (
    sentence_embeddings.mean(
        dim=0
    )
)


print(
    "Sentence vector:"
)

print(
    sentence_vector
)

print()

print(
    "Sentence vector shape:",
    sentence_vector.shape
)

print()


# ==================================================
# 14. TWO SENTENCES
# ==================================================

print("TEST 14: Compare Sentences")
print()


sentence_a = [
    "machine",
    "temperature",
    "warning"
]


sentence_b = [
    "machine",
    "pressure",
    "warning"
]


ids_a = torch.tensor([
    vocabulary[word]
    for word in sentence_a
])


ids_b = torch.tensor([
    vocabulary[word]
    for word in sentence_b
])


embedding_a = embedding(
    ids_a
).mean(
    dim=0
).unsqueeze(0)


embedding_b = embedding(
    ids_b
).mean(
    dim=0
).unsqueeze(0)


sentence_similarity = (
    cosine_similarity(
        embedding_a,
        embedding_b
    )
)


print(
    "Sentence A:",
    sentence_a
)

print()

print(
    "Sentence B:",
    sentence_b
)

print()

print(
    "Similarity:",
    sentence_similarity.item()
)

print()


# ==================================================
# 15. RANDOM INITIAL EMBEDDINGS
# ==================================================

print("TEST 15: Initial Embedding Concept")
print()


print(
    "Embedding values initially start "
    "as learned numerical parameters."
)

print()

print(
    "During training, the model adjusts "
    "these values based on the learning objective."
)

print()


# ==================================================
# 16. EMBEDDINGS ARE LEARNED
# ==================================================

print("TEST 16: Trainable Embeddings")
print()


print(
    "Embedding parameters:"
)

print(
    embedding.weight
)

print()

print(
    "Gradient tracking:",
    embedding.weight.requires_grad
)

print()


# ==================================================
# 17. SIMPLE EMBEDDING TRAINING
# ==================================================

print("TEST 17: Train an Embedding")
print()


embedding_model = nn.Embedding(
    4,
    3
)


optimizer = torch.optim.SGD(
    embedding_model.parameters(),
    lr=0.1
)


input_ids = torch.tensor([
    0,
    1,
    2,
    3
])


target = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
])


loss_function = nn.MSELoss()


for epoch in range(300):

    vectors = embedding_model(
        input_ids
    )


    loss = loss_function(
        vectors,
        target
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
# 18. VIEW LEARNED EMBEDDINGS
# ==================================================

print("TEST 18: Learned Embeddings")
print()


with torch.no_grad():

    learned_embeddings = (
        embedding_model.weight
    )


print(
    learned_embeddings
)

print()


# ==================================================
# 19. SEMANTIC REPRESENTATION CONCEPT
# ==================================================

print("SEMANTIC REPRESENTATION")
print()

print(
    "An embedding represents an item as a vector."
)

print()

print(
    "During language-model training, these "
    "representations can encode useful relationships."
)

print()

print(
    "Items used in similar contexts can develop "
    "related representations."
)

print()


# ==================================================
# 20. TOKENIZATION TO EMBEDDING PIPELINE
# ==================================================

print("TOKENIZATION → EMBEDDING")
print()

print("Text")
print(" ↓")
print("Tokens")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Embedding lookup")
print(" ↓")
print("Vectors")

print()


# ==================================================
# 21. CONNECTION TO TRANSFORMERS
# ==================================================

print("CONNECTION TO TRANSFORMERS")
print()

print(
    "Transformers receive numerical representations "
    "of token sequences."
)

print()

print(
    "Embeddings provide an important starting "
    "representation before attention layers process "
    "the sequence."
)

print()


# ==================================================
# 22. CONNECTION TO LLMS
# ==================================================

print("CONNECTION TO LLMS")
print()

print(
    "A language model does not directly process "
    "raw English words as strings."
)

print()

print(
    "Text is converted into tokens and numerical "
    "representations that neural networks can process."
)

print()

print(
    "Large language models use much larger vocabularies, "
    "embedding dimensions, datasets, and architectures "
    "than this teaching example."
)

print()


# ==================================================
# 23. COMMUNICATIVE AI PIPELINE
# ==================================================

print("FUTURE COMMUNICATIVE AI PIPELINE")
print()

print("User text")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Token IDs")
print(" ↓")
print("Embeddings")
print(" ↓")
print("Transformer")
print(" ↓")
print("Language-model output")
print(" ↓")
print("Response")
print()


# ==================================================
# 24. SILVERWING PROGRESSION
# ==================================================

print("SILVERWING AI PROGRESSION")
print()

print("Classical ML")
print("      ↓")
print("Neural Networks")
print("      ↓")
print("Embeddings")
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

print("=== LESSON 34 COMPLETE ===")

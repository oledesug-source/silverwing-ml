# Silverwing ML
# Phase 5 - Lesson 72R
# Silverwing Own Foundation Model
# Own Training Dataset and Data Pipeline
#
# Raw Corpus
# -> Normalization
# -> Quality Filtering
# -> Deduplication
# -> Silverwing BPE Tokenizer
# -> Token IDs
# -> Sequence Packing
# -> Train / Validation Split
# -> PyTorch Dataset
# -> DataLoader
#
# No external pretrained language model is used.

import hashlib
import json
import re

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LESSON_66_DIR = BASE_DIR.parent / "lesson66R"
LESSON_71_DIR = BASE_DIR.parent / "lesson71R"

VOCABULARY_FILE = (
        LESSON_66_DIR / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        LESSON_66_DIR / "silverwing_bpe_merges.json"
)

MODEL_CONFIG_FILE = (
        LESSON_71_DIR / "silverwing_decoder_config.json"
)

DATASET_CONFIG_FILE = (
        BASE_DIR / "silverwing_dataset_config.json"
)

CORPUS_FILE = (
        BASE_DIR / "silverwing_training_corpus.jsonl"
)

TRAIN_FILE = (
        BASE_DIR / "silverwing_train.jsonl"
)

VALIDATION_FILE = (
        BASE_DIR / "silverwing_validation.jsonl"
)

SEED = 42
SEQUENCE_LENGTH = 64
BATCH_SIZE = 2
VALIDATION_RATIO = 0.20
MIN_QUALITY_SCORE = 0.45

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 72R")
print("Silverwing Own Foundation Model")
print("Own Training Dataset and Data Pipeline")
print()


# ==================================================
# 2. CONFIGURATION TEST
# ==================================================

print("TEST 1: Configuration")
print()

print("Vocabulary:", VOCABULARY_FILE)
print("BPE merges:", MERGES_FILE)
print("Model config:", MODEL_CONFIG_FILE)
print("Sequence length:", SEQUENCE_LENGTH)
print("Batch size:", BATCH_SIZE)
print("Validation ratio:", VALIDATION_RATIO)
print("Device:", DEVICE)

print()


# ==================================================
# 3. VERIFY ARTIFACTS
# ==================================================

print("TEST 2: Verify Foundation Artifacts")
print()

required_files = [
    VOCABULARY_FILE,
    MERGES_FILE,
    MODEL_CONFIG_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{file_path}\n"
            "Run the corresponding previous lesson first."
        )

    print("FOUND:", file_path)

print()


# ==================================================
# 4. LOAD VOCABULARY
# ==================================================

print("TEST 3: Load Silverwing Vocabulary")
print()

with open(
        VOCABULARY_FILE,
        "r",
        encoding="utf-8"
) as file:
    vocabulary_data = json.load(file)

TOKEN_TO_ID: Dict[str, int] = {
    token: int(token_id)
    for token, token_id in vocabulary_data["token_to_id"].items()
}

ID_TO_TOKEN: Dict[int, str] = {
    token_id: token
    for token, token_id in TOKEN_TO_ID.items()
}

SPECIAL_TOKENS = vocabulary_data.get(
    "special_tokens",
    [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
        "<MASK>",
    ],
)

PAD_ID = TOKEN_TO_ID["<PAD>"]
UNK_ID = TOKEN_TO_ID["<UNK>"]
BOS_ID = TOKEN_TO_ID["<BOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]

VOCABULARY_SIZE = len(TOKEN_TO_ID)

print("Vocabulary size:", VOCABULARY_SIZE)
print("PAD ID:", PAD_ID)
print("UNK ID:", UNK_ID)
print("BOS ID:", BOS_ID)
print("EOS ID:", EOS_ID)

print()


# ==================================================
# 5. LOAD MODEL CONFIGURATION
# ==================================================

print("TEST 4: Load Model Configuration")
print()

with open(
        MODEL_CONFIG_FILE,
        "r",
        encoding="utf-8"
) as file:
    model_config = json.load(file)

configured_vocab_size = model_config.get(
    "vocabulary_size"
)

configured_max_sequence_length = model_config.get(
    "maximum_sequence_length",
    256
)

if configured_vocab_size is not None:
    if configured_vocab_size != VOCABULARY_SIZE:
        print("WARNING: Vocabulary size mismatch.")
        print("Decoder config:", configured_vocab_size)
        print("Tokenizer:", VOCABULARY_SIZE)

if SEQUENCE_LENGTH > configured_max_sequence_length:
    raise ValueError(
        "SEQUENCE_LENGTH exceeds the model maximum "
        "sequence length."
    )

print(
    "Configured vocabulary size:",
    configured_vocab_size
)

print(
    "Maximum sequence length:",
    configured_max_sequence_length
)

print()


# ==================================================
# 6. LOAD BPE MERGES
# ==================================================

print("TEST 5: Load Silverwing BPE Merges")
print()

with open(
        MERGES_FILE,
        "r",
        encoding="utf-8"
) as file:
    merge_data = json.load(file)

MERGE_RANKS: Dict[Tuple[str, str], int] = {}

for item in merge_data:
    pair = item.get("pair")
    rank = item.get("rank")

    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(
            f"Invalid BPE merge pair: {pair}"
        )

    MERGE_RANKS[(pair[0], pair[1])] = int(rank)

print(
    "Loaded merge operations:",
    len(MERGE_RANKS)
)

print()


# ==================================================
# 7. TEXT NORMALIZATION
# ==================================================

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(
            "Document text must be a string."
        )

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==================================================
# 8. WORD SPLITTING
# ==================================================

def split_words(text: str) -> List[str]:
    normalized = normalize_text(text).lower()

    return re.findall(
        r"\w+|[^\w\s]",
        normalized,
        flags=re.UNICODE,
    )


# ==================================================
# 9. BPE SYMBOLS
# ==================================================

BPE_END = "</w>"


def word_to_symbols(
        word: str
) -> Tuple[str, ...]:
    if not word:
        return tuple()

    symbols = list(word)
    symbols[-1] = symbols[-1] + BPE_END

    return tuple(symbols)


# ==================================================
# 10. APPLY BPE MERGE
# ==================================================

def merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str]
) -> Tuple[str, ...]:

    merged = []
    index = 0

    while index < len(symbols):
        if (
                index < len(symbols) - 1
                and (
                symbols[index],
                symbols[index + 1]
        ) == pair
        ):
            merged.append(
                symbols[index] + symbols[index + 1]
            )
            index += 2
        else:
            merged.append(symbols[index])
            index += 1

    return tuple(merged)


# ==================================================
# 11. SILVERWING BPE TOKENIZER
# ==================================================

def tokenize_word(
        word: str
) -> List[str]:

    symbols = word_to_symbols(word)

    if not symbols:
        return []

    while True:
        candidates = []

        for index in range(len(symbols) - 1):
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


# ==================================================
# 12. ENCODE / DECODE
# ==================================================

def encode_text(
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
) -> List[int]:

    tokens = tokenize_text(text)
    ids = []

    if add_bos:
        ids.append(BOS_ID)

    for token in tokens:
        ids.append(
            TOKEN_TO_ID.get(
                token,
                UNK_ID
            )
        )

    if add_eos:
        ids.append(EOS_ID)

    return ids


def decode_ids(
        ids: List[int]
) -> str:

    pieces = []

    for token_id in ids:
        token = ID_TO_TOKEN.get(
            int(token_id),
            "<UNK>"
        )

        if token in SPECIAL_TOKENS:
            continue

        if token.endswith(BPE_END):
            token = token[:-len(BPE_END)]

        pieces.append(token)

    text = ""

    for piece in pieces:
        if not text:
            text = piece
        elif re.fullmatch(
                r"[^\w\s]",
                piece,
                flags=re.UNICODE
        ):
            text += piece
        else:
            text += " " + piece

    return text


# ==================================================
# 13. TOKENIZER TEST
# ==================================================

print("TEST 6: Silverwing Tokenizer")
print()

sample_text = "Silverwing learns from data."

sample_tokens = tokenize_text(
    sample_text
)

sample_ids = encode_text(
    sample_text
)

print("Text:", sample_text)
print("Tokens:", sample_tokens)
print("Token IDs:", sample_ids)
print("Decoded:", decode_ids(sample_ids))

print()


# ==================================================
# 14. RAW TRAINING CORPUS
# ==================================================

print("TEST 7: Build Raw Corpus")
print()

raw_documents = [
    {
        "id": "doc_001",
        "domain": "artificial_intelligence",
        "text": (
            "Artificial intelligence systems learn useful "
            "representations from data. A language model "
            "learns statistical relationships between "
            "tokens and uses those relationships to predict "
            "future tokens."
        ),
    },
    {
        "id": "doc_002",
        "domain": "machine_learning",
        "text": (
            "Machine learning uses data to construct models "
            "that discover patterns. Training minimizes an "
            "objective function and optimization changes "
            "model parameters according to observed error."
        ),
    },
    {
        "id": "doc_003",
        "domain": "deep_learning",
        "text": (
            "Deep learning uses layered neural networks. "
            "Each layer transforms representations and "
            "produces features that are useful for later "
            "computations."
        ),
    },
    {
        "id": "doc_004",
        "domain": "transformers",
        "text": (
            "Transformer models use attention to route "
            "information between sequence positions. "
            "Queries, keys, and values determine how "
            "representations interact."
        ),
    },
    {
        "id": "doc_005",
        "domain": "language_models",
        "text": (
            "Autoregressive language modeling predicts the "
            "next token from previous tokens. The training "
            "target is shifted by one position relative to "
            "the model input."
        ),
    },
    {
        "id": "doc_006",
        "domain": "silverwing",
        "text": (
            "Silverwing is designed as a personal artificial "
            "intelligence architecture containing memory, "
            "reasoning, tools, learning systems, evaluation, "
            "and controlled adaptation."
        ),
    },
    {
        "id": "doc_007",
        "domain": "memory",
        "text": (
            "Persistent memory allows a system to retain "
            "useful information beyond one execution. "
            "Semantic retrieval helps select relevant "
            "information for future tasks."
        ),
    },
    {
        "id": "doc_008",
        "domain": "agents",
        "text": (
            "An intelligent agent receives goals, chooses "
            "actions, observes results, evaluates outcomes, "
            "and continues until the task reaches a useful "
            "completion state."
        ),
    },
    {
        "id": "doc_009",
        "domain": "systems",
        "text": (
            "Reliable artificial intelligence requires "
            "monitoring, error handling, validation, "
            "observability, recovery, and controlled "
            "deployment of new versions."
        ),
    },
    {
        "id": "doc_010",
        "domain": "science",
        "text": (
            "Scientific learning requires observations, "
            "hypotheses, measurements, experiments, "
            "analysis, replication, and revision when "
            "evidence contradicts previous assumptions."
        ),
    },
    {
        "id": "doc_011",
        "domain": "biology",
        "text": (
            "Biological systems continuously interact with "
            "their environments. Adaptation changes system "
            "behavior in response to internal and external "
            "conditions."
        ),
    },
    {
        "id": "doc_012",
        "domain": "physics",
        "text": (
            "Physical systems are described using quantities, "
            "relationships, conservation laws, and models "
            "that explain measurable behavior."
        ),
    },
    {
        "id": "doc_013",
        "domain": "continual_learning",
        "text": (
            "Continual learning requires mechanisms for "
            "receiving new information, evaluating it, "
            "updating representations, and preventing "
            "damaging changes to previously learned skills."
        ),
    },
    {
        "id": "doc_014",
        "domain": "self_improvement",
        "text": (
            "Controlled self improvement should produce a "
            "candidate change, run an experiment, measure "
            "the result, compare it against the previous "
            "version, and promote the candidate only when "
            "evaluation criteria are satisfied."
        ),
    },
    {
        "id": "doc_015",
        "domain": "engineering",
        "text": (
            "Engineering systems should be modular, "
            "testable, observable, reproducible, and "
            "recoverable. Components should communicate "
            "through well-defined interfaces."
        ),
    },
    {
        "id": "doc_016",
        "domain": "engineering",
        "text": (
            "Engineering systems should be modular, "
            "testable, observable, reproducible, and "
            "recoverable. Components should communicate "
            "through well-defined interfaces."
        ),
    },
]

print(
    "Raw documents:",
    len(raw_documents)
)

print()


# ==================================================
# 15. CLEANING
# ==================================================

def clean_document(
        document: Dict
) -> Dict:

    return {
        "id": document["id"],
        "domain": document.get(
            "domain",
            "general"
        ),
        "text": normalize_text(
            document["text"]
        ),
    }


cleaned_documents = [
    clean_document(document)
    for document in raw_documents
]

print("TEST 8: Cleaning")
print()

print(
    "Documents after cleaning:",
    len(cleaned_documents)
)

print()


# ==================================================
# 16. QUALITY SCORING
# ==================================================

def quality_score(
        text: str
) -> float:

    words = re.findall(
        r"\b\w+\b",
        text
    )

    if not words:
        return 0.0

    score = 0.0
    word_count = len(words)

    if word_count >= 12:
        score += 0.35
    elif word_count >= 8:
        score += 0.25
    elif word_count >= 4:
        score += 0.10

    if len(text) >= 100:
        score += 0.30
    elif len(text) >= 60:
        score += 0.20
    elif len(text) >= 30:
        score += 0.10

    if "." in text:
        score += 0.10

    unique_words = len(
        set(
            word.lower()
            for word in words
        )
    )

    lexical_diversity = (
            unique_words / word_count
    )

    score += 0.25 * lexical_diversity

    return min(
        score,
        1.0
    )


def quality_filter(
        documents: List[Dict],
        minimum_score: float
):

    accepted = []
    rejected = []

    for document in documents:

        score = quality_score(
            document["text"]
        )

        enriched = dict(
            document
        )

        enriched[
            "quality_score"
        ] = score

        if score >= minimum_score:
            accepted.append(
                enriched
            )
        else:
            rejected.append(
                enriched
            )

    return accepted, rejected


accepted_documents, rejected_documents = (
    quality_filter(
        cleaned_documents,
        MIN_QUALITY_SCORE
    )
)

print("TEST 9: Quality Filtering")
print()

print(
    "Accepted:",
    len(accepted_documents)
)

print(
    "Rejected:",
    len(rejected_documents)
)

print()


# ==================================================
# 17. DEDUPLICATION
# ==================================================

def content_hash(
        text: str
) -> str:

    normalized = normalize_text(
        text
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def deduplicate_documents(
        documents: List[Dict]
):

    seen = set()
    unique_documents = []
    duplicate_documents = []

    for document in documents:

        digest = content_hash(
            document["text"]
        )

        enriched = dict(
            document
        )

        enriched[
            "content_hash"
        ] = digest

        if digest in seen:
            duplicate_documents.append(
                enriched
            )
        else:
            seen.add(digest)
            unique_documents.append(
                enriched
            )

    return (
        unique_documents,
        duplicate_documents
    )


unique_documents, duplicate_documents = (
    deduplicate_documents(
        accepted_documents
    )
)

print("TEST 10: Deduplication")
print()

print(
    "Unique documents:",
    len(unique_documents)
)

print(
    "Duplicates removed:",
    len(duplicate_documents)
)

print()


# ==================================================
# 18. DOMAIN DISTRIBUTION
# ==================================================

print("TEST 11: Domain Distribution")
print()

domain_counts = Counter(
    document["domain"]
    for document in unique_documents
)

for domain, count in sorted(
        domain_counts.items()
):
    print(
        domain,
        "->",
        count
    )

print()


# ==================================================
# 19. TOKENIZE DOCUMENTS
# ==================================================

print("TEST 12: Tokenize Documents")
print()

tokenized_documents = []

for document in unique_documents:

    token_ids = encode_text(
        document["text"]
    )

    tokenized_documents.append(
        {
            "id": document["id"],
            "domain": document["domain"],
            "text": document["text"],
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "quality_score": document[
                "quality_score"
            ],
            "content_hash": document[
                "content_hash"
            ],
        }
    )

total_tokens = sum(
    document["token_count"]
    for document in tokenized_documents
)

print(
    "Tokenized documents:",
    len(tokenized_documents)
)

print(
    "Total tokens:",
    total_tokens
)

print()


# ==================================================
# 20. TOKEN STATISTICS
# ==================================================

print("TEST 13: Token Statistics")
print()

token_lengths = [
    document["token_count"]
    for document in tokenized_documents
]

if token_lengths:
    print(
        "Minimum:",
        min(token_lengths)
    )

    print(
        "Maximum:",
        max(token_lengths)
    )

    print(
        "Average:",
        round(
            sum(token_lengths)
            / len(token_lengths),
            2
        )
    )

print()


# ==================================================
# 21. GLOBAL TOKEN FREQUENCY
# ==================================================

print("TEST 14: Token Frequency")
print()

token_frequency = Counter()

for document in tokenized_documents:
    token_frequency.update(
        document["token_ids"]
    )

print(
    "Unique token IDs observed:",
    len(token_frequency)
)

print()

for token_id, count in token_frequency.most_common(15):
    print(
        token_id,
        "->",
        count,
        "->",
        ID_TO_TOKEN.get(
            token_id,
            "<UNK>"
        )
    )

print()


# ==================================================
# 22. BUILD TOKEN STREAM
# ==================================================

def build_token_stream(
        documents: List[Dict]
) -> List[int]:

    stream = []

    for document in documents:
        stream.extend(
            document["token_ids"]
        )

    return stream


token_stream = build_token_stream(
    tokenized_documents
)

print("TEST 15: Token Stream")
print()

print(
    "Token stream length:",
    len(token_stream)
)

print()


# ==================================================
# 23. SEQUENCE PACKING
# ==================================================

def pack_sequences(
        tokens: List[int],
        sequence_length: int
) -> List[Dict]:

    samples = []

    chunk_length = (
            sequence_length + 1
    )

    if len(tokens) < chunk_length:
        return samples

    start = 0

    while start + chunk_length <= len(tokens):

        chunk = tokens[
            start:
            start + chunk_length
        ]

        samples.append(
            {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
            }
        )

        start += sequence_length

    return samples


packed_sequences = pack_sequences(
    token_stream,
    SEQUENCE_LENGTH
)

print("TEST 16: Sequence Packing")
print()

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Training sequences:",
    len(packed_sequences)
)

print()


# ==================================================
# 24. TRAINING EXAMPLE
# ==================================================

print("TEST 17: Training Example")
print()

if packed_sequences:

    example = packed_sequences[0]

    print(
        "Input IDs:",
        example["input_ids"][:20]
    )

    print(
        "Label IDs:",
        example["labels"][:20]
    )

    print()

    print(
        "Decoded input:"
    )

    print(
        decode_ids(
            example["input_ids"]
        )[:300]
    )

    print()

    print(
        "Decoded target:"
    )

    print(
        decode_ids(
            example["labels"]
        )[:300]
    )

else:

    print(
        "No complete training sequence was created."
    )

print()


# ==================================================
# 25. TRAIN / VALIDATION SPLIT
# ==================================================

def split_dataset(
        samples: List[Dict],
        validation_ratio: float
):

    if not samples:
        return [], []

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if len(samples) == 1:
        return samples, []

    split_index = int(
        len(samples)
        * (1.0 - validation_ratio)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(samples) - 1
        )
    )

    return (
        samples[:split_index],
        samples[split_index:]
    )


train_samples, validation_samples = (
    split_dataset(
        packed_sequences,
        VALIDATION_RATIO
    )
)

print("TEST 18: Train/Validation Split")
print()

print(
    "Training sequences:",
    len(train_samples)
)

print(
    "Validation sequences:",
    len(validation_samples)
)

print()


# ==================================================
# 26. PYTORCH DATASET
# ==================================================

class SilverwingLanguageDataset(
    Dataset
):

    def __init__(
            self,
            samples: List[Dict]
    ):
        self.samples = samples

    def __len__(
            self
    ):
        return len(
            self.samples
        )

    def __getitem__(
            self,
            index: int
    ):
        sample = self.samples[index]

        return {
            "input_ids": torch.tensor(
                sample["input_ids"],
                dtype=torch.long
            ),
            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.long
            ),
        }


train_dataset = SilverwingLanguageDataset(
    train_samples
)

validation_dataset = SilverwingLanguageDataset(
    validation_samples
)

print("TEST 19: PyTorch Dataset")
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
# 27. DATALOADERS
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

print("TEST 20: DataLoaders")
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
# 28. BATCH INSPECTION
# ==================================================

print("TEST 21: Batch Inspection")
print()

if len(train_loader) > 0:

    batch = next(
        iter(train_loader)
    )

    input_batch = batch[
        "input_ids"
    ]

    label_batch = batch[
        "labels"
    ]

    print(
        "Input batch shape:",
        tuple(input_batch.shape)
    )

    print(
        "Label batch shape:",
        tuple(label_batch.shape)
    )

    print(
        "Input dtype:",
        input_batch.dtype
    )

    print(
        "Label dtype:",
        label_batch.dtype
    )

else:

    print(
        "No training batches available."
    )

print()


# ==================================================
# 29. INPUT / LABEL ALIGNMENT
# ==================================================

print("TEST 22: Input/Label Alignment")
print()

if len(train_loader) > 0:

    batch = next(
        iter(train_loader)
    )

    inputs = batch[
        "input_ids"
    ]

    labels = batch[
        "labels"
    ]

    if inputs.shape != labels.shape:
        raise RuntimeError(
            "Input and label shapes are not identical."
        )

    print(
        "Shape equality:",
        inputs.shape == labels.shape
    )

    print(
        "Input first tokens:",
        inputs[0, :10].tolist()
    )

    print(
        "Target first tokens:",
        labels[0, :10].tolist()
    )

print()


# ==================================================
# 30. TOKEN RANGE VALIDATION
# ==================================================

print("TEST 23: Token ID Range Validation")
print()

invalid_ids = []

for sample in packed_sequences:

    for token_id in sample["input_ids"]:
        if not 0 <= token_id < VOCABULARY_SIZE:
            invalid_ids.append(token_id)

    for token_id in sample["labels"]:
        if not 0 <= token_id < VOCABULARY_SIZE:
            invalid_ids.append(token_id)

if invalid_ids:
    raise RuntimeError(
        f"Invalid token IDs detected: "
        f"{invalid_ids[:20]}"
    )

print(
    "All token IDs are within vocabulary range."
)

print()


# ==================================================
# 31. DATASET HASH
# ==================================================

def dataset_hash(
        samples: List[Dict]
) -> str:

    serialized = json.dumps(
        samples,
        sort_keys=True
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


train_hash = dataset_hash(
    train_samples
)

validation_hash = dataset_hash(
    validation_samples
)

print("TEST 24: Dataset Reproducibility")
print()

print(
    "Training hash:",
    train_hash
)

print(
    "Validation hash:",
    validation_hash
)

print()


# ==================================================
# 32. QUALITY REPORT
# ==================================================

print("TEST 25: Dataset Quality Report")
print()

quality_values = [
    document["quality_score"]
    for document in unique_documents
]

average_quality = (
    sum(quality_values)
    / len(quality_values)
    if quality_values
    else 0.0
)

quality_report = {
    "raw_documents": len(raw_documents),
    "clean_documents": len(cleaned_documents),
    "quality_accepted": len(accepted_documents),
    "quality_rejected": len(rejected_documents),
    "unique_documents": len(unique_documents),
    "duplicates_removed": len(duplicate_documents),
    "total_tokens": total_tokens,
    "sequence_length": SEQUENCE_LENGTH,
    "packed_sequences": len(packed_sequences),
    "training_sequences": len(train_samples),
    "validation_sequences": len(validation_samples),
    "average_quality": round(
        average_quality,
        4
    ),
    "vocabulary_size": VOCABULARY_SIZE,
}

print(
    json.dumps(
        quality_report,
        indent=4
    )
)

print()


# ==================================================
# 33. SAVE CORPUS
# ==================================================

print("TEST 26: Save Clean Corpus")
print()

with open(
        CORPUS_FILE,
        "w",
        encoding="utf-8"
) as file:

    for document in unique_documents:

        file.write(
            json.dumps(
                document,
                ensure_ascii=False
            )
            + "\n"
        )

print(
    "Saved:",
    CORPUS_FILE
)

print()


# ==================================================
# 34. SAVE TRAINING DATA
# ==================================================

print("TEST 27: Save Training Dataset")
print()

with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
) as file:

    for sample in train_samples:

        file.write(
            json.dumps(sample)
            + "\n"
        )

print(
    "Saved:",
    TRAIN_FILE
)

print()


# ==================================================
# 35. SAVE VALIDATION DATA
# ==================================================

print("TEST 28: Save Validation Dataset")
print()

with open(
        VALIDATION_FILE,
        "w",
        encoding="utf-8"
) as file:

    for sample in validation_samples:

        file.write(
            json.dumps(sample)
            + "\n"
        )

print(
    "Saved:",
    VALIDATION_FILE
)

print()


# ==================================================
# 36. SAVE DATASET CONFIGURATION
# ==================================================

print("TEST 29: Save Dataset Configuration")
print()

dataset_config = {
    "dataset": "Silverwing-Corpus-v1",
    "tokenizer": "Silverwing-BPE-v1",
    "vocabulary_size": VOCABULARY_SIZE,
    "sequence_length": SEQUENCE_LENGTH,
    "batch_size": BATCH_SIZE,
    "validation_ratio": VALIDATION_RATIO,
    "minimum_quality_score": MIN_QUALITY_SCORE,
    "raw_documents": len(raw_documents),
    "clean_documents": len(cleaned_documents),
    "quality_accepted": len(accepted_documents),
    "quality_rejected": len(rejected_documents),
    "unique_documents": len(unique_documents),
    "duplicates_removed": len(duplicate_documents),
    "total_tokens": total_tokens,
    "training_sequences": len(train_samples),
    "validation_sequences": len(validation_samples),
    "training_hash": train_hash,
    "validation_hash": validation_hash,
    "seed": SEED,
    "device": str(DEVICE),
}

with open(
        DATASET_CONFIG_FILE,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        dataset_config,
        file,
        indent=4
    )

print(
    "Saved:",
    DATASET_CONFIG_FILE
)

print()


# ==================================================
# 37. BATCH MEMORY INSPECTION
# ==================================================

print("TEST 30: Batch Memory Footprint")
print()

if len(train_loader) > 0:

    batch = next(
        iter(train_loader)
    )

    input_elements = (
        batch["input_ids"].numel()
    )

    label_elements = (
        batch["labels"].numel()
    )

    input_bytes = (
            input_elements
            * batch["input_ids"].element_size()
    )

    label_bytes = (
            label_elements
            * batch["labels"].element_size()
    )

    print(
        "Input elements:",
        input_elements
    )

    print(
        "Label elements:",
        label_elements
    )

    print(
        "Input bytes:",
        input_bytes
    )

    print(
        "Label bytes:",
        label_bytes
    )

    print(
        "Total bytes:",
        input_bytes + label_bytes
    )

else:

    print(
        "No batch available."
    )

print()


# ==================================================
# 38. DATASET SHUFFLING
# ==================================================

print("TEST 31: Training Shuffle")
print()

if len(train_dataset) >= 2:

    loader_a = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    loader_b = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    order_a = []

    for batch in loader_a:
        order_a.extend(
            batch["input_ids"][:, 0].tolist()
        )

    order_b = []

    for batch in loader_b:
        order_b.extend(
            batch["input_ids"][:, 0].tolist()
        )

    print(
        "First shuffled order:",
        order_a[:10]
    )

    print(
        "Second shuffled order:",
        order_b[:10]
    )

else:

    print(
        "Not enough samples for a shuffle test."
    )

print()


# ==================================================
# 39. PIPELINE SUMMARY
# ==================================================

print("TEST 32: Pipeline Summary")
print()

pipeline_steps = [
    "Raw corpus created",
    "Text normalized",
    "Quality scored",
    "Quality filtering completed",
    "Duplicates removed",
    "Domain distribution measured",
    "Silverwing BPE vocabulary loaded",
    "BPE merge rules loaded",
    "Documents tokenized",
    "Token stream created",
    "Fixed-length training sequences created",
    "Input/target pairs created",
    "Train/validation split created",
    "PyTorch datasets created",
    "DataLoaders created",
    "Dataset artifacts persisted",
]

for step in pipeline_steps:
    print(
        "PASS:",
        step
    )

print()


# ==================================================
# 40. SILVERWING DATA PIPELINE
# ==================================================

print("SILVERWING DATA PIPELINE")
print()

print("Raw Information")
print("      ↓")
print("Normalization")
print("      ↓")
print("Quality Evaluation")
print("      ↓")
print("Deduplication")
print("      ↓")
print("Domain Classification")
print("      ↓")
print("Own BPE Tokenizer")
print("      ↓")
print("Token IDs")
print("      ↓")
print("Sequence Packing")
print("      ↓")
print("Input / Target Pairs")
print("      ↓")
print("Train / Validation")
print("      ↓")
print("DataLoader")
print("      ↓")
print("Silverwing Decoder")

print()


# ==================================================
# 41. CAUSAL TRAINING OBJECTIVE
# ==================================================

print("CAUSAL TRAINING OBJECTIVE")
print()

print(
    "Input:  [t0, t1, t2, ... tn-1]"
)

print(
    "Target: [t1, t2, t3, ... tn]"
)

print()

print(
    "Silverwing learns:"
)

print(
    "P(next_token | previous_tokens)"
)

print()


# ==================================================
# 42. DATA PROVENANCE
# ==================================================

print("DATA PROVENANCE")
print()

print(
    "Future training records should retain:"
)

print(
    "- source"
)

print(
    "- license / usage rights"
)

print(
    "- acquisition date"
)

print(
    "- processing version"
)

print(
    "- quality score"
)

print(
    "- content hash"
)

print()


# ==================================================
# 43. FUTURE DATA SOURCES
# ==================================================

print("FUTURE SILVERWING DATA SOURCES")
print()

future_sources = [
    "technical documentation",
    "engineering",
    "mathematics",
    "physics",
    "computer science",
    "programming",
    "machine learning",
    "scientific material with appropriate rights",
    "public-domain material",
    "properly licensed datasets",
    "synthetic training examples",
    "validated Silverwing experiences",
]

for source in future_sources:
    print(
        "-",
        source
    )

print()


# ==================================================
# 44. AUTONOMOUS GROWTH DATA LOOP
# ==================================================

print("FUTURE AUTONOMOUS GROWTH DATA LOOP")
print()

print("New Experience")
print("      ↓")
print("Data Capture")
print("      ↓")
print("Provenance")
print("      ↓")
print("Quality Evaluation")
print("      ↓")
print("Deduplication")
print("      ↓")
print("Training Candidate")
print("      ↓")
print("Experiment")
print("      ↓")
print("Evaluation")
print("      ↓")
print("Candidate Model")
print("      ↓")
print("Promotion Gate")

print()


# ==================================================
# 45. ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Silverwing should not blindly train on every "
    "piece of information it encounters."
)

print()

print(
    "Information should pass through validation, "
    "provenance, quality, and dataset-versioning "
    "before becoming training material."
)

print()

print(
    "Model changes should be reproducible and "
    "reversible."
)

print()


# ==================================================
# 46. CURRENT LIMITATION
# ==================================================

print("CURRENT LIMITATION")
print()

print(
    "This is a small educational seed corpus."
)

print()

print(
    "It is nowhere near sufficient for a capable "
    "general-purpose foundation model."
)

print()

print(
    "The goal of this lesson is to establish the "
    "engineering pipeline before scaling data."
)

print()


# ==================================================
# 47. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "The native model architecture exists and "
    "the native training data pipeline exists."
)

print()

print(
    "The next lesson builds Silverwing's actual "
    "pretraining engine."
)

print()

print(
    "It will implement:"
)

print(
    "- forward pass"
)

print(
    "- causal language-model loss"
)

print(
    "- backpropagation"
)

print(
    "- optimizer"
)

print(
    "- gradient monitoring"
)

print(
    "- validation"
)

print(
    "- checkpointing"
)

print()


# ==================================================
# 48. FOUNDATION MODEL PROGRESS
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
print("OWN TRAINING DATA PIPELINE  ← COMPLETE")
print(" ↓")
print("Pretraining Engine  ← NEXT")
print(" ↓")
print("Evaluation System")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Agent Integration")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 72R COMPLETE ===")
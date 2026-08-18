# Silverwing ML
# Phase 5 - Lesson 66R
# Silverwing Own Foundation Model
# Subword Vocabulary Algorithm
#
# Goal:
# Build a BPE-style subword tokenizer from scratch.
#
# This lesson does NOT use:
# - GPT-2 tokenizer
# - Qwen tokenizer
# - tokenizers library
# - pretrained vocabulary
#
# The algorithm is implemented locally so the
# mechanics of subword learning are visible.


import json
import re

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 66R")
print("Silverwing Own Foundation Model")
print("Subword Vocabulary Algorithm")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

VOCABULARY_FILE = (
        BASE_DIR / "silverwing_subword_vocabulary.json"
)

MERGES_FILE = (
        BASE_DIR / "silverwing_bpe_merges.json"
)

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<MASK>"
]

MAX_VOCAB_SIZE = 300

BPE_END = "</w>"


print("TEST 1: Configuration")
print()

print(
    "Vocabulary file:",
    VOCABULARY_FILE
)

print(
    "Merges file:",
    MERGES_FILE
)

print(
    "Maximum vocabulary:",
    MAX_VOCAB_SIZE
)

print()


# ==================================================
# 2. NORMALIZATION
# ==================================================

def normalize_text(
        text: str
) -> str:

    if not isinstance(
            text,
            str
    ):

        raise TypeError(
            "Text must be a string."
        )


    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# 3. WORD EXTRACTION
# ==================================================

def split_words(
        text: str
) -> List[str]:

    text = normalize_text(
        text
    )


    # Keep words and punctuation separate.

    return re.findall(
        r"\w+|[^\w\s]",
        text
    )


# ==================================================
# 4. INITIAL CHARACTER REPRESENTATION
# ==================================================

def word_to_symbols(
        word: str
) -> Tuple[str, ...]:

    if not word:

        return tuple()


    symbols = list(
        word
    )


    symbols[-1] = (
            symbols[-1]
            +
            BPE_END
    )


    return tuple(
        symbols
    )


# ==================================================
# 5. CORPUS PREPARATION
# ==================================================

def build_word_frequency(
        texts: List[str]
):

    frequencies = Counter()


    for text in texts:

        words = split_words(
            text
        )


        for word in words:

            frequencies[word] += 1


    return frequencies


# ==================================================
# 6. SYMBOL VOCABULARY
# ==================================================

def build_initial_symbol_vocab(
        word_frequencies
):

    vocabulary = set()


    for word in word_frequencies:

        symbols = word_to_symbols(
            word
        )


        vocabulary.update(
            symbols
        )


    return vocabulary


# ==================================================
# 7. BPE PAIR COUNTS
# ==================================================

def get_pair_counts(
        word_symbols,
        word_frequencies
):

    pair_counts = Counter()


    for symbols, frequency in (
            word_symbols.items()
    ):

        for index in range(
                len(symbols) - 1
        ):

            pair = (
                symbols[index],
                symbols[index + 1]
            )


            pair_counts[pair] += (
                frequency
            )


    return pair_counts


# ==================================================
# 8. APPLY MERGE
# ==================================================

def merge_pair(
        symbols,
        pair
):

    merged = []

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

            merged.append(
                symbols[index]
                +
                symbols[index + 1]
            )

            index += 2

        else:

            merged.append(
                symbols[index]
            )

            index += 1


    return tuple(
        merged
    )


# ==================================================
# 9. TRAIN BPE
# ==================================================

def train_bpe(
        texts,
        max_vocab_size
):

    word_frequencies = (
        build_word_frequency(
            texts
        )
    )


    word_symbols = {
        word:
            word_to_symbols(
                word
            )
        for word
        in word_frequencies
    }


    symbol_vocab = (
        build_initial_symbol_vocab(
            word_frequencies
        )
    )


    merges = []


    target_size = max(
        max_vocab_size
        -
        len(SPECIAL_TOKENS),
        len(symbol_vocab)
    )


    iteration = 0


    while (
            len(symbol_vocab)
            <
            target_size
    ):

        pair_counts = get_pair_counts(
            {
                symbols:
                    word_frequencies[word]
                for word, symbols
                in word_symbols.items()
            },
            word_frequencies
        )


        if not pair_counts:

            break


        best_pair, best_count = (
            pair_counts.most_common(
                1
            )[0]
        )


        if best_count <= 0:

            break


        new_symbol = (
                best_pair[0]
                +
                best_pair[1]
        )


        if new_symbol in symbol_vocab:

            break


        merges.append(
            {
                "pair": [
                    best_pair[0],
                    best_pair[1]
                ],

                "count":
                    best_count,

                "new_symbol":
                    new_symbol
            }
        )


        updated_symbols = {}


        for word, symbols in (
                word_symbols.items()
        ):

            updated_symbols[word] = (
                merge_pair(
                    symbols,
                    best_pair
                )
            )


        word_symbols = updated_symbols

        symbol_vocab.add(
            new_symbol
        )


        iteration += 1


        if iteration > 10000:

            break


    return {
        "word_frequencies":
            word_frequencies,

        "word_symbols":
            word_symbols,

        "symbol_vocab":
            symbol_vocab,

        "merges":
            merges
    }


# ==================================================
# 10. TOKENIZER
# ==================================================

class SilverwingBPETokenizer:

    def __init__(
            self,
            special_tokens=None
    ):

        self.special_tokens = (
            special_tokens
            if special_tokens
               is not None
            else SPECIAL_TOKENS.copy()
        )


        self.token_to_id = {}

        self.id_to_token = {}

        self.merges = []

        self.merge_ranks = {}

        self.fitted = False


    # ----------------------------------------------
    # Add vocabulary item
    # ----------------------------------------------

    def add_token(
            self,
            token
    ):

        if token in self.token_to_id:

            return


        token_id = len(
            self.token_to_id
        )


        self.token_to_id[
            token
        ] = token_id


        self.id_to_token[
            token_id
        ] = token


    # ----------------------------------------------
    # Fit
    # ----------------------------------------------

    def fit(
            self,
            texts,
            max_vocab_size
    ):

        result = train_bpe(
            texts,
            max_vocab_size
        )


        # Special tokens first.

        for token in (
                self.special_tokens
        ):

            self.add_token(
                token
            )


        # Add learned base symbols.

        for symbol in sorted(
                result["symbol_vocab"]
        ):

            if (
                    len(
                        self.token_to_id
                    )
                    >= max_vocab_size
            ):

                break


            self.add_token(
                symbol
            )


        self.merges = [
            tuple(
                item["pair"]
            )
            for item
            in result["merges"]
        ]


        self.merge_ranks = {
            pair:
                rank
            for rank, pair
            in enumerate(
                self.merges
            )
        }


        # Add merged vocabulary items.

        for item in result["merges"]:

            token = item[
                "new_symbol"
            ]


            if (
                    len(
                        self.token_to_id
                    )
                    <
                    max_vocab_size
            ):

                self.add_token(
                    token
                )


        self.fitted = True

        return result


    # ----------------------------------------------
    # Apply a single merge
    # ----------------------------------------------

    def apply_merge(
            self,
            symbols,
            pair
    ):

        return merge_pair(
            symbols,
            pair
        )


    # ----------------------------------------------
    # Tokenize a word
    # ----------------------------------------------

    def tokenize_word(
            self,
            word
    ):

        symbols = word_to_symbols(
            word
        )


        if not symbols:

            return []


        while True:

            candidate_pairs = []


            for index in range(
                    len(symbols) - 1
            ):

                pair = (
                    symbols[index],
                    symbols[index + 1]
                )


                if pair in (
                        self.merge_ranks
                ):

                    candidate_pairs.append(
                        (
                            self.merge_ranks[
                                pair
                            ],

                            pair
                        )
                    )


            if not candidate_pairs:

                break


            _, best_pair = min(
                candidate_pairs,
                key=lambda item:
                item[0]
            )


            symbols = self.apply_merge(
                symbols,
                best_pair
            )


        return list(
            symbols
        )


    # ----------------------------------------------
    # Full text tokenization
    # ----------------------------------------------

    def tokenize(
            self,
            text
    ):

        words = split_words(
            text
        )


        tokens = []


        for word in words:

            tokens.extend(
                self.tokenize_word(
                    word
                )
            )


        return tokens


    # ----------------------------------------------
    # Encode
    # ----------------------------------------------

    def encode(
            self,
            text,
            add_bos=True,
            add_eos=True
    ):

        if not self.fitted:

            raise RuntimeError(
                "Tokenizer has not been fitted."
            )


        tokens = self.tokenize(
            text
        )


        ids = []


        if add_bos:

            ids.append(
                self.token_to_id[
                    "<BOS>"
                ]
            )


        unknown_id = (
            self.token_to_id[
                "<UNK>"
            ]
        )


        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    unknown_id
                )
            )


        if add_eos:

            ids.append(
                self.token_to_id[
                    "<EOS>"
                ]
            )


        return ids


    # ----------------------------------------------
    # Decode
    # ----------------------------------------------

    def decode(
            self,
            ids,
            skip_special_tokens=True
    ):

        tokens = []


        for token_id in ids:

            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )


            if (
                    skip_special_tokens
                    and
                    token in self.special_tokens
            ):

                continue


            tokens.append(
                token
            )


        words = []

        current_word = ""


        for token in tokens:

            if token.endswith(
                    BPE_END
            ):

                clean_token = token[
                    :-
                    len(BPE_END)
                ]


                current_word += (
                    clean_token
                )


                words.append(
                    current_word
                )


                current_word = ""

            else:

                current_word += token


        if current_word:

            words.append(
                current_word
            )


        return " ".join(
            words
        )


    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    def save(
            self,
            vocabulary_path,
            merges_path
    ):

        vocabulary = {
            "model":
                "Silverwing-BPE-v1",

            "special_tokens":
                self.special_tokens,

            "token_to_id":
                self.token_to_id
        }


        merges = [
            {
                "rank":
                    rank,

                "pair":
                    [
                        pair[0],
                        pair[1]
                    ]
            }
            for rank, pair
            in enumerate(
                self.merges
            )
        ]


        with open(
                vocabulary_path,
                "w",
                encoding="utf-8"
        ) as file:

            json.dump(
                vocabulary,
                file,
                indent=4,
                ensure_ascii=False
            )


        with open(
                merges_path,
                "w",
                encoding="utf-8"
        ) as file:

            json.dump(
                merges,
                file,
                indent=4,
                ensure_ascii=False
            )


    # ----------------------------------------------
    # Vocabulary size
    # ----------------------------------------------

    def __len__(self):

        return len(
            self.token_to_id
        )


# ==================================================
# 11. TRAINING CORPUS
# ==================================================

print("TEST 2: Corpus")
print()


training_texts = [

    "Silverwing is an artificial intelligence system.",

    "Silverwing learns from data and experience.",

    "Machine learning discovers patterns in data.",

    "Deep learning uses neural networks.",

    "Transformer models process sequences.",

    "A language model predicts the next token.",

    "Silverwing maintains persistent memory.",

    "Silverwing can use tools and services.",

    "The agent plans and executes tasks.",

    "The system evaluates its performance.",

    "Learning requires feedback and adaptation.",

    "The architecture supports autonomous improvement.",

    "Biological systems adapt to changing environments.",

    "Physics describes interactions between matter and energy.",

    "Bio-inspired computation can use adaptive loops.",

    "Silverwing studies mathematics, science, and engineering.",

    "The foundation model learns representations from text.",

    "A neural network transforms numerical representations.",

    "Attention allows the model to relate tokens.",

    "Memory allows useful information to persist."
]


print(
    "Documents:",
    len(training_texts)
)

print()


# ==================================================
# 12. TRAIN
# ==================================================

print("TEST 3: Train BPE Vocabulary")
print()


tokenizer = SilverwingBPETokenizer()


training_result = tokenizer.fit(
    training_texts,
    MAX_VOCAB_SIZE
)


print(
    "Vocabulary size:",
    len(tokenizer)
)

print(
    "Learned merges:",
    len(
        tokenizer.merges
    )
)

print()


# ==================================================
# 13. MERGE INSPECTION
# ==================================================

print("TEST 4: Learned Merge Operations")
print()


for rank, pair in list(
        enumerate(
            tokenizer.merges
        )
)[:30]:

    print(
        rank,
        "->",
        pair
    )

print()


# ==================================================
# 14. SUBWORD TOKENIZATION
# ==================================================

print("TEST 5: Subword Tokenization")
print()


examples = [
    "learning",
    "learners",
    "learningbased",
    "biological",
    "bioinspired",
    "physics",
    "physical",
    "Silverwing"
]


for word in examples:

    tokens = tokenizer.tokenize_word(
        normalize_text(
            word
        )
    )


    print(
        word,
        "->",
        tokens
    )

print()


# ==================================================
# 15. UNKNOWN WORD TEST
# ==================================================

print("TEST 6: Unseen Word")
print()


unseen = (
    "bio-computationally"
)


tokens = tokenizer.tokenize(
    unseen
)


print(
    "Text:",
    unseen
)

print(
    "Subwords:",
    tokens
)

print()


# ==================================================
# 16. ENCODING
# ==================================================

print("TEST 7: Encode")
print()


sample = (
    "Silverwing learns from data."
)


sample_tokens = tokenizer.tokenize(
    sample
)


sample_ids = tokenizer.encode(
    sample
)


print(
    "Text:",
    sample
)

print(
    "Tokens:",
    sample_tokens
)

print(
    "IDs:",
    sample_ids
)

print()


# ==================================================
# 17. DECODING
# ==================================================

print("TEST 8: Decode")
print()


decoded = tokenizer.decode(
    sample_ids
)


print(
    "Original:",
    sample
)

print(
    "Decoded:",
    decoded
)

print()


# ==================================================
# 18. ROUND TRIP
# ==================================================

print("TEST 9: Round Trip")
print()


round_trip_ids = tokenizer.encode(
    decoded
)


print(
    "First encoding:",
    sample_ids
)

print(
    "Second encoding:",
    round_trip_ids
)

print(
    "Stable:",
    sample_ids
    ==
    round_trip_ids
)

print()


# ==================================================
# 19. SPECIAL TOKENS
# ==================================================

print("TEST 10: Special Tokens")
print()


for token in SPECIAL_TOKENS:

    print(
        token,
        "->",
        tokenizer.token_to_id.get(
            token
        )
    )

print()


# ==================================================
# 20. BPE REPRESENTATION
# ==================================================

print("TEST 11: Representation Example")
print()


word = "biophysical"


initial = word_to_symbols(
    word
)


final = tokenizer.tokenize_word(
    word
)


print(
    "Word:",
    word
)

print(
    "Initial symbols:",
    initial
)

print(
    "After BPE:",
    final
)

print()


# ==================================================
# 21. TOKEN EFFICIENCY
# ==================================================

print("TEST 12: Token Efficiency")
print()


evaluation_words = [
    "machine",
    "learning",
    "biological",
    "physics",
    "engineering",
    "adaptation",
    "computational"
]


for word in evaluation_words:

    characters = len(
        word
    )


    token_count = len(
        tokenizer.tokenize_word(
            word
        )
    )


    print(
        f"{word:18}",
        "characters:",
        characters,
        "subwords:",
        token_count
    )

print()


# ==================================================
# 22. VOCABULARY INSPECTION
# ==================================================

print("TEST 13: Vocabulary Inspection")
print()


for token_id in range(
        min(
            len(tokenizer),
            80
        )
):

    print(
        token_id,
        "->",
        tokenizer.id_to_token[
            token_id
        ]
    )

print()


# ==================================================
# 23. SAVE VOCABULARY
# ==================================================

print("TEST 14: Save Silverwing Vocabulary")
print()


tokenizer.save(
    VOCABULARY_FILE,
    MERGES_FILE
)


print(
    "Vocabulary saved:"
)

print(
    VOCABULARY_FILE
)

print()

print(
    "Merges saved:"
)

print(
    MERGES_FILE
)

print()


# ==================================================
# 24. BPE TRAINING SUMMARY
# ==================================================

print("TEST 15: BPE Training Summary")
print()


summary = {
    "algorithm":
        "BPE-style",

    "documents":
        len(training_texts),

    "vocabulary_size":
        len(tokenizer),

    "merge_operations":
        len(
            tokenizer.merges
        ),

    "special_tokens":
        SPECIAL_TOKENS
}


print(
    json.dumps(
        summary,
        indent=4
    )
)

print()


# ==================================================
# 25. WHY BPE MATTERS
# ==================================================

print("WHY SUBWORD TOKENIZATION MATTERS")
print()

print(
    "The model does not need one vocabulary entry "
    "for every possible human word."
)

print()

print(
    "Frequent patterns become reusable units."
)

print()

print(
    "Rare words can be represented by combinations "
    "of learned pieces."
)

print()

print(
    "This reduces dependence on a huge "
    "word-level vocabulary."
)

print()


# ==================================================
# 26. FOUNDATION MODEL CONNECTION
# ==================================================

print("SILVERWING FOUNDATION MODEL CONNECTION")
print()

print("Raw text")
print("   ↓")
print("Normalization")
print("   ↓")
print("BPE Subword Tokenizer")
print("   ↓")
print("Token IDs")
print("   ↓")
print("Embedding Matrix")
print("   ↓")
print("Transformer")
print("   ↓")
print("Next-token probabilities")

print()


# ==================================================
# 27. FUTURE TOKENIZER EVOLUTION
# ==================================================

print("FUTURE TOKENIZER EVOLUTION")
print()

print(
    "This implementation is educational."
)

print()

print(
    "A stronger Silverwing tokenizer can later "
    "incorporate byte-level handling, multilingual "
    "text, whitespace rules, special control tokens, "
    "and more efficient training."
)

print()

print(
    "The important principle is that the tokenizer "
    "and vocabulary belong to Silverwing's own "
    "foundation-model stack."
)

print()


# ==================================================
# 28. BIO-INSPIRED DEVELOPMENT CONNECTION
# ==================================================

print("BIO-INSPIRED DEVELOPMENT CONNECTION")
print()

print(
    "Vocabulary growth can eventually become part "
    "of controlled developmental learning."
)

print()

print(
    "New domains introduce new linguistic patterns."
)

print()

print(
    "The system can identify candidate vocabulary "
    "extensions and evaluate them before creating "
    "a new model version."
)

print()


# ==================================================
# 29. VERSIONED GROWTH
# ==================================================

print("VERSIONED VOCABULARY GROWTH")
print()

print("Corpus v1")
print("   ↓")
print("Vocabulary v1")
print("   ↓")
print("Train model v1")
print("   ↓")
print("Evaluate")
print("   ↓")
print("New corpus")
print("   ↓")
print("Candidate vocabulary v2")
print("   ↓")
print("Evaluate compatibility")
print("   ↓")
print("Vocabulary v2")
print("   ↓")
print("Train model v2")

print()


# ==================================================
# 30. IMPORTANT ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Self-growth does not mean uncontrolled mutation."
)

print()

print(
    "Every vocabulary or model change should be "
    "represented by a versioned artifact and "
    "validated before promotion."
)

print()


# ==================================================
# 31. NEXT FOUNDATION COMPONENT
# ==================================================

print("NEXT FOUNDATION COMPONENT")
print()

print(
    "The tokenizer converts language into token IDs."
)

print()

print(
    "The next component must convert those token IDs "
    "into learned numerical representations."
)

print()

print(
    "That component is Silverwing's own embedding "
    "system."
)

print()


# ==================================================
# 32. SILVERWING FOUNDATION PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own Tokenizer")
print(" ↓")
print("Own Subword Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("OWN EMBEDDING SYSTEM")
print(" ↓")
print("Positional Representation")
print(" ↓")
print("Self-Attention")
print(" ↓")
print("Transformer Blocks")
print(" ↓")
print("Decoder Model")
print(" ↓")
print("Training System")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Agent Integration")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 66R COMPLETE ===")
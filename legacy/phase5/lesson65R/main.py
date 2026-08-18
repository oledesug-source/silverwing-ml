# Silverwing ML
# Phase 5 - Lesson 65R
# Silverwing Own Foundation Model
# Tokenizer Architecture
#
# This lesson intentionally does NOT use:
# - GPT-2 tokenizer
# - Qwen tokenizer
# - Hugging Face pretrained tokenizer
#
# The goal is to create Silverwing's own
# tokenizer architecture.


import json
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 65R")
print("Silverwing Own Foundation Model")
print("Tokenizer Architecture")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

VOCABULARY_FILE = (
        BASE_DIR / "silverwing_vocabulary.json"
)

MODEL_NAME = "Silverwing-Foundation-Tokenizer"

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<MASK>"
]

DEFAULT_MAX_VOCAB_SIZE = 5000


print("TEST 1: Configuration")
print()

print(
    "Component:",
    MODEL_NAME
)

print(
    "Vocabulary file:",
    VOCABULARY_FILE
)

print(
    "Special tokens:",
    SPECIAL_TOKENS
)

print()


# ==================================================
# 2. TEXT NORMALIZATION
# ==================================================

class TextNormalizer:
    """
    Converts raw text into a predictable representation
    before tokenization.
    """

    @staticmethod
    def normalize(
            text: str
    ) -> str:

        if not isinstance(
                text,
                str
        ):

            raise TypeError(
                "Text must be a string."
            )


        # Unicode normalization.

        text = unicodedata.normalize(
            "NFKC",
            text
        )


        # Normalize whitespace.

        text = re.sub(
            r"\s+",
            " ",
            text
        )


        # Remove surrounding whitespace.

        text = text.strip()


        return text


normalizer = TextNormalizer()


# ==================================================
# 3. BASIC TOKENIZER
# ==================================================

class SilverwingTokenizer:

    def __init__(
            self,
            special_tokens=None
    ):

        self.special_tokens = (
            special_tokens
            if special_tokens is not None
            else SPECIAL_TOKENS.copy()
        )


        self.token_to_id: Dict[
            str,
            int
        ] = {}


        self.id_to_token: Dict[
            int,
            str
        ] = {}


        self.fitted = False


    # ----------------------------------------------
    # Token splitting
    # ----------------------------------------------

    def split_text(
            self,
            text: str
    ) -> List[str]:

        text = normalizer.normalize(
            text
        )


        # Preserve words, numbers and punctuation
        # as separate units.

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE
        )


        return tokens


    # ----------------------------------------------
    # Build vocabulary
    # ----------------------------------------------

    def fit(
            self,
            texts: List[str],
            max_vocab_size=DEFAULT_MAX_VOCAB_SIZE
    ):

        if not texts:

            raise ValueError(
                "Training texts cannot be empty."
            )


        counter = Counter()


        for text in texts:

            tokens = self.split_text(
                text
            )


            counter.update(
                tokens
            )


        self.token_to_id.clear()

        self.id_to_token.clear()


        # Special tokens always receive
        # the first IDs.

        for token in self.special_tokens:

            self._add_token(
                token
            )


        remaining_slots = (
                max_vocab_size
                -
                len(
                    self.special_tokens
                )
        )


        vocabulary_tokens = [
            token
            for token, _ in counter.most_common(
                remaining_slots
            )
        ]


        for token in vocabulary_tokens:

            if token not in self.token_to_id:

                self._add_token(
                    token
                )


        self.fitted = True


        return self


    # ----------------------------------------------
    # Internal vocabulary insertion
    # ----------------------------------------------

    def _add_token(
            self,
            token
    ):

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
    # Encode
    # ----------------------------------------------

    def encode(
            self,
            text: str,
            add_bos=True,
            add_eos=True
    ):

        if not self.fitted:

            raise RuntimeError(
                "Tokenizer must be fitted first."
            )


        tokens = self.split_text(
            text
        )


        ids = []


        if add_bos:

            ids.append(
                self.token_to_id[
                    "<BOS>"
                ]
            )


        unknown_id = self.token_to_id[
            "<UNK>"
        ]


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
            ids: List[int],
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


        result = ""


        for token in tokens:

            # Attach punctuation without a
            # preceding space.

            if (
                    result
                    and
                    re.match(
                        r"[^\w\s]",
                        token,
                        flags=re.UNICODE
                    )
            ):

                result += token

            else:

                if result:

                    result += " "

                result += token


        return result


    # ----------------------------------------------
    # Vocabulary size
    # ----------------------------------------------

    def __len__(self):

        return len(
            self.token_to_id
        )


    # ----------------------------------------------
    # Save vocabulary
    # ----------------------------------------------

    def save(
            self,
            path: Path
    ):

        data = {
            "model":
                MODEL_NAME,

            "special_tokens":
                self.special_tokens,

            "token_to_id":
                self.token_to_id
        }


        with open(
                path,
                "w",
                encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )


    # ----------------------------------------------
    # Load vocabulary
    # ----------------------------------------------

    @classmethod
    def load(
            cls,
            path: Path
    ):

        with open(
                path,
                "r",
                encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        tokenizer = cls(
            special_tokens=data[
                "special_tokens"
            ]
        )


        for token, token_id in (
                data["token_to_id"].items()
        ):

            tokenizer.token_to_id[
                token
            ] = int(
                token_id
            )

            tokenizer.id_to_token[
                int(token_id)
            ] = token


        tokenizer.fitted = True


        return tokenizer


# ==================================================
# 4. TRAINING CORPUS
# ==================================================

print("TEST 2: Training Corpus")
print()


training_texts = [

    "Silverwing is an artificial intelligence system.",

    "Silverwing learns from data and experience.",

    "Machine learning allows systems to discover patterns.",

    "Deep learning uses neural networks.",

    "Transformers process sequences of tokens.",

    "A language model predicts useful continuations.",

    "Memory allows Silverwing to preserve information.",

    "Tools allow Silverwing to interact with its environment.",

    "The agent can plan and execute tasks.",

    "Silverwing evaluates its own performance.",

    "Learning requires data, evaluation, and adaptation.",

    "A model improves through training and feedback.",

    "The system monitors its internal state.",

    "The system can discover new capabilities.",

    "Reliable AI requires verification."
]


print(
    "Training documents:",
    len(training_texts)
)

print()


# ==================================================
# 5. NORMALIZATION TEST
# ==================================================

print("TEST 3: Text Normalization")
print()


raw_text = (
    "  Silverwing   learns  from   data.  "
)


normalized_text = normalizer.normalize(
    raw_text
)


print(
    "Raw:",
    repr(raw_text)
)

print(
    "Normalized:",
    repr(normalized_text)
)

print()


# ==================================================
# 6. TRAIN TOKENIZER
# ==================================================

print("TEST 4: Build Vocabulary")
print()


tokenizer = SilverwingTokenizer()


tokenizer.fit(
    training_texts,
    max_vocab_size=DEFAULT_MAX_VOCAB_SIZE
)


print(
    "Vocabulary size:",
    len(tokenizer)
)

print()


# ==================================================
# 7. SPECIAL TOKEN IDs
# ==================================================

print("TEST 5: Special Token IDs")
print()


for token in SPECIAL_TOKENS:

    print(
        token,
        "->",
        tokenizer.token_to_id[
            token
        ]
    )

print()


# ==================================================
# 8. TOKENIZATION
# ==================================================

print("TEST 6: Tokenization")
print()


sample_text = (
    "Silverwing learns from data."
)


tokens = tokenizer.split_text(
    sample_text
)


print(
    "Input:",
    sample_text
)

print(
    "Tokens:",
    tokens
)

print()


# ==================================================
# 9. ENCODING
# ==================================================

print("TEST 7: Token IDs")
print()


token_ids = tokenizer.encode(
    sample_text
)


print(
    "Input:",
    sample_text
)

print(
    "Token IDs:",
    token_ids
)

print()


# ==================================================
# 10. DECODING
# ==================================================

print("TEST 8: Decode")
print()


decoded_text = tokenizer.decode(
    token_ids
)


print(
    "Decoded:",
    decoded_text
)

print()


# ==================================================
# 11. ROUND-TRIP TEST
# ==================================================

print("TEST 9: Encode/Decode Round Trip")
print()


round_trip = (
    tokenizer.decode(
        tokenizer.encode(
            sample_text
        )
    )
)


print(
    "Original:",
    sample_text
)

print(
    "Round trip:",
    round_trip
)

print()


# ==================================================
# 12. UNKNOWN TOKEN
# ==================================================

print("TEST 10: Unknown Token")
print()


unknown_text = (
    "Silverwing quantum-bio-mechanical system"
)


unknown_ids = tokenizer.encode(
    unknown_text
)


print(
    "Text:",
    unknown_text
)

print(
    "IDs:",
    unknown_ids
)

print(
    "Decoded:",
    tokenizer.decode(
        unknown_ids
    )
)

print()


# ==================================================
# 13. VOCABULARY INSPECTION
# ==================================================

print("TEST 11: Vocabulary Inspection")
print()


for token_id in range(
        min(
            len(tokenizer),
            40
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
# 14. TOKEN FREQUENCY ANALYSIS
# ==================================================

print("TEST 12: Token Frequency Analysis")
print()


counter = Counter()


for text in training_texts:

    counter.update(
        tokenizer.split_text(
            text
        )
    )


for token, frequency in (
        counter.most_common(20)
):

    print(
        token,
        "->",
        frequency
    )

print()


# ==================================================
# 15. VOCABULARY SAVE
# ==================================================

print("TEST 13: Save Vocabulary")
print()


tokenizer.save(
    VOCABULARY_FILE
)


print(
    "Saved:",
    VOCABULARY_FILE
)

print()


# ==================================================
# 16. VOCABULARY RELOAD
# ==================================================

print("TEST 14: Reload Vocabulary")
print()


loaded_tokenizer = (
    SilverwingTokenizer.load(
        VOCABULARY_FILE
    )
)


print(
    "Reloaded vocabulary:",
    len(loaded_tokenizer)
)

print()


# ==================================================
# 17. CONSISTENCY CHECK
# ==================================================

print("TEST 15: Vocabulary Consistency")
print()


original_ids = tokenizer.encode(
    sample_text
)


reloaded_ids = loaded_tokenizer.encode(
    sample_text
)


print(
    "Original IDs:",
    original_ids
)

print(
    "Reloaded IDs:",
    reloaded_ids
)

print(
    "Identical:",
    original_ids == reloaded_ids
)

print()


# ==================================================
# 18. BATCH ENCODING
# ==================================================

print("TEST 16: Batch Encoding")
print()


batch_texts = [
    "Silverwing learns.",
    "Machine learning is useful.",
    "Transformers process tokens."
]


batch_ids = [
    tokenizer.encode(
        text
    )
    for text
    in batch_texts
]


for text, ids in zip(
        batch_texts,
        batch_ids
):

    print(
        text
    )

    print(
        ids
    )

    print()


# ==================================================
# 19. PADDING
# ==================================================

print("TEST 17: Padding")
print()


def pad_sequences(
        sequences: List[List[int]],
        pad_id: int
):

    if not sequences:

        return []


    max_length = max(
        len(sequence)
        for sequence
        in sequences
    )


    padded = []


    for sequence in sequences:

        padding = (
                [pad_id]
                *
                (
                        max_length
                        -
                        len(sequence)
                )
        )


        padded.append(
            sequence
            +
            padding
        )


    return padded


padded_batch = pad_sequences(
    batch_ids,
    tokenizer.token_to_id[
        "<PAD>"
    ]
)


for sequence in padded_batch:

    print(
        sequence
    )

print()


# ==================================================
# 20. ATTENTION MASK
# ==================================================

print("TEST 18: Attention Mask")
print()


def create_attention_mask(
        padded_sequences,
        pad_id
):

    masks = []


    for sequence in padded_sequences:

        mask = [
            0
            if token_id == pad_id
            else 1
            for token_id
            in sequence
        ]


        masks.append(
            mask
        )


    return masks


attention_masks = (
    create_attention_mask(
        padded_batch,
        tokenizer.token_to_id[
            "<PAD>"
        ]
    )
)


for mask in attention_masks:

    print(
        mask
    )

print()


# ==================================================
# 21. AUTOREGRESSIVE TRAINING PAIR
# ==================================================

print("TEST 19: Autoregressive Training Pair")
print()


example_ids = tokenizer.encode(
    "Silverwing learns from data."
)


input_ids = (
    example_ids[:-1]
)


target_ids = (
    example_ids[1:]
)


print(
    "Full sequence:",
    example_ids
)

print(
    "Input IDs:",
    input_ids
)

print(
    "Target IDs:",
    target_ids
)

print()


# ==================================================
# 22. NEXT-TOKEN CONCEPT
# ==================================================

print("TEST 20: Next-Token Prediction")
print()


decoded_inputs = tokenizer.decode(
    input_ids,
    skip_special_tokens=False
)


decoded_targets = tokenizer.decode(
    target_ids,
    skip_special_tokens=False
)


print(
    "Input sequence:",
    decoded_inputs
)

print(
    "Target sequence:",
    decoded_targets
)

print()


# ==================================================
# 23. VOCABULARY GROWTH
# ==================================================

print("TEST 21: Vocabulary Growth Concept")
print()


new_training_texts = [
    "Silverwing studies biology.",
    "Silverwing studies physics.",
    "Silverwing studies energy systems.",
    "Silverwing studies computation."
]


new_counter = Counter()


for text in new_training_texts:

    new_counter.update(
        tokenizer.split_text(
            text
        )
    )


new_tokens = [
    token
    for token in new_counter
    if token
       not in tokenizer.token_to_id
]


print(
    "Potential new tokens:"
)

for token in new_tokens:

    print(
        "-",
        token
    )

print()


# ==================================================
# 24. TOKENIZER LIMITATION
# ==================================================

print("TEST 22: Current Tokenizer Limitation")
print()

print(
    "This tokenizer currently uses word and "
    "punctuation units."
)

print()

print(
    "It is intentionally simple so the internal "
    "mechanics are visible."
)

print()

print(
    "A production foundation model will need a "
    "more sophisticated subword tokenizer capable "
    "of representing rare and unseen words efficiently."
)

print()


# ==================================================
# 25. WHY SUBWORDS MATTER
# ==================================================

print("WHY SUBWORDS MATTER")
print()

print(
    "Consider an unfamiliar word:"
)

print(
    "bio-physical"
)

print()

print(
    "A word-level tokenizer may treat the entire "
    "word as unknown."
)

print()

print(
    "A subword tokenizer can decompose it into "
    "reusable pieces."
)

print()

print(
    "This greatly reduces unknown-token problems "
    "for large vocabularies."
)

print()


# ==================================================
# 26. SILVERWING FOUNDATION MODEL PIPELINE
# ==================================================

print("SILVERWING FOUNDATION MODEL PIPELINE")
print()

print("Raw Corpus")
print("   ↓")
print("Text Normalization")
print("   ↓")
print("Tokenizer Training")
print("   ↓")
print("Silverwing Vocabulary")
print("   ↓")
print("Token IDs")
print("   ↓")
print("Embedding Layer")
print("   ↓")
print("Positional Representation")
print("   ↓")
print("Transformer")
print("   ↓")
print("Language Modeling Head")

print()


# ==================================================
# 27. COGNITIVE DEVELOPMENT CONNECTION
# ==================================================

print("COGNITIVE DEVELOPMENT CONNECTION")
print()

print(
    "The tokenizer does not constitute intelligence."
)

print()

print(
    "It creates the numerical language interface "
    "through which Silverwing's neural model will "
    "learn statistical and semantic structure."
)

print()

print(
    "The actual learning system begins with the "
    "embedding and transformer architecture built "
    "in the following lessons."
)

print()


# ==================================================
# 28. OWNERSHIP PRINCIPLE
# ==================================================

print("SILVERWING OWNERSHIP PRINCIPLE")
print()

print(
    "No pretrained tokenizer is used as Silverwing's "
    "native tokenizer."
)

print()

print(
    "The vocabulary is generated from Silverwing's "
    "own training corpus."
)

print()

print(
    "Later model versions can create new vocabulary "
    "versions as the training corpus expands."
)

print()


# ==================================================
# 29. FUTURE SELF-GROWTH DIRECTION
# ==================================================

print("FUTURE VOCABULARY GROWTH")
print()

print("New Experiences")
print("      ↓")
print("New Training Data")
print("      ↓")
print("Vocabulary Analysis")
print("      ↓")
print("Candidate Tokens")
print("      ↓")
print("Evaluation")
print("      ↓")
print("Vocabulary Version")
print("      ↓")
print("Retraining / Adaptation")

print()


# ==================================================
# 30. IMPORTANT SAFETY / ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Silverwing should never silently mutate its "
    "production vocabulary or neural weights."
)

print()

print(
    "Future self-improvement should occur through "
    "versioned artifacts, evaluation, checkpoints, "
    "and explicit promotion of validated versions."
)

print()


# ==================================================
# 31. CURRENT PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own Tokenizer")
print(" ↓")
print("Own Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("Embedding Layer")
print(" ↓")
print("Attention")
print(" ↓")
print("Transformer Blocks")
print(" ↓")
print("Language Model")
print(" ↓")
print("Training")
print(" ↓")
print("Instruction Tuning")
print(" ↓")
print("Agent Integration")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 65R COMPLETE ===")
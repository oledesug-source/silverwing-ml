"""Embeddings — text embedding models for RAG and similarity search.

Provides from-scratch implementations of text embedding techniques:

    - ``TFIDFEmbedder``: TF-IDF vectorization with cosine similarity
    - ``WordEmbedding``: Word-level embeddings (random init, trainable)
    - ``SentenceEmbedding``: Averaged word embeddings for sentences
    - ``hash_vectorize``: Hashing-based feature vectors (fastText style)

All implementations are numpy-based and do not require torch or
transformers libraries at import time.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text)
    return [t for t in tokens if t.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (or each row of a 2-D matrix)."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return np.where(norms > 0, vec / norms, vec)


# ---------------------------------------------------------------------------
# TF-IDF Embedder
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """TF-IDF text embedder with cosine similarity search.

    Builds a vocabulary from training documents and produces
    TF-IDF-weighted vectors.  Supports efficient similarity search
    via precomputed IDF weights and normalized document vectors.

    Args:
        max_features: Maximum vocabulary size (top-k by frequency).
        min_df:       Minimum document frequency for a term to be included.
        max_df:       Maximum document frequency ratio (0.0–1.0).

    Attributes:
        vocab:        List of terms in the vocabulary.
        term_to_idx:  Mapping from term → column index.
        idf:          IDF vector (1-D array).
        doc_vectors:  Normalized document-term matrix (n_docs × vocab_size).
    """

    def __init__(
        self,
        max_features: int = 10000,
        min_df: int = 1,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocab: list[str] = []
        self.term_to_idx: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.doc_vectors: np.ndarray | None = None
        self._fitted = False
        self._doc_tokens: list[list[str]] = []
        self._n_docs = 0

    def fit(self, documents: list[str]) -> TFIDFEmbedder:
        """Fit the TF-IDF model on a list of documents.

        Args:
            documents: List of text documents.

        Returns:
            self (fitted embedder).
        """
        tokenized_docs = [tokenize(doc) for doc in documents]
        self._doc_tokens = tokenized_docs
        self._n_docs = len(documents)

        # Count document frequencies
        df_counter: Counter = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                df_counter[tok] += 1

        # Filter by df thresholds
        valid_terms = [
            term for term, df in df_counter.items()
            if df >= self.min_df and df / self._n_docs <= self.max_df
        ]

        # Sort by document frequency descending, truncate to max_features
        valid_terms.sort(key=lambda t: df_counter[t], reverse=True)
        self.vocab = valid_terms[:self.max_features]
        self.term_to_idx = {t: i for i, t in enumerate(self.vocab)}

        # Compute IDF: log((1 + N) / (1 + df)) + 1 (smoothed)
        idf_vals = np.zeros(len(self.vocab))
        for term, idx in self.term_to_idx.items():
            df = df_counter[term]
            idf_vals[idx] = math.log((1 + self._n_docs) / (1 + df)) + 1.0
        self.idf = idf_vals

        # Build document-term matrix
        n_docs = len(tokenized_docs)
        vocab_size = len(self.vocab)
        doc_term = np.zeros((n_docs, vocab_size), dtype=np.float64)
        for doc_idx, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.term_to_idx:
                    col = self.term_to_idx[term]
                    doc_term[doc_idx, col] = count / len(tokens)

        # Apply TF-IDF weighting
        tfidf = doc_term * self.idf[np.newaxis, :]
        self.doc_vectors = normalize(tfidf)
        self._fitted = True
        return self

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a TF-IDF vector.

        Returns:
            Normalized TF-IDF vector of shape (vocab_size,).
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        tokens = tokenize(text)
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            if term in self.term_to_idx:
                vec[self.term_to_idx[term]] = count / len(tokens)
        vec = vec * self.idf
        return normalize(vec)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search for the most similar documents to a query.

        Args:
            query:  Query text string.
            top_k:  Number of top results to return.

        Returns:
            List of (doc_index, similarity_score) tuples, sorted descending.
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        query_vec = self.embed(query)
        # Cosine similarity = dot product (since vectors are normalized)
        sims = np.dot(self.doc_vectors, query_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    @property
    def vector_dim(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Word Embedding & Sentence Embedding
# ---------------------------------------------------------------------------

class WordEmbedding:
    """Randomly-initialized word embedding lookup table.

    In a full implementation these weights would be trained via backprop.
    Here we provide a functional lookup table suitable for testing
    embedding arithmetic (e.g., analogy tasks).

    Args:
        vocab_size: Size of the vocabulary.
        dim:        Embedding dimension.
        seed:       Random seed for reproducible initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        seed: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (vocab_size + dim))
        self.weights = rng.uniform(-limit, limit, (vocab_size, dim))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings for given token IDs.

        Args:
            token_ids: Array of integer token IDs.

        Returns:
            Array of shape (len(token_ids), dim).
        """
        return self.weights[token_ids]

    def similarity(self, id1: int, id2: int) -> float:
        """Cosine similarity between two token embeddings."""
        return cosine_similarity(self.weights[id1], self.weights[id2])

    def analogy(self, a: int, b: int, c: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Solve analogy: a - b + c ≈ ?

        Returns top_k closest tokens by cosine similarity.
        """
        target = self.weights[a] - self.weights[b] + self.weights[c]
        target = normalize(target)
        sims = np.dot(self.weights, target)
        top = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top if i not in (a, b, c)]


class SentenceEmbedding:
    """Sentence embedding via averaged word embeddings.

    Args:
        vocab:    Dictionary mapping word → index.
        dim:      Embedding dimension.
        seed:     Random seed.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        dim: int = 100,
        seed: int | None = None,
    ) -> None:
        self.vocab = vocab
        self.dim = dim
        self.word_embedder = WordEmbedding(len(vocab) + 1, dim, seed)

    def embed(self, text: str) -> np.ndarray:
        """Embed a sentence as the average of its word embeddings."""
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.dim)
        indices = [self.vocab.get(t, len(self.vocab)) for t in tokens]
        word_vecs = self.word_embedder.forward(np.array(indices))
        return np.mean(word_vecs, axis=0)

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two sentence embeddings."""
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return cosine_similarity(v1, v2)


# ---------------------------------------------------------------------------
# Hashing-based embedding (fastText-style)
# ---------------------------------------------------------------------------

def hash_vectorize(text: str, num_features: int = 1024, ngram_range: tuple[int, int] = (1, 2)) -> np.ndarray:
    """Hashing-based feature vectorization (fastText style).

    Uses the hashing trick to map tokens and n-grams to fixed-size
    feature vectors without requiring a precomputed vocabulary.

    Args:
        text:         Input text.
        num_features: Dimensionality of the output vector.
        ngram_range:  (min_n, max_n) for n-gram features.

    Returns:
        Binary/count feature vector of shape (num_features,).
    """
    tokens = tokenize(text)
    vec = np.zeros(num_features, dtype=np.float64)

    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            # Hash the n-gram to a feature index
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = h % num_features
            vec[idx] += 1.0

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

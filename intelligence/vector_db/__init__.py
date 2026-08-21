"""In-memory vector database with similarity search and hybrid retrieval.

Provides:

    - ``VectorStore``: A simple in-memory vector database supporting
      insertion, deletion, nearest-neighbor search via cosine similarity,
      and metadata filtering.
    - ``HybridSearch``: Combines dense vector similarity with sparse
      (keyword/TF-IDF) retrieval for improved RAG performance.
    - ``VectorIndex``: IVF (Inverted File) index for approximate nearest
      neighbor search on large corpora.

All implementations are numpy-based and do not require external vector
databases (e.g., FAISS, Chroma, Pinecone) at import time.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embeddings import TFIDFEmbedder, cosine_similarity, normalize, tokenize

# ---------------------------------------------------------------------------
# VectorStore — core in-memory vector database
# ---------------------------------------------------------------------------

@dataclass
class VectorEntry:
    """A single entry in the vector store.

    Attributes:
        id:        Unique identifier for the vector.
        vector:    Dense vector embedding.
        metadata:  Associated metadata (text, source, timestamp, etc.).
    """

    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """In-memory vector database with similarity search.

    Supports cosine similarity and Euclidean distance for nearest-neighbor
    retrieval.  Vectors are kept normalized for efficient cosine computation.

    Args:
        dim: Expected dimensionality of vectors.  If provided, vectors
             of mismatched dimension will raise an error.

    Example:
        >>> store = VectorStore(dim=128)
        >>> store.add("vec1", vec1, {"text": "hello world"})
        >>> store.add("vec2", vec2, {"text": "goodbye"})
        >>> results = store.search(query_vec, top_k=5)
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self._entries: dict[str, VectorEntry] = {}
        self._normalized_vectors: dict[str, np.ndarray] = {}

    def _validate_vector(self, vector: np.ndarray) -> None:
        if self.dim is not None and vector.shape[-1] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[-1]} does not match "
                f"expected dimension {self.dim}."
            )

    def add(
        self,
        vector_id: str | None,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a vector with optional metadata.

        Args:
            vector_id:  Unique ID. Auto-generated if None.
            vector:     Dense vector array.
            metadata:   Optional metadata dict.

        Returns:
            The vector ID used.
        """
        vector = np.asarray(vector, dtype=np.float64)
        self._validate_vector(vector)
        if vector_id is None:
            vector_id = str(uuid.uuid4())

        entry = VectorEntry(
            id=vector_id,
            vector=vector,
            metadata=metadata or {},
        )
        self._entries[vector_id] = entry
        self._normalized_vectors[vector_id] = normalize(vector)
        return vector_id

    def add_batch(
        self,
        vectors: list[np.ndarray],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add multiple vectors at once."""
        ids = []
        for i, vec in enumerate(vectors):
            meta = metadatas[i] if metadatas else None
            ids.append(self.add(None, vec, meta))
        return ids

    def get(self, vector_id: str) -> VectorEntry | None:
        """Retrieve a vector entry by ID."""
        return self._entries.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector entry by ID. Returns True if deleted."""
        if vector_id in self._entries:
            del self._entries[vector_id]
            del self._normalized_vectors[vector_id]
            return True
        return False

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Find the nearest neighbors to the query vector.

        Args:
            query:    Query vector.
            top_k:    Number of results to return.
            metric:   Distance metric: "cosine" or "euclidean".

        Returns:
            List of (vector_id, score, metadata) tuples sorted by score
            (descending for cosine, ascending for euclidean).
        """
        query = np.asarray(query, dtype=np.float64)
        self._validate_vector(query)

        if not self._entries:
            return []

        results: list[tuple[float, str, dict]] = []

        if metric == "cosine":
            query_norm = normalize(query)
            for vid, store_norm in self._normalized_vectors.items():
                score = float(np.dot(query_norm, store_norm))
                results.append((score, vid, self._entries[vid].metadata))
            results.sort(key=lambda x: x[0], reverse=True)

        elif metric == "euclidean":
            for vid, entry in self._entries.items():
                dist = float(np.linalg.norm(query - entry.vector))
                results.append((dist, vid, entry.metadata))
            results.sort(key=lambda x: x[0])

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

        return [(vid, score, meta) for score, vid, meta in results[:top_k]]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def ids(self) -> list[str]:
        """List all vector IDs in the store."""
        return list(self._entries.keys())


# ---------------------------------------------------------------------------
# HybridSearch — combines dense + sparse retrieval
# ---------------------------------------------------------------------------

class HybridSearch:
    """Hybrid search combining dense vector similarity and sparse keyword matching.

    Uses a dual-encoder approach: dense vectors (from an embedding function)
    for semantic search, and TF-IDF sparse vectors for keyword matching.
    Results are fused using Reciprocal Rank Fusion (RRF).

    Args:
        embed_fn:     Callable that converts text → dense vector.
        alpha:        Weight for dense score in fusion (1-alpha = sparse weight).
        rrf_k:        RRF parameter k (controls rank decay).
        top_k:        Default number of results to return.
        tfidf_kwargs: Keyword args for the internal TFIDFEmbedder.

    Example:
        >>> search = HybridSearch(embed_fn=lambda t: my_embedder(t), alpha=0.7)
        >>> search.fit(["doc1 text", "doc2 text"])
        >>> results = search.search("query text")
    """

    def __init__(
        self,
        embed_fn: Any,
        alpha: float = 0.7,
        rrf_k: int = 60,
        top_k: int = 5,
        tfidf_kwargs: dict | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.tfidf_kwargs = tfidf_kwargs or {}
        self._documents: list[str] = []
        self._dense_vectors: list[np.ndarray] = []
        self._tfidf: TFIDFEmbedder | None = None
        self._fitted = False

    def fit(self, documents: list[str]) -> HybridSearch:
        """Index a corpus of documents for hybrid search.

        Args:
            documents: List of text documents to index.
        """
        self._documents = list(documents)
        self._dense_vectors = [normalize(self.embed_fn(doc)) for doc in documents]
        self._tfidf = TFIDFEmbedder(**self.tfidf_kwargs)
        self._tfidf.fit(documents)
        self._fitted = True
        return self

    def _dense_scores(self, query: str) -> np.ndarray:
        """Compute dense (vector) similarity scores for all documents."""
        q_vec = normalize(self.embed_fn(query))
        scores = np.array([
            float(np.dot(q_vec, doc_vec)) for doc_vec in self._dense_vectors
        ])
        return scores

    def _sparse_scores(self, query: str) -> dict[int, float]:
        """Compute sparse (TF-IDF) similarity scores for all documents."""
        # search returns list of (doc_index, score) tuples
        sparse_results = self._tfidf.search(query, top_k=len(self._documents))
        return dict(sparse_results)

    @staticmethod
    def _ranks(scores: np.ndarray, descending: bool = True) -> np.ndarray:
        """Convert scores to ranks (1-based)."""
        order = np.argsort(-scores if descending else scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float, str]]:
        """Hybrid search: combine dense and sparse retrieval via RRF.

        Args:
            query:  Query text.
            top_k:  Number of results (defaults to configured value).

        Returns:
            List of (doc_index, fused_score, document_text) sorted by score.
        """
        if not self._fitted:
            raise RuntimeError("HybridSearch not fitted. Call fit() first.")

        k = top_k or self.top_k

        # Dense scores (cosine similarity)
        dense_scores = self._dense_scores(query)
        dense_ranks = self._ranks(dense_scores, descending=True)

        # Sparse (TF-IDF) scores
        sparse_map = self._sparse_scores(query)
        sparse_scores = np.array([
            sparse_map.get(i, 0.0) for i in range(len(self._documents))
        ])
        sparse_ranks = self._ranks(sparse_scores, descending=True)

        # Reciprocal Rank Fusion
        rrf_scores = np.array([
            self.alpha / (self.rrf_k + dense_ranks[i])
            + (1 - self.alpha) / (self.rrf_k + sparse_ranks[i])
            for i in range(len(self._documents))
        ])

        top_indices = np.argsort(rrf_scores)[::-1][:k]
        return [
            (int(idx), float(rrf_scores[idx]), self._documents[idx])
            for idx in top_indices
        ]


# ---------------------------------------------------------------------------
# VectorIndex — IVF (Inverted File) approximate nearest neighbor index
# ---------------------------------------------------------------------------

class VectorIndex:
    """IVF (Inverted File) approximate nearest neighbor index.

    Partitions the vector space into ``nlist`` clusters using k-means
    initialization, then searches only the nearest clusters for
    efficiency.  This trades a small amount of recall for significant
    speed gains on large datasets.

    Args:
        dim:   Dimensionality of vectors.
        nlist: Number of clusters (partitions).
    """

    def __init__(self, dim: int, nlist: int = 100) -> None:
        self.dim = dim
        self.nlist = nlist
        self._centroids: np.ndarray | None = None
        self._cluster_assignments: dict[int, list[tuple[str, np.ndarray, dict]]] = {}
        self._entries: dict[str, VectorEntry] = {}
        self._built = False

    def add(
        self,
        vector_id: str | None,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a vector to the index (must rebuild after adding)."""
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape[-1] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[-1]} != expected {self.dim}"
            )
        if vector_id is None:
            vector_id = str(uuid.uuid4())

        entry = VectorEntry(
            id=vector_id, vector=vector, metadata=metadata or {},
        )
        self._entries[vector_id] = entry
        self._built = False
        return vector_id

    def _kmeans_init(self, vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """k-means++ initialization."""
        n = len(vectors)
        centroids = np.empty((k, vectors.shape[1]))
        # Pick first centroid randomly
        centroids[0] = vectors[rng.integers(n)]
        closest_dist = np.full(n, np.inf)

        for i in range(1, k):
            dists = np.sum((vectors - centroids[i - 1]) ** 2, axis=1)
            closest_dist = np.minimum(closest_dist, dists)
            if closest_dist.sum() == 0:
                centroids[i] = vectors[rng.integers(n)]
            else:
                probs = closest_dist / closest_dist.sum()
                idx = rng.choice(n, p=probs)
                centroids[i] = vectors[idx]

        return centroids

    def build(self, max_iters: int = 20, seed: int | None = None) -> None:
        """Build the IVF index by clustering all vectors.

        Args:
            max_iters: Maximum k-means iterations.
            seed:      Random seed for initialization.
        """
        if not self._entries:
            self._built = True
            return

        vectors = np.array([e.vector for e in self._entries.values()])
        n = len(vectors)
        k = min(self.nlist, n)
        rng = np.random.default_rng(seed)

        # Normalize for cosine clustering
        vectors_norm = normalize(vectors)

        centroids = self._kmeans_init(vectors_norm, k, rng)

        for _ in range(max_iters):
            # Assign each vector to nearest centroid
            dists = np.zeros((n, k))
            for j in range(k):
                diff = vectors_norm - centroids[j]
                dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))

            assignments = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    new_centroids[j] = vectors_norm[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids

        self._centroids = centroids
        self._cluster_assignments = {i: [] for i in range(k)}

        entry_ids = list(self._entries.keys())
        for idx, vid in enumerate(entry_ids):
            cluster = assignments[idx]
            self._cluster_assignments[int(cluster)].append(
                (vid, normalize(vectors[idx]), self._entries[vid].metadata)
            )

        self._built = True

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for nearest neighbors.

        Args:
            query:   Query vector.
            top_k:   Number of results to return.
            nprobe:  Number of clusters to search (higher = better recall, slower).

        Returns:
            List of (vector_id, score, metadata) tuples.
        """
        if not self._built:
            self.build()

        query = np.asarray(query, dtype=np.float64)
        query_norm = normalize(query)

        # Find nearest centroids
        dists = np.sqrt(np.sum((self._centroids - query_norm) ** 2, axis=1))
        nearest_clusters = np.argsort(dists)[:min(nprobe, len(dists))]

        # Search within nearest clusters
        results: list[tuple[float, str, dict]] = []
        for cluster_id in nearest_clusters:
            for vid, vec_norm, meta in self._cluster_assignments.get(int(cluster_id), []):
                score = float(np.dot(query_norm, vec_norm))
                results.append((score, vid, meta))

        results.sort(key=lambda x: x[0], reverse=True)
        return [(vid, score, meta) for score, vid, meta in results[:top_k]]

    def __len__(self) -> int:
        return len(self._entries)

"""RAG (Retrieval-Augmented Generation) pipeline.

Combines chunking, embedding, vector search, and LLM generation into
an end-to-end RAG system.

Pipeline:
    1. Document ingestion: chunk text into passages
    2. Indexing: embed passages and store in a vector database
    3. Retrieval: find top-k relevant passages for a query
    4. Augmentation: insert retrieved passages into an LLM prompt
    5. Generation: produce the final answer

Provides:

    - ``Chunker``: Text chunking strategies (fixed-size, sentence-based,
      token-aware)
    - ``RAGPipeline``: End-to-end RAG orchestrator
    - ``Retriever``: Query → top-k relevant chunks
    - ``RAGPromptBuilder``: Assembles context + query into LLM prompt

Example::

    chunker = Chunker(strategy="sentence", chunk_size=200, overlap=50)
    retriever = Retriever(embedding_fn=my_embedder, vector_store=store)
    pipeline = RAGPipeline(chunker, retriever, llm_fn=my_llm)
    pipeline.ingest(documents)
    answer = pipeline.query("What is the capital of France?")
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..embeddings import normalize, tokenize

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Text chunking with multiple strategies.

    Args:
        strategy:  Chunking strategy:
            - "fixed": Fixed-size character chunks with overlap
            - "sentence": Sentence-based chunks (respects sentence boundaries)
            - "word": Word-based chunks
            - "recursive": Recursive splitting on markdown/JSON structure
        chunk_size:  Size of each chunk (chars for fixed, words for word, target tokens for recursive).
        overlap:     Overlap between consecutive chunks (same unit).
        separator:   Primary separator for recursive splitting.
    """

    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 200,
        overlap: int = 50,
        separator: str = "\n\n",
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks based on the configured strategy.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text)
        elif self.strategy == "word":
            return self._chunk_word(text)
        elif self.strategy == "recursive":
            return self._chunk_recursive(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _chunk_fixed(self, text: str) -> list[str]:
        """Fixed-size character chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            if end >= len(text):
                break
            # Step back to find a good break point
            start = end - self.overlap
        return [c for c in chunks if c]

    def _chunk_sentence(self, text: str) -> list[str]:
        """Sentence-based chunking — groups sentences into chunks of ~max_tokens."""
        # Split into sentences
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_words = 0
        target_words = self.chunk_size

        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > target_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)[-self.overlap:] if self.overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_words = len(overlap_text.split()) if overlap_text else 0

            current_chunk.append(sent)
            current_words += sent_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]

    def _chunk_word(self, text: str) -> list[str]:
        """Word-based chunking."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start = end - self.overlap
        return chunks

    def _chunk_recursive(self, text: str) -> list[str]:
        """Recursive splitting — tries separator, then sentence, then fixed."""
        # First split by separator
        parts = text.split(self.separator)
        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(self.separator) <= self.chunk_size:
                current += (self.separator + part if current else part)
            else:
                if current:
                    chunks.append(current.strip())
                # Part too large — sub-chunk it
                if len(part) > self.chunk_size:
                    sub_chunks = self._chunk_sentence(part)
                    chunks.extend(sub_chunks)
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]

    def chunk_batch(self, texts: list[str]) -> list[list[str]]:
        """Chunk a batch of texts."""
        return [self.chunk(t) for t in texts]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    text: str
    score: float
    metadata: dict[str, Any]
    doc_index: int


class Retriever:
    """Retriever that combines dense and sparse retrieval with hybrid search.

    Args:
        embedding_fn:    Function (text → np.ndarray) for dense embeddings.
        chunker:         Chunker for splitting documents.
        top_k:           Number of chunks to retrieve per query.
        alpha:           Dense score weight in hybrid fusion (0=sparse only).
        rrf_k:           RRF parameter for rank fusion.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray],
        chunker: Chunker | None = None,
        top_k: int = 5,
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.chunker = chunker or Chunker(strategy="sentence", chunk_size=200, overlap=50)
        self.top_k = top_k
        self.alpha = alpha
        self.rrf_k = rrf_k
        self._chunks: list[str] = []
        self._chunk_embeddings: list[np.ndarray] = []
        self._chunk_metadata: list[dict[str, Any]] = []
        self._tfidf_vocab: dict[str, int] = {}
        self._tfidf_idf: np.ndarray | None = None
        self._doc_vectors: np.ndarray | None = None
        self._doc_tokenized: list[list[str]] = []
        self._n_docs = 0

    def ingest(self, documents: list[str]) -> None:
        """Chunk and index a list of documents.

        Args:
            documents: List of document strings.
        """
        self._chunks = []
        self._chunk_metadata = []

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunker.chunk(doc)
            for chunk_idx, chunk in enumerate(chunks):
                self._chunks.append(chunk)
                self._chunk_metadata.append({
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "doc_length": len(doc),
                })

        # Dense embeddings
        self._chunk_embeddings = [
            normalize(self.embedding_fn(chunk)) for chunk in self._chunks
        ]

        # Sparse (TF-IDF) index
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        """Build TF-IDF vectors for all chunks."""
        tokenized = [tokenize(c) for c in self._chunks]
        self._doc_tokenized = tokenized
        self._n_docs = len(self._chunks)

        # Document frequencies
        df: Counter = Counter()
        for tokens in tokenized:
            unique = set(tokens)
            for tok in unique:
                df[tok] += 1

        # Vocabulary (sorted by frequency)
        all_terms = sorted(df.keys(), key=lambda t: df[t], reverse=True)
        self._tfidf_vocab = {t: i for i, t in enumerate(all_terms)}
        vocab_size = len(all_terms)

        # IDF
        self._tfidf_idf = np.array([
            math.log((1 + self._n_docs) / (1 + df[t])) + 1 for t in all_terms
        ])

        # Build document-term matrix
        doc_term = np.zeros((self._n_docs, vocab_size), dtype=np.float64)
        for doc_idx, tokens in enumerate(tokenized):
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self._tfidf_vocab:
                    col = self._tfidf_vocab[term]
                    doc_term[doc_idx, col] = count / len(tokens)

        tfidf = doc_term * self._tfidf_idf[np.newaxis, :]
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        self._doc_vectors = np.where(norms > 0, tfidf / norms, tfidf)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query.

        Uses hybrid search (dense + sparse with RRF).

        Args:
            query:  Query text.
            top_k:  Number of results (defaults to configured value).

        Returns:
            List of RetrievalResult, sorted by relevance.
        """
        if not self._chunks:
            raise RuntimeError("No documents ingested. Call ingest() first.")

        k = top_k or self.top_k
        n = len(self._chunks)

        # Dense scores
        q_dense = normalize(self.embedding_fn(query))
        dense_scores = np.array([
            float(np.dot(q_dense, doc_vec)) for doc_vec in self._chunk_embeddings
        ])

        # Sparse (TF-IDF) scores
        query_tokens = tokenize(query)
        q_vec = np.zeros(len(self._tfidf_vocab), dtype=np.float64)
        term_counts = Counter(query_tokens)
        for term, count in term_counts.items():
            if term in self._tfidf_vocab:
                col = self._tfidf_vocab[term]
                q_vec[col] = count / len(query_tokens)
        q_vec = q_vec * self._tfidf_idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        sparse_scores = np.array([
            float(np.dot(q_vec, dv)) for dv in self._doc_vectors
        ])

        # RRF fusion
        dense_ranks = np.argsort(-dense_scores)
        sparse_ranks = np.argsort(-sparse_scores)

        rrf_scores = np.zeros(n, dtype=np.float64)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += self.alpha / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_ranks):
            rrf_scores[idx] += (1 - self.alpha) / (self.rrf_k + rank + 1)

        top_indices = np.argsort(-rrf_scores)[:k]
        return [
            RetrievalResult(
                text=self._chunks[i],
                score=float(rrf_scores[i]),
                metadata=self._chunk_metadata[i],
                doc_index=i,
            )
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# RAGPromptBuilder
# ---------------------------------------------------------------------------

class RAGPromptBuilder:
    """Builds RAG prompts by injecting retrieved context into a template.

    Args:
        template:         Prompt template with {context} and {query} placeholders.
        max_context_tokens: Maximum tokens to include in the context window.
        separator:        String between retrieved context items.
    """

    def __init__(
        self,
        template: str = "Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:",
        max_context_tokens: int = 3000,
        separator: str = "\n\n",
    ) -> None:
        self.template = template
        self.max_context_tokens = max_context_tokens
        self.separator = separator

    def build(
        self,
        query: str,
        retrieved: list[RetrievalResult],
    ) -> str:
        """Build a RAG prompt from retrieved results.

        Args:
            query:     The user's question.
            retrieved: List of retrieved context results.

        Returns:
            Formatted prompt string ready for LLM.
        """
        context_parts = []
        token_count = 0

        for result in retrieved:
            text = result.text.strip()
            # Rough token estimate (4 chars per token)
            text_tokens = len(text) // 4
            if token_count + text_tokens > self.max_context_tokens:
                break
            context_parts.append(f"[{result.score:.3f}] {text}")
            token_count += text_tokens

        context = self.separator.join(context_parts)
        return self.template.replace("{context}", context).replace("{query}", query)


# ---------------------------------------------------------------------------
# RAGPipeline — end-to-end orchestrator
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline: ingest → retrieve → augment → generate.

    Args:
        retriever:      The Retriever instance.
        prompt_builder: The RAGPromptBuilder instance.
        llm_fn:         Function (prompt: str) → response: str.
        chunker:        Chunker for the retriever (if retriever doesn't have one).

    Example::

        pipeline = RAGPipeline(retriever, prompt_builder, llm_fn=my_llm)
        pipeline.ingest(["The capital of France is Paris.", ...])
        answer = pipeline.query("What is the capital of France?")
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: RAGPromptBuilder | None = None,
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.llm_fn = llm_fn

    def ingest(self, documents: list[str]) -> None:
        """Ingest documents into the retriever's index."""
        self.retriever.ingest(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant context for a query."""
        return self.retriever.retrieve(query, top_k)

    def augment(self, query: str, retrieved: list[RetrievalResult]) -> str:
        """Build the RAG prompt from query and retrieved context."""
        return self.prompt_builder.build(query, retrieved)

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM. Must have an llm_fn set."""
        if self.llm_fn is None:
            raise RuntimeError("No LLM function set. Pass llm_fn to RAGPipeline.")
        return self.llm_fn(prompt)

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Run the full RAG pipeline for a question.

        Args:
            question: The user's question.
            top_k:    Number of documents to retrieve.

        Returns:
            Dict with:
                - "question": The input question
                - "retrieved": List of retrieved context items
                - "prompt": The generated RAG prompt
                - "answer": The LLM response
        """
        retrieved = self.retrieve(question, top_k)
        prompt = self.augment(question, retrieved)
        answer = self.generate(prompt) if self.llm_fn else ""
        return {
            "question": question,
            "retrieved": [{"text": r.text, "score": r.score} for r in retrieved],
            "prompt": prompt,
            "answer": answer,
        }

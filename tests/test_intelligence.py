"""Tests for the intelligence/ package.

Covers all 13 intelligence submodules that map to the 16-step Gen AI
engineering roadmap:

  - intelligence.transformers  (Steps 01-02: Python, Attention, Transformers, BPE)
  - intelligence.embeddings    (Steps 05-06: Embeddings, cosine similarity, hashing)
  - intelligence.vector_db     (Steps 05-06: VectorStore, HybridSearch, VectorIndex)
  - intelligence.mcp           (Steps 07-09: MCPServer, MCPClient, tools/resources/prompts)
  - intelligence.peft          (Steps 10-11: LoRA, LoRAAdapter, LoRATrainer)
  - intelligence.prompt        (Step 03: Prompt templates, CoT, structured output)
  - intelligence.rag           (Step 05-06: Chunking, Retriever, RAGPipeline)
  - intelligence.multimodal    (Step 10-11: Vision/Audio encoders)
  - intelligence.observability (Steps 14-16: Tracing, Metrics, Guardrails, RedTeam)

All modules are stdlib + numpy only — no torch at import time.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# Transformers module — Steps 01-02 (Attention, Transformer, BPE, Positional Encoding)
# ===========================================================================


class TestSoftmax:
    def test_softmax_basic(self):
        from intelligence.transformers.attention import softmax

        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert abs(result.sum() - 1.0) < 1e-10
        assert result[2] > result[0]

    def test_softmax_temperature(self):
        from intelligence.transformers.attention import softmax

        x = np.array([1.0, 2.0, 3.0])
        soft = softmax(x, temperature=10.0)
        hard = softmax(x, temperature=0.01)
        # High temperature → more uniform
        assert abs(soft[0] - soft[2]) < abs(hard[0] - hard[2])

    def test_softmax_2d(self):
        from intelligence.transformers.attention import softmax

        x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        result = softmax(x, axis=1)
        assert result.shape == (2, 3)
        for row in result:
            assert abs(row.sum() - 1.0) < 1e-10

    def test_softmax_3d(self):
        from intelligence.transformers.attention import softmax

        x = np.random.randn(2, 3, 4)
        result = softmax(x, axis=-1)
        assert result.shape == (2, 3, 4)
        for mat in result:
            for row in mat:
                assert abs(row.sum() - 1.0) < 1e-10


class TestScaledDotProductAttention:
    def test_attention_shapes(self):
        from intelligence.transformers.attention import scaled_dot_product_attention

        seq_len = 4
        d_k = 8
        q = np.random.randn(seq_len, d_k)
        k = np.random.randn(seq_len, d_k)
        v = np.random.randn(seq_len, d_k)
        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (seq_len, d_k)
        assert weights.shape == (seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        from intelligence.transformers.attention import scaled_dot_product_attention

        q = np.random.randn(3, 8)
        k = np.random.randn(3, 8)
        v = np.random.randn(3, 8)
        _, weights = scaled_dot_product_attention(q, k, v)
        for row in weights:
            assert abs(row.sum() - 1.0) < 1e-10

    def test_attention_with_mask(self):
        from intelligence.transformers.attention import scaled_dot_product_attention

        q = np.random.randn(4, 8)
        k = np.random.randn(4, 8)
        v = np.random.randn(4, 8)
        mask = np.triu(np.full((4, 4), -1e9), k=1)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        # Upper triangle (future positions) should have ~0 attention
        for i in range(4):
            for j in range(i + 1, 4):
                assert weights[i, j] < 1e-6

    def test_attention_batch_dims(self):
        from intelligence.transformers.attention import scaled_dot_product_attention

        q = np.random.randn(2, 3, 8)  # (batch, seq, d_k)
        k = np.random.randn(2, 3, 8)
        v = np.random.randn(2, 3, 8)
        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (2, 3, 8)
        assert weights.shape == (2, 3, 3)


class TestMultiHeadAttention:
    def test_mha_output_shape(self):
        from intelligence.transformers.attention import MultiHeadAttention

        d_model = 64
        num_heads = 8
        seq_len = 10
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
        x = np.random.randn(seq_len, d_model)
        output, weights = mha(x, x, x)
        assert output.shape == (seq_len, d_model)
        assert weights.shape == (num_heads, seq_len, seq_len)

    def test_mha_invalid_d_model(self):
        from intelligence.transformers.attention import MultiHeadAttention

        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=65, num_heads=8)

    def test_mha_different_qkv_lengths(self):
        from intelligence.transformers.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=64, num_heads=8, seed=42)
        q = np.random.randn(5, 64)
        k = np.random.randn(10, 64)
        v = np.random.randn(10, 64)
        output, weights = mha(q, k, v)
        assert output.shape == (5, 64)
        assert weights.shape == (8, 5, 10)

    def test_mha_with_mask(self):
        from intelligence.transformers.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
        mha.training = False
        x = np.random.randn(6, 32)
        mask = np.zeros((6, 6))
        mask[0, 3:] = -1e9
        output, weights = mha.forward(x, x, x, mask=mask)
        assert output.shape == (6, 32)

    def test_mha_forward_method(self):
        from intelligence.transformers.attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(5, 32)
        output, weights = mha.forward(x, x, x)
        assert output.shape == (5, 32)
        assert weights.shape == (4, 5, 5)


class TestMaskedMultiHeadAttention:
    def test_causal_mask(self):
        from intelligence.transformers.attention import MaskedMultiHeadAttention

        mha = MaskedMultiHeadAttention(d_model=64, num_heads=8, seed=42)
        mha.training = False
        x = np.random.randn(6, 64)
        output, weights = mha(x, x, x)
        assert output.shape == (6, 64)
        # Check causality: no attention to future positions
        for h in range(8):
            for i in range(6):
                for j in range(i + 1, 6):
                    assert weights[h, i, j] < 1e-6

    def test_masked_mha_output_shape(self):
        from intelligence.transformers.attention import MaskedMultiHeadAttention

        mha = MaskedMultiHeadAttention(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(5, 32)
        output, weights = mha.forward(x, x, x)
        assert output.shape == (5, 32)


class TestPositionalEncoding:
    def test_sinusoidal_shape(self):
        from intelligence.transformers.positional_encoding import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        # The internal _encoding attribute holds (max_len, d_model)
        assert pe._encoding.shape == (100, 64)

    def test_sinusoidal_call(self):
        from intelligence.transformers.positional_encoding import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=4, max_len=10)
        positions = np.array([0, 1, 2, 3])
        result = pe(positions)
        assert result.shape == (4, 4)

    def test_sinusoidal_values(self):
        from intelligence.transformers.positional_encoding import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=4, max_len=10)
        enc = pe._encoding
        # Position 0: sin(0)=0 for even indices, cos(0)=1 for odd indices
        assert abs(enc[0, 0]) < 1e-10  # sin(0) = 0
        assert abs(enc[0, 1] - 1.0) < 1e-10  # cos(0) = 1

    def test_learned_shape(self):
        from intelligence.transformers.positional_encoding import LearnedPositionalEncoding

        pe = LearnedPositionalEncoding(d_model=64, max_len=100, seed=42)
        assert pe._encoding.shape == (100, 64)

    def test_learned_get_all(self):
        from intelligence.transformers.positional_encoding import LearnedPositionalEncoding

        pe = LearnedPositionalEncoding(d_model=32, max_len=50, seed=42)
        all_enc = pe.get_all()
        assert all_enc.shape == (50, 32)

    def test_sinusoidal_extrapolate_error(self):
        from intelligence.transformers.positional_encoding import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=4, max_len=5)
        with pytest.raises(ValueError, match="exceeds max_len"):
            pe(np.array([10]))

    def test_positional_encoding_invalid_dims(self):
        from intelligence.transformers.positional_encoding import SinusoidalPositionalEncoding

        with pytest.raises(ValueError):
            SinusoidalPositionalEncoding(d_model=0)
        with pytest.raises(ValueError):
            SinusoidalPositionalEncoding(d_model=4, max_len=0)


class TestEncoderLayer:
    def test_encoder_layer_output_shape(self):
        from intelligence.transformers.model import TransformerEncoderLayer

        layer = TransformerEncoderLayer(d_model=64, num_heads=8, d_ff=128, seed=42)
        layer.set_training(False)
        x = np.random.randn(10, 64)
        output, weights = layer.forward(x)
        assert output.shape == (10, 64)
        assert weights.shape == (8, 10, 10)

    def test_encoder_layer_with_mask(self):
        from intelligence.transformers.model import TransformerEncoderLayer

        layer = TransformerEncoderLayer(d_model=64, num_heads=8, d_ff=128, seed=42)
        layer.set_training(False)
        x = np.random.randn(10, 64)
        mask = np.zeros((10, 10))
        output, weights = layer.forward(x, src_mask=mask)
        assert output.shape == (10, 64)


class TestDecoderLayer:
    def test_decoder_layer_output_shape(self):
        from intelligence.transformers.model import TransformerDecoderLayer

        layer = TransformerDecoderLayer(d_model=64, num_heads=8, d_ff=128, seed=42)
        layer.set_training(False)
        x = np.random.randn(10, 64)
        enc = np.random.randn(20, 64)
        output, self_weights, cross_weights = layer.forward(x, enc)
        assert output.shape == (10, 64)
        assert self_weights.shape == (8, 10, 10)
        assert cross_weights.shape == (8, 10, 20)


class TestTransformerLayers:
    def test_transformer_encoder_output_shape(self):
        from intelligence.transformers.model import TransformerEncoder

        enc = TransformerEncoder(
            num_layers=2, d_model=64, num_heads=8, d_ff=128, seed=42
        )
        enc.set_training(False)
        x = np.random.randn(10, 64)
        output, weights_list = enc.forward(x)
        assert output.shape == (10, 64)
        assert len(weights_list) == 2  # one per layer

    def test_transformer_decoder_output_shape(self):
        from intelligence.transformers.model import TransformerDecoder

        dec = TransformerDecoder(
            num_layers=2, d_model=64, num_heads=8, d_ff=128, seed=42
        )
        dec.set_training(False)
        x = np.random.randn(8, 64)
        enc_out = np.random.randn(10, 64)
        output, self_weights, cross_weights = dec.forward(x, enc_out)
        assert output.shape == (8, 64)
        assert len(self_weights) == 2

    def test_transformer_full_forward(self):
        from intelligence.transformers.model import Transformer

        model = Transformer(
            vocab_size=100, d_model=64, num_encoder_layers=2,
            num_decoder_layers=2, num_heads=8, d_ff=128,
            max_seq_len=32, dropout_rate=0.0, seed=42,
        )
        model.set_training(False)
        src = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        tgt = np.array([10, 11, 12, 13])
        out, enc_w, dec_sw, dec_cw = model.forward(src, tgt)
        assert out.shape == (4, 100)

    def test_transformer_encode_decode(self):
        from intelligence.transformers.model import Transformer

        model = Transformer(
            vocab_size=100, d_model=32, num_encoder_layers=1,
            num_decoder_layers=1, num_heads=4, d_ff=64,
            max_seq_len=32, seed=42,
        )
        model.set_training(False)
        src = np.array([1, 2, 3, 4])
        tgt = np.array([5, 6, 7, 8])
        enc_out, enc_w = model.encode(src)
        assert enc_out.shape == (4, 32)
        dec_out, dec_sw, dec_cw = model.decode(tgt, enc_out)
        assert dec_out.shape == (4, 32)
        projected = model.project(dec_out)
        assert projected.shape == (4, 100)

    def test_transformer_set_training(self):
        from intelligence.transformers.model import Transformer

        model = Transformer(
            vocab_size=50, d_model=16, num_encoder_layers=1,
            num_decoder_layers=1, num_heads=4, d_ff=32, seed=42,
        )
        model.set_training(True)
        assert model.encoder.layers[0].self_attn.training is True
        model.set_training(False)
        assert model.encoder.layers[0].self_attn.training is False


class TestBPETokenizer:
    def test_train_and_encode(self):
        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        tokenizer.train("hello world hello world test test test")
        ids = tokenizer.encode("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self):
        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=60, min_frequency=1)
        text = "the quick brown fox jumps over the lazy dog"
        tokenizer.train(text)
        ids = tokenizer.encode("the quick brown")
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_vocab_size(self):
        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=80, min_frequency=1)
        tokenizer.train("a b c a b a a a b c b b b c")
        assert tokenizer.actual_vocab_size > 0
        assert len(tokenizer) == tokenizer.actual_vocab_size

    def test_encode_batch(self):
        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=50, min_frequency=1)
        tokenizer.train("hello world foo bar hello world")
        batch = tokenizer.encode_batch(["hello world", "foo bar hello"])
        assert len(batch) == 2
        # All sequences should be padded to same length
        assert len(batch[0]) == len(batch[1])

    def test_roundtrip(self):
        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        tokenizer.train("the quick brown fox jumps over the lazy dog")
        text = "the quick brown fox"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_get_stats_returns_counter(self):
        from collections import Counter

        from intelligence.transformers.bpe_tokenizer import BPETokenizer

        vocab = {"hello": ["he", "llo"], "world": ["wo", "rld"]}
        stats = BPETokenizer._get_stats(vocab)
        assert isinstance(stats, Counter)


# ===========================================================================
# Embeddings module — Steps 05-06 (Embeddings, cosine similarity, hashing)
# ===========================================================================


class TestTokenize:
    def test_tokenize_basic(self):
        from intelligence.embeddings import tokenize

        tokens = tokenize("Hello, world!")
        assert len(tokens) >= 2
        assert isinstance(tokens, list)

    def test_tokenize_empty(self):
        from intelligence.embeddings import tokenize

        tokens = tokenize("")
        assert tokens == []

    def test_tokenize_lowercase(self):
        from intelligence.embeddings import tokenize

        tokens = tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from intelligence.embeddings import cosine_similarity

        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        from intelligence.embeddings import cosine_similarity

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-10

    def test_opposite_vectors(self):
        from intelligence.embeddings import cosine_similarity

        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-10

    def test_cosine_similarity_range(self):
        from intelligence.embeddings import cosine_similarity

        v1 = np.random.randn(100)
        v2 = np.random.randn(100)
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0


class TestNormalize:
    def test_normalize_unit_length(self):
        from intelligence.embeddings import normalize

        v = np.array([3.0, 4.0])
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_normalize_zero_vector(self):
        from intelligence.embeddings import normalize

        v = np.zeros(5)
        n = normalize(v)
        assert np.all(n == 0.0)

    def test_normalize_already_unit(self):
        from intelligence.embeddings import normalize

        v = np.array([1.0, 0.0, 0.0])
        n = normalize(v)
        assert np.allclose(n, v)


class TestTFIDFEmbedder:
    def test_fit_transform(self):
        from intelligence.embeddings import TFIDFEmbedder

        docs = ["hello world foo", "world bar baz", "foo bar"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        vec = embedder.embed("hello world")
        assert embedder.vector_dim > 0
        assert vec.shape == (embedder.vector_dim,)

    def test_search(self):
        from intelligence.embeddings import TFIDFEmbedder

        docs = ["apple banana", "banana cherry", "apple cherry"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        results = embedder.search("banana", top_k=2)
        assert len(results) == 2
        for doc_idx, score in results:
            assert isinstance(doc_idx, int)
            assert isinstance(score, float)
            assert 0 <= doc_idx < len(docs)

    def test_search_returns_scores_descending(self):
        from intelligence.embeddings import TFIDFEmbedder

        docs = ["apple banana", "banana cherry", "apple cherry"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        results = embedder.search("apple", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_vector_dim(self):
        from intelligence.embeddings import TFIDFEmbedder

        docs = ["hello world", "foo bar"]
        embedder = TFIDFEmbedder()
        embedder.fit(docs)
        assert isinstance(embedder.vector_dim, int)
        assert embedder.vector_dim > 0


class TestWordEmbedding:
    def test_word_embedding_shape(self):
        from intelligence.embeddings import WordEmbedding

        emb = WordEmbedding(vocab_size=100, dim=50, seed=42)
        vec = emb.forward(np.array([1, 2, 3]))
        assert vec.shape == (3, 50)

    def test_word_embedding_similarity(self):
        from intelligence.embeddings import WordEmbedding

        emb = WordEmbedding(vocab_size=50, dim=32, seed=42)
        sim = emb.similarity(0, 1)
        assert -1.0 <= sim <= 1.0

    def test_word_embedding_analogy(self):
        from intelligence.embeddings import WordEmbedding

        emb = WordEmbedding(vocab_size=50, dim=32, seed=42)
        results = emb.analogy(0, 1, 2, top_k=3)
        assert len(results) <= 3
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestSentenceEmbedding:
    def test_sentence_embedding_shape(self):
        from intelligence.embeddings import SentenceEmbedding

        vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        emb = SentenceEmbedding(vocab=vocab, dim=64, seed=42)
        vec = emb.embed("hello world")
        assert vec.shape == (64,)

    def test_sentence_embedding_consistency(self):
        from intelligence.embeddings import SentenceEmbedding

        vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        emb = SentenceEmbedding(vocab=vocab, dim=64, seed=42)
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        assert np.allclose(v1, v2)

    def test_sentence_embedding_similarity(self):
        from intelligence.embeddings import SentenceEmbedding, cosine_similarity

        vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3,
                 "machine": 4, "learning": 5, "cooking": 6, "recipes": 7}
        emb = SentenceEmbedding(vocab=vocab, dim=64, seed=42)
        v1 = emb.embed("hello world")
        v2 = emb.embed("foo bar")
        sim = emb.similarity("hello world", "foo bar")
        assert -1.0 <= sim <= 1.0
        assert abs(sim - cosine_similarity(v1, v2)) < 1e-10

    def test_sentence_embedding_unknown_words(self):
        from intelligence.embeddings import SentenceEmbedding

        vocab = {"hello": 0, "world": 1}
        emb = SentenceEmbedding(vocab=vocab, dim=32, seed=42)
        # Unknown words should not crash
        vec = emb.embed("unknown words here")
        assert vec.shape == (32,)


class TestHashVectorize:
    def test_hash_vectorize(self):
        from intelligence.embeddings import hash_vectorize

        vec = hash_vectorize("hello world", num_features=128)
        assert vec.shape == (128,)
        assert np.linalg.norm(vec) > 0

    def test_hash_vectorize_different_texts(self):
        from intelligence.embeddings import hash_vectorize

        v1 = hash_vectorize("hello world", num_features=64)
        v2 = hash_vectorize("goodbye universe", num_features=64)
        assert v1.shape == (64,)
        assert v2.shape == (64,)

    def test_hash_vectorize_deterministic(self):
        from intelligence.embeddings import hash_vectorize

        v1 = hash_vectorize("hello world", num_features=64)
        v2 = hash_vectorize("hello world", num_features=64)
        assert np.allclose(v1, v2)

    def test_hash_vectorize_ngram_range(self):
        from intelligence.embeddings import hash_vectorize

        vec = hash_vectorize("hello world", num_features=32, ngram_range=(1, 3))
        assert vec.shape == (32,)


# ===========================================================================
# Vector DB module — Steps 05-06 (VectorStore, HybridSearch, VectorIndex)
# ===========================================================================


class TestVectorStore:
    def test_add_and_search(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=4)
        store.add("vec1", np.array([1.0, 0.0, 0.0, 0.0]), {"text": "a"})
        store.add("vec2", np.array([0.0, 1.0, 0.0, 0.0]), {"text": "b"})

        results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert len(results) == 1
        assert results[0][0] == "vec1"
        assert results[0][1] > 0.99  # high cosine similarity

    def test_delete(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=4)
        store.add("v1", np.array([1.0, 0.0, 0.0, 0.0]))
        assert store.delete("v1") is True
        assert store.delete("nonexistent") is False
        assert len(store) == 0

    def test_euclidean_distance(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=2)
        store.add("a", np.array([0.0, 0.0]))
        store.add("b", np.array([3.0, 4.0]))
        results = store.search(np.array([0.0, 0.0]), top_k=1, metric="euclidean")
        assert results[0][0] == "a"

    def test_dim_mismatch(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=4)
        with pytest.raises(ValueError):
            store.add("bad", np.array([1.0, 0.0]))

    def test_get_and_ids(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=3)
        store.add("v1", np.array([1.0, 0.0, 0.0]), metadata={"x": 1})
        entry = store.get("v1")
        assert entry is not None
        assert entry.id == "v1"
        assert entry.metadata == {"x": 1}
        assert "v1" in store.ids

    def test_add_batch(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=4)
        ids = store.add_batch([
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ])
        assert len(ids) == 2
        assert len(store) == 2

    def test_search_cosine(self):
        from intelligence.vector_db import VectorStore

        store = VectorStore(dim=3)
        store.add("a", np.array([1.0, 0.0, 0.0]))
        store.add("b", np.array([0.0, 1.0, 0.0]))
        store.add("c", np.array([1.0, 1.0, 0.0]))
        results = store.search(np.array([1.0, 0.0, 0.0]), top_k=2, metric="cosine")
        assert len(results) == 2
        # Most similar should be "c" (has the same direction component)
        # Actually "a" is identical so should be first
        assert results[0][0] == "a"


class TestHybridSearch:
    def test_fit_and_search(self):
        from intelligence.vector_db import HybridSearch

        docs = [
            "The cat sat on the mat",
            "Dogs are loyal animals",
            "The quick brown fox jumps",
        ]
        search = HybridSearch(
            embed_fn=lambda t: np.random.RandomState(42).randn(8),
            alpha=0.5,
        )
        search.fit(docs)
        results = search.search("cat and dog", top_k=2)
        assert len(results) == 2
        for doc_idx, score, text in results:
            assert isinstance(doc_idx, int)
            assert isinstance(score, float)
            assert isinstance(text, str)

    def test_not_fitted(self):
        from intelligence.vector_db import HybridSearch

        search = HybridSearch(embed_fn=lambda t: np.zeros(8))
        with pytest.raises(RuntimeError):
            search.search("test")

    def test_search_default_top_k(self):
        from intelligence.vector_db import HybridSearch

        docs = ["alpha beta", "gamma delta", "epsilon zeta"]
        search = HybridSearch(
            embed_fn=lambda t: np.random.RandomState(42).randn(4),
            alpha=0.5,
            top_k=3,
        )
        search.fit(docs)
        results = search.search("alpha gamma")
        assert len(results) == 3

    def test_search_returns_doc_text(self):
        from intelligence.vector_db import HybridSearch

        docs = ["first document", "second document", "third document"]
        search = HybridSearch(
            embed_fn=lambda t: np.ones(4),
            alpha=0.5,
        )
        search.fit(docs)
        results = search.search("first", top_k=1)
        assert len(results) == 1
        doc_idx, score, text = results[0]
        assert text in docs

    def test_sparse_scores(self):
        from intelligence.vector_db import HybridSearch

        docs = ["hello world", "foo bar", "hello foo"]
        search = HybridSearch(
            embed_fn=lambda t: np.random.RandomState(42).randn(4),
            alpha=0.5,
            top_k=3,
        )
        search.fit(docs)
        # sparse scores for a query should return dict of doc_idx → score
        sparse = search._sparse_scores("hello")
        assert isinstance(sparse, dict)
        assert len(sparse) == 3  # all docs


class TestVectorIndex:
    def test_build_and_search(self):
        from intelligence.vector_db import VectorIndex

        idx = VectorIndex(dim=4, nlist=3)
        rng = np.random.default_rng(42)
        for i in range(10):
            vec = rng.standard_normal(4)
            idx.add(f"v{i}", vec)
        idx.build(seed=42)
        results = idx.search(rng.standard_normal(4), top_k=3)
        assert len(results) <= 3
        for _vid, _score, _meta in results:
            assert isinstance(_score, float)

    def test_empty_index(self):
        from intelligence.vector_db import VectorIndex

        idx = VectorIndex(dim=4, nlist=5)
        idx.build()
        assert len(idx) == 0

    def test_index_len(self):
        from intelligence.vector_db import VectorIndex

        idx = VectorIndex(dim=4, nlist=5)
        idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
        idx.add("b", np.array([0.0, 1.0, 0.0, 0.0]))
        assert len(idx) == 2

    def test_index_search_returns_tuples(self):
        from intelligence.vector_db import VectorIndex

        idx = VectorIndex(dim=8, nlist=2)
        rng = np.random.default_rng(42)
        for i in range(5):
            idx.add(f"v{i}", rng.standard_normal(8))
        idx.build(seed=42)
        query = rng.standard_normal(8)
        results = idx.search(query, top_k=3)
        assert len(results) == 3
        for vid, score, meta in results:
            assert isinstance(vid, str)
            assert isinstance(score, float)
            assert isinstance(meta, dict)


# ===========================================================================
# MCP module — Steps 07-09 (MCPServer, MCPClient, tools/resources/prompts)
# ===========================================================================


class TestMCPServer:
    def test_add_tool(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(args: dict) -> dict:
            return {"result": args["a"] + args["b"]}

        server.add_tool("add", "Add two numbers", handler=handler)
        assert "add" in [t.name for t in server.tools]

    def test_add_resource(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler() -> str:
            return "hello world"

        server.add_resource("test://greeting", "greeting", handler=handler)
        assert "test://greeting" in [r.uri for r in server.resources]

    def test_add_prompt(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(args: dict) -> str:
            return f"Summarize: {args['text']}"

        server.add_prompt(
            "summarize", "Summarize text",
            arguments=[{"name": "text", "description": "text to summarize", "required": True}],
            handler=handler,
        )
        assert "summarize" in [p.name for p in server.prompts]

    def test_call_tool(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(a: int, b: int) -> dict:
            return {"result": a + b}

        server.add_tool("add", "Add two numbers", handler=handler)
        result = server.call_tool("add", arguments={"a": 2, "b": 3})
        assert result["isError"] is False
        assert result["content"][0]["json"]["result"] == 5

    def test_call_tool_not_found(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")
        result = server.call_tool("nonexistent", arguments={})
        assert result["isError"] is True

    def test_read_resource(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(uri: str) -> str:
            return "hello world"

        server.add_resource("test://greeting", "greeting", handler=handler)
        result = server.read_resource("test://greeting")
        assert result["isError"] is False
        assert "hello" in result["contents"][0]["text"]

    def test_render_prompt(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(args: dict) -> str:
            return f"Summarize: {args['text']}"

        server.add_prompt(
            "summarize", "Summarize text",
            arguments=[{"name": "text", "description": "text", "required": True}],
            handler=handler,
        )
        result = server.render_prompt("summarize", arguments={"text": "hello"})
        assert "hello" in str(result)

    def test_list_tools(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")
        server.add_tool("tool1", "desc1", handler=lambda a: {"ok": True})
        server.add_tool("tool2", "desc2", handler=lambda a: {"ok": True})
        tools = server.list_tools()
        assert len(tools) == 2

    def test_list_resources(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")
        server.add_resource("res1://a", "a1", handler=lambda: "data1")
        server.add_resource("res2://b", "b2", handler=lambda: "data2")
        resources = server.list_resources()
        assert len(resources) == 2

    def test_list_prompts(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")
        server.add_prompt("p1", "d1", handler=lambda a: "result1")
        server.add_prompt("p2", "d2", handler=lambda a: "result2")
        prompts = server.list_prompts()
        assert len(prompts) == 2

    def test_handle_request_tool(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")

        def handler(text: str) -> dict:
            return {"echo": text}

        server.add_tool("echo", "echo tool", handler=handler)
        # Initialize first
        server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        }
        result = server.handle_request(request)
        assert result["result"]["isError"] is False
        assert result["result"]["content"][0]["json"]["echo"] == "hello"

    def test_handle_request_resource(self):
        from intelligence.mcp import MCPServer

        server = MCPServer("test_server")
        server.add_resource("static://data", "data", handler=lambda uri: "static data")
        # Initialize first
        server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "static://data"},
        }
        result = server.handle_request(request)
        assert result["result"]["isError"] is False
        assert "static data" in result["result"]["contents"][0]["text"]


class TestMCPClient:
    def test_connect_in_process(self):
        from intelligence.mcp import MCPClient, MCPServer

        server = MCPServer("test_server")

        def handler(args: dict) -> dict:
            return {"result": args["a"] + args["b"]}

        server.add_tool("add", "Add", handler=handler)

        client = MCPClient()
        client.connect_in_process(server)
        tools = client.list_tools()
        assert len(tools) >= 1

    def test_client_call_tool(self):
        from intelligence.mcp import MCPClient, MCPServer

        server = MCPServer("test_server")

        def handler(a: int, b: int) -> dict:
            return {"result": a + b}

        server.add_tool("add", "Add", handler=handler)

        client = MCPClient()
        client.connect_in_process(server)
        result = client.call_tool("add", arguments={"a": 1, "b": 2})
        assert result["isError"] is False
        assert result["content"][0]["json"]["result"] == 3

    def test_client_read_resource(self):
        from intelligence.mcp import MCPClient, MCPServer

        server = MCPServer("test_server")
        server.add_resource("test://data", "data", handler=lambda: "hello from resource")
        client = MCPClient()
        client.connect_in_process(server)
        resources = client.list_resources()
        assert len(resources) >= 1

    def test_client_list_tools(self):
        from intelligence.mcp import MCPClient, MCPServer

        server = MCPServer("test_server")
        server.add_tool("tool1", "desc1", handler=lambda a: {"ok": True})
        server.add_tool("tool2", "desc2", handler=lambda a: {"ok": True})
        client = MCPClient()
        client.connect_in_process(server)
        tools = client.list_tools()
        assert len(tools) == 2

    def test_client_disconnect(self):
        from intelligence.mcp import MCPClient, MCPServer

        server = MCPServer("test_server")
        server.add_tool("test", "test", handler=lambda a: {"ok": True})
        client = MCPClient()
        client.connect_in_process(server)
        client.disconnect()
        # Should not raise after disconnect
        assert not client._connected


class TestMCPTool:
    def test_mcp_tool_fields(self):
        from intelligence.mcp import MCPTool

        def handler(args: dict) -> dict:
            return {"result": 42}

        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=handler,
            annotations={},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert callable(tool.handler)


class TestMCPResource:
    def test_mcp_resource_fields(self):
        from intelligence.mcp import MCPResource

        def handler() -> str:
            return "data"

        resource = MCPResource(
            uri="test://data",
            name="data",
            description="a data resource",
            mime_type="text/plain",
            handler=handler,
        )
        assert resource.uri == "test://data"
        assert resource.name == "data"
        assert callable(resource.handler)


class TestMCPPrompt:
    def test_mcp_prompt_fields(self):
        from intelligence.mcp import MCPPrompt

        def handler(args: dict) -> str:
            return "result"

        prompt = MCPPrompt(
            name="test_prompt",
            description="a test prompt",
            arguments=[{"name": "input", "description": "input text", "required": True}],
            handler=handler,
        )
        assert prompt.name == "test_prompt"
        assert prompt.description == "a test prompt"
        assert callable(prompt.handler)


# ===========================================================================
# PEFT module — Step 10-11 (LoRA, LoRAAdapter, LoRATrainer)
# ===========================================================================


class TestLoRALayer:
    def test_init_shapes(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=16, d_out=8, rank=4, alpha=1.0, seed=42)
        assert layer.A.shape == (4, 16)  # (rank, d_in)
        assert layer.B.shape == (8, 4)  # (d_out, rank)

    def test_init_zeros(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=16, d_out=8, rank=4, alpha=1.0, init="zeros", seed=42)
        assert np.all(layer.A == 0.0)
        assert np.all(layer.B == 0.0)

    def test_forward_output_shape(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=16, d_out=8, rank=4, alpha=2.0, seed=42)
        x = np.random.randn(5, 16)
        base_out = np.random.randn(5, 8)
        result = layer.forward(base_out, x, training=False)
        assert result.shape == (5, 8)

    def test_forward_no_change_when_b_zero(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=16, d_out=8, rank=4, alpha=2.0, init="zeros", seed=42)
        x = np.random.randn(5, 16)
        base_out = np.random.randn(5, 8)
        result = layer.forward(base_out, x, training=False)
        assert np.allclose(result, base_out)

    def test_rank_too_large(self):
        from intelligence.peft import LoRALayer

        with pytest.raises(ValueError):
            LoRALayer(d_in=4, d_out=4, rank=8)

    def test_merge_unmerge(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=8, d_out=4, rank=2, alpha=2.0, seed=42)
        base_weights = np.random.randn(4, 8)
        merged = layer.merge(base_weights)
        assert merged.shape == (4, 8)
        unmerged = layer.unmerge(merged)
        assert np.allclose(unmerged, base_weights)

    def test_scaling(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=8, d_out=4, rank=2, alpha=4.0, seed=42)
        assert layer.scaling == 4.0 / 2  # alpha / rank

    def test_update(self):
        from intelligence.peft import LoRALayer

        layer = LoRALayer(d_in=8, d_out=4, rank=2, alpha=1.0, seed=42)
        grad_A = np.random.randn(2, 8)
        grad_B = np.random.randn(4, 2)
        layer.update(grad_A, grad_B, lr=0.01)
        assert layer.A.shape == (2, 8)
        assert layer.B.shape == (4, 2)


class TestLoRAAdapter:
    def test_init_shapes(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q", "v"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        assert adapter.count_base_parameters() == 16 * 8

    def test_add_adapter(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q", "v"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        layer = adapter.add_adapter("lora1", rank=4, alpha=2.0, seed=42)
        assert adapter.get_adapter("lora1") is layer

    def test_forward(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, alpha=2.0, seed=42)
        x = np.random.randn(5, 8)
        result = adapter.forward(x, adapter_names=["lora1"], return_attention=True)
        assert result[0].shape == (5, 16)

    def test_forward_no_adapters(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        x = np.random.randn(5, 8)
        result = adapter.forward(x)
        assert result.shape == (5, 16)

    def test_count_parameters(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q", "v"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, seed=42)
        assert adapter.count_parameters() > 0
        assert adapter.count_base_parameters() == 16 * 8

    def test_parameter_ratio(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, seed=42)
        ratio = adapter.parameter_ratio
        assert 0.0 < ratio < 1.0

    def test_merge_all(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, alpha=2.0, seed=42)
        merged = adapter.merge_all()
        assert merged.shape == (16, 8)


class TestLoRATrainer:
    def test_init(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig, LoRATrainer

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, alpha=2.0, seed=42)
        trainer = LoRATrainer(adapter, lr=0.01, momentum=0.9, weight_decay=0.001)
        assert trainer.lr == 0.01

    def test_train_step(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig, LoRATrainer

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(16, 8)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=4, alpha=2.0, seed=42)
        trainer = LoRATrainer(adapter, lr=0.01)
        x = np.random.randn(4, 8)
        target = np.random.randn(4, 16)
        loss = trainer.train_step(x, target, "lora1")
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train(self):
        from intelligence.peft import LoRAAdapter, LoRAConfig, LoRATrainer

        config = LoRAConfig(r=4, alpha=1.0, dropout=0.0, target_modules=["q"])
        base_weights = np.random.randn(8, 4)
        adapter = LoRAAdapter(base_weights=base_weights, config=config)
        adapter.add_adapter("lora1", rank=2, alpha=2.0, seed=42)
        trainer = LoRATrainer(adapter, lr=0.01)
        train_data = [(np.random.randn(4, 4), np.random.randn(4, 8)) for _ in range(3)]
        history = trainer.train(train_data, "lora1", epochs=3)
        assert isinstance(history, list)
        assert len(history) == 3
        assert all(isinstance(h, float) for h in history)

    def test_lora_config_fields(self):
        from intelligence.peft import LoRAConfig

        config = LoRAConfig(r=8, alpha=16.0, dropout=0.1, target_modules=["q", "v", "k"])
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q", "v", "k"]


class TestPEFTFunctions:
    def test_merge_lora(self):
        from intelligence.peft import LoRALayer, merge_lora

        layer1 = LoRALayer(d_in=8, d_out=4, rank=2, alpha=1.0, seed=42)
        layer2 = LoRALayer(d_in=8, d_out=4, rank=2, alpha=1.0, seed=99)
        base_weights = np.random.randn(4, 8)
        merged = merge_lora(base_weights, {"lora1": layer1, "lora2": layer2})
        assert merged.shape == (4, 8)

    def test_estimate_lora_parameters(self):
        from intelligence.peft import estimate_lora_parameters

        count = estimate_lora_parameters(d_model=768, num_layers=12, rank=8, target_modules=3)
        assert isinstance(count, int)
        assert count > 0

    def test_estimate_memory_savings(self):
        from intelligence.peft import estimate_memory_savings

        savings = estimate_memory_savings(d_model=768, num_layers=12, rank=8, target_modules=3)
        assert isinstance(savings, dict)
        assert "full_params" in savings
        assert "lora_params" in savings
        assert "reduction_ratio" in savings
        assert savings["full_params"] > savings["lora_params"]


# ===========================================================================
# Prompt module — Step 03 (Prompt Engineering, Structured Outputs)
# ===========================================================================


class TestPromptTemplate:
    def test_render(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(template="Hello {name}!", variables={"name": "alice"})
        result = template.render()
        assert "alice" in result

    def test_render_override(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(template="Hello {name}!", variables={"name": "default"})
        result = template.render(name="bob")
        assert "bob" in result
        assert "default" not in result

    def test_with_prefix_suffix(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(
            template="Question: {question}\nAnswer:",
            prefix="SYSTEM: You are a helpful assistant.\n",
            suffix="\nFollow up:",
            variables={"question": "What is 2+2?"},
        )
        result = template.render()
        assert "SYSTEM:" in result
        assert "What is 2+2?" in result
        assert "Follow up:" in result

    def test_variable_names(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(template="Hello {name}, you are {age}", variables={})
        names = template.variable_names
        assert "name" in names
        assert "age" in names

    def test_with_examples(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(
            template="{input}",
            examples=["example1", "example2"],
            example_separator=" | ",
            variables={"input": "test"},
        )
        result = template.render()
        assert "example1" in result
        assert "example2" in result

    def test_template_as_callable(self):
        from intelligence.prompt import PromptTemplate

        template = PromptTemplate(template="Result: {value}", variables={"value": 42})
        result = template()
        assert "42" in str(result)


class TestFewShotBuilder:
    def test_add_example_and_build(self):
        from intelligence.prompt import FewShotBuilder

        builder = FewShotBuilder(instruction="Classify sentiment:")
        builder.add_example("I love this!", "positive")
        builder.add_example("I hate this.", "negative")
        builder.set_query("{input}")
        prompt = builder.build(input="It's okay.")
        assert "Classify sentiment:" in prompt
        assert "positive" in prompt
        assert "negative" in prompt
        assert "It's okay." in prompt

    def test_build_with_kwargs(self):
        from intelligence.prompt import FewShotBuilder

        builder = FewShotBuilder(instruction="Translate to French:")
        builder.add_example("hello", "bonjour")
        builder.set_query("{input}")
        prompt = builder.build(input="goodbye")
        assert "bonjour" in prompt
        assert "goodbye" in prompt


class TestChainOfThought:
    def test_build(self):
        from intelligence.prompt import ChainOfThought

        cot = ChainOfThought(max_steps=3)
        prompt = cot.build("What is 2+2?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_verification(self):
        from intelligence.prompt import ChainOfThought

        cot = ChainOfThought(max_steps=3)
        prompt = cot.build_verification("What is the capital of France?", "Paris")
        assert isinstance(prompt, str)
        assert "Paris" in prompt


class TestStructuredPrompt:
    def test_build_prompt(self):
        from intelligence.prompt import StructuredPrompt

        class Response(BaseModel):
            name: str
            age: int

        sp = StructuredPrompt(template="Extract info: {text}", model=Response)
        prompt = sp.build_prompt(text="John is 30 years old")
        assert "John is 30" in prompt
        assert "name" in prompt.lower()

    def test_validate_output(self):
        from intelligence.prompt import StructuredPrompt

        class Response(BaseModel):
            name: str
            age: int

        sp = StructuredPrompt(template="Extract: {text}", model=Response)
        success, parsed, error = sp.validate_output('{"name": "Jane", "age": 25}')
        assert success is True
        assert error == ""
        assert parsed is not None

    def test_validate_output_failure(self):
        from intelligence.prompt import StructuredPrompt

        class Response(BaseModel):
            name: str
            age: int

        sp = StructuredPrompt(template="Extract: {text}", model=Response)
        success, parsed, error = sp.validate_output('{"name": "Jane"}')
        assert success is False
        assert isinstance(error, str) and len(error) > 0


class TestPromptVariant:
    def test_prompt_variant_fields(self):
        from intelligence.prompt import PromptVariant

        variant = PromptVariant(name="v1", prompt="Hello {name}", template="Hello {name}")
        assert variant.name == "v1"
        assert variant.prompt == "Hello {name}"


class TestPromptOptimizer:
    def test_evaluate(self):
        from intelligence.prompt import PromptOptimizer, PromptTemplate, PromptVariant

        optimizer = PromptOptimizer()
        variants = [
            PromptVariant(name="v1", prompt="Answer: {query}",
                          template=PromptTemplate(template="Answer: {query}")),
            PromptVariant(name="v2", prompt="Response: {query}",
                          template=PromptTemplate(template="Response: {query}")),
        ]
        queries = ["What is AI?", "Explain ML."]

        def response_fn(prompt: str, query: str) -> str:
            return f"Response to: {query}"

        results = optimizer.evaluate(variants, queries, response_fn, query="test")
        assert isinstance(results, dict)
        assert "v1" in results
        assert "v2" in results

    def test_best_variant(self):
        from intelligence.prompt import PromptOptimizer, PromptTemplate, PromptVariant

        def scorer(prompt: str, query: str, response: str) -> float:
            return len(response)

        optimizer = PromptOptimizer(scorer=scorer)
        variants = [
            PromptVariant(name="short", prompt="Hi {query}",
                          template=PromptTemplate(template="Hi {query}")),
            PromptVariant(name="long", prompt="Hello there {query} how are you",
                          template=PromptTemplate(template="Hello there {query} how are you")),
        ]
        queries = ["world"]

        def response_fn(prompt: str, query: str) -> str:
            return prompt.replace("{query}", query)

        best = optimizer.best_variant(variants, queries, response_fn, query="world")
        assert best.name == "long"



# ===========================================================================
# RAG module — Steps 05-06 (Chunking, Retriever, RAGPipeline)
# ===========================================================================


class TestChunker:
    def test_fixed_chunking(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="fixed", chunk_size=10, overlap=2)
        chunks = chunker.chunk("abcdefghij klm nop qrst uvwxyz")
        assert len(chunks) >= 1
        assert all(len(c) <= 12 for c in chunks)

    def test_sentence_chunking(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="sentence", chunk_size=10, overlap=2)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_word_chunking(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="word", chunk_size=3, overlap=1)
        text = "one two three four five six seven"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 3 + 1  # chunk_size + overlap

    def test_recursive_chunking(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="recursive", chunk_size=20, overlap=5, separator="\n\n")
        text = "Para 1 text here.\n\nPara 2 more text here.\n\nPara 3 final text."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_batch(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="word", chunk_size=3, overlap=1)
        texts = ["one two three", "four five six"]
        results = chunker.chunk_batch(texts)
        assert len(results) == 2
        assert all(len(r) > 0 for r in results)

    def test_unknown_strategy(self):
        from intelligence.rag import Chunker

        chunker = Chunker(strategy="unknown")
        with pytest.raises(ValueError):
            chunker.chunk("test text")


class TestRetriever:
    def test_ingest_and_retrieve(self):
        from intelligence.rag import Chunker, Retriever

        def embed(text: str) -> np.ndarray:
            rng = np.random.RandomState(hash(text) % 2**31)
            return rng.randn(8)

        chunker = Chunker(strategy="sentence", chunk_size=50, overlap=10)
        retriever = Retriever(embedding_fn=embed, chunker=chunker, top_k=3)
        docs = ["Hello world foo bar", "Goodbye universe test data", "Another doc here"]
        retriever.ingest(docs)
        results = retriever.retrieve("hello world")
        assert len(results) <= 3
        for r in results:
            assert hasattr(r, "text")
            assert hasattr(r, "score")
            assert hasattr(r, "doc_index")

    def test_retrieve_no_docs(self):
        from intelligence.rag import Retriever

        def embed(text: str) -> np.ndarray:
            return np.zeros(8)

        retriever = Retriever(embedding_fn=embed, top_k=3)
        with pytest.raises(RuntimeError):
            retriever.retrieve("test")


class TestRAGPromptBuilder:
    def test_build_prompt(self):
        from intelligence.rag import RAGPromptBuilder, RetrievalResult

        builder = RAGPromptBuilder(max_context_tokens=100)
        results = [
            RetrievalResult(text="Paris is the capital of France.", score=0.9,
                            metadata={}, doc_index=0),
            RetrievalResult(text="France is in Europe.", score=0.8,
                            metadata={}, doc_index=1),
        ]
        prompt = builder.build("What is the capital of France?", results)
        assert "capital of France" in prompt
        assert "Paris" in prompt
        assert "What is the capital" in prompt

    def test_max_context_truncation(self):
        from intelligence.rag import RAGPromptBuilder, RetrievalResult

        builder = RAGPromptBuilder(max_context_tokens=5)
        results = [
            RetrievalResult(text="A" * 100, score=0.9, metadata={}, doc_index=0),
            RetrievalResult(text="B" * 100, score=0.8, metadata={}, doc_index=1),
        ]
        prompt = builder.build("query", results)
        assert "query" in prompt


class TestRAGPipeline:
    def _make_pipeline(self, llm_fn=None):
        from intelligence.rag import Chunker, RAGPipeline, Retriever

        def embed(text: str) -> np.ndarray:
            rng = np.random.RandomState(hash(text) % 2**31)
            return rng.randn(8)

        chunker = Chunker(strategy="sentence", chunk_size=50)
        retriever = Retriever(embedding_fn=embed, chunker=chunker, top_k=3)
        return RAGPipeline(retriever=retriever, llm_fn=llm_fn)

    def test_pipeline_without_llm(self):
        pipeline = self._make_pipeline(llm_fn=None)
        pipeline.ingest(["The capital of France is Paris.", "France is known for wine."])
        result = pipeline.query("What is the capital of France?")
        assert "question" in result
        assert "retrieved" in result
        assert "prompt" in result
        assert result["answer"] == ""

    def test_pipeline_with_llm(self):
        def llm(prompt: str) -> str:
            return "The capital of France is Paris."

        pipeline = self._make_pipeline(llm_fn=llm)
        pipeline.ingest(["The capital of France is Paris."])
        result = pipeline.query("What is the capital of France?")
        assert result["answer"] == "The capital of France is Paris."
        assert "Paris" in result["prompt"]

    def test_pipeline_generate_without_llm(self):
        pipeline = self._make_pipeline(llm_fn=None)
        with pytest.raises(RuntimeError):
            pipeline.generate("some prompt")

    def test_pipeline_augment(self):
        from intelligence.rag import RetrievalResult

        pipeline = self._make_pipeline()
        results = [
            RetrievalResult(text="Paris is capital.", score=0.9, metadata={}, doc_index=0),
        ]
        prompt = pipeline.augment("What is the capital?", results)
        assert "capital" in prompt
        assert "Paris" in prompt

    def test_rag_pipeline_retrieve(self):
        pipeline = self._make_pipeline()
        pipeline.ingest(["alpha beta gamma", "delta epsilon zeta"])
        results = pipeline.retrieve("alpha beta")
        assert len(results) > 0


# ===========================================================================
# Multimodal module — Steps 10-11 (Vision/Audio Models)
# ===========================================================================


class TestImage:
    def test_creation_hwc(self):
        from intelligence.multimodal import Image

        data = np.random.randn(10, 12, 3)
        img = Image.from_array(data)
        assert img.width == 12
        assert img.height == 10
        assert img.channels == 3
        assert img.mode == "rgb"

    def test_creation_grayscale(self):
        from intelligence.multimodal import Image

        data = np.random.randn(10, 12)
        img = Image.from_array(data)
        assert img.channels == 1
        assert img.mode == "grayscale"

    def test_random(self):
        from intelligence.multimodal import Image

        img = Image.random(width=8, height=8, channels=3)
        assert img.width == 8
        assert img.height == 8
        assert img.channels == 3

    def test_to_grayscale(self):
        from intelligence.multimodal import Image

        data = np.random.randn(10, 12, 3)
        img = Image.from_array(data)
        gray = img.to_grayscale()
        assert gray.channels == 1
        assert gray.mode == "grayscale"

    def test_normalize(self):
        from intelligence.multimodal import Image

        data = np.random.rand(10, 12, 3)
        img = Image.from_array(data)
        norm = img.normalize()
        assert norm.data.shape == img.data.shape

    def test_to_patches(self):
        from intelligence.multimodal import Image

        data = np.random.randn(32, 32, 3)
        img = Image.from_array(data)
        patches = img.to_patches(patch_size=16)
        # 32/16 = 2 patches per dim, so 2*2=4 patches
        assert patches.shape == (4, 16 * 16 * 3)

    def test_resolution(self):
        from intelligence.multimodal import Image

        img = Image.random(width=20, height=15, channels=3)
        assert img.resolution == (20, 15)

    def test_repr(self):
        from intelligence.multimodal import Image

        img = Image.random(width=10, height=10, channels=3)
        assert "Image" in repr(img)


class TestImageEncoder:
    def test_patch_embed(self):
        from intelligence.multimodal import Image, ImageEncoder

        img = Image.random(width=32, height=32, channels=3)
        encoder = ImageEncoder(patch_size=16, dim=64, depth=1, heads=4, mlp_dim=128, image_size=32)
        patches = encoder.patch_embed_forward(img)
        # num_patches = (32//16)^2 = 4, plus CLS token = 5
        assert patches.shape == (5, 64)

    def test_forward(self):
        from intelligence.multimodal import Image, ImageEncoder

        img = Image.random(width=32, height=32, channels=3)
        encoder = ImageEncoder(patch_size=16, dim=64, depth=2, heads=4, mlp_dim=128, image_size=32)
        features = encoder.forward(img)
        # num_patches = 4, plus CLS = 5
        assert features.shape == (5, 64)

    def test_cls_token(self):
        from intelligence.multimodal import Image, ImageEncoder

        img = Image.random(width=32, height=32, channels=3)
        encoder = ImageEncoder(patch_size=16, dim=64, depth=1, heads=4, mlp_dim=128, image_size=32)
        cls = encoder.get_cls_token(img)
        assert cls.shape == (64,)


class TestAudio:
    def test_creation(self):
        from intelligence.multimodal import Audio

        samples = [0.1, 0.2, 0.3, 0.4, -0.1]
        audio = Audio.from_list(samples, sample_rate=8000)
        assert audio.num_samples == 5
        assert audio.sample_rate == 8000

    def test_random(self):
        from intelligence.multimodal import Audio

        audio = Audio.random(duration=1.0, sample_rate=8000)
        assert audio.num_samples == 8000
        assert audio.duration == 1.0

    def test_resample(self):
        from intelligence.multimodal import Audio

        audio = Audio.random(duration=1.0, sample_rate=16000)
        resampled = audio.resample(8000)
        assert resampled.sample_rate == 8000
        assert resampled.num_samples < audio.num_samples

    def test_spectrogram(self):
        from intelligence.multimodal import Audio

        audio = Audio.random(duration=0.5, sample_rate=8000)
        spec = audio.spectrogram(window_size=256, hop_length=128, n_fft=256)
        assert spec.ndim == 2
        assert spec.shape[1] == 129  # n_fft // 2 + 1

    def test_mel_spectrogram(self):
        from intelligence.multimodal import Audio

        audio = Audio.random(duration=0.5, sample_rate=8000)
        mel = audio.mel_spectrogram(n_mels=40, window_size=256, hop_length=128, n_fft=256)
        assert mel.ndim == 2
        assert mel.shape[1] == 40


class TestAudioEncoder:
    def test_forward(self):
        from intelligence.multimodal import Audio, AudioEncoder

        audio = Audio.random(duration=1.0, sample_rate=16000)
        encoder = AudioEncoder(
            input_dim=1, hidden_dim=32, num_layers=2,
            kernel_size=64, stride=3,
        )
        features = encoder.forward(audio)
        assert features.ndim == 2
        assert features.shape[1] == 32
        assert features.shape[0] > 0


class TestMultimodalEncoder:
    def test_forward(self):
        from intelligence.multimodal import MultimodalEncoder

        encoder = MultimodalEncoder(vision_dim=32, text_dim=32, fuse_dim=16)
        vision = np.random.randn(5, 32)
        text = np.random.randn(7, 32)
        result = encoder.forward(vision, text)
        assert result.shape == (7, 16)


# ===========================================================================
# Observability module — Steps 14-16 (Tracing, Metrics, Guardrails, RedTeam)
# ===========================================================================


class TestSpan:
    def test_span_to_dict(self):
        from intelligence.observability import Span

        span = Span(
            name="test_op",
            trace_id="trace1",
            span_id="span1",
            parent_id=None,
            start_time=1000.0,
            end_time=1001.0,
            attributes={"key": "value"},
        )
        d = span.to_dict()
        assert d["name"] == "test_op"
        assert d["trace_id"] == "trace1"
        assert d["duration_ms"] == 1000.0

    def test_span_duration_none(self):
        from intelligence.observability import Span

        span = Span(
            name="op", trace_id="t1", span_id="s1",
            parent_id=None, start_time=1000.0,
        )
        assert span.duration_ms is None


class TestTraceProvider:
    def test_start_end_trace(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        tracer.start_trace("test_trace", {"user": "alice"})
        span = tracer.start_span("operation", attributes={"op": "compute"})
        tracer.end_span(span)
        tracer.end_trace()
        assert tracer.trace_count == 1
        assert tracer.span_count == 2

    def test_export(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        tracer.start_trace("trace1")
        span = tracer.start_span("op1")
        tracer.end_span(span)
        tracer.end_trace()
        traces = tracer.export()
        assert len(traces) == 1
        assert len(traces[0]["spans"]) == 2

    def test_export_specific_trace(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        tid1 = tracer.start_trace("trace1")
        tracer.end_trace()
        tracer.start_trace("trace2")
        tracer.end_trace()
        traces = tracer.export(trace_id=tid1)
        assert len(traces) == 1

    def test_record_event(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        tracer.start_trace("test")
        span = tracer.start_span("op")
        tracer.record_event(span, "checkpoint", {"step": 1})
        tracer.end_span(span)
        tracer.end_trace()
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"

    def test_get_trace(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        trace_id = tracer.start_trace("test")
        tracer.end_trace()
        spans = tracer.get_trace(trace_id)
        assert len(spans) == 1

    def test_multiple_traces(self):
        from intelligence.observability import TraceProvider

        tracer = TraceProvider()
        t1 = tracer.start_trace("t1")
        tracer.end_trace()
        t2 = tracer.start_trace("t2")
        tracer.end_trace()
        assert tracer.trace_count == 2
        assert len(tracer.get_trace(t1)) == 1
        assert len(tracer.get_trace(t2)) == 1


class TestMetricRegistry:
    def test_counter(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        val = m.counter("requests")
        assert val == 0.0
        m.inc("requests")
        m.inc("requests")
        assert m.get_counter("requests") == 2.0

    def test_gauge(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        m.gauge("active_users", value=42)
        assert m.get_gauge("active_users") == 42.0

    def test_histogram(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        for v in [10.0, 20.0, 30.0]:
            m.histogram("latency", v)
        stats = m.get_histogram("latency")
        assert stats["count"] == 3
        assert stats["sum"] == 60.0
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert abs(stats["avg"] - 20.0) < 1e-10

    def test_observe_alias(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        m.observe("duration", 5.0)
        assert m.get_histogram("duration")["count"] == 1

    def test_labels(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        m.inc("requests", endpoint="/api", method="GET")
        m.inc("requests", endpoint="/api", method="GET")
        m.inc("requests", endpoint="/health", method="GET")
        assert m.get_counter("requests", endpoint="/api", method="GET") == 2.0
        assert m.get_counter("requests", endpoint="/health", method="GET") == 1.0

    def test_summary(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        m.inc("counter1")
        m.gauge("gauge1", value=5.0)
        summary = m.summary()
        assert "counters" in summary
        assert "gauges" in summary
        assert "histograms" in summary

    def test_reset(self):
        from intelligence.observability import MetricRegistry

        m = MetricRegistry()
        m.inc("counter1")
        m.gauge("gauge1", value=5.0)
        m.reset()
        assert m.get_counter("counter1") == 0.0
        assert m.get_gauge("gauge1") == 0.0


class TestGuardrail:
    def test_safe_input(self):
        from intelligence.observability import Guardrail

        guard = Guardrail()
        result = guard.check("Please summarize the quarterly sales report.")
        assert result.passed is True

    def test_pii_ssn(self):
        from intelligence.observability import Guardrail

        guard = Guardrail()
        result = guard.check("Please process SSN 123-45-6789 for verification.")
        assert result.passed is False
        assert "PII" in str(result.violations)

    def test_pii_email(self):
        from intelligence.observability import Guardrail

        guard = Guardrail()
        result = guard.check("My email is admin@company.com please use it.")
        assert result.passed is False

    def test_toxicity(self):
        from intelligence.observability import Guardrail

        guard = Guardrail()
        # Use an exact default keyword
        result = guard.check("I want to die today.")
        assert result.passed is False
        assert any("Toxic" in v or "toxic" in v.lower() for v in result.violations)

    def test_length_limit(self):
        from intelligence.observability import Guardrail

        guard = Guardrail(max_length=10)
        result = guard.check("This text is way too long for the limit.")
        assert result.passed is False

    def test_blocked_pattern(self):
        from intelligence.observability import Guardrail

        guard = Guardrail(blocked_patterns=[r"password\s*=\s*\S+"])
        result = guard.check("password = secret123")
        assert result.passed is False

    def test_sanitize(self):
        from intelligence.observability import Guardrail

        guard = Guardrail()
        result = guard.sanitize("Email me at admin@company.com")
        assert "admin@company.com" not in result
        assert "REDACTED" in result

    def test_guardrail_result_to_dict(self):
        from intelligence.observability import GuardrailResult

        result = GuardrailResult(passed=True, reason="OK", violations=[], metadata={"k": "v"})
        d = result.to_dict()
        assert d["passed"] is True
        assert d["reason"] == "OK"

    def test_default_toxicity_keywords(self):
        from intelligence.observability import Guardrail

        assert "kill yourself" in Guardrail.DEFAULT_TOXICITY_KEYWORDS
        assert "i want to die" in Guardrail.DEFAULT_TOXICITY_KEYWORDS

    def test_no_pii_blocking(self):
        from intelligence.observability import Guardrail

        guard = Guardrail(block_pii=False, toxicity_keywords=[])
        result = guard.check("My SSN is 123-45-6789")
        assert result.passed is True


class TestRedTeam:
    def test_safe_input_passes(self):
        from intelligence.observability import Guardrail, RedTeam

        guard = Guardrail()
        red_team = RedTeam(guard)
        result = red_team.run_test("safe_test", "Please help me with my project.", expect_violation=False)
        assert result.passed is True

    def test_pii_input_caught(self):
        from intelligence.observability import Guardrail, RedTeam

        guard = Guardrail()
        red_team = RedTeam(guard)
        result = red_team.run_test(
            "pii_test",
            "Please process SSN 123-45-6789 for verification.",
            severity="high",
            expect_violation=True,
        )
        assert result.passed is True

    def test_run_all(self):
        from intelligence.observability import Guardrail, RedTeam

        guard = Guardrail()
        red_team = RedTeam(guard)
        results = red_team.run_all()
        assert len(results) == len(red_team.DEFAULT_TEST_CASES)
        for r in results:
            assert r.test_name
            assert r.severity in ("low", "medium", "high", "critical")

    def test_summary(self):
        from intelligence.observability import Guardrail, RedTeam

        guard = Guardrail()
        red_team = RedTeam(guard)
        red_team.run_all()
        summary = red_team.summary()
        assert "passed" in summary
        assert "failed" in summary
        assert "total" in summary
        assert "pass_rate" in summary
        assert summary["total"] > 0

    def test_red_team_result_to_dict(self):
        from intelligence.observability import GuardrailResult, RedTeamResult

        guard_result = GuardrailResult(passed=True, reason="OK")
        result = RedTeamResult(
            test_name="test",
            input_text="hello",
            passed=True,
            guardrail_result=guard_result,
            severity="high",
        )
        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["passed"] is True
        assert d["severity"] == "high"
        assert "guardrail_result" in d

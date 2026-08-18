"""Tests for M14: native inference with KV cache."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from foundation.inference import (
    GenerationResult,
    Generator,
    InferenceConfig,
)
from foundation.inference.generator import (
    _apply_repetition_penalty,
    _pad_sequences,
    _sample_token,
    _top_k_logits,
    _top_p_logits,
)
from foundation.model import ModelConfig, SilverwingDecoder, build_model
from foundation.tokenizer import TokenizerV2

ROOT = Path(__file__).resolve().parents[1]


def make_tokenizer() -> TokenizerV2:
    return TokenizerV2(merges=[])


def small_config(**overrides) -> ModelConfig:
    base = {
        "vocab_size": 260,  # 4 specials + 256 byte-level
        "block_size": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "n_embd": 64,
        "mlp_hidden_size": 128,
    }
    base.update(overrides)
    return ModelConfig.from_dict(base)


def build_tiny_model() -> SilverwingDecoder:
    cfg = small_config()
    model = build_model(cfg)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# KV-cache correctness
# --------------------------------------------------------------------------- #


def test_kv_cache_matches_no_cache_logits():
    """Forward with use_cache must produce identical logits to plain forward."""
    model = build_tiny_model()
    ids = torch.tensor([[5, 10, 15, 8, 3]])

    with torch.no_grad():
        logits_no_cache = model(ids)
        logits_cache, past = model(ids, use_cache=True)

    assert logits_cache.shape == (1, 5, 260)
    assert torch.allclose(logits_no_cache, logits_cache, atol=1e-6)
    assert len(past) == model.cfg.n_layer
    assert past[0][0].shape == (1, 2, 5, 16)  # (B, n_kv_head, T, head_dim)
    assert past[0][1].shape == (1, 2, 5, 16)


def test_kv_cache_incremental_matches_full():
    """Feeding tokens one at a time with KV cache == feeding them all at once."""
    model = build_tiny_model()
    ids = torch.tensor([[5, 10, 15, 8, 3, 7, 12]])

    with torch.no_grad():
        full_logits, _ = model(ids, use_cache=True)

        # Step 1: feed first 3 tokens (pre-fill with causal mask)
        step1_logits, past = model(ids[:, :3], use_cache=True)
        assert torch.allclose(step1_logits, full_logits[:, :3, :], atol=1e-6)

        # Steps 2-7: feed one token at a time with KV cache
        for i in range(3, 7):
            step_logits, past = model(
                ids[:, i : i + 1], use_cache=True, past_key_values=past
            )
            assert torch.allclose(step_logits, full_logits[:, i : i + 1, :], atol=1e-6)


def test_kv_cache_single_token_decode():
    """Single-token decode with cache matches re-encoding the full sequence."""
    model = build_tiny_model()
    ids = torch.tensor([[5, 10, 15, 8]])

    with torch.no_grad():
        # Method A: re-encode the full sequence + new token
        full = torch.cat([ids, torch.tensor([[3]])], dim=-1)
        logits_a = model(full)[:, -1, :]

        # Method B: warm cache with ids, then decode one token
        _, past = model(ids, use_cache=True)
        logits_b, _ = model(torch.tensor([[3]]), use_cache=True, past_key_values=past)

    assert torch.allclose(logits_a, logits_b, atol=1e-6)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_generator_greedy_deterministic():
    """Greedy sampling (temperature=0) must be deterministic."""
    model = build_tiny_model()
    tokenizer = make_tokenizer()
    gen = Generator(
        model, tokenizer, max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0
    )

    prompt = "hi"
    result = gen._generate_single(
        prompt, max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0
    )
    assert isinstance(result, GenerationResult)
    assert result.token_ids  # non-empty
    assert len(result.token_ids) <= 5

    # Run again — must be identical
    result2 = gen._generate_single(
        prompt, max_new_tokens=5, temperature=0.0, top_k=0, top_p=1.0
    )
    assert result.token_ids == result2.token_ids


def test_generator_batched_matches_single():
    """Batched generation of same-length prompts == individual generations."""
    model = build_tiny_model()
    tokenizer = make_tokenizer()
    gen = Generator(model, tokenizer, max_new_tokens=5, temperature=0.0, top_k=1)

    # All prompts are 2 characters → same token length with this tokenizer.
    prompts = ["hi", "go", "ok"]
    singles = [gen.generate(p) for p in prompts]
    batched = gen.generate(prompts)

    assert len(batched) == 3
    for s, b in zip(singles, batched):
        assert s.token_ids == b.token_ids


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


def test_top_k_logits():
    logits = torch.tensor([1.0, 5.0, 3.0, 8.0, 2.0])
    result = _top_k_logits(logits, k=2)
    # Only the top-2 (8.0 and 5.0) should remain
    nonzero = (result > -torch.inf).sum().item()
    assert nonzero == 2
    surviving = result[result > -torch.inf]
    expected = torch.tensor([5.0, 8.0])
    assert torch.equal(torch.sort(surviving).values, torch.sort(expected).values)


def test_top_k_logits_zero_is_identity():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(_top_k_logits(logits, k=0), logits)


def test_top_p_logits():
    # Two tokens have equal prob (0.5 each) and rest are negligible
    logits = torch.tensor([10.0, 10.0, -1000.0, -1000.0])
    result = _top_p_logits(logits, p=0.5)
    surviving = (result > -torch.inf).sum().item()
    assert surviving == 2


def test_top_p_logits_full_range_is_identity():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(_top_p_logits(logits, p=1.0), logits)


def test_repetition_penalty_reduces_repeats():
    logits = torch.tensor([1.0, 5.0, -1.0])
    generated = torch.tensor([0, 1])
    result = _apply_repetition_penalty(logits, generated, penalty=1.5)
    assert result[0].item() < logits[0].item()
    assert result[1].item() < logits[1].item()
    assert result[2].item() == logits[2].item()


def test_repetition_penalty_identity():
    logits = torch.tensor([1.0, 2.0, 3.0])
    generated = torch.tensor([0])
    result = _apply_repetition_penalty(logits, generated, penalty=1.0)
    assert torch.equal(result, logits)


def test_sample_token_greedy():
    logits = torch.tensor([1.0, 10.0, -5.0])
    token_id = _sample_token(logits, temperature=0.0, top_k=0, top_p=1.0)
    assert token_id == 1  # argmax


def test_sample_token_greedy_with_top_k():
    logits = torch.tensor([10.0, 5.0, 3.0, -100.0])
    token_id = _sample_token(logits, temperature=0.0, top_k=1, top_p=1.0)
    assert token_id == 0


def test_pad_sequences():
    seqs = [[1, 2], [3, 4, 5], [6]]
    result = _pad_sequences(seqs, pad_id=0)
    assert result.shape == (3, 3)
    assert torch.equal(result[0], torch.tensor([1, 2, 0]))
    assert torch.equal(result[1], torch.tensor([3, 4, 5]))
    assert torch.equal(result[2], torch.tensor([6, 0, 0]))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.version == "inference-v1"
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == 0.8
    assert cfg.top_k == 50
    assert cfg.top_p == 0.9
    assert cfg.device == "cpu"
    assert isinstance(cfg.digest(), str)
    assert len(cfg.digest()) == 64


def test_inference_config_yaml(tmp_path: Path):
    cfg_path = tmp_path / "inference.yaml"
    cfg_path.write_text(
        "inference:\n  max_new_tokens: 32\n  temperature: 0.5\n  top_k: 10\n  device: cpu\n",
        encoding="utf-8",
    )
    cfg = InferenceConfig.from_yaml(cfg_path)
    assert cfg.max_new_tokens == 32
    assert cfg.temperature == 0.5
    assert cfg.top_k == 10
    assert cfg.version == "inference-v1"
    assert (
        cfg.digest()
        == InferenceConfig(max_new_tokens=32, temperature=0.5, top_k=10).digest()
    )


def test_inference_config_to_dict_round_trip():
    cfg = InferenceConfig(temperature=0.1, top_p=0.5)
    d = cfg.to_dict()
    assert d["temperature"] == 0.1
    assert d["top_p"] == 0.5
    assert InferenceConfig.from_dict(d) == cfg


# --------------------------------------------------------------------------- #
# Generator from config (with checkpoint)
# --------------------------------------------------------------------------- #


def _save_checkpoint_and_tokenizer(tmp_path: Path) -> tuple[Path, Path, Path]:
    from foundation.training import save_checkpoint

    model = build_tiny_model()
    tokenizer = make_tokenizer()

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt_path = save_checkpoint(
        ckpt_dir,
        step=1,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        run_id="inference-test",
        config_digest="abc",
        tokenizer_hash=tokenizer.digest(),
        dataset_hash="def",
        git_commit="test",
    )

    tok_dir = tmp_path / "tokenizer"
    tokenizer.save(tok_dir)

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        json.dumps({"model": small_config().to_dict()}),
        encoding="utf-8",
    )
    return ckpt_path, model_cfg, tok_dir


def test_generator_from_config(tmp_path: Path):
    ckpt_path, model_cfg, tok_dir = _save_checkpoint_and_tokenizer(tmp_path)
    cfg = InferenceConfig(
        checkpoint_path=str(ckpt_path),
        model_config_path=str(model_cfg),
        tokenizer_dir=str(tok_dir),
        max_new_tokens=5,
        temperature=0.0,
        top_k=1,
    )
    gen = Generator.from_config(cfg)
    result = gen.generate("hi", max_new_tokens=3, temperature=0.0, top_k=1)
    assert isinstance(result, GenerationResult)
    assert len(result.token_ids) <= 3


def test_generator_batch_from_config(tmp_path: Path):
    ckpt_path, model_cfg, tok_dir = _save_checkpoint_and_tokenizer(tmp_path)
    cfg = InferenceConfig(
        checkpoint_path=str(ckpt_path),
        model_config_path=str(model_cfg),
        tokenizer_dir=str(tok_dir),
        max_new_tokens=5,
        temperature=0.0,
        top_k=1,
    )
    gen = Generator.from_config(cfg)
    results = gen.generate(["hi", "ok"], max_new_tokens=3, temperature=0.0, top_k=1)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, GenerationResult)
        assert len(r.token_ids) <= 3

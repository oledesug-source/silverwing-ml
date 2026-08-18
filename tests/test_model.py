"""Tests for M06: Silverwing Decoder V2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from foundation.model import ModelConfig, SilverwingDecoder, apply_rope, build_model, precompute_rope_cache
from foundation.model.layers import RMSNorm
from foundation.model.rope import apply_rope as rope_apply

ROOT = Path(__file__).resolve().parents[1]


def small_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        mlp_hidden_size=128,
    )
    base.update(overrides)
    return ModelConfig.from_dict(base)


def test_forward_shape() -> None:
    cfg = small_config()
    model = SilverwingDecoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_causal_attention() -> None:
    cfg = small_config()
    model = SilverwingDecoder(cfg).eval()
    seq_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    seq_b = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 99]])
    with torch.no_grad():
        logits_a = model(seq_a)
        logits_b = model(seq_b)
    # Future tokens must not influence earlier positions.
    assert torch.equal(logits_a[0, :7], logits_b[0, :7])


def test_causal_mask_bounds_sequence_length() -> None:
    cfg = small_config(block_size=16)
    model = SilverwingDecoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        model(ids)  # should not raise


def test_rope_preserves_norm() -> None:
    head_dim = 8
    cos, sin = precompute_rope_cache(block_size=8, head_dim=head_dim)
    # Shape is (B, H, T, D) — the layout used by Attention after transpose(1, 2).
    x = torch.randn(2, 3, 8, head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5)
    # rotation depends on position
    assert not torch.allclose(rope_apply(x, cos, sin), rope_apply(x, torch.flip(cos, [0]), torch.flip(sin, [0])))


def test_rope_rotation_is_position_dependent() -> None:
    """Identical input at different positions must receive different rotations."""
    head_dim = 8
    cos, sin = precompute_rope_cache(block_size=16, head_dim=head_dim)
    # Shape (B=1, H=1, T=4, D=head_dim) — all four positions identical
    x = torch.randn(1, 1, 1, head_dim).expand(1, 1, 4, head_dim).clone()
    y = apply_rope(x, cos, sin)
    # Position 0 uses cos[0]=1, sin[0]=0 → identity. Later positions use
    # non-trivial cos/sin, so outputs must differ.
    assert not torch.allclose(y[0, 0, 0], y[0, 0, 1])
    assert not torch.allclose(y[0, 0, 0], y[0, 0, 3])


def test_gqa_forward() -> None:
    cfg = small_config(n_kv_head=2)
    model = SilverwingDecoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_mlp_activations() -> None:
    for activation in ("gelu", "swiglu"):
        cfg = small_config(mlp_activation=activation)
        model = SilverwingDecoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, cfg.vocab_size)


def test_rmsnorm_scale_invariance() -> None:
    norm = RMSNorm(16)
    x = torch.randn(3, 5, 16)
    y1 = norm(x)
    y2 = norm(x * 7.3)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)


def test_tied_embeddings_counted_once() -> None:
    cfg = small_config()
    model = SilverwingDecoder(cfg)
    assert model.lm_head.weight is model.token_embedding.weight
    sd = model.state_dict()
    assert torch.equal(sd["lm_head.weight"], sd["token_embedding.weight"])
    assert model.num_parameters() == cfg.expected_parameter_count()


def test_parameter_count_matches_formula() -> None:
    for cfg in [small_config(), small_config(n_layer=3, n_head=8, n_kv_head=4, n_embd=128, mlp_hidden_size=256)]:
        model = SilverwingDecoder(cfg)
        assert model.num_parameters() == cfg.expected_parameter_count()


def test_default_config_about_100m() -> None:
    cfg = ModelConfig.from_yaml(ROOT / "configs" / "model.yaml")
    assert cfg.model_name == "silverwing-decoder-v2"
    model = SilverwingDecoder(cfg)
    params = model.num_parameters()
    assert 95_000_000 <= params <= 110_000_000
    ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 16, cfg.vocab_size)


def test_config_yaml_round_trip() -> None:
    cfg = ModelConfig.from_yaml(ROOT / "configs" / "model.yaml")
    again = ModelConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.digest() == cfg.digest()


def test_config_invalid_rejected() -> None:
    with pytest.raises(ValueError):
        small_config(n_head=5, n_embd=64)  # 64 % 5 != 0
    with pytest.raises(ValueError):
        small_config(mlp_activation="bogus")


def test_state_dict_round_trip(tmp_path: Path) -> None:
    cfg = small_config()
    model_a = SilverwingDecoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        expected = model_a(ids)
    checkpoint = tmp_path / "model.pt"
    torch.save(model_a.state_dict(), checkpoint)
    model_b = SilverwingDecoder(cfg).eval()
    model_b.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        actual = model_b(ids)
    assert torch.equal(expected, actual)
    model_b.train()


def test_build_model_variants(tmp_path: Path) -> None:
    cfg = small_config()
    assert isinstance(build_model(cfg), SilverwingDecoder)
    assert isinstance(build_model(cfg.to_dict()), SilverwingDecoder)
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(json.dumps({"model": cfg.to_dict()}), encoding="utf-8")
    assert isinstance(build_model(cfg_path), SilverwingDecoder)

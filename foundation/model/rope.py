"""Rotary position embeddings (RoPE) helpers."""

from __future__ import annotations

import torch


def precompute_rope_cache(
    block_size: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) caches of shape (block_size, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(block_size, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply RoPE to x of shape (B, H, T, D); cos/sin shape (block_size, D // 2).

    Rotates each head's last dimension by interleaved halves: x' = (x1*cos - x2*sin, x2*cos + x1*sin).
    The positional slice cos/sin is indexed along the sequence dimension starting
    at ``start_pos`` (for KV-cache autoregressive decoding, where new tokens
    need positions *after* the cached prefix), and broadcast across batch and
    heads.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    t = x.shape[2]
    c = cos[start_pos : start_pos + t].unsqueeze(0).unsqueeze(1)
    s = sin[start_pos : start_pos + t].unsqueeze(0).unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

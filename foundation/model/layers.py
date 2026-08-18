"""Transformer building blocks for Silverwing Decoder V2.

Modern, GPT-2-free decoder stack: RMSNorm pre-norm, RoPE attention with
Grouped Query Attention, and SwiGLU (or GELU) feed-forward, all without
bias terms unless enabled. Suitable for CPU training at the ~100M scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.activation = cfg.mlp_activation
        hidden = cfg.mlp_hidden_size
        if cfg.mlp_activation == "swiglu":
            self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        else:
            self.w_in = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.w_out = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w_out(F.gelu(self.w_in(x)))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        q_dim = cfg.n_head * self.head_dim
        kv_dim = cfg.n_kv_head * self.head_dim
        self.qkv = nn.Linear(cfg.n_embd, q_dim + 2 * kv_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        b, t, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(
            [
                cfg.n_head * self.head_dim,
                cfg.n_kv_head * self.head_dim,
                cfg.n_kv_head * self.head_dim,
            ],
            dim=-1,
        )
        q = q.view(b, t, cfg.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, cfg.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, cfg.n_kv_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            # Autoregressive decode: x contains only the new token(s), position = past_len
            past_len = cache[0].shape[-2]
            q = apply_rope(q, cos, sin, start_pos=past_len)
            k = apply_rope(k, cos, sin, start_pos=past_len)
            # Concatenate with cached K/V (stored in n_kv_head form — repeat later).
            # k_cat / v_cat accumulate the full history across decode steps.
            k_cat = torch.cat([cache[0], k], dim=-2)
            v_cat = torch.cat([cache[1], v], dim=-2)
            k_full = k_cat
            v_full = v_cat
        else:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            k_full = k
            v_full = v

        # Grouped Query Attention: repeat K/V heads to match number of query heads
        if cfg.n_kv_head != cfg.n_head:
            repeat = cfg.n_head // cfg.n_kv_head
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        att = q @ k_full.transpose(-2, -1) * (self.head_dim**-0.5)
        if cache is None:
            att = att.masked_fill(~self.causal_mask[:t, :t], torch.finfo(att.dtype).min)
        # When using a cache, the keys are already in causal order so no explicit mask is needed
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=cfg.dropout, training=self.training)
        y = att @ v_full
        y = y.transpose(1, 2).contiguous().view(b, t, cfg.n_embd)

        if use_cache:
            # Store the pre-repeat K/V so the cache retains n_kv_head form.
            # On first call (cache=None) this captures all prompt positions;
            # on subsequent calls it accumulates the full decode history.
            new_cache: tuple[torch.Tensor, torch.Tensor] = (
                k_cat if cache is not None else k,
                v_cat if cache is not None else v,
            )
            return self.o_proj(y), new_cache
        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Returns ``(x, cache)`` where *cache* is ``None`` if KV caching is
        inactive (i.e. ``use_cache=False``)."""
        if use_cache:
            x_attn, new_cache = self.attn(self.ln1(x), cos, sin, cache, use_cache=True)
            x = x + x_attn
            x = x + self.mlp(self.ln2(x))
            return x, new_cache
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x, None

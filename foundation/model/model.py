"""Silverwing Decoder V2: causal decoder-only LM with tied embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig
from .layers import Block, RMSNorm
from .rope import precompute_rope_cache


class SilverwingDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        cos, sin = precompute_rope_cache(
            cfg.block_size, cfg.head_dim, base=cfg.rope_base
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: bool = False,
        use_cache: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.token_embedding(input_ids)
        layers = [x] if hidden_states else None

        if not use_cache:
            for block in self.blocks:
                x, _ = block(x, self.rope_cos, self.rope_sin)
                if hidden_states:
                    layers.append(x)
        else:
            if past_key_values is None:
                past_key_values = [None] * len(self.blocks)
            for i, block in enumerate(self.blocks):
                x, past_key_values[i] = block(
                    x, self.rope_cos, self.rope_sin, past_key_values[i], use_cache=True
                )

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if hidden_states:
            return logits, layers
        if use_cache:
            return logits, past_key_values
        return logits


def build_model(config: ModelConfig | dict | str | Path) -> SilverwingDecoder:
    if isinstance(config, ModelConfig):
        return SilverwingDecoder(config)
    if isinstance(config, dict):
        return SilverwingDecoder(ModelConfig.from_dict(config))
    return SilverwingDecoder(ModelConfig.from_yaml(config))

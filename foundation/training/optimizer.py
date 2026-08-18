"""AdamW with the standard LLM grouping: decay 2-D weights, no decay on 1-D
norms/biases."""

from __future__ import annotations

import torch


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> tuple[torch.optim.Optimizer, dict]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    report = {
        "decay_params": len(decay),
        "no_decay_params": len(no_decay),
        "decay_parameters": sum(p.numel() for p in decay),
        "no_decay_parameters": sum(p.numel() for p in no_decay),
    }
    return optimizer, report

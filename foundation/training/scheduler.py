"""Linear warmup followed by cosine decay to min_lr * lr."""

from __future__ import annotations

import math


def schedule_lr(
    step: int,
    lr: float,
    warmup_steps: int = 0,
    max_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> float:
    """Learning rate for a 0-based `step`."""
    min_lr = lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

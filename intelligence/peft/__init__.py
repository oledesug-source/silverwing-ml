"""Parameter-Efficient Fine-Tuning (PEFT) — LoRA adapter layers.

Implements Low-Rank Adaptation (LoRA) as described in
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021).

Key idea: freeze the pretrained model weights and inject trainable
low-rank decomposition matrices into attention layers.  This reduces
the number of trainable parameters by 10,000×+ while matching full
fine-tuning quality on many tasks.

Provides:

    - ``LoRALayer``: A single LoRA adapter that wraps a base layer
    - ``LoRAAdapter``: Manages multiple LoRA adapters on a model
    - ``LoRATrainer``: Simple training loop for LoRA parameters
    - ``merge_lora``: Merges LoRA weights into the base model

All implementations are numpy-based and demonstrate the mathematical
principles without requiring torch.

Example::

    adapter = LoRAAdapter(base_weights=model_weights, rank=8, alpha=16)
    output = adapter.forward(x, adapter_names=["task_a"])
    adapter.train(train_data, learning_rate=0.001, epochs=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (used in feed-forward sublayers)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + x**3 / 3.0)))


class LoRALayer:
    """A single LoRA adapter that injects low-rank updates into a base layer.

    The LoRA decomposition replaces a weight update ΔW with:
        ΔW = B @ A
    where A is (r, d_in) and B is (d_out, r), with r << min(d_in, d_out).

    The forward pass computes:
        output = base_weights @ x + scaling * B @ A @ x

    Args:
        d_in:     Input dimension (must match base layer).
        d_out:    Output dimension (must match base layer).
        rank:     Low-rank dimension (r). Typical: 1–32.
        alpha:    Scaling factor. Effective learning rate scales with alpha/rank.
        dropout:  Dropout probability for the LoRA branch.
        init:     Initialization strategy: "gaussian" (default) or "zeros".

    Attributes:
        rank, alpha, scaling, dropout_rate.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init: str = "gaussian",
        seed: int | None = None,
    ) -> None:
        if rank > min(d_in, d_out):
            raise ValueError(
                f"rank ({rank}) must be <= min(d_in, d_out) = {min(d_in, d_out)}"
            )
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_rate = dropout
        self.training = True
        rng = np.random.default_rng(seed)

        # LoRA matrices: A (rank, d_in), B (d_out, rank)
        # By default, A is initialized with small random values and B is zero
        # so that the initial LoRA contribution is zero (no change to base model).
        if init == "zeros":
            self.A = np.zeros((rank, d_in), dtype=np.float64)
            self.B = np.zeros((d_out, rank), dtype=np.float64)
        else:  # gaussian / default
            self.A = rng.standard_normal((rank, d_in)) * 0.01
            self.B = np.zeros((d_out, rank), dtype=np.float64)

        # Gradients (populated during backward pass)
        self.grad_A: np.ndarray | None = None
        self.grad_B: np.ndarray | None = None

    def forward(
        self,
        base_output: np.ndarray,
        x: np.ndarray,
        training: bool | None = None,
    ) -> np.ndarray:
        """Forward pass: base_output + LoRA contribution.

        Args:
            base_output: Output from the frozen base layer (d_out,) or (..., d_out).
            x:           Input to the base layer (d_in,) or (..., d_in).

        Returns:
            Combined output: base_output + scaling * B @ A @ x
        """
        # LoRA: lora_output = scaling * x @ A^T @ B^T
        # A: (rank, d_in), B: (d_out, rank)
        # x: (..., d_in) → x @ A^T → (..., rank) → @ B^T → (..., d_out)
        h = np.matmul(x, self.A.T)  # (..., rank)
        lora_contribution = self.scaling * np.matmul(h, self.B.T)  # (..., d_out)

        use_training = training if training is not None else self.training
        if use_training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            rng = np.random.default_rng()
            mask = (rng.random(lora_contribution.shape) < keep_prob) / keep_prob
            lora_contribution = lora_contribution * mask

        return base_output + lora_contribution

    def __call__(self, base_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.forward(base_output, x)

    def merge(self, base_weights: np.ndarray) -> np.ndarray:
        """Merge LoRA weights into the base weight matrix.

        Returns W_merged = W_base + scaling * B @ A

        Args:
            base_weights: Original weight matrix (d_out, d_in).

        Returns:
            Merged weight matrix (d_out, d_in).
        """
        delta = self.scaling * np.matmul(self.B, self.A)
        return base_weights + delta

    def unmerge(self, merged_weights: np.ndarray) -> np.ndarray:
        """Remove LoRA contribution from merged weights."""
        delta = self.scaling * np.matmul(self.B, self.A)
        return merged_weights - delta

    def update(self, grad_A: np.ndarray, grad_B: np.ndarray, lr: float) -> None:
        """Simple SGD update of LoRA parameters.

        Args:
            grad_A: Gradient for matrix A (rank, d_in).
            grad_B: Gradient for matrix B (d_out, rank).
            lr:     Learning rate.
        """
        self.A -= lr * grad_A
        self.B -= lr * grad_B


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_norm: str = "none"  # "none", "layer_norm", "rms_norm"
    trainable: bool = True
    seed: int | None = None


# ---------------------------------------------------------------------------
# LoRA Adapter Manager
# ---------------------------------------------------------------------------

class LoRAAdapter:
    """Manager for multiple LoRA adapters on a single set of base weights.

    Supports:
        - Multiple adapters for different tasks
        - Merging/unmerging individual or all adapters
        - Forward pass with adapter selection

    Args:
        base_weights: The frozen base weight matrix (d_out, d_in).
        base_bias:    Optional bias vector.
        config:       LoRAConfig for default adapter settings.
    """

    def __init__(
        self,
        base_weights: np.ndarray,
        base_bias: np.ndarray | None = None,
        config: LoRAConfig | None = None,
    ) -> None:
        self.base_weights = base_weights
        self.base_bias = base_bias
        self.config = config or LoRAConfig()
        self.d_out, self.d_in = base_weights.shape
        self.adapters: dict[str, LoRALayer] = {}

    def add_adapter(
        self,
        name: str,
        rank: int | None = None,
        alpha: float | None = None,
        dropout: float | None = None,
        seed: int | None = None,
    ) -> LoRALayer:
        """Add a new LoRA adapter.

        Args:
            name:     Adapter name (unique).
            rank:     Override config rank.
            alpha:    Override config alpha.
            dropout:  Override config dropout.
            seed:     Override config seed.

        Returns:
            The created LoRALayer.
        """
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already exists")
        adapter = LoRALayer(
            d_in=self.d_in,
            d_out=self.d_out,
            rank=rank or self.config.r,
            alpha=alpha or self.config.alpha,
            dropout=dropout or self.config.dropout,
            seed=seed or self.config.seed,
        )
        self.adapters[name] = adapter
        return adapter

    def get_adapter(self, name: str) -> LoRALayer:
        """Retrieve an adapter by name."""
        return self.adapters[name]

    def merge_adapter(self, name: str) -> np.ndarray:
        """Merge a single adapter's LoRA weights into the base weights.

        Returns the merged weight matrix (does not modify base_weights).
        """
        adapter = self.adapters[name]
        return adapter.merge(self.base_weights)

    def merge_all(self) -> np.ndarray:
        """Merge all adapters' LoRA weights into the base weights.

        Returns the merged weight matrix.
        """
        merged = self.base_weights.copy()
        for adapter in self.adapters.values():
            delta = adapter.scaling * np.matmul(adapter.B, adapter.A)
            merged = merged + delta
        return merged

    def unmerge(self, merged_weights: np.ndarray, adapter_names: list[str] | None = None) -> np.ndarray:
        """Remove specified adapters' contributions from merged weights."""
        result = merged_weights.copy()
        names = adapter_names or list(self.adapters.keys())
        for name in names:
            adapter = self.adapters[name]
            delta = adapter.scaling * np.matmul(adapter.B, adapter.A)
            result = result - delta
        return result

    def forward(
        self,
        x: np.ndarray,
        adapter_names: list[str] | None = None,
        return_attention: bool = False,
    ) -> np.ndarray:
        """Forward pass: base layer + selected adapter(s).

        Args:
            x:             Input array (seq_len, d_in) or (batch, seq_len, d_in).
            adapter_names: Names of adapters to apply (default: all).
            return_attention: If True, also return attention weights.

        Returns:
            Output array (seq_len, d_out) or (batch, seq_len, d_out).
        """
        # Base layer
        if x.ndim == 2:
            base_output = np.matmul(x, self.base_weights.T)
        else:
            base_output = np.matmul(x, self.base_weights.T)

        if self.base_bias is not None:
            base_output = base_output + self.base_bias

        names = adapter_names or list(self.adapters.keys())
        for name in names:
            if name in self.adapters:
                adapter = self.adapters[name]
                base_output = adapter.forward(base_output, x)

        if return_attention:
            return base_output, None
        return base_output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters across all adapters."""
        return sum(a.A.size + a.B.size for a in self.adapters.values())

    def count_base_parameters(self) -> int:
        """Count the frozen base model parameters."""
        return self.base_weights.size + (self.base_bias.size if self.base_bias is not None else 0)

    @property
    def parameter_ratio(self) -> float:
        """Ratio of LoRA parameters to base parameters."""
        base = self.count_base_parameters()
        if base == 0:
            return 0.0
        return self.count_parameters() / base


# ---------------------------------------------------------------------------
# LoRA Trainer
# ---------------------------------------------------------------------------

class LoRATrainer:
    """Simple trainer for LoRA adapter parameters.

    Uses SGD with optional momentum.  Only LoRA matrices (A, B) are
    updated — the base model remains frozen.

    Args:
        adapter:   The LoRAAdapter to train.
        lr:        Learning rate (default: 0.001).
        momentum:  SGD momentum (default: 0.9).
        weight_decay: L2 regularization (default: 0.01).
    """

    def __init__(
        self,
        adapter: LoRAAdapter,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
    ) -> None:
        self.adapter = adapter
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[str, np.ndarray] = {}
        self._init_velocity()

    def _init_velocity(self) -> None:
        for name, adapter in self.adapter.adapters.items():
            self._velocity[f"{name}_A"] = np.zeros_like(adapter.A)
            self._velocity[f"{name}_B"] = np.zeros_like(adapter.B)

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
        adapter_name: str,
    ) -> float:
        """Perform a single training step.

        Uses gradient computation via finite differences for the LoRA
        matrices (since we use numpy, not autograd).

        Args:
            x:             Input (seq_len, d_in) or (batch, d_in).
            target:        Target output (seq_len, d_out).
            adapter_name:  Name of the adapter to update.

        Returns:
            Loss value (MSE).
        """
        adapter = self.adapter.adapters[adapter_name]
        eps = 1e-4

        # Forward pass
        output = self.adapter.forward(x, adapter_names=[adapter_name])
        loss = float(np.mean((output - target) ** 2))

        # Numerical gradients via finite differences
        grad_A = np.zeros_like(adapter.A)
        grad_B = np.zeros_like(adapter.B)

        # Gradient for A (perturb each element)
        for i in range(min(adapter.rank, 4)):  # limit for performance
            for j in range(min(adapter.d_in, 8)):
                orig = adapter.A[i, j]
                adapter.A[i, j] = orig + eps
                out_p = self.adapter.forward(x, adapter_names=[adapter_name])
                loss_p = np.mean((out_p - target) ** 2)
                adapter.A[i, j] = orig - eps
                out_m = self.adapter.forward(x, adapter_names=[adapter_name])
                loss_m = np.mean((out_m - target) ** 2)
                adapter.A[i, j] = orig
                grad_A[i, j] = (loss_p - loss_m) / (2 * eps)

        # Gradient for B (perturb each element)
        for i in range(min(adapter.d_out, 8)):
            for j in range(min(adapter.rank, 4)):
                orig = adapter.B[i, j]
                adapter.B[i, j] = orig + eps
                out_p = self.adapter.forward(x, adapter_names=[adapter_name])
                loss_p = np.mean((out_p - target) ** 2)
                adapter.B[i, j] = orig - eps
                out_m = self.adapter.forward(x, adapter_names=[adapter_name])
                loss_m = np.mean((out_m - target) ** 2)
                adapter.B[i, j] = orig
                grad_B[i, j] = (loss_p - loss_m) / (2 * eps)

        # Add weight decay
        grad_A += self.weight_decay * adapter.A
        grad_B += self.weight_decay * adapter.B

        # Momentum update
        va = self._velocity[f"{adapter_name}_A"]
        vb = self._velocity[f"{adapter_name}_B"]
        va = self.momentum * va + self.lr * grad_A
        vb = self.momentum * vb + self.lr * grad_B
        self._velocity[f"{adapter_name}_A"] = va
        self._velocity[f"{adapter_name}_B"] = vb

        adapter.A -= va
        adapter.B -= vb

        return loss

    def train(
        self,
        train_data: list[tuple[np.ndarray, np.ndarray]],
        adapter_name: str,
        epochs: int = 10,
    ) -> list[float]:
        """Train for multiple epochs.

        Args:
            train_data: List of (input, target) pairs.
            adapter_name: Name of the adapter to train.
            epochs: Number of epochs.

        Returns:
            List of average loss per epoch.
        """
        losses = []
        for _epoch in range(epochs):
            epoch_losses = []
            for x, target in train_data:
                loss = self.train_step(x, target, adapter_name)
                epoch_losses.append(loss)
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            losses.append(avg_loss)
        return losses


def merge_lora(
    base_weights: np.ndarray,
    adapters: dict[str, LoRALayer],
) -> np.ndarray:
    """Merge all LoRA adapters into base weights.

    Args:
        base_weights: Original weight matrix.
        adapters:     Dict of {name: LoRALayer}.

    Returns:
        Merged weight matrix = base + sum(scaling * B @ A).
    """
    merged = base_weights.copy()
    for adapter in adapters.values():
        delta = adapter.scaling * np.matmul(adapter.B, adapter.A)
        merged = merged + delta
    return merged


def estimate_lora_parameters(
    d_model: int,
    num_layers: int,
    rank: int,
    target_modules: int = 2,
) -> int:
    """Estimate the number of trainable LoRA parameters.

    Args:
        d_model:         Model dimension.
        num_layers:      Number of transformer layers.
        rank:            LoRA rank.
        target_modules:  Number of modules per layer that get LoRA (e.g., q_proj, v_proj).

    Returns:
        Estimated number of trainable parameters.
    """
    # Each module has A (rank × d_model) + B (d_model × rank) = 2 * rank * d_model
    per_module = 2 * rank * d_model
    return per_module * num_layers * target_modules


def estimate_memory_savings(
    d_model: int,
    num_layers: int,
    rank: int,
    target_modules: int = 2,
) -> dict[str, Any]:
    """Estimate memory and parameter savings from using LoRA vs full fine-tuning.

    Returns a dict with:
        - full_params: Parameters if fully fine-tuning all weights
        - lora_params: Parameters with LoRA
        - reduction_ratio: full_params / lora_params
        - reduction_percent: Percentage reduction
    """
    full_params = d_model * d_model * num_layers * target_modules  # W_q, W_v per layer
    lora_params = estimate_lora_parameters(d_model, num_layers, rank, target_modules)
    ratio = full_params / lora_params if lora_params > 0 else float("inf")
    return {
        "full_params": full_params,
        "lora_params": lora_params,
        "reduction_ratio": ratio,
        "reduction_percent": (1 - lora_params / full_params) * 100 if full_params > 0 else 0,
    }

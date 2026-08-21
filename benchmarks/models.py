"""Model adapters for the benchmark harness.

A ModelAdapter is anything that can complete a prompt and optionally score the
log-probability of a continuation. Adapters make the harness model-agnostic:
legacy Silverwing checkpoints, external transformers models and future
Silverwing releases all plug in the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str

    def complete(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a completion for the prompt."""
        ...

    def log_prob(self, prompt: str, continuation: str) -> float:
        """Log-probability of continuation given prompt. Raise if unsupported."""
        ...


class DummyModel:
    """Deterministic stub model for tests and pipeline smoke runs."""

    model_id = "dummy"

    def __init__(self, answer: str = "42") -> None:
        self._answer = answer

    def complete(self, prompt: str, max_new_tokens: int = 128) -> str:
        return self._answer

    def log_prob(self, prompt: str, continuation: str) -> float:
        return -0.1


class SilverwingModel:
    """Adapter around a Silverwing Decoder V2 checkpoint (foundation model)."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str = "experiments/tokenizer",
        model_config: str = "configs/model.yaml",
        device: str | None = None,
        prompt_template: str | None = None,
    ) -> None:
        import torch

        from foundation.model import ModelConfig, build_model
        from foundation.tokenizer import TokenizerV2
        from foundation.training import load_checkpoint

        self._torch = torch
        self._cfg = ModelConfig.from_yaml(model_config)
        self._model = build_model(self._cfg)
        self._tokenizer = TokenizerV2.load(tokenizer_dir)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        load_checkpoint(checkpoint_path, self._model, None, self._device)
        self._model.to(self._device)
        self._model.eval()
        self._eos_id = self._tokenizer.special_ids["<|endoftext|>"]
        self._prompt_template = prompt_template
        self.model_id = f"silverwing:{Path(checkpoint_path).name}"

    def _render(self, prompt: str) -> str:
        if self._prompt_template is None:
            return prompt
        return self._prompt_template.format(prompt=prompt)

    def complete(self, prompt: str, max_new_tokens: int = 128) -> str:
        ids = self._tokenizer.encode(self._render(prompt))
        generated: list[int] = []
        with self._torch.no_grad():
            for _ in range(max_new_tokens):
                window = ids[-self._cfg.block_size :]
                x = self._torch.tensor([window], dtype=self._torch.long, device=self._device)
                logits = self._model(x)
                next_id = int(logits[0, -1].argmax())
                generated.append(next_id)
                ids = ids + [next_id]
                if next_id == self._eos_id:
                    break
        n_special = len(self._tokenizer.special_ids)
        plain = [token_id for token_id in generated if token_id >= n_special]
        return self._tokenizer.decode(plain).strip()

    def log_prob(self, prompt: str, continuation: str) -> float:
        import torch.nn.functional as F

        text_ids = self._tokenizer.encode(self._render(prompt) + continuation)
        prefix_ids = self._tokenizer.encode(self._render(prompt))
        context = self._torch.tensor([text_ids], dtype=self._torch.long, device=self._device)
        with self._torch.no_grad():
            logits = self._model(context)[0]
        logp = F.log_softmax(logits[:-1], dim=-1)
        positions = self._torch.arange(len(prefix_ids) - 1, len(text_ids) - 1, device=self._device)
        targets = self._torch.tensor(text_ids[len(prefix_ids) :], dtype=self._torch.long, device=self._device)
        selected = logp[positions, targets]
        return float(selected.sum())


class TransformersModel:
    """Adapter around a transformers causal LM (optional dependency)."""

    def __init__(self, model_name: str, device: str | None = None, max_new_tokens: int = 128) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - requires optional deps
            raise ImportError("TransformersModel requires `pip install transformers torch`") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_id = model_name
        self.max_new_tokens = max_new_tokens
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)
        self._model.eval()

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        limit = max_new_tokens or self.max_new_tokens
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs.to(self._device),
                max_new_tokens=limit,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def log_prob(self, prompt: str, continuation: str) -> float:
        text = prompt + continuation
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        prefix_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)
        labels = inputs["input_ids"].clone()
        labels[:, : prefix_ids.shape[1]] = -100
        with self._torch.no_grad():
            out = self._model(**inputs, labels=labels)
        loss = out.loss.item()
        n_tokens = (labels != -100).sum().item()
        return -loss * n_tokens if n_tokens > 0 else 0.0

"""Native inference engine for Silverwing Decoder V2.

Provides efficient autoregressive generation with:

* **KV-cache** — keys and values from previous positions are cached so each
  decode step only processes the single new token (O(1) per-token latency
  instead of the O(T²) re-encode used by the benchmark adapter).
* **Configurable sampling** — greedy, top-k, top-p (nucleus), temperature,
  and repetition penalty.
* **Batched generation** — multiple prompts padded to the same length and
  generated in parallel with per-sequence KV caches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from foundation.model import ModelConfig, SilverwingDecoder, build_model
from foundation.tokenizer import TokenizerV2

from .config import InferenceConfig


@dataclass
class GenerationResult:
    """Result of a single generation call.

    ``text`` is the decoded completion (excluding the prompt, with special
    tokens stripped). ``token_ids`` are the raw token IDs of the generated
    continuation (before special-token stripping).
    """

    text: str
    token_ids: list[int]


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set all but the top-k logits to -inf."""
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    threshold = v[-1]
    return torch.where(logits < threshold, torch.full_like(logits, -torch.inf), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Set to -inf the smallest tokens whose cumulative probability exceeds p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, -torch.inf)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Apply the HF-style repetition penalty to ``logits``.

    Tokens that have already appeared in ``generated_ids`` are scaled: if the
    logit is positive, divide by ``penalty``; otherwise multiply by
    ``penalty``.
    """
    if penalty == 1.0 or generated_ids.numel() == 0:
        return logits
    unique_ids = torch.unique(generated_ids)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(0, unique_ids, True)
    logits = logits.clone()
    logits[mask] = torch.where(
        logits[mask] > 0,
        logits[mask] / penalty,
        logits[mask] * penalty,
    )
    return logits


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    """Pad a list of variable-length token ID lists to a rectangular tensor."""
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample a single token ID from a 1-D logits tensor.

    Greedy when ``temperature <= 0`` (optionally narrowed to top-k).
    """
    logits = logits.clone()
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if 0.0 < top_p < 1.0:
        logits = _top_p_logits(logits, top_p)
    if temperature <= 0.0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits / temperature, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())


class Generator:
    """Efficient batched autoregressive generator with KV cache.

    Usage::

        cfg = InferenceConfig.from_yaml("configs/inference.yaml")
        gen = Generator.from_config(cfg)
        result = gen.generate("Hello, how are you?")
    """

    def __init__(
        self,
        model: SilverwingDecoder,
        tokenizer: TokenizerV2,
        *,
        device: str | torch.device = "cpu",
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_on_eos: bool = True,
        prompt_template: str | None = None,
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_new_tokens = max_new_tokens
        self._min_new_tokens = min_new_tokens
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._stop_on_eos = stop_on_eos
        self._prompt_template = prompt_template
        self._eos_id = tokenizer.special_ids["<|endoftext|>"]
        self._pad_id = tokenizer.special_ids.get("<|pad|>", self._eos_id)
        self._n_special = len(tokenizer.special_ids)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: InferenceConfig) -> Generator:
        """Build a generator from an :class:`InferenceConfig`."""
        from foundation.training import load_checkpoint

        model_cfg = ModelConfig.from_yaml(cfg.model_config_path)
        model = build_model(model_cfg)
        load_checkpoint(cfg.checkpoint_path, model, None, cfg.device)
        tokenizer = TokenizerV2.load(cfg.tokenizer_dir)
        return cls(
            model,
            tokenizer,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            stop_on_eos=cfg.stop_on_eos,
            prompt_template=cfg.prompt_template,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _render_prompt(self, prompt: str) -> str:
        if self._prompt_template is None:
            return prompt
        return self._prompt_template.format(prompt=prompt)

    def generate(
        self,
        prompt: str | Sequence[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> GenerationResult | list[GenerationResult]:
        """Generate completion(s) for a prompt or batch of prompts.

        If ``prompt`` is a string, returns a single :class:`GenerationResult`.
        If ``prompt`` is a sequence of strings, returns one result per input.
        """
        eff_max_new = (
            max_new_tokens if max_new_tokens is not None else self._max_new_tokens
        )
        eff_temp = temperature if temperature is not None else self._temperature
        eff_top_k = top_k if top_k is not None else self._top_k
        eff_top_p = top_p if top_p is not None else self._top_p

        if isinstance(prompt, str):
            return self._generate_single(
                prompt, eff_max_new, eff_temp, eff_top_k, eff_top_p
            )
        return self._generate_batch(
            list(prompt), eff_max_new, eff_temp, eff_top_k, eff_top_p
        )

    # ------------------------------------------------------------------
    # Single-prompt generation (KV cache warm-started from full prompt)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GenerationResult:
        rendered = self._render_prompt(prompt)
        input_ids = self._tokenizer.encode(rendered)
        if len(input_ids) == 0:
            input_ids = [self._eos_id]

        prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        generated: list[int] = []

        # Warm KV cache: encode the full prompt. We only need the cache and
        # the logits at the last position — we do NOT re-feed the last
        # prompt token in the decode loop.
        step_logits, past_key_values = self._model(prompt_tensor, use_cache=True)
        next_logits = step_logits[0, -1, :]

        # Generate token by token.
        for step in range(max_new_tokens):
            gen_tensor = (
                torch.tensor(generated, dtype=torch.long, device=self._device)
                if generated
                else torch.zeros(0, dtype=torch.long, device=self._device)
            )
            penalized = _apply_repetition_penalty(
                next_logits, gen_tensor, self._repetition_penalty
            )
            token_id = _sample_token(penalized, temperature, top_k, top_p)

            if (
                token_id == self._eos_id
                and self._stop_on_eos
                and step >= self._min_new_tokens
            ):
                break

            generated.append(token_id)

            # Feed only the newly generated token, reuse KV cache.
            x = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
            step_logits, past_key_values = self._model(
                x, use_cache=True, past_key_values=past_key_values
            )
            next_logits = step_logits[0, -1, :]

        plain = [t for t in generated if t >= self._n_special]
        text = self._tokenizer.decode(plain).strip()
        return GenerationResult(text=text, token_ids=generated)

    # ------------------------------------------------------------------
    # Batched generation (all prompts padded to same length)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[GenerationResult]:
        rendered = [self._render_prompt(p) for p in prompts]
        encoded = [self._tokenizer.encode(r) for r in rendered]
        for i in range(len(encoded)):
            if len(encoded[i]) == 0:
                encoded[i] = [self._eos_id]

        batch_size = len(encoded)
        input_ids = _pad_sequences(encoded, self._pad_id).to(self._device)
        attention_mask = input_ids != self._pad_id  # (B, T)

        # results[b] = list of generated token IDs for sequence b
        results: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        # Pre-fill: encode the entire prompt batch at once.
        step_logits, past_key_values = self._model(input_ids, use_cache=True)

        # The last real token position for each sequence (before padding).
        last_real = attention_mask.sum(dim=1) - 1  # (B,)

        for step in range(max_new_tokens):
            # Get logits for the position we need to sample from.
            if step == 0:
                # After pre-fill: use the logits at each sequence's last real position.
                batch_idx = torch.arange(batch_size, device=self._device)
                next_logits = step_logits[batch_idx, last_real, :].clone()  # (B, V)
            else:
                next_logits = step_logits[:, -1, :].clone()  # (B, V)
            step_logits = None

            next_ids: list[int] = []
            for b in range(batch_size):
                if finished[b].item():
                    next_ids.append(self._pad_id)
                    continue
                logit = next_logits[b]
                gen_tensor = (
                    torch.tensor(results[b], dtype=torch.long, device=self._device)
                    if results[b]
                    else torch.zeros(0, dtype=torch.long, device=self._device)
                )
                logit = _apply_repetition_penalty(
                    logit, gen_tensor, self._repetition_penalty
                )
                token_id = _sample_token(logit, temperature, top_k, top_p)

                if (
                    token_id == self._eos_id
                    and self._stop_on_eos
                    and step >= self._min_new_tokens
                ):
                    finished[b] = True
                    next_ids.append(self._pad_id)
                else:
                    results[b].append(token_id)
                    next_ids.append(token_id)

            if finished.all():
                break

            next_input = torch.tensor(
                next_ids, dtype=torch.long, device=self._device
            ).unsqueeze(1)  # (B, 1)
            step_logits, past_key_values = self._model(
                next_input, use_cache=True, past_key_values=past_key_values
            )

        # Decode results.
        out: list[GenerationResult] = []
        for b in range(batch_size):
            generated_ids = results[b]
            plain = [t for t in generated_ids if t >= self._n_special]
            text = self._tokenizer.decode(plain).strip()
            out.append(GenerationResult(text=text, token_ids=generated_ids))
        return out

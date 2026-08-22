"""Pydantic schemas for Silverwing YAML configuration files.

Configs are validated at load time so malformed or inconsistent values fail
fast with actionable error messages instead of surfacing deep inside a
training run.  Unknown keys are tolerated (``extra="allow"``) so configs can
evolve without breaking older loaders; known fields are type-checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _SilverwingConfig(BaseModel):
    """Base config: strict about types of known fields, tolerant of extras."""

    model_config = ConfigDict(extra="allow")


class ModelConfig(_SilverwingConfig):
    """Schema for ``configs/model.yaml`` (section: ``model``)."""

    model_name: str = "silverwing-decoder-v2"
    vocab_size: int = Field(gt=0)
    block_size: int = Field(gt=0)
    n_layer: int = Field(gt=0)
    n_head: int = Field(gt=0)
    n_kv_head: int | None = Field(default=None, gt=0)
    n_embd: int = Field(gt=0)
    mlp_hidden_size: int | None = None
    mlp_activation: str = "swiglu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    norm_eps: float = Field(default=1e-5, gt=0.0)
    tie_embeddings: bool = True
    bias: bool = False
    init_std: float = Field(default=0.02, gt=0.0)
    rope_base: float = Field(default=10000.0, gt=0.0)


class TrainingConfig(_SilverwingConfig):
    """Schema for ``configs/training*.yaml`` (section: ``training``)."""

    version: str = "training-v1"
    model_config_path: str
    corpus_dir: str
    tokenizer_dir: str
    tokenizer_version: str | None = None
    checkpoint_dir: str
    resume_from: str | None = None
    batch_size: int = Field(ge=1)
    grad_accum_steps: int = Field(ge=1)
    block_size: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    lr: float = Field(gt=0.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    seed: int = 42
    log_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    eval_sequences: int = Field(default=8, ge=1)
    save_steps: int = Field(default=100, ge=1)
    max_tokens: int | None = None
    verify_dataset: bool = True
    expected_dataset_hash: str | None = None
    require_validation: bool = True
    require_clean_repo: bool = True
    device: str = "cpu"


class TokenizerConfig(_SilverwingConfig):
    """Schema for ``configs/tokenizer.yaml`` (section: ``tokenizer``)."""

    version: str = "tokenizer-v1"
    algorithm: str = "byte-level-bpe"
    vocab_size: int = Field(gt=0)
    min_frequency: int = Field(default=2, ge=1)
    seed: int = 42
    corpus_dir: str
    corpus_split: str = "train"
    max_documents: int | None = None
    max_bytes: int | None = None
    output_dir: str


# Filename prefix -> schema class + expected top-level YAML section.
CONFIG_REGISTRY: dict[str, tuple[type[BaseModel], str]] = {
    "model": (ModelConfig, "model"),
    "training": (TrainingConfig, "training"),
    "tokenizer": (TokenizerConfig, "tokenizer"),
}


def schema_for(path: Path) -> tuple[type[BaseModel], str] | None:
    """Return ``(schema_class, section)`` for a config path, or ``None``."""
    name = path.name.lower()
    for prefix, entry in CONFIG_REGISTRY.items():
        if name.startswith(prefix):
            return entry
    return None


def load_config(
    path: str | Path,
    schema: type[BaseModel] | None = None,
    section: str | None = None,
) -> BaseModel:
    """Load and validate a YAML config file.

    Parameters:
        path: Path to the YAML file.
        schema: Pydantic model to validate against.  If omitted, one is
            selected from :data:`CONFIG_REGISTRY` by filename prefix.
        section: Top-level YAML key holding the config body (auto-selected
            alongside the default schema when omitted).

    Returns:
        A validated config model instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: On YAML syntax errors, missing sections, or validation
            failures (message lists every offending field).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    if schema is None or section is None:
        resolved = schema_for(p)
        if resolved is not None:
            default_schema, default_section = resolved
            schema = schema or default_schema
            section = section or default_section

    try:
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{p}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a mapping at the top level")

    data: Any = raw
    if section is not None:
        if section not in raw:
            raise ValueError(
                f"{p}: missing required section '{section}' "
                f"(found sections: {sorted(raw)})"
            )
        data = raw[section]

    assert schema is not None  # narrowed above
    try:
        return schema.model_validate(data)
    except Exception as exc:
        raise ValueError(f"{p}: config validation failed:\n{exc}") from exc

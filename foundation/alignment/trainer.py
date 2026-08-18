"""DPO trainer: init-from checkpoint -> preference-aligned checkpoint (M12).

Implements Direct Preference Optimization (DPO) [Rafailov et al., 2023]:

    L_DPO = -log σ( β * (logπ_θ(y_w|x) - logπ_θ(y_l|x)
                          - logπ_ref(y_w|x) + logπ_ref(y_l|x)) )

The reference model π_ref is a frozen copy of the init checkpoint.  Only the
policy model π_θ is trained.  Checkpoints and the alignment report follow the
same M01 provenance discipline as pretraining and SFT.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model import ModelConfig, build_model
from ..tokenizer import TokenizerV2
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.optimizer import build_optimizer
from ..training.repo import git_commit, require_clean_repo
from ..training.scheduler import schedule_lr
from .config import AlignmentConfig
from .dataset import PreferenceDataset, dataset_hash, load_preferences

BEST_FILENAME = "best.pt"
FINAL_FILENAME = "final.pt"


def _log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Sum of token log-probs over response positions (labels != IGNORE_INDEX).

    logits: (B, T, V)  labels: (B, T)
    Returns per-example scalar sum of log p(chosen token) over supervised positions.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Shift: predict token t+1 from position t
    shift_logits = log_probs[:, :-1, :].clone()
    shift_labels = labels[:, 1:].clone()
    mask = shift_labels != -100
    # Gather log-probs of the actual next tokens
    gathered = torch.gather(
        shift_logits,
        dim=-1,
        index=shift_labels.unsqueeze(-1).clamp(min=0),
    ).squeeze(-1)
    gathered = gathered * mask.float()
    return gathered.sum(dim=-1)


def compute_dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids_w: torch.Tensor,
    labels_w: torch.Tensor,
    input_ids_l: torch.Tensor,
    labels_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO loss for a single pair of (chosen, rejected) sequences.

    All tensors are (B, T).  Returns a scalar loss averaged over the batch.
    """
    logits_w = model(input_ids_w)
    logits_l = model(input_ids_l)
    with torch.no_grad():
        ref_logits_w = ref_model(input_ids_w)
        ref_logits_l = ref_model(input_ids_l)

    logp_w = _log_probs_from_logits(logits_w, labels_w)
    logp_l = _log_probs_from_logits(logits_l, labels_l)
    ref_logp_w = _log_probs_from_logits(ref_logits_w, labels_w)
    ref_logp_l = _log_probs_from_logits(ref_logits_l, labels_l)

    # DPO loss: -log σ(β * (logπ(y_w) - logπ(y_l) - logπ_ref(y_w) + logπ_ref(y_l)))
    logits = beta * (logp_w - logp_l - ref_logp_w + ref_logp_l)
    loss = -F.logsigmoid(logits).mean()
    return loss


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    data_eval: PreferenceDataset,
    n_examples: int,
    beta: float,
    device: torch.device | str,
) -> float | None:
    if data_eval.n_blocks == 0:
        return None
    pairs = data_eval.ordered_blocks(n_examples)
    w_ids = torch.stack([p[0] for p in pairs]).to(device)
    w_lbl = torch.stack([p[1] for p in pairs]).to(device)
    l_ids = torch.stack([p[2] for p in pairs]).to(device)
    l_lbl = torch.stack([p[3] for p in pairs]).to(device)

    model.eval()
    loss = compute_dpo_loss(model, ref_model, w_ids, w_lbl, l_ids, l_lbl, beta)
    model.train()
    return float(loss.item())


def train_alignment(cfg: AlignmentConfig, log: Callable[[str], None] = print) -> dict:
    commit = require_clean_repo() if cfg.require_clean_repo else git_commit()

    model_cfg = ModelConfig.from_yaml(cfg.model_config_path)
    if cfg.block_size > model_cfg.block_size:
        raise ValueError(
            f"alignment block_size {cfg.block_size} exceeds model block_size {model_cfg.block_size}"
        )
    tokenizer = TokenizerV2.load(cfg.tokenizer_dir)
    if tokenizer.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} exceeds model vocab {model_cfg.vocab_size}"
        )
    if not Path(cfg.init_from).exists():
        raise ValueError(f"init checkpoint does not exist: {cfg.init_from}")

    examples = load_preferences(cfg.dataset_path)
    data_train = PreferenceDataset(
        examples,
        tokenizer,
        cfg.block_size,
        seed=cfg.seed,
        eval_fraction=cfg.eval_fraction,
        split="train",
    )
    data_eval = PreferenceDataset(
        examples,
        tokenizer,
        cfg.block_size,
        seed=cfg.seed,
        eval_fraction=cfg.eval_fraction,
        split="eval",
    )
    if data_train.n_blocks == 0:
        raise ValueError("alignment train split is empty")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_model(model_cfg).to(device)
    ref_model = copy.deepcopy(model)
    load_checkpoint(cfg.init_from, model, None, cfg.device)
    # Reference model = init checkpoint (frozen)
    load_checkpoint(cfg.init_from, ref_model, None, cfg.device)
    ref_model.eval()

    tokenizer_hash = tokenizer.digest()
    dataset_h = dataset_hash(cfg.dataset_path)

    optimizer, opt_report = build_optimizer(
        model, cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()

    batch_iter = data_train.shuffled_batches(cfg.batch_size, cfg.seed)
    best_path: Path | None = None
    best_eval = float("inf")
    train_loss = float("nan")
    grad_norm: float | None = None
    tokens_per_pair = 2 * cfg.block_size

    def persist(
        step: int, *, eval_loss: float | None = None, filename: str | None = None
    ) -> Path:
        return save_checkpoint(
            cfg.checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
            config_digest=cfg.digest(),
            tokenizer_hash=tokenizer_hash,
            dataset_hash=dataset_h,
            git_commit=commit,
            model_config_digest=model_cfg.digest(),
            best_eval_loss=best_eval if math.isfinite(best_eval) else None,
            eval_loss=eval_loss,
            filename=filename,
        )

    for step in range(1, cfg.max_steps + 1):
        batch = next(batch_iter)
        w_ids = torch.stack([b[0] for b in batch]).to(device)
        w_lbl = torch.stack([b[1] for b in batch]).to(device)
        l_ids = torch.stack([b[2] for b in batch]).to(device)
        l_lbl = torch.stack([b[3] for b in batch]).to(device)

        loss = compute_dpo_loss(
            model, ref_model, w_ids, w_lbl, l_ids, l_lbl, cfg.dpo_beta
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            )
        lr = schedule_lr(
            step - 1, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        train_loss = float(loss.item())

        if cfg.log_steps and step % cfg.log_steps == 0:
            elapsed = time.perf_counter() - start_time
            throughput = tokens_per_pair * step / elapsed if elapsed > 0 else 0.0
            grad_norm_str = f"{grad_norm:.4f}" if grad_norm is not None else "n/a"
            log(
                f"step {step}/{cfg.max_steps} lr {lr:.2e} loss {train_loss:.4f} "
                f"grad_norm {grad_norm_str} ({throughput:.0f} tok/s)"
            )

        if cfg.eval_steps and (step % cfg.eval_steps == 0 or step == cfg.max_steps):
            eval_loss = _evaluate(
                model, ref_model, data_eval, cfg.eval_examples, cfg.dpo_beta, device
            )
            if eval_loss is not None:
                log(f"step {step} eval_loss {eval_loss:.4f}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    best_path = persist(
                        step, eval_loss=eval_loss, filename=BEST_FILENAME
                    )
                    log(
                        f"step {step} new best eval_loss {eval_loss:.4f} -> {best_path!s}"
                    )

        if cfg.save_steps and step % cfg.save_steps == 0:
            persist(step)

    elapsed = time.perf_counter() - start_time
    final_eval = (
        _evaluate(model, ref_model, data_eval, cfg.eval_examples, cfg.dpo_beta, device)
        if data_eval.n_blocks > 0
        else None
    )
    final_path = persist(
        cfg.max_steps,
        eval_loss=final_eval,
        filename=FINAL_FILENAME,
    )

    finished_at = datetime.now(UTC).isoformat()
    report = {
        "run_id": run_id,
        "stage": "alignment",
        "model_name": model_cfg.model_name,
        "git_commit": commit,
        "require_clean_repo": cfg.require_clean_repo,
        "model_config_path": str(Path(cfg.model_config_path)),
        "model_config_digest": model_cfg.digest(),
        "tokenizer_hash": tokenizer_hash,
        "dataset_path": str(Path(cfg.dataset_path)),
        "dataset_hash": dataset_h,
        "init_from": cfg.init_from,
        "alignment_config_digest": cfg.digest(),
        "alignment_config": cfg.to_dict(),
        "num_parameters": model.num_parameters(),
        "num_preference_pairs": data_train.n_examples,
        "num_eval_pairs": data_eval.n_examples,
        "max_steps": cfg.max_steps,
        "tokens_per_pair": tokens_per_pair,
        "tokens_seen": tokens_per_pair * cfg.max_steps,
        "final_train_loss": train_loss,
        "final_grad_norm": grad_norm,
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval if best_eval != float("inf") else None,
        "best_checkpoint": str(best_path) if best_path else None,
        "final_checkpoint": str(final_path),
        "optimizer": opt_report,
        "device": cfg.device,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_tokens_per_sec": round(tokens_per_pair * cfg.max_steps / elapsed, 1)
        if elapsed > 0
        else 0.0,
        "started_at": started_at,
        "finished_at": finished_at,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint_dir": str(Path(cfg.checkpoint_dir)),
        "dpo_beta": cfg.dpo_beta,
    }
    report_path = Path(cfg.checkpoint_dir) / "alignment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

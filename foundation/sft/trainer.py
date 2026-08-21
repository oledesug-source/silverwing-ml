"""SFT trainer: pretrained weights -> instruction-tuned checkpoint (M11).

Loads a pretrained Silverwing checkpoint, trains with response-masked
cross-entropy on a packed SFT dataset, and writes best/final checkpoints plus
a provenance report under the same M01 reproducibility rules as pretraining.
"""

from __future__ import annotations

import json
import math
import sys
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
from .config import SftConfig
from .dataset import IGNORE_INDEX, SftDataset, dataset_hash, load_examples

BEST_FILENAME = "best.pt"
FINAL_FILENAME = "final.pt"


def _evaluate(
    model: torch.nn.Module,
    data_eval: SftDataset,
    n_blocks: int,
    device: torch.device | str,
) -> tuple[float, float, int]:
    model.eval()
    x, y = data_eval.ordered_blocks(n_blocks)
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    model.train()
    n_supervised = int((y != IGNORE_INDEX).sum().item())
    return float(loss.item()), float(math.exp(loss.item())), n_supervised


def train_sft(cfg: SftConfig, log: Callable[[str], None] = print) -> dict:
    commit = require_clean_repo() if cfg.require_clean_repo else git_commit()

    model_cfg = ModelConfig.from_yaml(cfg.model_config_path)
    if cfg.block_size > model_cfg.block_size:
        raise ValueError(f"SFT block_size {cfg.block_size} exceeds model block_size {model_cfg.block_size}")
    tokenizer = TokenizerV2.load(cfg.tokenizer_dir)
    if tokenizer.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} exceeds model vocab {model_cfg.vocab_size}"
        )
    if not Path(cfg.init_from).exists():
        raise ValueError(f"init checkpoint does not exist: {cfg.init_from}")

    examples = load_examples(cfg.dataset_path)
    data_train = SftDataset(
        examples, tokenizer, cfg.block_size, seed=cfg.seed, eval_fraction=cfg.eval_fraction, split="train"
    )
    data_eval = SftDataset(
        examples, tokenizer, cfg.block_size, seed=cfg.seed, eval_fraction=cfg.eval_fraction, split="eval"
    )
    if data_train.n_blocks == 0:
        raise ValueError("SFT train split is empty")
    if data_eval.n_blocks == 0:
        raise ValueError("SFT eval split is empty")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.amp_dtype]
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    model = build_model(model_cfg).to(device)

    load_checkpoint(cfg.init_from, model, None, cfg.device)
    tokenizer_hash = tokenizer.digest()
    sft_dataset_hash = dataset_hash(cfg.dataset_path)

    optimizer, opt_report = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()

    batch_iter = data_train.shuffled_batches(cfg.batch_size, cfg.seed)
    tokens_per_step = cfg.batch_size * cfg.block_size
    best_path: Path | None = None
    best_eval = float("inf")
    train_loss = float("nan")
    grad_norm: float | None = None

    def persist(step: int, *, eval_loss: float | None = None, filename: str | None = None) -> Path:
        return save_checkpoint(
            cfg.checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
            config_digest=cfg.digest(),
            tokenizer_hash=tokenizer_hash,
            dataset_hash=sft_dataset_hash,
            git_commit=commit,
            model_config_digest=model_cfg.digest(),
            best_eval_loss=best_eval if math.isfinite(best_eval) else None,
            eval_loss=eval_loss,
            filename=filename,
        )

    for step in range(1, cfg.max_steps + 1):
        x, y = next(batch_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
        lr = schedule_lr(step - 1, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        train_loss = float(loss.item())

        if cfg.log_steps and step % cfg.log_steps == 0:
            elapsed = time.perf_counter() - start_time
            throughput = tokens_per_step * step / elapsed if elapsed > 0 else 0.0
            grad_norm_str = f"{grad_norm:.4f}" if grad_norm is not None else "n/a"
            log(
                f"step {step}/{cfg.max_steps} lr {lr:.2e} loss {train_loss:.4f} "
                f"grad_norm {grad_norm_str} ({throughput:.0f} tok/s)"
            )

        if cfg.eval_steps and (step % cfg.eval_steps == 0 or step == cfg.max_steps):
            eval_loss, ppl, n_supervised = _evaluate(model, data_eval, cfg.eval_examples, device)
            log(f"step {step} eval_loss {eval_loss:.4f} ppl {ppl:.2f} (n={n_supervised})")
            if eval_loss < best_eval:
                best_eval = eval_loss
                best_path = persist(step, eval_loss=eval_loss, filename=BEST_FILENAME)
                log(f"step {step} new best eval_loss {eval_loss:.4f} -> {str(best_path)}")

        if cfg.save_steps and step % cfg.save_steps == 0:
            persist(step)

    elapsed = time.perf_counter() - start_time
    final_eval = _evaluate(model, data_eval, cfg.eval_examples, device)
    final_path = persist(cfg.max_steps, eval_loss=final_eval[0], filename=FINAL_FILENAME)

    finished_at = datetime.now(UTC).isoformat()
    report = {
        "run_id": run_id,
        "stage": "sft",
        "model_name": model_cfg.model_name,
        "git_commit": commit,
        "require_clean_repo": cfg.require_clean_repo,
        "model_config_path": str(Path(cfg.model_config_path)),
        "model_config_digest": model_cfg.digest(),
        "tokenizer_hash": tokenizer_hash,
        "init_from": cfg.init_from,
        "dataset_path": str(Path(cfg.dataset_path)),
        "dataset_hash": sft_dataset_hash,
        "sft_config_digest": cfg.digest(),
        "sft_config": cfg.to_dict(),
        "num_parameters": model.num_parameters(),
        "num_train_examples": data_train.n_examples,
        "num_eval_examples": data_eval.n_examples,
        "supervised_tokens_train": data_train.supervised_tokens,
        "max_steps": cfg.max_steps,
        "tokens_per_step": tokens_per_step,
        "tokens_seen": tokens_per_step * cfg.max_steps,
        "final_train_loss": train_loss,
        "final_grad_norm": grad_norm,
        "final_eval_loss": final_eval[0],
        "final_perplexity": final_eval[1],
        "best_eval_loss": best_eval if best_eval != float("inf") else None,
        "best_checkpoint": str(best_path) if best_path else None,
        "final_checkpoint": str(final_path),
        "optimizer": opt_report,
        "device": cfg.device,
        "amp": {"enabled": use_amp, "dtype": cfg.amp_dtype if use_amp else None},
        "elapsed_seconds": round(elapsed, 3),
        "throughput_tokens_per_sec": round(tokens_per_step * cfg.max_steps / elapsed, 1) if elapsed > 0 else 0.0,
        "started_at": started_at,
        "finished_at": finished_at,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "checkpoint_dir": str(Path(cfg.checkpoint_dir)),
    }
    report_path = Path(cfg.checkpoint_dir) / "sft_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

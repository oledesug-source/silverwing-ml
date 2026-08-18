"""Curriculum trainer (M14).

Runs SFT stages sequentially with increasing difficulty. Each stage loads
from the previous stage's best checkpoint and saves its own best/final
checkpoints. Produces a curriculum report after all stages complete.
"""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..model import ModelConfig, build_model
from ..sft.dataset import IGNORE_INDEX, SftDataset, dataset_hash, load_examples
from ..tokenizer import TokenizerV2
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.optimizer import build_optimizer
from ..training.repo import git_commit, require_clean_repo
from ..training.scheduler import schedule_lr
from .config import CurriculumConfig

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


def _train_stage(
    model: torch.nn.Module,
    tokenizer: TokenizerV2,
    stage: StageConfig,
    model_cfg: ModelConfig,
    tokenizer_hash: str,
    commit: str,
    device: torch.device,
    global_step: int,
    log: Callable[[str], None],
    best_eval_global: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    eval_examples: int = 32,
    eval_fraction: float = 0.1,
) -> tuple[float, int, float]:
    """Train one stage. Returns (best_eval_loss, global_step, train_loss)."""
    examples = load_examples(stage.dataset_path)
    stage_hash = dataset_hash(stage.dataset_path)
    data_train = SftDataset(
        examples, tokenizer, model_cfg.block_size, seed=42,
        eval_fraction=eval_fraction, split="train",
    )
    data_eval = SftDataset(
        examples, tokenizer, model_cfg.block_size, seed=42,
        eval_fraction=eval_fraction, split="eval",
    )
    if data_train.n_blocks == 0:
        raise ValueError(f"Stage '{stage.name}': train split empty")
    if data_eval.n_blocks == 0:
        raise ValueError(f"Stage '{stage.name}': eval split empty")

    optimizer, _ = build_optimizer(model, stage.lr, weight_decay, betas, eps)
    tokens_per_step = model_cfg.block_size
    start_time = time.perf_counter()
    best_stage_eval = float("inf")
    best_path: Path | None = None

    batch_iter = data_train.shuffled_batches(1, 42)
    for step in range(1, stage.max_steps + 1):
        global_step += 1
        x, y = next(batch_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if stage.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage.grad_clip)
        lr = schedule_lr(step - 1, stage.lr, stage.warmup_steps, stage.max_steps, 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        train_loss = float(loss.item())

        if stage.log_steps and step % stage.log_steps == 0:
            elapsed = time.perf_counter() - start_time
            log(f"  [{stage.name}] step {step}/{stage.max_steps} lr {lr:.2e} loss {train_loss:.4f}")

        if stage.eval_steps and (step % stage.eval_steps == 0 or step == stage.max_steps):
            eval_loss, ppl, _ = _evaluate(model, data_eval, eval_examples, device)
            log(f"  [{stage.name}] step {step} eval_loss {eval_loss:.4f} ppl {ppl:.2f}")
            if eval_loss < best_stage_eval:
                best_stage_eval = eval_loss
                best_path = Path(stage.checkpoint_dir)
                best_path.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    str(best_path), step=global_step, model=model, optimizer=optimizer,
                    run_id=f"curriculum-{stage.name}", config_digest="",
                    tokenizer_hash=tokenizer_hash, dataset_hash=stage_hash,
                    git_commit=commit, model_config_digest=model_cfg.digest(),
                    best_eval_loss=best_stage_eval if math.isfinite(best_stage_eval) else None,
                    eval_loss=eval_loss, filename=BEST_FILENAME,
                )
                log(f"  [{stage.name}] new best {eval_loss:.4f}")

        if stage.save_steps and step % stage.save_steps == 0:
            Path(stage.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                stage.checkpoint_dir, step=global_step, model=model, optimizer=optimizer,
                run_id=f"curriculum-{stage.name}", config_digest="",
                tokenizer_hash=tokenizer_hash, dataset_hash=stage_hash,
                git_commit=commit, model_config_digest=model_cfg.digest(),
                best_eval_loss=None, eval_loss=None, filename=f"step-{global_step}.pt",
            )

    return best_stage_eval, global_step, train_loss


def train_curriculum(
    cfg: CurriculumConfig,
    log: Callable[[str], None] = print,
) -> dict:
    commit = require_clean_repo() if cfg.require_clean_repo else git_commit()

    model_cfg = ModelConfig.from_yaml(cfg.model_config_path)
    tokenizer = TokenizerV2.load(cfg.tokenizer_dir)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(f"vocab mismatch: tokenizer={tokenizer.vocab_size}, model={model_cfg.vocab_size}")

    device = torch.device(cfg.device)
    model = build_model(model_cfg).to(device)

    if not Path(cfg.init_from).exists():
        raise ValueError(f"init_from checkpoint does not exist: {cfg.init_from}")
    load_checkpoint(cfg.init_from, model, None, cfg.device)
    log(f"Loaded init checkpoint: {cfg.init_from}")

    tokenizer_hash = tokenizer.digest()
    commit_str = commit or "no-git"

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    global_step = 0
    best_eval = float("inf")
    stage_reports = []

    for i, stage in enumerate(cfg.stages):
        log(f"\n=== Stage {i+1}/{len(cfg.stages)}: {stage.name} ===")
        log(f"  dataset: {stage.dataset_path}")
        log(f"  steps: {stage.max_steps}, lr: {stage.lr}")

        stage_start = time.perf_counter()
        stage_best, global_step, stage_train_loss = _train_stage(
            model=model,
            tokenizer=tokenizer,
            stage=stage,
            model_cfg=model_cfg,
            tokenizer_hash=tokenizer_hash,
            commit=commit_str,
            device=device,
            global_step=global_step,
            log=log,
            best_eval_global=best_eval,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
            eval_examples=cfg.eval_examples,
            eval_fraction=cfg.eval_fraction,
        )
        stage_elapsed = time.perf_counter() - stage_start
        if stage_best < best_eval:
            best_eval = stage_best

        stage_reports.append({
            "name": stage.name,
            "dataset_path": stage.dataset_path,
            "max_steps": stage.max_steps,
            "lr": stage.lr,
            "best_eval_loss": stage_best if stage_best != float("inf") else None,
            "elapsed_seconds": round(stage_elapsed, 3),
        })
        log(f"  Stage {stage.name} done in {stage_elapsed:.1f}s, best_eval={stage_best:.4f}")

    elapsed = time.perf_counter() - start_time
    final_path = save_checkpoint(
        cfg.stages[-1].checkpoint_dir if cfg.stages else "experiments/checkpoints/curriculum",
        step=global_step, model=model, optimizer=None,
        run_id="curriculum-final", config_digest=cfg.digest(),
        tokenizer_hash=tokenizer_hash, dataset_hash=None,
        git_commit=commit_str, model_config_digest=model_cfg.digest(),
        best_eval_loss=best_eval if math.isfinite(best_eval) else None,
        eval_loss=None, filename=FINAL_FILENAME,
    )

    report = {
        "run_id": f"curriculum-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": cfg.version,
        "config_digest": cfg.digest(),
        "git_commit": commit_str,
        "model_config_path": cfg.model_config_path,
        "model_config_digest": model_cfg.digest(),
        "num_parameters": model.num_parameters(),
        "init_from": cfg.init_from,
        "total_steps": global_step,
        "num_stages": len(cfg.stages),
        "stages": stage_reports,
        "best_eval_loss": best_eval if best_eval != float("inf") else None,
        "final_checkpoint": str(final_path),
        "device": cfg.device,
        "elapsed_seconds": round(elapsed, 3),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    report_path = Path(cfg.stages[-1].checkpoint_dir if cfg.stages else "experiments/checkpoints/curriculum") / "curriculum_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nCurriculum complete. {len(cfg.stages)} stages, {global_step} total steps.")
    log(f"Report: {report_path}")
    return report

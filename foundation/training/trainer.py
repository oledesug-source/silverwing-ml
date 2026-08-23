"""The training engine: data -> model -> loss -> optimizer -> checkpoint."""

from __future__ import annotations

import json
import math
import sys as _stdlib_sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model import build_model
from ..tokenizer import TokenizerV2
from .checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from .config import TrainConfig
from .data import PretrainingData
from .optimizer import build_optimizer
from .preflight import preflight_train
from .repo import git_commit, require_clean_repo
from .scheduler import schedule_lr

BEST_FILENAME = "best.pt"
FINAL_FILENAME = "final.pt"


def _pad_id(tokenizer: TokenizerV2) -> int:
    return tokenizer.special_ids["<|pad|>"]


def evaluate(
    model: torch.nn.Module,
    tokenizer: TokenizerV2,
    data_val: PretrainingData,
    n_sequences: int,
    device: torch.device | str,
    *,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float] | None:
    batch = data_val.ordered_batch(n_sequences)
    if batch is None:
        return None
    x, y = batch
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None
        ):
            logits = model(x.to(device))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.to(device).reshape(-1),
            ignore_index=_pad_id(tokenizer),
        )
    model.train()
    eval_loss = float(loss.item())
    return eval_loss, float(math.exp(eval_loss))


def _validate_resume_checkpoint(
    checkpoint: dict,
    *,
    model_config_digest: str,
    tokenizer_hash: str,
    dataset_hash: str | None,
    resume_config_digest: str,
) -> None:
    """Reject a resume whose immutable assets differ from this run."""
    expected = {
        "model_config_digest": model_config_digest,
        "tokenizer_hash": tokenizer_hash,
        "dataset_hash": dataset_hash,
        "resume_config_digest": resume_config_digest,
    }
    for name, expected_value in expected.items():
        actual_value = checkpoint.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"resume checkpoint {name} does not match this training run: "
                f"checkpoint={actual_value!r}, current={expected_value!r}"
            )
    if checkpoint.get("data_state") is None or checkpoint.get("rng_state") is None:
        raise ValueError("resume checkpoint lacks exact data/RNG state; start a fresh M10 run instead")


def train(cfg: TrainConfig, log: Callable[[str], None] = print) -> dict:
    commit = require_clean_repo() if cfg.require_clean_repo else git_commit()

    inputs = preflight_train(cfg)
    model_cfg = inputs.model_config
    tokenizer = inputs.tokenizer
    data_train = inputs.train_data
    data_val = inputs.validation_data
    dataset_hash = inputs.dataset_hash

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.amp_dtype]
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    model = build_model(model_cfg).to(device)
    tokenizer_hash = tokenizer.digest()
    if cfg.init_from:
        if not Path(cfg.init_from).exists():
            raise ValueError(f"init_from checkpoint does not exist: {cfg.init_from}")
        load_checkpoint(cfg.init_from, model, None, cfg.device)

    optimizer, opt_report = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps)

    # Initialize experiment trackers (MLflow local file backend + offline W&B).
    # All trackers are optional: training proceeds without them when absent.
    mlflow_tracker = None
    wandb_run = None
    try:
        from foundation.ops.mlflow_tracker import MLflowTracker

        mlflow_tracker = MLflowTracker(
            experiment="silverwing-training", tracking_uri="experiments/mlruns"
        )
    except Exception:
        mlflow_tracker = None

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        import wandb

        wandb_run = wandb.init(
            project="silverwing-training",
            name=f"{model_cfg.model_name}-{run_id}",
            config=cfg.to_dict(),
            dir=str(Path("experiments/wandb").resolve()),
            mode="offline",
            anonymous="never",
            force=False,
        )
        use_wandb = True
    except ImportError:
        use_wandb = False
        wandb_run = None

    if mlflow_tracker is not None:
        _mlflow_run = mlflow_tracker.start_run(run_name=f"{model_cfg.model_name}-{run_id}", config=cfg.to_dict())
        _mlflow_run.__enter__()

    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()

    start_step = 1
    best_eval = float("inf")
    batch_stream = data_train.batch_stream(cfg.batch_size, cfg.seed)
    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from, model, optimizer, cfg.device)
        _validate_resume_checkpoint(
            ckpt,
            model_config_digest=model_cfg.digest(),
            tokenizer_hash=tokenizer_hash,
            dataset_hash=dataset_hash,
            resume_config_digest=cfg.resume_digest(),
        )
        start_step = ckpt["step"] + 1
        run_id = str(ckpt.get("run_id", run_id))
        best_eval = float(ckpt.get("best_eval_loss") or ckpt.get("eval_loss") or float("inf"))
        batch_stream.load_state_dict(ckpt["data_state"])
        restore_rng_state(ckpt["rng_state"])

    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
    best_path: Path | None = None
    train_loss = float("nan")
    grad_norm: torch.Tensor | None = None
    pending_loss: torch.Tensor | None = None

    def persist(step: int, *, eval_loss: float | None = None, filename: str | None = None) -> Path:
        return save_checkpoint(
            cfg.checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
            config_digest=cfg.digest(),
            tokenizer_hash=tokenizer_hash,
            dataset_hash=dataset_hash,
            git_commit=commit,
            model_config_digest=model_cfg.digest(),
            resume_config_digest=cfg.resume_digest(),
            data_state=batch_stream.state_dict(),
            rng_state=capture_rng_state(),
            best_eval_loss=best_eval if math.isfinite(best_eval) else None,
            eval_loss=eval_loss,
            filename=filename,
        )

    for step in range(start_step, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        for _ in range(cfg.grad_accum_steps):
            x, y = next(batch_stream)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=_pad_id(tokenizer),
                )
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            micro_losses.append(loss.detach())
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = schedule_lr(step - 1, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        # Keep the mean loss on GPU; only sync with .item() when a log line
        # actually needs it, so each step avoids a full pipeline stall.
        pending_loss = torch.stack(micro_losses).mean()

        if cfg.log_steps and step % cfg.log_steps == 0:
            train_loss = float(pending_loss)
            gn = float(grad_norm) if grad_norm is not None else None
            grad_norm_str = f"{gn:.4f}" if gn is not None else "n/a"
            elapsed = time.perf_counter() - start_time
            throughput = tokens_per_step * step / elapsed if elapsed > 0 else 0.0
            log(
                f"step {step}/{cfg.max_steps} lr {lr:.2e} loss {train_loss:.4f} "
                f"grad_norm {grad_norm_str} ({throughput:.0f} tok/s)"
            )
            # Log to W&B
            if use_wandb:
                wandb.log({
                    "step": step,
                    "train_loss": train_loss,
                    "lr": lr,
                    "grad_norm": gn,
                    "throughput": throughput,
                })
            if mlflow_tracker is not None:
                mlflow_tracker.log_metrics({
                    "step": step,
                    "train_loss": train_loss,
                    "lr": lr,
                    "grad_norm": gn,
                    "throughput": throughput,
                }, step=step)

        if cfg.eval_steps and (step % cfg.eval_steps == 0 or step == cfg.max_steps):
            result = evaluate(model, tokenizer, data_val, cfg.eval_sequences, device, amp_dtype=amp_dtype if use_amp else None)
            if result is not None:
                eval_loss, ppl = result
                log(f"step {step} eval_loss {eval_loss:.4f} ppl {ppl:.2f}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    best_path = persist(step, eval_loss=eval_loss, filename=BEST_FILENAME)
                    log(f"step {step} new best eval_loss {eval_loss:.4f} -> {str(best_path)}")
                # Log eval to W&B
                if use_wandb:
                    wandb.log({
                        "step": step,
                        "eval_loss": eval_loss,
                        "eval_perplexity": ppl,
                    })
                if mlflow_tracker is not None:
                    mlflow_tracker.log_metrics({
                        "eval_loss": eval_loss,
                        "eval_perplexity": ppl,
                    }, step=step)

        if cfg.save_steps and step % cfg.save_steps == 0:
            persist(step)

    elapsed = time.perf_counter() - start_time
    if math.isnan(train_loss) and pending_loss is not None:
        train_loss = float(pending_loss)
    final_result = evaluate(model, tokenizer, data_val, cfg.eval_sequences, device, amp_dtype=amp_dtype if use_amp else None)
    final_path = persist(
        cfg.max_steps,
        eval_loss=final_result[0] if final_result else None,
        filename=FINAL_FILENAME,
    )

    finished_at = datetime.now(UTC).isoformat()
    report = {
        "run_id": run_id,
        "model_name": model_cfg.model_name,
        "git_commit": commit,
        "require_clean_repo": cfg.require_clean_repo,
        "model_config_path": str(Path(cfg.model_config_path)),
        "model_config_digest": model_cfg.digest(),
        "tokenizer_hash": tokenizer_hash,
        "dataset_hash": dataset_hash,
        "train_config_digest": cfg.digest(),
        "train_config": cfg.to_dict(),
        "num_parameters": model.num_parameters(),
        "num_train_blocks": data_train.n_blocks,
        "num_train_documents": data_train.n_documents,
        "train_tokens_in_corpus": data_train.num_tokens(),
        "max_steps": cfg.max_steps,
        "steps_done": cfg.max_steps,
        "tokens_seen": tokens_per_step * cfg.max_steps,
        "tokens_per_step": tokens_per_step,
        "micro_batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "final_train_loss": train_loss,
        "final_grad_norm": float(grad_norm) if grad_norm is not None else None,
        "final_eval_loss": final_result[0] if final_result else None,
        "final_perplexity": final_result[1] if final_result else None,
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
        "python": f"{_stdlib_sys.version_info.major}.{_stdlib_sys.version_info.minor}.{_stdlib_sys.version_info.micro}",
        "torch": torch.__version__,
        "checkpoint_dir": str(Path(cfg.checkpoint_dir)),
        "resumed_from": cfg.resume_from,
        "init_from": cfg.init_from,
        "start_step": start_step,
        "preflight": inputs.report(),
    }
    report_path = Path(cfg.checkpoint_dir) / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    
     # Finish trackers
    if use_wandb:
        try:
            wandb.run.summary["final_train_loss"] = train_loss
            wandb.run.summary["final_eval_loss"] = final_result[0] if final_result else None
            wandb.run.summary["best_eval_loss"] = best_eval
            wandb.finish()
        except Exception:
            pass
    if mlflow_tracker is not None:
        try:
            mlflow_tracker.log_metric("final_train_loss", train_loss)
            mlflow_tracker.log_metric("final_eval_loss", final_result[0] if final_result else 0.0)
            mlflow_tracker.log_metric("best_eval_loss", best_eval)
            mlflow_tracker.log_artifact(report_path)
            mlflow_tracker.end_run()
            _mlflow_run.__exit__(None, None, None)
        except Exception:
            pass

    return report

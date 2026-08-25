"""Silverwing end-to-end GPU training pipeline (SFT-v3 -> DPO-v2).

ONE command on any GPU machine:

    python scripts/train_pipeline.py                 # full production run
    python scripts/train_pipeline.py --smoke         # 30+20 step validation run
    python scripts/train_pipeline.py --dry-run       # print plan only
    python scripts/train_pipeline.py --stage dpo     # resume at alignment stage

Stages:
    1. environment check   (GPU, torch)
    2. repo sync           (clone/hard-sync to origin/main unless already inside)
    3. artifact staging    (tokenizer + pretrained checkpoint + datasets)
    4. SFT-v3              (unified lesson track L01-L16, fp16 AMP)
    5. DPO-v2              (preference alignment, init from SFT best)
    6. collect outputs     (best.pt copies + reports into --output-dir)

Artifact lookup order: existing local files -> /kaggle/input mounts ->
Kaggle dataset download (videlisndichi/silverwing-state).
Credentials: KAGGLE_USERNAME/KAGGLE_KEY env vars, else ~/.kaggle/kaggle.json,
else embedded fallback below.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "videlisndichi")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "KGAT_4252a1b11f1dc8ea8a0d48a8bee2986d")
STATE_DS = f"{KAGGLE_USERNAME}/silverwing-state"
REPO_URL = "https://github.com/oledesug-source/silverwing-ml.git"

SFT_CONFIG = {
    "version": "sft-v3",
    "model_config_path": "configs/model.yaml",
    "batch_size": 16,
    "block_size": 512,
    "max_steps": 1200,
    "warmup_steps": 100,
    "lr": 1.0e-4,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "grad_clip": 1.0,
    "seed": 42,
    "log_steps": 25,
    "eval_steps": 200,
    "eval_examples": 64,
    "save_steps": 500,
    "eval_fraction": 0.05,
}

DPO_CONFIG = {
    "version": "alignment-v2",
    "model_config_path": "configs/model.yaml",
    "batch_size": 4,
    "block_size": 512,
    "max_steps": 800,
    "warmup_steps": 40,
    "lr": 1.0e-5,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.0,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "grad_clip": 1.0,
    "dpo_beta": 0.1,
    "label_smoothing": 0.0,
    "seed": 42,
    "log_steps": 10,
    "eval_steps": 100,
    "eval_examples": 16,
    "save_steps": 200,
    "eval_fraction": 0.05,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    log(">>> " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    for line in proc.stdout or []:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"FAILED ({rc}): {' '.join(str(c) for c in cmd)}")


# ----------------------------------------------------------------- stages

def check_gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("NO GPU visible - run on Colab T4 / Kaggle GPU / CUDA box")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {props.name} ({props.total_memory / 1e9:.0f} GB)")
    return "cuda"


def sync_repo() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "train_pipeline.py").exists():
        log(f"already inside repo: {cwd}")
        return cwd
    target = Path("/content/Silverwing-ML" if os.path.exists("/content") else "Silverwing-ML")
    if not target.exists():
        run(["git", "clone", REPO_URL, str(target)])
    os.chdir(target)
    run(["git", "fetch", "origin"])
    run(["git", "reset", "--hard", "origin/main"])
    return target


def _kaggle_cli() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def ensure_kaggle_creds() -> None:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    kg = Path.home() / ".kaggle"
    kg.mkdir(exist_ok=True)
    cred = kg / "kaggle.json"
    if not cred.exists():
        cred.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}))
    try:
        os.chmod(cred, 0o600)
    except OSError:
        pass


def find_file(name: str, subpath: list[str]) -> Path | None:
    """Locate an artifact: repo-relative, then /kaggle/input mounts."""
    direct = Path(name)
    if direct.exists():
        return direct
    mounts = Path("/kaggle/input")
    if mounts.is_dir():
        hits = sorted(mounts.rglob("/".join(subpath)))
        if hits:
            return hits[0]
    return None


def stage_artifacts(tok_dir: str, ckpt_out: Path, args_init_from: str | None = None) -> dict[str, str]:
    resolved: dict[str, str] = {}

    tok = Path(tok_dir)
    if not (tok / "vocab.json").exists():
        mount = find_file("vocab.json", ["tokenizer-v2", "vocab.json"])
        if mount:
            shutil.rmtree(tok, ignore_errors=True)
            shutil.copytree(mount.parent, tok)
            log(f"tokenizer staged from {mount.parent}")
        else:
            ensure_kaggle_creds()
            if not _kaggle_cli():
                run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
            dl = Path("_state_dl")
            shutil.rmtree(dl, ignore_errors=True)
            dl.mkdir(parents=True)
            r = subprocess.run(
                ["kaggle", "datasets", "download", STATE_DS, "-p", str(dl), "--unzip"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise SystemExit(f"state dataset pull failed:\n{(r.stderr or '')[-800:]}")
            state = dl / "state" if (dl / "state").exists() else dl
            src = next(iter(state.rglob("tokenizer-v2/vocab.json"))).parent
            shutil.copytree(src, tok)
            pre = next(iter(state.rglob("checkpoints/pretrain/best.pt")))
            shutil.copy2(pre, ckpt_out / "best.pt")
            shutil.rmtree(dl, ignore_errors=True)
            log(f"tokenizer + checkpoint staged from Kaggle ({STATE_DS})")
    resolved["tokenizer"] = tok_dir
    if not (tok / "vocab.json").exists():
        raise SystemExit(f"tokenizer still missing under {tok_dir}")

    init_ckpt = find_file(str(ckpt_out / "best.pt"), ["checkpoints", "pretrain", "best.pt"])
    if init_ckpt is None and args_init_from:
        if Path(args_init_from).exists():
            init_ckpt = Path(args_init_from)
    if init_ckpt is None:
        raise SystemExit("pretrained best.pt not found (local, mount, dataset, or --init-from)")
    resolved["init_from"] = str(init_ckpt)
    return resolved


def write_yaml(path: Path, section: str, cfg: dict, extra: dict) -> None:
    import yaml

    payload = {section: {**cfg, **extra}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    log(f"wrote {path}")


def prune_steps(directory: Path, keep: int = 1) -> None:
    steps = sorted(
        directory.glob("step-*.pt"), key=lambda p: int(re.sub(r"\D", "", p.name))
    )
    for stale in steps[:-keep]:
        stale.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["all", "sft", "dpo"], default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts to validate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tok-dir", default="experiments/tokenizer-v2")
    ap.add_argument("--ckpt-dir", default=None,
                    help="base dir for checkpoints (set to Drive path on Colab)")
    ap.add_argument("--init-from", default=None,
                    help="explicit pretrained checkpoint path (skips artifact lookup)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (validation runs only - very slow)")
    ap.add_argument("--out-dir", default="pipeline_output")
    args = ap.parse_args()

    if args.cpu:
        device = "cpu"
        log("CPU mode - validation only, expect slow steps")
    else:
        device = check_gpu()
    repo = sync_repo()

    base_ckpts = Path(args.ckpt_dir) if args.ckpt_dir else Path("experiments/checkpoints")
    sft_dir = base_ckpts / "sft"
    align_dir = base_ckpts / "alignment"
    out_dir = Path(args.out_dir)

    sft_cfg = dict(SFT_CONFIG)
    dpo_cfg = dict(DPO_CONFIG)
    if args.smoke:
        sft_cfg.update(max_steps=30, warmup_steps=5, eval_steps=15, save_steps=30,
                       eval_examples=8, batch_size=2, block_size=256, log_steps=5)
        dpo_cfg.update(max_steps=20, warmup_steps=4, eval_steps=10, save_steps=20,
                       eval_examples=8, batch_size=2, block_size=256, log_steps=5)

    plan = {
        "device": device,
        "repo": str(repo),
        "checkpoints": {"sft": str(sft_dir), "dpo": str(align_dir)},
        "sft_steps": sft_cfg["max_steps"],
        "dpo_steps": dpo_cfg["max_steps"],
        "stages": args.stage,
    }
    log("PLAN " + json.dumps(plan))
    if args.dry_run:
        artifacts = stage_artifacts(args.tok_dir, base_ckpts / "pretrain", args.init_from)
        log("DRY-RUN resolved artifacts: " + json.dumps(artifacts))
        return 0

    sft_best = sft_dir / "best.pt"
    align_best = align_dir / ("final.pt" if align_dir.joinpath("final.pt").exists() else "best.pt")

    if args.stage in ("all", "sft"):
        for d in (sft_dir, base_ckpts / "pretrain"):
            d.mkdir(parents=True, exist_ok=True)
        artifacts = stage_artifacts(args.tok_dir, base_ckpts / "pretrain", args.init_from)

        write_yaml(Path("configs/_pipe_sft.yaml"), "sft", sft_cfg, {
            "tokenizer_dir": artifacts["tokenizer"],
            "init_from": artifacts["init_from"],
            "dataset_path": "experiments/sft/sft-v3-all.jsonl",
            "checkpoint_dir": str(sft_dir),
            "require_clean_repo": False,
            "device": "cpu",
            "amp": True,
            "amp_dtype": "float16",
        })
        assert Path("experiments/sft/sft-v3-all.jsonl").exists(), \
            "sft-v3-all.jsonl missing - is the repo synced?"
        t0 = time.time()
        run([sys.executable, "scripts/train_sft.py", "--config", "configs/_pipe_sft.yaml",
             "--device", device, "--no-clean-repo-check"])
        while True:
            prune_steps(sft_dir)
            if not (sft_dir / "best.pt").exists():
                time.sleep(2)
                continue
            break
        prune_steps(sft_dir)
        log(f"SFT done in {(time.time() - t0) / 60:.1f} min")
    elif not sft_best.exists():
        raise SystemExit("--stage dpo but SFT best.pt missing; run stage sft first")

    if args.stage in ("all", "dpo"):
        align_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(Path("configs/_pipe_dpo.yaml"), "alignment", dpo_cfg, {
            "tokenizer_dir": args.tok_dir,
            "init_from": str(sft_best),
            "dataset_path": "experiments/alignment/dpo-v2-all.jsonl",
            "checkpoint_dir": str(align_dir),
            "require_clean_repo": False,
            "device": "cpu",
            "amp": True,
            "amp_dtype": "float16",
        })
        assert Path("experiments/alignment/dpo-v2-all.jsonl").exists(), \
            "dpo-v2-all.jsonl missing - is the repo synced?"
        t0 = time.time()
        run([sys.executable, "scripts/train_alignment.py", "--config", "configs/_pipe_dpo.yaml",
             "--device", device, "--no-clean-repo-check"])
        prune_steps(align_dir)
        log(f"DPO done in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------ collect outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sft").mkdir(exist_ok=True)
    (out_dir / "alignment").mkdir(exist_ok=True)
    shutil.copy2(sft_best, out_dir / "sft/best.pt")
    final_align = align_dir / "final.pt"
    chosen = final_align if final_align.exists() else align_dir / "best.pt"
    shutil.copy2(chosen, out_dir / "alignment/best.pt")
    for src, dst in [
        (sft_dir / "sft_report.json", out_dir / "sft/sft_report.json"),
        (Path("experiments/alignment/alignment_report.json"),
         out_dir / "alignment/alignment_report.json"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)

    summary = {
        f: f"{(out_dir / rel).stat().st_size / 1e9:.2f} GB"
        for f, rel in [("sft", "sft/best.pt"), ("aligned", "alignment/best.pt")]
    }
    log("PIPELINE COMPLETE " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

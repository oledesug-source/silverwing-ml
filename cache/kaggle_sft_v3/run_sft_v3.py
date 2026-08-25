# Silverwing SFT-v3 + DPO-v2 end-to-end GPU run (Kaggle kernel)
# Stage 1: SFT on the unified lesson track (L01-L16)  ~40 min on T4/P100
# Stage 2: DPO alignment from SFT best                ~45 min
# Outputs: /kaggle/working/out/{sft,alignment}/best.pt + reports

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

print("=== Silverwing unified training kernel ===", flush=True)
os.environ["KAGGLE_USERNAME"] = "videlisndichi"
os.environ["KAGGLE_KEY"] = "KGAT_4252a1b11f1dc8ea8a0d48a8bee2986d"

import torch

assert torch.cuda.is_available(), "NO GPU"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} ({p.total_memory/1e9:.0f} GB)", flush=True)

# ---------------------------------------------------------------- clone repo
REPO = "/kaggle/working/Silverwing-ML"
if not os.path.exists(REPO):
    subprocess.run(
        ["git", "clone", "https://github.com/oledesug-source/silverwing-ml.git", REPO],
        check=True,
    )
os.chdir(REPO)
subprocess.run(["git", "fetch", "origin"], check=True)
subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
head = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()
print("repo at:", head, flush=True)

# ---------------------------------------------------------------- stage data
INPUT = Path("/kaggle/input")


def find(rel_parts):
    hits = list(INPUT.rglob("/".join(rel_parts)))
    return str(hits[0]) if hits else None


state_best = find(("checkpoints", "pretrain", "best.pt"))
assert state_best, f"pretrain best.pt not found under {INPUT}"
tok_src = find(("tokenizer-v2",))
assert tok_src and (Path(tok_src) / "vocab.json").exists(), "tokenizer not found"

corpus_dst = Path("experiments/corpus-external")
corpus_dst.mkdir(parents=True, exist_ok=True)
shards = [pth for pth in INPUT.rglob("train.*.jsonl") if "tokcache" not in str(pth)]
assert shards, "no corpus shards found"
for f in shards:
    shutil.copy2(f, corpus_dst / f.name)
for pattern in ("*.bin", "*.npy"):
    for cache_file in INPUT.rglob(pattern):
        shutil.copy2(cache_file, corpus_dst / cache_file.name)

TOK = "experiments/tokenizer-v2"
shutil.rmtree(TOK, ignore_errors=True)
shutil.copytree(tok_src, TOK)
print("data staged; corpus dir entries:", len(list(corpus_dst.glob("*"))), flush=True)

# ---------------------------------------------------------------- helpers
def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    print(f"<<< exit={rc} elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    assert rc == 0, f"command failed: {' '.join(str(c) for c in cmd)}"


import yaml

SFT_BEST = "experiments/checkpoints/sft/best.pt"
ALIGN_DIR = "experiments/checkpoints/alignment"

sft_cfg = {"sft": {
    "version": "sft-v3",
    "model_config_path": "configs/model.yaml",
    "tokenizer_dir": TOK,
    "init_from": state_best,
    "dataset_path": "experiments/sft/sft-v3-all.jsonl",
    "checkpoint_dir": "experiments/checkpoints/sft",
    "batch_size": 16, "block_size": 512,
    "max_steps": 1200,
    "warmup_steps": 100,
    "lr": 1.0e-4, "min_lr_ratio": 0.1, "weight_decay": 0.1,
    "betas": [0.9, 0.95], "eps": 1.0e-8, "grad_clip": 1.0,
    "seed": 42, "log_steps": 25, "eval_steps": 200, "eval_examples": 64,
    "save_steps": 500, "eval_fraction": 0.05,
    "require_clean_repo": False, "device": "cpu",
    "amp": True, "amp_dtype": "float16",
}}
with open("configs/sft_production.yaml", "w") as f:
    yaml.safe_dump(sft_cfg, f)

run([sys.executable, "scripts/train_sft.py",
     "--config", "configs/sft_production.yaml",
     "--device", "cuda", "--no-clean-repo-check"])
assert os.path.exists(SFT_BEST), "SFT best.pt missing"

align_cfg = {"alignment": {
    "version": "alignment-v2",
    "model_config_path": "configs/model.yaml",
    "tokenizer_dir": TOK,
    "init_from": SFT_BEST,
    "dataset_path": "experiments/alignment/dpo-v2-all.jsonl",
    "checkpoint_dir": ALIGN_DIR,
    "batch_size": 4, "block_size": 512,
    "max_steps": 800,
    "warmup_steps": 40,
    "lr": 1.0e-5, "min_lr_ratio": 0.1, "weight_decay": 0.0,
    "betas": [0.9, 0.95], "eps": 1.0e-8, "grad_clip": 1.0,
    "dpo_beta": 0.1, "label_smoothing": 0.0,
    "seed": 42, "log_steps": 10, "eval_steps": 100, "eval_examples": 16,
    "save_steps": 200, "eval_fraction": 0.05,
    "require_clean_repo": False, "device": "cpu",
    "amp": True, "amp_dtype": "float16",
}}
with open("configs/alignment_production.yaml", "w") as f:
    yaml.safe_dump(align_cfg, f)

run([sys.executable, "scripts/train_alignment.py",
     "--config", "configs/alignment_production.yaml",
     "--device", "cuda", "--no-clean-repo-check"])

# ---------------------------------------------------------------- outputs
out = Path("/kaggle/working/out")
(out / "sft").mkdir(parents=True, exist_ok=True)
(out / "alignment").mkdir(parents=True, exist_ok=True)
shutil.copy2(SFT_BEST, out / "sft/best.pt")
rep = Path("experiments/checkpoints/sft/sft_report.json")
if rep.exists():
    shutil.copy2(rep, out / "sft/sft_report.json")
align_best = Path(ALIGN_DIR) / "best.pt"
align_final = Path(ALIGN_DIR) / "final.pt"
src = align_best if align_best.exists() else align_final
assert src.exists(), "alignment checkpoint missing"
shutil.copy2(src, out / "alignment/best.pt")
rep = Path("experiments/alignment/alignment_report.json")
if rep.exists():
    shutil.copy2(rep, out / "alignment/alignment_report.json")

print("\n=== PIPELINE COMPLETE ===", flush=True)
print(json.dumps({
    "sft_best_gb": round((out / "sft/best.pt").stat().st_size / 1e9, 2),
    "aligned_best_gb": round((out / "alignment/best.pt").stat().st_size / 1e9, 2),
}, indent=2), flush=True)

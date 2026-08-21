#!/usr/bin/env python3
"""Check available foundation modules and training infrastructure."""
import os

paths = [
    "foundation/model/config.py",
    "foundation/model/model.py",
    "foundation/model/layers.py",
    "foundation/training/config.py",
    "foundation/training/training.py",
    "foundation/training/optimizer.py",
    "foundation/training/checkpoint.py",
    "foundation/training/scheduler.py",
    "foundation/tokenizer/tokenizer.py",
    "foundation/tokenizer/train.py",
    "foundation/inference/generator.py",
    "foundation/curriculum/config.py",
    "foundation/curriculum/trainer.py",
    "foundation/evaluation/evaluator.py",
    "foundation/database/store.py",
    "foundation/sft/trainer.py",
    "foundation/alignment/trainer.py",
    "scripts/train.py",
    "scripts/train_tokenizer.py",
    "scripts/build_corpus.py",
]

for path in paths:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {path} ({size} bytes)")

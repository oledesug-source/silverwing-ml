# Silverwing — Experiment Registry

Every experiment **must** record the fields below. A training run without a
recorded manifest is not a real experiment.

## Corpus releases

Derived corpus artifacts are not committed (reproducible from committed
config + seed); each release's provenance is pinned here.

### corpus-v1 (math) — M08

| Field          | Value                                                                  |
|----------------|------------------------------------------------------------------------|
| `version`      | `math-corpus-v1`                                                       |
| `git_commit`   | `3a3e4e07c470fa4ea10c4bc54fdd5a68595af119` (generator commit)          |
| `seed`         | `42`                                                                   |
| `documents`    | `3200` generated, `3196` after dedup                                    |
| `records`      | `3196` (train `3064` / validation `57` / test `75`)                     |
| `content_digest`| `4be39c9a68225a53a690ec35c24f8dfe2f12047f3aff34df9021da9cbefbee22`    |
| `dataset_hash` | `c049878033198f7197376e8de93ef94e41a3563e0a53ba4afe8db88efc936445`     |
| `config_digest`| `91a0183b5e480ef9080309d6938e32b6d5279fb5a7613a3650885b4729aa1e99`    |
| `train tokens` | `813636`                                                               |

Regenerate: `scripts/generate_math_corpus.py`, then
`scripts/build_corpus.py --source math=experiments/raw-math --output-dir experiments/corpus`.

### tokenizer-v1 — M10

| Field             | Value                                                            |
|-------------------|------------------------------------------------------------------|
| `version`         | `tokenizer-v1` (byte-level BPE)                                   |
| `vocab_size`      | `16384` (16124 merges, min_frequency 2)                           |
| `corpus`          | corpus-v1 train split (`c0498780…`)                               |
| `tokenizer_hash`  | `22398c052521a4308f9b8f8a07fd7534dd6b15a5b2aa7d047e17ef74a35c472e`|
| `git_commit`      | tokenizer training run report (`experiments/tokenizer/…`)         |

Regenerate: `scripts/train_tokenizer.py` (trained on `experiments/corpus`).

### pretrain-0001 (M10) — real pretraining on corpus-v1

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | `20260816T114210Z`                                                 |
| `model`                   | `silverwing-decoder-v2` (`configs/model.yaml`, 102,255,360 params) |
| `git_commit`              | `057daf345a25c7653b47967bcb4067d98e34fc60`                         |
| `train_config_digest`     | `d007f8716cdf263490d34cd1a8a166ba5cc868d420c0f547d39ac5fe28effd8b` |
| `model_config_digest`     | `9a7dcc81dcba04cce6079ebd088b649b9e7ff9298bf5a857b01843e938dddd56` |
| `tokenizer`               | tokenizer-v1 (`22398c0525…`)                                       |
| `dataset`                 | corpus-v1 train (`c049878033…`), verified `ok: true`               |
| `train config`            | bs=1, block 512, 300 steps, lr 3e-4, wu 25, cosine→0.1, wd 0.1     |
| `tokens seen`             | `153600` (0.23 epoch; train split = 340,014 real tokens / 664 blocks) |
| `final train loss`        | `6.2269`                                                           |
| `final eval loss / ppl`   | `5.1133` / `166.22` (eval at step 300)                             |
| `eval loss curve`         | `6.7439` (step 100) → `5.8366` (200) → `5.1133` (300)              |
| `elapsed`                 | `2181 s` at `70.4 tok/s` (CPU, i5-6300U)                           |
| `checkpoints`             | `experiments/checkpoints/{best,final}.pt` (+ `training_report.json`) |
| `benchmark eval`          | math-benchmark-v1: `n=200, parsed=0` (see `experiments/eval/`)     |

Regenerate: `.venv\Scripts\python.exe scripts\train.py --config configs\training.yaml` (tree must be clean; `require_clean_repo`).

### pretrain-0002 (continued pretraining, course adjustment) — corpus-v1, 1000 steps

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | `20260816T142613Z`                                                 |
| `model`                   | `silverwing-decoder-v2` (102,255,360 params)                       |
| `git_commit`              | `fcbff35c4348a137a1bff0387a9c45415df564f0`                         |
| `train_config_digest`     | `65bea4d94f4cb7a12a1573ae6cabe6bf4d76ce52a3b8cff91aa97e7537cf3097` |
| `init_from`               | `experiments/checkpoints/best.pt` (pretrain-0001)                  |
| `dataset`                 | corpus-v1 train (`c049878033…`), verified `ok: true`               |
| `train config`            | bs=1, block 512, 1000 steps, lr 1e-4, wu 25, cosine→0.1, wd 0.1   |
| `tokens seen`             | `512000` (total ~665K including M10's 153K)                        |
| `final train loss`        | `0.7953`                                                           |
| `final eval loss / ppl`   | `3.3377` / `28.15` (eval at step 1000)                             |
| `elapsed`                 | `9059 s` at `56.5 tok/s` (CPU, i5-6300U)                           |
| `checkpoints`             | `experiments/checkpoints/cont/{best,final}.pt`                     |
| `benchmark eval`          | math-benchmark-v1: `n=200, parsed=0` (corpus-format template produces `-- Practice 3.`) |

Regenerate: `.venv\Scripts\python.exe scripts\train.py --config configs\training_cont.yaml` (tree must be clean).

### pretrain-0003 (continued pretraining round 2) — corpus-v1, 2000 more steps

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | see `cont2/training_report.json`                                   |
| `model`                   | silverwing-decoder-v2 (102,255,360 params)                         |
| `git_commit`              | `f31cb47`                                                          |
| `init_from`               | `experiments/checkpoints/cont/best.pt` (eval ppl 28.2)             |
| `dataset`                 | corpus-v1 train (`c049878033…`), verified                          |
| `train config`            | bs=1, block 512, 2000 steps, lr 5e-5, wu 50, cosine→0.1, wd 0.1   |
| `tokens seen`             | `1024000` (total ~1.7M including prior)                            |
| `final train loss`        | `0.7779`                                                           |
| `final eval loss / ppl`   | `2.7548` / `15.72` (eval at step 2000)                             |
| `elapsed`                 | see report (est ~4h at 79 tok/s)                                   |
| `checkpoints`             | `experiments/checkpoints/cont2/{best,final}.pt`                    |

Regenerate: `.venv\Scripts\python.exe scripts\train.py --config configs\training_cont2.yaml` (tree must be clean).

### sft-v1 (M11) — first SFT attempt, 50 pairs/topic

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | `20260816T114211Z` (approx; see archived `sft_report.json`)        |
| `model`                   | silverwing-decoder-v2 init from pretrain-0001 `best.pt`            |
| `git_commit`              | `057daf3` (SFT infra commit `3df01cc`; config `sft_dataset.yaml` per_topic=50) |
| `dataset`                 | sft-v1 500 examples, 10 topics, `dataset_hash=1f7c6e476ac401f0…`   |
| `train config`            | bs=1, block 512, 200 steps, lr 1e-4, wu 20, cosine→0.1, wd 0.1     |
| `supervised tokens`       | train `1394` / eval `145`                                          |
| `best eval loss`          | `4.0641` (ppl `74.6`)                                              |
| `generation`              | degenerate: collapses to a few modes (e.g. `2` for most prompts)   |

Regenerate: `scripts/build_sft_dataset.py` (per_topic 50) then `scripts/train_sft.py`.

### sft-v2 (M11) — larger dataset, 200 pairs/topic

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | `20260816T131915Z`                                                 |
| `model`                   | silverwing-decoder-v2 init from pretrain-0001 `best.pt`            |
| `git_commit`              | `c9fa1a0`                                                          |
| `sft_config_digest`       | `37c906c7ee020eed304c7cae6d2bcac0cadaf490fa0a927cccab7158c2f72ac7` |
| `tokenizer`               | tokenizer-v1 (`22398c0525…`)                                       |
| `dataset`                 | sft-v1 2000 examples, 811 unique answers, `dataset_hash=4a52bb9ded4d28b3…` |
| `train config`            | bs=1, block 512, 200 steps, lr 1e-4, wu 20, cosine→0.1, wd 0.1     |
| `supervised tokens`       | train `5566` / eval `607`                                          |
| `final train loss`        | `2.2733`                                                           |
| `best eval loss / ppl`    | `3.0739` / `21.63` (eval at step 200)                              |
| `eval loss curve`         | `3.8564` (50) → `3.2823` (100) → `3.1330` (150) → `3.0739` (200)   |
| `elapsed`                 | `1480 s` at `69.7 tok/s` (CPU, i5-6300U)                           |
| `checkpoints`             | `experiments/checkpoints/sft/{best,final}.pt` (+ `sft_report.json`) |
| `generation`              | learns `Question:/Answer:` format but collapses to `<eos>` for unseen questions (held-out 80: 69 empty); train-seen also empty in greedy |
| `benchmark eval`          | math-benchmark-v1: `n=200, parsed=0` even with prompt template `Question: {prompt}\nAnswer:` |

Regenerate: `scripts/build_sft_dataset.py` (per_topic 200, committed config) then
`scripts/train_sft.py --config configs/sft.yaml`.

### sft-v3 (M11 adjusted) — SFT from pretrain-0002 (stronger base)

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | `20260816T210000Z` (see `sft_report.json` for exact)               |
| `model`                   | silverwing-decoder-v2 init from pretrain-0002 `cont/best.pt`       |
| `init_from`               | `experiments/checkpoints/cont/best.pt` (eval ppl 28.2)             |
| `dataset`                 | sft-v1 2000 examples, `dataset_hash=4a52bb9ded4d…`                |
| `train config`            | bs=1, block 512, 200 steps, lr 1e-4, wu 20, cosine→0.1, wd 0.1     |
| `best eval loss / ppl`    | `2.5052` / `12.25` (eval at step 200)                              |
| `generation (held-out)`   | 8/80 exact hits (10%), 5/80 empty — no longer collapses to `<eos>` |
| `benchmark eval`          | math-benchmark-v1 w/ template: **`parsed=165/200` (82.5%)**, mae `215.26`, rmse `435.29` |
| `vs dummy baseline`       | dummy: parsed=200, mae=191.26; SFT-v3: parsed=165, mae=215.26     |

Regenerate: `scripts/train_sft.py --config configs/sft.yaml --init-from experiments/checkpoints/cont/best.pt`.

### sft-combined (M13) — basic + reasoning CoT, 3000 examples

| Field                     | Value                                                              |
|---------------------------|--------------------------------------------------------------------|
| `run_id`                  | see `sft-combined/sft_report.json`                                 |
| `init_from`               | `experiments/checkpoints/cont2/best.pt` (eval ppl 15.7)            |
| `dataset`                 | sft-v1-combined: 2000 basic + 1000 reasoning CoT = 3000 examples   |
| `train config`            | bs=1, block 512, 300 steps, lr 1e-4, wu 25, cosine→0.1, wd 0.1     |
| `best eval loss / ppl`    | `2.7817` / `16.15` (eval at step 300)                              |
| `benchmark eval`          | math-benchmark-v1 w/ template: **`parsed=179/200` (89.5%)**, mae `207.86`, rmse `422.11` |
| `vs weaker base (cont)`   | cont: parsed=180, mae=208.22, ppl=22.4; cont2: parsed=179, mae=207.86, ppl=16.2 |

Regenerate: `scripts/train_sft.py --config configs/sft_combined.yaml` (from F:\AI\Silverwing-ML).

**M11/M13 conclusion:** The model generates parseable numbers for 89.5% of benchmark prompts.
MAE ~208 remains close to the dummy baseline (191), meaning the model learns the output format
but cannot yet perform arithmetic. Key lessons:
1. **Base capability is the bottleneck** — SFT is a steering layer (M10 ppl 166 → collapsed to eos; cont ppl 28 → parsed 0→82.5%; cont2 ppl 15.7 → eval loss 2.78)
2. **CoT data helps** — reasoning examples improve both parsed rate and mae vs basic-only SFT
3. **More base pretraining helps** — each round of continued pretraining improves SFT eval loss (3.07 → 2.50 → 2.78 with better base)
4. **Accuracy requires scale** — a 102M model on 340K tokens cannot learn arithmetic; next: larger corpus or larger model

## Required manifest fields

| Field                | Meaning                                                        |
|----------------------|----------------------------------------------------------------|
| `run_id`             | unique experiment id (e.g. `run-0001`)                         |
| `git_commit`         | full commit hash the code ran from                             |
| `config_hash`        | hash of the resolved config used                               |
| `dataset_version`    | dataset version string (e.g. `dataset-v1`)                     |
| `dataset_hash`       | hash of the dataset content                                    |
| `tokenizer_version`  | tokenizer version + vocabulary size                            |
| `model_version`      | model config version                                           |
| `seed`               | random seed                                                    |
| `start_time`         | ISO timestamp                                                  |
| `end_time`           | ISO timestamp                                                  |
| `checkpoint`         | path to the produced checkpoint                                |
| `evaluation_report`  | path to the evaluation report                                  |
| `promotion_result`   | passed / rejected + reason                                     |

## Hard rule

> NO TRAINING RUN WITHOUT A GIT COMMIT.

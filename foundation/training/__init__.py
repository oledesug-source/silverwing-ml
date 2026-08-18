"""Training Engine V2: pretraining loop for Silverwing Decoder V2.

Reproducibility (M01 rule): a run only starts from a committed repository, and
every checkpoint and the training report pin the git commit, model config
digest, tokenizer hash and corpus dataset hash.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TRAINING_VERSION, TrainConfig
from .data import PretrainingData
from .optimizer import build_optimizer
from .preflight import TrainingInputs, preflight_train
from .repo import git_commit, git_is_clean, require_clean_repo
from .scheduler import schedule_lr
from .trainer import BEST_FILENAME, FINAL_FILENAME, evaluate, train

__all__ = [
    "TRAINING_VERSION",
    "TrainConfig",
    "PretrainingData",
    "build_optimizer",
    "TrainingInputs",
    "preflight_train",
    "schedule_lr",
    "git_commit",
    "git_is_clean",
    "require_clean_repo",
    "save_checkpoint",
    "load_checkpoint",
    "BEST_FILENAME",
    "FINAL_FILENAME",
    "evaluate",
    "train",
]

"""Preference alignment via DPO (M12).

Direct Preference Optimization: trains a policy model from the pretrained
checkpoint using human preference pairs (chosen > rejected), with a frozen
reference model for the KL penalty implicit in the DPO objective.
"""

from .config import ALIGNMENT_VERSION, AlignmentConfig
from .dataset import (
    IGNORE_INDEX,
    PreferenceDataset,
    PreferenceExample,
    dataset_hash,
    load_preferences,
)
from .trainer import BEST_FILENAME, FINAL_FILENAME, compute_dpo_loss, train_alignment

__all__ = [
    "ALIGNMENT_VERSION",
    "BEST_FILENAME",
    "FINAL_FILENAME",
    "IGNORE_INDEX",
    "AlignmentConfig",
    "PreferenceDataset",
    "PreferenceExample",
    "compute_dpo_loss",
    "dataset_hash",
    "load_preferences",
    "train_alignment",
]

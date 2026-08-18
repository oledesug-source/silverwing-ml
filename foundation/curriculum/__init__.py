"""Curriculum learning (M14)."""

from .config import CurriculumConfig, StageConfig
from .trainer import train_curriculum

__all__ = ["CurriculumConfig", "StageConfig", "train_curriculum"]

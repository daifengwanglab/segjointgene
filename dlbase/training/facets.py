"""Hydra-driven switches for the standard training stack (defaults: all on)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf

_DEFAULT_TRAIN_FACETS: dict[str, Any] = {
    "clear_experiment_leaf": True,
    "tqdm_epoch_progress": True,
    "epoch_eta_hours": True,
    "learning_rate_monitor": True,
    "model_checkpoint": True,
    "early_stopping": True,
    "lightning_loggers": True,
    "tensorboard_auto_launch_after_fit": True,
    "summary_log": True,
}

_DEFAULT_TASK_FACETS_CLASSIFICATION: dict[str, Any] = {
    "phase_two_extension": True,
    "post_train_eval_and_summary": True,
}

_DEFAULT_TASK_FACETS_DYNAMIC_SEGMENTATION: dict[str, Any] = {
    "post_train_eval_and_summary": True,
}


def merge_train_facets(cfg: DictConfig) -> DictConfig:
    """Merge ``cfg.train.facets`` onto defaults (missing keys stay default-on)."""
    base = OmegaConf.create(deepcopy(_DEFAULT_TRAIN_FACETS))
    custom = cfg.train.get("facets")
    if custom is not None:
        base = OmegaConf.merge(base, custom)
    return base


def merge_classification_task_facets(cfg: DictConfig) -> DictConfig:
    """Merge ``cfg.task.facets`` for classification (only when ``task.name==classification``)."""
    base = OmegaConf.create(deepcopy(_DEFAULT_TASK_FACETS_CLASSIFICATION))
    custom = cfg.task.get("facets") if OmegaConf.is_config(cfg.task) else None
    if custom is not None:
        base = OmegaConf.merge(base, custom)
    return base


def merge_dynamic_segmentation_task_facets(cfg: DictConfig) -> DictConfig:
    """Merge ``cfg.task.facets`` for ``dynamic_segmentation`` / ``dynamic_segmentation_CID``."""
    base = OmegaConf.create(deepcopy(_DEFAULT_TASK_FACETS_DYNAMIC_SEGMENTATION))
    custom = cfg.task.get("facets") if OmegaConf.is_config(cfg.task) else None
    if custom is not None:
        base = OmegaConf.merge(base, custom)
    return base

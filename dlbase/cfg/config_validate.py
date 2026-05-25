"""Lightweight Hydra config checks before training (no full schema migration)."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from dlbase.registry import get_dataset_plugin, get_model_plugin, get_task_plugin


def validate_hydra_cfg(cfg: DictConfig) -> None:
    """Fail fast on missing top-level groups, unknown registry keys, and bad scalars."""
    required = ("dataset", "model", "train", "task", "paths", "experiment", "seed")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    if not OmegaConf.is_config(cfg.dataset) or "name" not in cfg.dataset:
        raise ValueError("cfg.dataset must be a config node with ``name``")
    if not OmegaConf.is_config(cfg.model) or "name" not in cfg.model:
        raise ValueError("cfg.model must be a config node with ``name``")
    if not OmegaConf.is_config(cfg.task) or "name" not in cfg.task:
        raise ValueError("cfg.task must be a config node with ``name``")

    ds_name = str(cfg.dataset.name)
    model_name = str(cfg.model.name)
    task_name = str(cfg.task.name)

    get_dataset_plugin(ds_name)
    get_model_plugin(model_name)
    get_task_plugin(task_name)

    if "max_epochs" not in cfg.train:
        raise ValueError("cfg.train.max_epochs is required")
    me = cfg.train.max_epochs
    try:
        me_int = int(me)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cfg.train.max_epochs must be an integer, got {me!r}") from exc
    if me_int < 1:
        raise ValueError(f"cfg.train.max_epochs must be >= 1, got {me_int}")

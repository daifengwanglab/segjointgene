"""Dispatch ``cfg.task.name`` to a concrete task package."""

from __future__ import annotations

from omegaconf import DictConfig


def run_task(cfg: DictConfig) -> None:
    from dlbase.registry import get_task_plugin

    get_task_plugin(str(cfg.task.name)).task_class().run(cfg)

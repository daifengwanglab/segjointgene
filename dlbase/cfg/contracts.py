"""Generic cfg-level assertions (no hard-coded dataset / model / task names)."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig


def assert_task_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    t = str(cfg.task.name)
    if t not in allowed:
        raise ValueError(f"{context}: unsupported task.name={t!r} (allowed: {sorted(allowed)})")


def assert_dataset_name_in(cfg: DictConfig, allowed: FrozenSet[str], context: str) -> None:
    ds = str(cfg.dataset.name)
    if ds not in allowed:
        raise ValueError(f"{context}: unsupported dataset.name={ds!r}")

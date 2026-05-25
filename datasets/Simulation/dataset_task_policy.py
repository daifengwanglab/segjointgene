"""Per-task dataset policy for Simulation (see README §3.2)."""

from __future__ import annotations

from omegaconf import DictConfig


def assert_task_supported(cfg: DictConfig) -> None:
    """Raise if ``cfg.task.name`` is not supported for this dataset."""
    task = str(cfg.task.name)
    if task in ("dynamic_segmentation", "dynamic_segmentation_CID", "genesegnet"):
        return
    raise ValueError(f"Simulation: unsupported task.name={task!r}.")

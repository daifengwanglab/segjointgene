"""task × dataset compatibility for ``unet`` (see README §3.2)."""

from __future__ import annotations

from omegaconf import DictConfig


def assert_compatible(cfg: DictConfig) -> None:
    task = str(cfg.task.name)
    ds = str(cfg.dataset.name)
    if task not in ("dynamic_segmentation", "dynamic_segmentation_CID"):
        raise ValueError(f"unet: unsupported task.name={task!r} (only 'dynamic_segmentation' / 'dynamic_segmentation_CID' are wired).")
    if ds != "Simulation":
        raise ValueError(
            f"unet (standalone): unsupported dataset.name={ds!r} (expected 'Simulation')."
        )

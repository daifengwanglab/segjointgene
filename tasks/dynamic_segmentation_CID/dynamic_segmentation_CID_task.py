"""TaskModule wrapper for dynamic segmentation with CID attribution."""

from __future__ import annotations

from omegaconf import DictConfig

from dlbase.registry import TaskModule
from tasks.dynamic_segmentation_CID.run_CID import run_dynamic_segmentation_CID_training


class DynamicSegmentationCIDTask(TaskModule):
    @property
    def name(self) -> str:
        return "dynamic_segmentation_CID"

    def run(self, cfg: DictConfig) -> None:
        run_dynamic_segmentation_CID_training(cfg)

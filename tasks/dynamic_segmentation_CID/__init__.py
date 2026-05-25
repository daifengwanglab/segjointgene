"""Dynamic segmentation with CID attribution (extends dynamic_segmentation)."""

from tasks.dynamic_segmentation_CID.dynamic_segmentation_CID_task import DynamicSegmentationCIDTask
from tasks.dynamic_segmentation_CID.run_CID import run_dynamic_segmentation_CID_training

__all__ = ["DynamicSegmentationCIDTask", "run_dynamic_segmentation_CID_training"]

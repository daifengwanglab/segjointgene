"""Register dynamic_segmentation_CID task (standalone Simulation + CID package)."""

from __future__ import annotations

from dlbase.registry import register_task


def _register_builtin_tasks() -> None:
    from tasks.dynamic_segmentation_CID import DynamicSegmentationCIDTask

    register_task("dynamic_segmentation_CID", DynamicSegmentationCIDTask)


_register_builtin_tasks()

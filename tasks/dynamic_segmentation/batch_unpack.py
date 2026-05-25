"""Unpack dataloader batches: Simulation adds ``ground_truth_instance`` as 14th element."""

from __future__ import annotations


def unpack_seg_batch(batch_data):
    """Support Simulation 14-tuple (``ground_truth_instance``) vs 13-tuple for other datasets."""
    bl = list(batch_data)
    if len(bl) == 14:
        gt_inst = bl.pop()
        return tuple(bl), gt_inst
    return tuple(bl), None

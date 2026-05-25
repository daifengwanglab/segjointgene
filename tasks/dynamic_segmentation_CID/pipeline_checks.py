"""
Pre-training checks for dynamic_segmentation_CID.

Reuses the dataset/model gates and shape dry-run from dynamic_segmentation,
accepting both ``dynamic_segmentation`` and ``dynamic_segmentation_CID``
as valid task names.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from models.build import build_model
from tasks.dynamic_segmentation.pipeline_checks import (
    _invoke_dataset_task_check,
    _invoke_model_task_check,
)


def _get_output(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _cid_shape_dry_run(cfg: DictConfig) -> None:
    net = build_model(cfg)
    ps = int(cfg.dynamic_seg.patch_size)
    c_in = int(cfg.model.in_channels)
    nc = int(cfg.model.num_classes)
    B = 2
    x = torch.zeros(B, c_in, ps, ps)
    out = net(x)
    logits = _get_output(out)
    if logits.dim() != 4 or logits.shape[0] != B or logits.shape[1] != nc:
        raise RuntimeError(
            f"dynamic_segmentation_CID dry-run: expected logits [B={B}, {nc}, H, W], "
            f"got {tuple(logits.shape)}"
        )
    if logits.shape[-2:] != (ps, ps):
        raise RuntimeError(
            f"dynamic_segmentation_CID dry-run: expected spatial size ({ps}, {ps}), "
            f"got {tuple(logits.shape[-2:])}"
        )


def run_final_pre_training_checks(cfg: DictConfig) -> None:
    _invoke_dataset_task_check(cfg)
    _invoke_model_task_check(cfg)
    _cid_shape_dry_run(cfg)

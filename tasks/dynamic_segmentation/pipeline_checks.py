"""
Run before training: dataset task gate, model task×dataset gate, and a forward shape dry-run.
Standalone package: Simulation + unet + dynamic_segmentation_CID only.
"""

from __future__ import annotations

from typing import Union

import torch
from omegaconf import DictConfig

from models.build import build_model


def _get_output(outputs: Union[torch.Tensor, tuple]) -> torch.Tensor:
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _invoke_dataset_task_check(cfg: DictConfig) -> None:
    name = str(cfg.dataset.name)
    if name != "Simulation":
        raise ValueError(f"Standalone package supports dataset.name='Simulation' only, got {name!r}")
    from datasets.Simulation.dataset_task_policy import assert_task_supported

    assert_task_supported(cfg)


def _invoke_model_task_check(cfg: DictConfig) -> None:
    name = str(cfg.model.name)
    if name != "unet":
        raise ValueError(f"Standalone package supports model.name='unet' only, got {name!r}")
    from models.unet.model_task_dataset_compatibility import assert_compatible

    assert_compatible(cfg)


def _dynamic_segmentation_shape_dry_run(cfg: DictConfig) -> None:
    if str(cfg.task.name) != "dynamic_segmentation_CID":
        return

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
            f"dynamic_segmentation pipeline_checks: expected logits [B={B}, {nc}, H, W], "
            f"got {tuple(logits.shape)}"
        )
    if logits.shape[-2:] != (ps, ps):
        raise RuntimeError(
            f"dynamic_segmentation pipeline_checks: expected spatial size ({ps}, {ps}), "
            f"got {tuple(logits.shape[-2:])}"
        )


def run_final_pre_training_checks(cfg: DictConfig) -> None:
    """Call after seed, before DataModule heavy use / training loop."""
    _invoke_dataset_task_check(cfg)
    _invoke_model_task_check(cfg)
    _dynamic_segmentation_shape_dry_run(cfg)

"""Simulation Lightning datamodule wiring (registered via ``datasets.plugins``)."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from datasets.Simulation import ImagePatchDataset


def setup_simulation_datamodule(dm: Any, cfg: DictConfig, stage: Any = None) -> None:
    ds = cfg.dataset
    root = ds.data_path
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    dm.train_set = ImagePatchDataset(npz_dir=train_dir)
    dm.val_set = ImagePatchDataset(npz_dir=val_dir)
    dm.test_set = ImagePatchDataset(npz_dir=test_dir)


def simulation_eval_split_note(dm: Any) -> str:
    return (
        "note: Simulation NPZ — train/val/test come from separate folders after preprocess (~7:1:2 random patch split)."
    )

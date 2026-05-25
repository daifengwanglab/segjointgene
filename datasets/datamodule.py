"""Lightning DataModule for NPZ patch folders (train/ / test/).

Supports datasets: WMB, CA1, Simulation via registry (``datasets.plugins``).
"""

from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dlbase.registry import get_dataset_plugin


class DLBaseDataModule(pl.LightningDataModule):
    """Builds train/val/test loaders from Hydra ``dataset`` config (registry-driven setup)."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        name = str(self.cfg.dataset.name)
        plugin = get_dataset_plugin(name)
        plugin.assert_task_supported(self.cfg)
        plugin.setup(self, self.cfg, stage)

    def train_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.train_set,
            batch_size=ds.batch_size,
            shuffle=True,
            num_workers=ds.num_workers,
            pin_memory=ds.pin_memory,
        )

    def val_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.val_set,
            batch_size=ds.batch_size,
            shuffle=False,
            num_workers=ds.num_workers,
            pin_memory=ds.pin_memory,
        )

    def test_dataloader(self):
        ds = self.cfg.dataset
        return DataLoader(
            self.test_set,
            batch_size=ds.batch_size,
            shuffle=False,
            num_workers=ds.num_workers,
            pin_memory=ds.pin_memory,
        )

    def eval_split_note(self) -> str:
        plugin = get_dataset_plugin(str(self.cfg.dataset.name))
        return plugin.eval_split_note(self)

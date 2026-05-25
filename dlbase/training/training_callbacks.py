"""Training UX callbacks: ETA in hours, val loss history for two-phase schedule."""

from __future__ import annotations

import time
from typing import Any, List, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from typing_extensions import override


def val_still_strictly_decreasing(
    vals: List[float], min_delta: float, window: int = 3
) -> bool:
    """True if the last ``window`` validation losses strictly decrease epoch-to-epoch."""
    if len(vals) < window:
        return False
    tail = vals[-window:]
    for i in range(1, window):
        if min_delta <= 0.0:
            if not (tail[i] < tail[i - 1]):
                return False
        else:
            if not (tail[i] < tail[i - 1] - min_delta):
                return False
    return True


class ValLossHistoryAndPhaseLrCallback(pl.Callback):
    """Records val/loss each epoch (skips sanity check); captures LR at end of fit."""

    def __init__(self) -> None:
        self.val_losses: List[float] = []
        self.phase1_end_lr: Optional[float] = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val/loss")
        if v is None:
            return
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu().item())
        else:
            v = float(v)
        self.val_losses.append(v)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.optimizers:
            return
        lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        if self.phase1_end_lr is None:
            self.phase1_end_lr = lr


class EpochEtaHoursCallback(pl.Callback):
    """After each epoch, print estimated wall time (hours) to reach ``target_epochs`` for this run."""

    def __init__(self, target_epochs: int) -> None:
        self.target_epochs = max(1, int(target_epochs))
        self._t0: Optional[float] = None
        self._epochs_done = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t0 = time.perf_counter()
        self._epochs_done = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._epochs_done += 1
        if self._t0 is None:
            return
        elapsed = time.perf_counter() - self._t0
        if self._epochs_done <= 0:
            return
        est_total_s = elapsed / self._epochs_done * self.target_epochs
        est_h = est_total_s / 3600.0
        rank_zero_info(f"[ETA] {self._epochs_done}/{self.target_epochs} ~{est_h:.2f}h")


def _fmt_pb_float(v: Any) -> str:
    if hasattr(v, "detach"):
        v = float(v.detach().cpu().item())
    else:
        v = float(v)
    return f"{v:.3f}"


class CompactTQDMProgressBar(TQDMProgressBar):
    """Short bar line: compact postfix (L/a/vL/va), no logger v_num, no it/s in the bar."""

    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"

    @override
    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, Any]:
        items = dict(trainer.progress_bar_metrics)
        out: dict[str, Any] = {}
        # Train: epoch running mean (train/*_bar); val: epoch aggregates (official Lightning defaults).
        tl = items.get("train/loss_bar", items.get("train/loss"))
        ta = items.get("train/acc_bar", items.get("train/acc"))
        if tl is not None:
            out["L"] = _fmt_pb_float(tl)
        if ta is not None:
            out["a"] = _fmt_pb_float(ta)
        if "val/loss" in items:
            out["vL"] = _fmt_pb_float(items["val/loss"])
        if "val/acc" in items:
            out["va"] = _fmt_pb_float(items["val/acc"])
        return out

    @property
    @override
    def sanity_check_description(self) -> str:
        return "san"

    @property
    @override
    def validation_description(self) -> str:
        return "val"

    @property
    @override
    def test_description(self) -> str:
        return "test"

    @property
    @override
    def predict_description(self) -> str:
        return "pred"

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any) -> None:
        super().on_train_epoch_start(trainer, *_)
        self.train_progress_bar.set_description(f"E{trainer.current_epoch}")

    @override
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        if self.val_progress_bar is None or self.val_progress_bar.disable:
            return
        tag = "san" if trainer.sanity_checking else "val"
        self.val_progress_bar.set_description(f"{tag}:{dataloader_idx}" if dataloader_idx else tag)


def tqdm_epoch_progress_bar(cfg: DictConfig) -> TQDMProgressBar:
    """Per-epoch batch progress; postfix refresh every N batches (calmer when N>1)."""
    r = max(1, int(cfg.train.get("progress_bar_refresh_rate", 20)))
    return CompactTQDMProgressBar(refresh_rate=r)

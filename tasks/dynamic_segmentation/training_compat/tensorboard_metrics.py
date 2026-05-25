"""TensorBoard scalar logging for manual training loops (DLBase-style train/*, val/* names)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc, assignment]


class TensorBoardEpochLogger:
    """
    Writes one scalar step per epoch under log_dir.

    Uses names like train/loss, val/loss, train/ce, val/ce aligned with PyTorch Lightning conventions.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if SummaryWriter is None:
            raise ImportError("tensorboard package required; pip install tensorboard")
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, float(value), epoch)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> "TensorBoardEpochLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

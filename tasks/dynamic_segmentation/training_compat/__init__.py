"""Manual-loop training infrastructure: checkpoints, TensorBoard, epoch aggregation (not Lightning)."""

from .checkpoint_io import load_checkpoint, resolve_checkpoint_path, save_checkpoint
from .epoch_aggregates import EpochBatchAggregator, StatHistory
from .experiment_paths import tensorboard_log_dir
from .tensorboard_metrics import TensorBoardEpochLogger

__all__ = [
    "EpochBatchAggregator",
    "StatHistory",
    "TensorBoardEpochLogger",
    "load_checkpoint",
    "resolve_checkpoint_path",
    "save_checkpoint",
    "tensorboard_log_dir",
]

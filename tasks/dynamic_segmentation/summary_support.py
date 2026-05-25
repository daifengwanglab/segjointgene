"""Orchestrate ``summary.log`` for dynamic segmentation (aligned with DLBase ``tasks/classification/run.py``)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from dlbase.summary_log import build_eval_blocks_with_adapter, write_summary_log
from dlbase.training.facets import merge_dynamic_segmentation_task_facets, merge_train_facets
from tasks.dynamic_segmentation.post_train_summary import (
    dynamic_segmentation_stat_splits,
    last_epoch_metrics_line,
)
from tasks.dynamic_segmentation.training_compat.epoch_aggregates import StatHistory


def _count_parameters_m(module: torch.nn.Module) -> str:
    n = sum(p.numel() for p in module.parameters())
    return f"{n / 1e6:.2f}M"


def maybe_write_summary_log(
    *,
    cfg: DictConfig,
    dm: Any,
    stat_history: StatHistory,
    net: torch.nn.Module,
    train_start: datetime,
    train_end: datetime,
    start_epoch: int,
    best_epoch: int,
    best_test_loss: float,
    args: Any,
    num_classes: int,
) -> None:
    """Write canonical ``summary.log`` when train/task facets allow (see DLBase classification task)."""
    train_f = merge_train_facets(cfg)
    task_f = merge_dynamic_segmentation_task_facets(cfg)

    if train_f.summary_log and task_f.post_train_eval_and_summary:
        last_epoch_id = int(cfg.train.max_epochs)
        idx_best = best_epoch - start_epoch
        idx_last = last_epoch_id - start_epoch

        best_path = os.path.join(os.getcwd(), "best.ckpt")
        if os.path.isfile(best_path):
            eval_blocks = build_eval_blocks_with_adapter(
                task_name=str(cfg.task.name),
                adapter_name="dynamic_segmentation_stat_splits",
                adapter=dynamic_segmentation_stat_splits,
                adapter_args=(stat_history, idx_best, idx_last),
            )
        else:
            eval_blocks = "(evaluation skipped: no checkpoint file on disk)"

        best_ckpt = best_path if os.path.isfile(best_path) else "(none)"

        run_details: dict[str, Any] = {
            "input": f"n_gene={int(args.n_gene)} patch_size={int(args.patch_size)}",
            "output": f"num_classes={num_classes} (logits channels)",
            "optimization_loss": "HybridLoss (CE + Dice)",
            "num_parameters": _count_parameters_m(net),
            "manual_training_loop": "true",
            "epoch_range": f"{start_epoch}..{last_epoch_id} (inclusive)",
            "best_epoch": str(best_epoch),
            "best_test_loss_val_total": f"{best_test_loss:.6f}",
            "predict_epoch": f"{int(getattr(args, 'predict_epoch', 0))} (epochs <= this: NPZ clone; after: update_label)",
            "prediction_threshold": (
                f"{float(getattr(args, 'prediction_threshold', 0.5)):.4f} "
                "(update_label conf gate; training stdout: [Stats] Pred/Anchors/Δcls/Δinst per epoch batch0)"
            ),
        }
        last_line = last_epoch_metrics_line(stat_history, idx_last)
        if last_line:
            run_details["last_epoch_metrics"] = last_line

        write_summary_log(
            os.path.join(os.getcwd(), "summary.log"),
            cfg,
            train_start=train_start,
            train_end=train_end,
            run_details=run_details,
            best_ckpt_path=best_ckpt,
            eval_note=dm.eval_split_note(),
            eval_blocks=eval_blocks,
        )
        rank_zero_info(f"Wrote summary.log to {os.path.join(os.getcwd(), 'summary.log')}")
    elif not train_f.summary_log or not task_f.post_train_eval_and_summary:
        rank_zero_info(
            "summary.log skipped (train.facets.summary_log or task.facets.post_train_eval_and_summary is false)."
        )

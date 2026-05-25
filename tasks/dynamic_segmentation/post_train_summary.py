"""Post-training ``summary.log`` adapter payload for dynamic segmentation (``dlbase.summary_log``).

Mirrors ``tasks/classification/post_train_eval.classification_eval_splits``: split names must be
``train`` / ``val`` / ``test`` (``val`` only when distinct val metrics exist in stat history).
"""

from __future__ import annotations

from typing import Any, Mapping

from tasks.dynamic_segmentation.training_compat.epoch_aggregates import StatHistory


def _row_flat(idx: int, stat_history: StatHistory) -> dict[str, Any]:
    if idx < 0:
        return {}
    out: dict[str, Any] = {}
    for k, series in stat_history.stat.items():
        if idx >= len(series):
            continue
        out[str(k)] = float(series[idx])
    return out


def _assign_stat_key_to_split(key: str) -> tuple[str | None, str]:
    """Map a per-epoch stat key to (split, short_metric_name). Unmapped keys return (None, key)."""
    k = str(key)
    if k.startswith("train_"):
        return ("train", k[6:])
    if k.startswith("val_"):
        return ("val", k[4:])
    if k.startswith("test_"):
        return ("test", k[5:])
    if k.endswith("_train"):
        return ("train", k[: -len("_train")])
    if k.endswith("_val"):
        return ("val", k[: -len("_val")])
    if k.endswith("_test"):
        return ("test", k[: -len("_test")])
    return (None, k)


def _bucket_at_idx(idx: int, stat_history: StatHistory) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    train_d: dict[str, float] = {}
    val_d: dict[str, float] = {}
    test_d: dict[str, float] = {}
    if idx < 0:
        return train_d, val_d, test_d
    for k, series in stat_history.stat.items():
        if idx >= len(series):
            continue
        v = float(series[idx])
        split, metric = _assign_stat_key_to_split(k)
        if split == "train":
            train_d[metric] = v
        elif split == "val":
            val_d[metric] = v
        elif split == "test":
            test_d[metric] = v
    return train_d, val_d, test_d


def last_epoch_metrics_line(stat_history: StatHistory, idx_last: int, *, max_keys: int = 32) -> str:
    """Compact line for ``run_details`` (eval table uses best-epoch train/val/test columns)."""
    row = _row_flat(idx_last, stat_history)
    if not row:
        return ""
    parts: list[str] = []
    # Prefer ``cell_calling`` (often last in append order) so summary.log includes it when capped.
    ordered = list(row.items())
    if "cell_calling" in row:
        ordered = [("cell_calling", row["cell_calling"])] + [(k, v) for k, v in ordered if k != "cell_calling"]
    for k, v in ordered[:max_keys]:
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def dynamic_segmentation_stat_splits(
    stat_history: StatHistory,
    idx_best: int,
    idx_last: int,
) -> Mapping[str, Mapping[str, Any]]:
    """Task-level eval payload for ``dlbase.summary_log.build_eval_blocks_with_adapter``."""
    del idx_last  # last-epoch snapshot is passed to ``run_details`` via ``last_epoch_metrics_line``.
    train_d, val_d, test_d = _bucket_at_idx(idx_best, stat_history)
    out: dict[str, dict[str, Any]] = {}
    out["train"] = train_d if train_d else {"notes": "no_bucketed_train_metrics"}
    if val_d:
        out["val"] = val_d
    out["test"] = test_d if test_d else {"notes": "no_bucketed_test_metrics"}
    return out

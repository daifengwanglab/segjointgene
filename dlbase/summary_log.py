"""Task-agnostic helpers for writing ``summary.log``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from omegaconf import DictConfig


FormatterMap = Mapping[str, Callable[[Any], str]]
SummaryAdapter = Callable[..., Mapping[str, Mapping[str, Any]]]

# Adapters must use DLBase split names only: train / val / test (val optional).
_ALLOWED_SUMMARY_SPLITS = frozenset({"train", "val", "test"})


def _normalize_metric_keys_per_split(
    split_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Drop redundant ``{split}_`` prefix from metric keys inside each split bucket
    (e.g. ``train`` split: ``train_loss`` -> ``loss``) so row labels stay short.
    """
    out: dict[str, dict[str, Any]] = {}
    for split, metrics in split_metrics.items():
        prefix = f"{split}_"
        inner: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(k, str) and k.startswith(prefix):
                nk = k[len(prefix) :]
                inner[nk if nk else k] = v
            else:
                inner[k] = v
        out[split] = inner
    return out


def build_eval_blocks_with_adapter(
    *,
    task_name: str,
    adapter_name: str,
    adapter: Optional[SummaryAdapter],
    adapter_args: tuple[Any, ...] = (),
    split_order: Optional[Iterable[str]] = None,
    metric_order: Optional[Iterable[str]] = None,
    formatters: Optional[FormatterMap] = None,
) -> str:
    """
    Runtime-enforced entrypoint for task summary adapters.

    Raises:
        RuntimeError: if adapter is missing or returns malformed payload.

    Split names are restricted to ``train``, ``val``, and ``test``; ``train`` and
    ``test`` are required, ``val`` is optional (no separate validation set).
    """
    if adapter is None or not callable(adapter):
        raise RuntimeError(
            f"Task '{task_name}' must provide a callable summary adapter "
            f"('{adapter_name}') before writing summary.log."
        )
    split_metrics = adapter(*adapter_args)
    _validate_split_metrics_payload(task_name, adapter_name, split_metrics)
    return format_split_metrics_table(
        split_metrics,
        split_order=split_order,
        metric_order=metric_order,
        formatters=formatters,
    )


def format_split_metrics_table(
    split_metrics: Mapping[str, Mapping[str, Any]],
    *,
    split_order: Optional[Iterable[str]] = None,
    metric_order: Optional[Iterable[str]] = None,
    formatters: Optional[FormatterMap] = None,
) -> str:
    """
    Build one wide ASCII table: metrics x splits.

    - ``split_metrics`` example:
      ``{"train": {"loss": 0.1, "top1": 0.9}, "val": {...}, "test": {...}}``
    - By default, all discovered metrics are shown.
    - ``formatters`` allows per-metric pretty-print (e.g. percentages).
    - Inside each split, leading ``{split}_`` on metric keys is stripped for display
      (e.g. ``train_loss`` under ``train`` becomes row ``loss``). ``metric_order`` and
      ``formatters`` keys use these **display** names (after stripping).
    """
    split_metrics = _normalize_metric_keys_per_split(split_metrics)
    fmt_map = dict(formatters or {})
    splits = list(split_order) if split_order is not None else list(split_metrics.keys())
    if not splits:
        return "(no evaluation splits)"

    metrics_seen: List[str] = []
    if metric_order is not None:
        metrics_seen.extend(list(metric_order))
    for split in splits:
        for metric in split_metrics.get(split, {}).keys():
            if metric not in metrics_seen:
                metrics_seen.append(metric)
    if not metrics_seen:
        return "(no evaluation metrics)"

    header = ("metric", *splits)
    body_rows: List[List[str]] = []
    for metric in metrics_seen:
        row = [metric]
        for split in splits:
            value = split_metrics.get(split, {}).get(metric, "-")
            if metric in fmt_map and value != "-":
                row.append(fmt_map[metric](value))
            else:
                row.append(_default_metric_text(value))
        body_rows.append(row)

    rows = [list(header), *body_rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    lines = [f"[ {' | '.join(splits)} ]", sep, _format_row(rows[0], widths), sep]
    for row in body_rows:
        lines.append(_format_row(row, widths))
        lines.append(sep)
    return "\n".join(lines)


def write_summary_log(
    path: str,
    cfg: DictConfig,
    *,
    train_start: datetime,
    train_end: datetime,
    run_details: Mapping[str, Any],
    best_ckpt_path: str,
    eval_note: str,
    eval_blocks: str,
) -> None:
    """Write canonical summary text for any task."""
    lines: List[str] = [
        "=== Experiment ===",
        f"dataset: {cfg.dataset.name}",
        f"net (model): {cfg.model.name}",
        f"task: {cfg.task.name}",
        f"sub_path (exp_run_leaf): {cfg.exp_run_leaf}",
        f"experiment.sub_name: {cfg.experiment.sub_name}",
        f"random_seed: {cfg.seed}",
        "",
        "=== Training window (local time, minute precision) ===",
        f"train_start: {train_start.strftime('%Y-%m-%d %H:%M')}",
        f"train_end:   {train_end.strftime('%Y-%m-%d %H:%M')}",
        "",
        "=== Run configuration (training) ===",
    ]
    for k, v in run_details.items():
        lines.append(f"{k}: {v}")
    lines.extend(
        [
            "",
            "=== Final evaluation (model.eval, best checkpoint) ===",
            f"best_checkpoint: {best_ckpt_path}",
            eval_note,
            "",
            eval_blocks,
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _format_row(cols: List[str], widths: List[int]) -> str:
    padded = [f" {col:<{widths[i]}} " for i, col in enumerate(cols)]
    return "|" + "|".join(padded) + "|"


def _default_metric_text(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _validate_split_metrics_payload(
    task_name: str,
    adapter_name: str,
    split_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(split_metrics, Mapping) or not split_metrics:
        raise RuntimeError(
            f"Task '{task_name}' adapter '{adapter_name}' must return a non-empty "
            "mapping: split -> metric -> value."
        )
    for split, metrics in split_metrics.items():
        if not isinstance(split, str) or not split:
            raise RuntimeError(
                f"Task '{task_name}' adapter '{adapter_name}' returned invalid split name: {split!r}."
            )
        if not isinstance(metrics, Mapping):
            raise RuntimeError(
                f"Task '{task_name}' adapter '{adapter_name}' split '{split}' "
                "must map metric names to values."
            )
        for metric_name in metrics.keys():
            if not isinstance(metric_name, str) or not metric_name:
                raise RuntimeError(
                    f"Task '{task_name}' adapter '{adapter_name}' split '{split}' contains "
                    f"invalid metric key: {metric_name!r}."
                )
    for split in split_metrics.keys():
        if split not in _ALLOWED_SUMMARY_SPLITS:
            raise RuntimeError(
                f"Task '{task_name}' adapter '{adapter_name}' must only use split names "
                f"{sorted(_ALLOWED_SUMMARY_SPLITS)}; got {split!r}."
            )
    if "train" not in split_metrics:
        raise RuntimeError(
            f"Task '{task_name}' adapter '{adapter_name}' must include split 'train'."
        )
    if "test" not in split_metrics:
        raise RuntimeError(
            f"Task '{task_name}' adapter '{adapter_name}' must include split 'test'."
        )

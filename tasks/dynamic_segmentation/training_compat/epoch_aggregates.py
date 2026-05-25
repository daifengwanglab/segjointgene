"""Per-epoch batch-weighted metric aggregation and full-training stat history (replaces Logger.stat lists)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional


class RunningWeightedAverage:
    def __init__(self) -> None:
        self._sum = 0.0
        self._weight = 0.0

    def reset(self) -> None:
        self._sum = 0.0
        self._weight = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self._sum += float(value) * int(n)
        self._weight += int(n)

    @property
    def avg(self) -> float:
        if self._weight <= 0:
            return 0.0
        return self._sum / self._weight


class EpochBatchAggregator:
    """Weighted batch averages within one epoch (matches old Logger + AverageMeter behavior)."""

    def __init__(self) -> None:
        self._meters: Dict[str, RunningWeightedAverage] = {}

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def _get(self, key: str) -> RunningWeightedAverage:
        if key not in self._meters:
            self._meters[key] = RunningWeightedAverage()
        return self._meters[key]

    def update(self, key: str, value: float, n: int = 1) -> None:
        self._get(key).update(value, n)

    def averages(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self._meters.items()}


class StatHistory:
    """Ordered per-epoch lists (``stat_history.stat``) for printing and TensorBoard; optional checkpoint extra."""

    def __init__(self) -> None:
        self.stat: "OrderedDict[str, List[float]]" = OrderedDict()

    def reset(self) -> None:
        """Clear all series (e.g. fresh run)."""
        self.stat.clear()

    def append_epoch(self, values: Dict[str, float], epoch_index: int) -> None:
        """Append one value per key for this epoch (epoch_index unused; kept for API clarity)."""
        for k, v in values.items():
            if k not in self.stat:
                self.stat[k] = []
            self.stat[k].append(float(v))

    def stat_check(self) -> None:
        """Drop-in compatibility with Logger.stat_check (no-op validation)."""
        pass

    def to_checkpoint_extra(self) -> Dict[str, Any]:
        return {"stat_history": {k: list(v) for k, v in self.stat.items()}}

    @classmethod
    def from_checkpoint_extra(cls, extra: Optional[Dict[str, Any]]) -> "StatHistory":
        h = cls()
        if not extra or "stat_history" not in extra:
            return h
        raw = extra["stat_history"]
        if isinstance(raw, dict):
            for k, vals in raw.items():
                h.stat[k] = [float(x) for x in vals]
        return h

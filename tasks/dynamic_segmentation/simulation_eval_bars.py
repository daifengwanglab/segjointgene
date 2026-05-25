"""Bar order and labels for Simulation eval_a / eval_b (aligned with preprocess condition names)."""

from __future__ import annotations

# Internal keys match ``spots_{name}.csv`` / ``CONDITION_GRID`` in ``datasets.preprocess.Simulation``.
# Top-to-bottom display order after ``invert_yaxis()`` on barh (index 0 drawn at bottom, then inverted).
SIMULATION_BAR_CONDITION_KEYS: tuple[str, ...] = (
    "HighNoise_SparseCells",
    "LowNoise_SparseCells",
    "HighNoise_DenseCells",
    "LowNoise_DenseCells",
)

SIMULATION_BAR_LABELS: tuple[str, ...] = (
    "Sparse Cell + High Noise",
    "Sparse Cell + Low Noise",
    "Dense Cell + High Noise",
    "Dense Cell + Low Noise",
)

# Two-line legend text (same order as ``SIMULATION_BAR_LABELS`` / bars, top → bottom after ``invert_yaxis``).
SIMULATION_BAR_LEGEND_LINES: tuple[str, ...] = (
    "Sparse Cell\nHigh Noise",
    "Sparse Cell\nLow Noise",
    "Dense Cell\nHigh Noise",
    "Dense Cell\nLow Noise",
)

_BAR_GRID: tuple[tuple[int, int], ...] = (
    (1, 1),
    (0, 1),
    (1, 0),
    (0, 0),
)


def simulation_bar_index_from_gr_gc(group_row: int, group_col: int) -> int | None:
    """Return bar slot 0..3 for a patch ``(group_row, group_col)``, or ``None`` if unknown."""
    try:
        return _BAR_GRID.index((int(group_row), int(group_col)))
    except ValueError:
        return None

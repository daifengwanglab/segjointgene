"""Register Simulation dataset (standalone Simulation + CID package)."""

from __future__ import annotations

from dlbase.registry import register_dataset

from datasets.Simulation.datamodule_setup import (
    simulation_eval_split_note,
    setup_simulation_datamodule,
)
from datasets.Simulation.dataset_task_policy import assert_task_supported as simulation_assert_task_supported


def _register_builtin_datasets() -> None:
    register_dataset(
        "Simulation",
        setup_simulation_datamodule,
        simulation_assert_task_supported,
        simulation_eval_split_note,
    )


_register_builtin_datasets()

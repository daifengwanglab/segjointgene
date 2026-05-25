"""Run Simulation preprocess before training when required by config or empty outputs."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from datasets.preprocess.gate import (
    Simulation_both_output_dirs_empty,
    should_run_Simulation_preprocess,
)
from tasks.dynamic_segmentation.config_utils import hydra_cfg_to_preprocess_args


def maybe_run_dataset_preprocess(cfg: DictConfig) -> None:
    """Dispatch preprocess for Simulation when ``dataset.rerun_preprocess`` or outputs are empty."""
    name = str(cfg.dataset.name)
    if name != "Simulation":
        raise ValueError(f"Standalone package supports dataset.name='Simulation' only, got {name!r}")

    data_dir = str(cfg.paths.data_dir)
    rerun = bool(OmegaConf.select(cfg, "dataset.rerun_preprocess", default=False))
    pargs = hydra_cfg_to_preprocess_args(cfg)

    if not should_run_Simulation_preprocess(data_dir, rerun):
        rank_zero_info(
            "[preprocess Simulation] skipped: `Simulation` / `Simulation_image` already populated and "
            "dataset.rerun_preprocess=false"
        )
        return
    if Simulation_both_output_dirs_empty(data_dir):
        rank_zero_info("[preprocess Simulation] running (both output dirs empty or missing)")
    else:
        rank_zero_info("[preprocess Simulation] running (dataset.rerun_preprocess=true)")
    from datasets.preprocess.Simulation import step_preprocess_Simulation

    step_preprocess_Simulation(data_dir, pargs)

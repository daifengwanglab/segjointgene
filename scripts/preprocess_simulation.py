#!/usr/bin/env python3
"""Preprocess Simulation raw data only (no training loop)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.preprocess.Simulation import step_preprocess_Simulation
from datasets.preprocess.gate import Simulation_both_output_dirs_empty, should_run_Simulation_preprocess
from tasks.dynamic_segmentation.config_utils import hydra_cfg_to_preprocess_args


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess Simulation dataset into NPZ patches.")
    p.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    p.add_argument("--force", action="store_true", help="Always rerun preprocess")
    args = p.parse_args()

    conf_dir = str(ROOT / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.root={ROOT}",
                f"paths.data_dir={args.data_dir}",
                "dataset.rerun_preprocess=true" if args.force else "dataset.rerun_preprocess=false",
            ],
        )

    data_dir = str(cfg.paths.data_dir)
    rerun = bool(args.force) or bool(OmegaConf.select(cfg, "dataset.rerun_preprocess", default=False))
    if not should_run_Simulation_preprocess(data_dir, rerun):
        print("[preprocess] skipped: Simulation outputs already exist (use --force to rebuild)")
        return
    if Simulation_both_output_dirs_empty(data_dir):
        print("[preprocess] running (output dirs empty or missing)")
    else:
        print("[preprocess] running (forced rerun)")
    pargs = hydra_cfg_to_preprocess_args(cfg)
    print(f"[preprocess] data_dir={data_dir}")
    print(f"[preprocess] patch_size={pargs.patch_size}, n_gene={pargs.n_gene}, n_celltype={pargs.n_celltype}")
    step_preprocess_Simulation(data_dir, pargs)
    print("[preprocess] done")


if __name__ == "__main__":
    os.chdir(str(ROOT))
    main()

"""
CLI entry that resolves project root and forwards to :mod:`dlbase.train` (Hydra).

**SegJointGene-CID standalone — Simulation · UNet · dynamic_segmentation_CID**.

  python3 main.py --sub_path basic

Layout: raw data under ``${paths.data_dir}/Simulation_raw/``; NPZ patches under
``${paths.data_dir}/Simulation/{train,val,test}/`` (default ``paths.data_dir`` = ``./data``).
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from dlbase.runtime_check import verify_training_runtime


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="SegJointGene-CID: Simulation + UNet + CID self-training.")
    p.add_argument("--sub_path", type=str, default="basic", help="experiment.sub_name (Hydra)")
    p.add_argument(
        "--net_epoch",
        type=int,
        default=None,
        help="If set, overrides train.max_epochs.",
    )
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--random_seed", type=int, default=1)
    p.add_argument("--net_lr", type=float, default=1.0e-4)
    p.add_argument("--net_weight_decay", type=float, default=1.0e-2)
    p.add_argument(
        "--rerun_preprocess",
        action="store_true",
        help="Set dataset.rerun_preprocess=true (rebuild dataset NPZ + image dirs when not both empty).",
    )
    return p.parse_known_args()


def main():
    verify_training_runtime(context="main.py")
    args, unknown = parse_args()
    root = _project_root()
    overrides = [
        f"paths.root={root}",
        f"experiment.sub_name={args.sub_path}",
        f"seed={args.random_seed}",
        f"dataset.num_workers={args.num_workers}",
        f"train.optimizer.lr={args.net_lr}",
        f"train.optimizer.weight_decay={args.net_weight_decay}",
    ]
    if args.net_epoch is not None:
        overrides.append(f"train.max_epochs={args.net_epoch}")
    if args.rerun_preprocess:
        overrides.append("dataset.rerun_preprocess=true")

    print("[main] project root:", root)
    print("[main] Hydra overrides:", " ".join(overrides))
    if unknown:
        print("[main] extra Hydra args:", " ".join(unknown))
    os.chdir(str(root))
    sys.argv = ["dlbase.train", *overrides, *unknown]
    runpy.run_module("dlbase.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

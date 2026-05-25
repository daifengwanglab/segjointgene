"""Whether to run Simulation preprocessing before training."""

from __future__ import annotations

import os


def dir_missing_or_empty(path: str) -> bool:
    """True if ``path`` is not a directory or has no entries (empty)."""
    if not os.path.isdir(path):
        return True
    try:
        return len(os.listdir(path)) == 0
    except OSError:
        return True


def _both_output_dirs_empty(data_dir: str, npz_name: str, image_name: str) -> bool:
    d_npz = os.path.join(data_dir, npz_name)
    d_img = os.path.join(data_dir, image_name)
    return dir_missing_or_empty(d_npz) and dir_missing_or_empty(d_img)


def _should_run_preprocess(data_dir: str, npz_name: str, image_name: str, rerun: bool) -> bool:
    if _both_output_dirs_empty(data_dir, npz_name, image_name):
        return True
    return bool(rerun)


def Simulation_both_output_dirs_empty(data_dir: str) -> bool:
    return _both_output_dirs_empty(data_dir, "Simulation", "Simulation_image")


def should_run_Simulation_preprocess(data_dir: str, rerun_preprocess: bool) -> bool:
    return _should_run_preprocess(data_dir, "Simulation", "Simulation_image", rerun_preprocess)

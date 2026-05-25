"""Experiment directory layout helpers (DLBase-style four segments for future task routing)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def tensorboard_log_dir(net_sub_path: str) -> str:
    """TensorBoard event files under the run leaf, matching common DLBase layout."""
    p = os.path.join(net_sub_path, "tensorboard")
    os.makedirs(p, exist_ok=True)
    return p


def semantic_segments(
    args: Any,
    step_suffix: str,
    exp_run_leaf: Optional[str] = None,
) -> Dict[str, str]:
    """
    Map run to dataset / model / task / leaf names for documentation and future Hydra wiring.

    Current SpatialPheno paths use step_suffix where DLBase would use task; exp_run_leaf defaults
    to net_sub_suffix (sub-experiment name).
    """
    leaf = exp_run_leaf if exp_run_leaf is not None else str(getattr(args, "net_sub_suffix", "default"))
    return {
        "dataset": str(getattr(args, "datasets_name", "")),
        "model": str(getattr(args, "net_name", "")),
        "task": step_suffix,
        "exp_run_leaf": leaf,
    }


def path_dict_with_semantics(
    path_dict: Dict[str, str],
    args: Any,
    step_suffix: str,
) -> Dict[str, str]:
    """Non-breaking: add semantic keys without changing existing path_dict entries."""
    out = dict(path_dict)
    seg = semantic_segments(args, step_suffix)
    out["_semantic_dataset"] = seg["dataset"]
    out["_semantic_model"] = seg["model"]
    out["_semantic_task"] = seg["task"]
    out["_semantic_exp_run_leaf"] = seg["exp_run_leaf"]
    return out

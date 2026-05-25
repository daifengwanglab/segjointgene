"""
Global reproducibility policy for DLBase and SpatialPheno.

RNG state is set exclusively via :func:`pytorch_lightning.seed_everything` (Python ``random``,
NumPy, PyTorch, ``PL_GLOBAL_SEED`` / worker env). We then align cuDNN with the former standalone
``setup_seed`` behavior (deterministic kernels; disable benchmark), which Lightning does not set.
"""

from __future__ import annotations

import torch
from pytorch_lightning import seed_everything

__all__ = ["apply_lightning_reproducibility"]


def apply_lightning_reproducibility(seed: int, *, workers: bool = True, verbose: bool = True) -> int:
    """
    Thin wrapper around Lightning's ``seed_everything`` plus deterministic cuDNN flags.

    Returns the integer seed applied (same contract as ``seed_everything``).
    """
    s = seed_everything(int(seed), workers=workers, verbose=verbose)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return s

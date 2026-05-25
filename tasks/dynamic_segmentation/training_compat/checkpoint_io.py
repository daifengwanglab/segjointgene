"""Checkpoint save/load compatible with DLBase-style dict checkpoints; supports legacy list format."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: str,
    epoch: int,
    net: nn.Module,
    optimizer: optim.Optimizer,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a dict checkpoint (no serialized Logger objects).

    Keys: epoch, state_dict, optimizer_state_dict, optional extra (e.g. stat_history dict, args snapshot).
    """
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)


def _load_raw(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(
    path: str,
    net: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    *,
    strict: bool = True,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """
    Load weights (and optimizer if provided) from dict or legacy list checkpoint.

    Legacy format: [state_dict, optimizer_state_dict, logger_object] — logger is ignored.

    Returns (epoch_from_file, extra_dict_or_none). For legacy list checkpoints epoch is None
    (caller should use the requested ckpt_load_epoch).
    """
    raw = _load_raw(path)
    if isinstance(raw, dict) and "state_dict" in raw:
        net.load_state_dict(raw["state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in raw:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
        epoch = int(raw.get("epoch", 0))
        extra = raw.get("extra")
        return epoch, extra if isinstance(extra, dict) else None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        net.load_state_dict(raw[0], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(raw[1])
        return None, None
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def resolve_checkpoint_path(
    net_sub_path: str,
    epoch: int,
    *,
    prefer_new_name: bool = True,
) -> Optional[str]:
    """
    Return path to checkpoint for epoch if it exists.

    Tries epoch_{epoch}.ckpt first, then legacy net_{epoch}.ckpt.
    """
    new_p = os.path.join(net_sub_path, f"epoch_{epoch}.ckpt")
    old_p = os.path.join(net_sub_path, f"net_{epoch}.ckpt")
    if prefer_new_name:
        if os.path.isfile(new_p):
            return new_p
        if os.path.isfile(old_p):
            return old_p
    else:
        if os.path.isfile(old_p):
            return old_p
        if os.path.isfile(new_p):
            return new_p
    return None

"""Warmup + cosine decay (aligned with step_basic.set_lr)."""

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def make_warmup_cosine_lambda(max_epochs: int, warmup_epochs: float, min_lr_ratio: float):
    def lr_lambda(epoch: int) -> float:
        if max_epochs <= warmup_epochs:
            return (epoch + 1) / max(1, max_epochs)
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denom = max_epochs - warmup_epochs - 1
        if denom <= 0:
            return min_lr_ratio
        t = epoch - warmup_epochs
        progress = t / denom
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def build_warmup_cosine(optimizer: Optimizer, max_epochs: int, warmup_epochs: float, min_lr_ratio: float) -> LambdaLR:
    return LambdaLR(optimizer, make_warmup_cosine_lambda(max_epochs, warmup_epochs, min_lr_ratio))
